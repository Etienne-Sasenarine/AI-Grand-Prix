#!/usr/bin/env python3
"""Live telemetry view for a Betaflight FC, over MSP instead of the companion frame.

This is `companion_listener.py` rewritten on top of MSP. Same live table, same rate
graph, same stutter/stop detection -- but every number now comes from stock MSP
commands, so it works against unmodified Betaflight with no custom firmware.

Why the two are not the same shape
----------------------------------
The companion protocol is a *push stream*: the FC emits one 60-byte 0xEE frame per
loop and the listener's whole job is to resync on the sync byte and check a CRC. MSP
is *request/response*: nothing arrives unless we ask. So the listener stops being a
frame decoder and becomes a poller, and the health stats change meaning with it:

    companion                       MSP
    -------------------------       -------------------------------------------
    frames/s seen on the wire       poll cycles/s we managed to complete
    bad_crc_pct (false syncs)       crc_errors (real line corruption only)
    bad_incomplete_pct              timeouts -- the FC never answered
    (n/a)                           rejects -- the FC answered '!' (unsupported)

`bad_crc_pct` on the companion listener is mostly payload bytes that happened to be
0xEE. MSP frames are addressed by command id and only accepted after a checksum, so a
nonzero `crc_errors` here means the *link* is bad, not that the framing is ambiguous.

Field mapping
-------------
    orig_roll/pitch/yaw/throttle -> MSP_RC          (rcmap order, default AETR)
    armed, arming blockers       -> MSP_STATUS
    vbat, current, mah, rssi     -> MSP_ANALOG
    attitude                     -> MSP_ATTITUDE    (degrees, not the 1e-4 rad field)
    gyro / accel                 -> MSP_RAW_IMU     (raw FC units, not m/s^2)
    gps                          -> MSP_RAW_GPS
    altitude                     -> MSP_ALTITUDE    (estimator, not the raw +1000 field)
    companion_trpy               -> what *we* transmit, echoed back through MSP_RC
    ai_flight_mode / targeting   -> no stock MSP equivalent; shown as n/a

Polling cost is not uniform, so the loop splits into two groups: attitude / IMU / RC
every cycle, and status / analog / GPS / altitude every Nth (`--slow-divisor`). That
keeps the fast fields at the highest rate the link allows instead of dragging them
down to the rate of the slowest one. The loop is capped by `--poll-rate` (50 Hz by
default): unlike a push stream, polling as fast as possible floods the FC's MSP handler
and competes with the RC stream for the same port.

Transmit behaviour matches the companion listener: by default it streams RC with all
four axes ramping 1000 -> 2000, and `--listen-only` is the opt-out. Two MSP-specific
caveats on that ramp. Betaflight blocks arming while an MSP client is connected, so the
FC will normally refuse to arm unless something clears that lock (`msp.py` exposes
`set_arming_disabled`, which this tool never calls). And AUX1 is part of the RC frame we
send, held low, so streaming actively holds the FC disarmed rather than merely not
arming it. Neither is a substitute for taking the props off.

Usage, safest first:

    python3 src/companion_listener_msp.py --port /dev/ttyTHS1 --listen-only
    python3 src/companion_listener_msp.py --port /dev/ttyTHS1 --pin-throttle
    python3 src/companion_listener_msp.py --port /dev/ttyTHS1              # ramps throttle

Requires: pyserial and rich.
"""

import argparse
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# pyserial / rich / msp are needed to *run* the live view but not to use the pure
# helpers below. Import them lazily so importers on a bare machine still work.
Console = Live = Table = None
msp = msp_rc = None


def _load_deps():
    """Import the runtime dependencies, distinguishing the two ways this fails.

    Missing pip packages and a missing sibling module both raise ImportError, but the
    fix is completely different -- installing rich will never conjure up msp.py -- so
    they are reported separately.
    """
    global Console, Live, Table, msp, msp_rc
    try:
        import serial  # noqa: F401 - imported by msp; checked here for a clear message
        from rich.console import Console as _Console
        from rich.live import Live as _Live
        from rich.table import Table as _Table
    except ImportError as exc:
        sys.stderr.write(
            f"companion_listener_msp needs pyserial and rich ({exc}).\n"
            "  sudo apt install python3-serial python3-rich\n"
            "  or: pip3 install pyserial rich\n"
        )
        sys.exit(1)

    try:
        import msp as _msp
        import msp_rc as _msp_rc
    except ImportError as exc:
        sys.stderr.write(
            f"companion_listener_msp needs msp.py and msp_rc.py beside it ({exc}).\n"
            f"  looked in: {_HERE}\n"
            "  This script is part of the repo's src/ directory -- run it in place:\n"
            "      python3 /path/to/CommBetaFlight/src/companion_listener_msp.py\n"
            "  Copying it somewhere on its own will not work.\n"
        )
        sys.exit(1)

    Console, Live, Table = _Console, _Live, _Table
    msp, msp_rc = _msp, _msp_rc


DEFAULT_PORT = "/dev/ttyTHS1"
DEFAULT_BAUD = 115200

STUTTER_THRESHOLD_S = 0.1
STOP_THRESHOLD_S = 1.0

# MSP round-trips are slower than a pushed frame stream, so the graph is scaled to a
# realistic polling rate rather than the companion frame's 200 Hz.
GRAPH_MAX_HZ = 100.0
RATE_WINDOW_S = 1.0

RC_MIN, RC_MID, RC_MAX = 1000, 1500, 2000

# The companion listener stepped every channel by 1 per frame at 100 Hz, so 1000 -> 2000
# took ten seconds. Held here as a duration so the sweep looks the same at any tx rate.
RAMP_PERIOD_S = 10.0


def crc_errors_of(link):
    """Decoder CRC error count, tolerating an msp.py that predates the public property."""
    try:
        return link.crc_errors
    except AttributeError:
        pass
    try:
        return link._decoder.crc_errors
    except AttributeError:
        return 0


def raw_gps_via(link):
    """MSP_RAW_GPS decode, done here so a stale msp.py without raw_gps() still works."""
    fn = getattr(link, "raw_gps", None)
    if fn is not None:
        return fn()
    import struct

    p = link.request(msp.MSP_RAW_GPS)
    if len(p) < 16:
        raise msp.MSPError(f"MSP_RAW_GPS too short ({len(p)} bytes)")
    fix, num_sat, lat, lon, alt_m, speed_cms, course_ddeg = struct.unpack_from("<BBiiHHH", p, 0)
    return {
        "fix": fix,
        "num_sat": num_sat,
        "lat_deg": lat / 1e7,
        "lon_deg": lon / 1e7,
        "altitude_m": float(alt_m),
        "ground_speed_m_s": speed_cms / 100.0,
        "ground_course_deg": course_ddeg / 10.0,
    }


def stale_msp_helpers(link):
    """Names this script expects on MSPLink that an older msp.py does not provide."""
    return [n for n in ("crc_errors", "raw_gps") if not hasattr(link, n)]


def find_ftdi_port(vid=0x0403, pid=0x6001):
    from serial.tools import list_ports

    for port in list_ports.comports():
        if port.vid == vid and port.pid == pid:
            return port.device
    return None


def render_freq_graph(history, max_hz=GRAPH_MAX_HZ, width=50):
    """Sparkline of the last `width` rate samples. Identical to the companion view."""
    if width <= 0:
        return ""
    if len(history) > width:
        history = history[-width:]
    if len(history) < width:
        history = [0.0] * (width - len(history)) + history
    chars = []
    for value in history:
        clamped = max(0.0, min(max_hz, value))
        level = int(round((clamped / max_hz) * 8))
        if level <= 0:
            chars.append(" ")
        elif level <= 2:
            chars.append(".")
        elif level <= 4:
            chars.append(":")
        elif level <= 6:
            chars.append("=")
        else:
            chars.append("#")
    return "".join(chars)


class RateMeter:
    """Rolling rate over a fixed window.

    The companion listener divided a cumulative count by total elapsed time, which
    smears a stall across the whole run and makes the graph almost flat. A window
    shows the rate *now*, which is the number you actually watch for on a live link.
    """

    def __init__(self, window_s=RATE_WINDOW_S):
        self.window_s = window_s
        self._stamps = []

    def tick(self, now):
        self._stamps.append(now)
        self._trim(now)

    def rate(self, now):
        self._trim(now)
        if len(self._stamps) < 2:
            return 0.0
        span = now - self._stamps[0]
        # N stamps span N-1 intervals; dividing by N would over-report the rate.
        return (len(self._stamps) - 1) / span if span > 0 else 0.0

    def _trim(self, now):
        cutoff = now - self.window_s
        while self._stamps and self._stamps[0] < cutoff:
            self._stamps.pop(0)


class TelemetryPoller:
    """Polls a Betaflight FC over MSP and keeps the latest value of every field.

    A field that times out or is rejected keeps its previous value and bumps a counter
    instead of blanking the view or killing the loop -- an FC with no GPS should not
    stop you seeing attitude.
    """

    def __init__(self, link, slow_divisor=10):
        self.link = link
        self.slow_divisor = max(1, slow_divisor)
        self.data = {}
        self.cycles = 0
        self.good_cycles = 0
        self.timeouts = 0
        self.rejects = 0
        self.malformed = 0
        self._cycle = 0

    def _fetch(self, key, fn):
        try:
            self.data[key] = fn()
            return True
        except msp.MSPTimeout:
            self.timeouts += 1
        except msp.MSPError:
            self.rejects += 1
        except Exception:  # short/odd payload from a firmware variant
            self.malformed += 1
        return False

    def poll(self):
        """Run one cycle. Returns True if every fast field answered."""
        self.cycles += 1
        self._cycle += 1

        ok = self._fetch("attitude", self.link.attitude)
        ok &= self._fetch("imu", self.link.raw_imu)
        ok &= self._fetch("rc", self.link.rc_channels)

        if self._cycle % self.slow_divisor == 0:
            self._fetch("status", self.link.status)
            self._fetch("analog", self.link.analog)
            self._fetch("altitude", self.link.altitude)
            self._fetch("gps", lambda: raw_gps_via(self.link))

        if ok:
            self.good_cycles += 1
        return ok


def format_snapshot(
    data,
    poll_hz,
    tx_hz,
    freq_graph,
    cycles,
    pct_good,
    timeouts,
    rejects,
    malformed,
    crc_errors,
    stutter_count,
    stopped,
    since_last_good_s,
    sent_rc,
    tx_mode,
):
    """Build the live table. Mirrors the companion listener's row order."""
    table = Table(show_header=False, box=None)
    table.add_row("poll_hz", f"{poll_hz:.1f}")
    table.add_row("tx_hz", f"{tx_hz:.1f}")
    table.add_row("freq_graph", freq_graph)
    table.add_row("cycles_total", str(cycles))
    table.add_row("good_pct", f"{pct_good:.1f}%")
    # Counted per field, not per cycle -- a cycle asks for several -- so these stay raw
    # counts rather than percentages that could add up past 100.
    table.add_row("field_timeouts", str(timeouts))
    table.add_row("field_rejects", str(rejects))
    table.add_row("field_malformed", str(malformed))
    table.add_row("crc_errors", str(crc_errors))
    table.add_row("stutters", str(stutter_count))
    table.add_row("stopped", "yes" if stopped else "no")
    table.add_row("since_last_good_s", f"{since_last_good_s:.3f}")

    rc = data.get("rc")
    if rc and len(rc) >= 4:
        # rcmap default AETR: 0 roll, 1 pitch, 2 throttle, 3 yaw.
        table.add_row(
            "fc_rc_trpy",
            f"t={rc[2]} r={rc[0]} p={rc[1]} y={rc[3]}",
        )
        if len(rc) > 4:
            table.add_row("fc_rc_aux", " ".join(str(v) for v in rc[4:8]))
    else:
        table.add_row("fc_rc_trpy", "-")

    status = data.get("status")
    if status:
        table.add_row("armed", "yes" if status["armed"] else "no")
        blockers = status["arming_disable_reasons"]
        table.add_row("arming_blocked_by", ", ".join(blockers) if blockers else "-")
        table.add_row("sensors", ", ".join(msp.decode_sensor_flags(status["sensor_flags"])) or "-")
        load = status["system_load_pct"]
        table.add_row(
            "fc_load",
            f"{load}% cycle={status['cycle_time_us']}us" if load is not None
            else f"cycle={status['cycle_time_us']}us",
        )
    else:
        table.add_row("armed", "-")

    analog = data.get("analog")
    if analog:
        table.add_row("vbat", f"{analog['voltage_v']:.2f} V")
        table.add_row("current", f"{analog['current_a']:.2f} A ({analog['mah_drawn']} mAh)")
        table.add_row("rssi", str(analog["rssi"]))
    else:
        table.add_row("vbat", "-")

    att = data.get("attitude")
    if att:
        table.add_row(
            "att_deg", f"roll={att[0]:.1f} pitch={att[1]:.1f} yaw={att[2]:.1f}"
        )
    else:
        table.add_row("att_deg", "-")

    imu = data.get("imu")
    if imu:
        acc, gyro, mag = imu
        table.add_row("gyro_raw", f"roll={gyro[0]} pitch={gyro[1]} yaw={gyro[2]}")
        table.add_row("accel_raw", f"x={acc[0]} y={acc[1]} z={acc[2]}")
        table.add_row("mag_raw", f"x={mag[0]} y={mag[1]} z={mag[2]}")
    else:
        table.add_row("gyro_raw", "-")

    gps = data.get("gps")
    if gps:
        table.add_row("gps", f"lat={gps['lat_deg']:.7f} lon={gps['lon_deg']:.7f}")
        table.add_row(
            "gps_speed",
            f"{gps['ground_speed_m_s']:.1f} m/s course={gps['ground_course_deg']:.1f} deg",
        )
        table.add_row("gps_fix", f"fix={gps['fix']} sats={gps['num_sat']}")
    else:
        table.add_row("gps", "- (no GPS or command unsupported)")

    alt = data.get("altitude")
    if alt:
        table.add_row("altitude", f"{alt[0]:.2f} m vario={alt[1]:.2f} m/s")
    else:
        table.add_row("altitude", "-")

    if sent_rc:
        table.add_row(
            "sent_trpy",
            f"t={sent_rc[2]} r={sent_rc[0]} p={sent_rc[1]} y={sent_rc[3]}  [{tx_mode}]",
        )
    else:
        table.add_row("sent_trpy", f"- [{tx_mode}]")

    table.add_row("ai_modes", "n/a (companion firmware field, no MSP equivalent)")
    return table


def make_setpoint(elapsed, ramp, period=RAMP_PERIOD_S):
    """The transmitted setpoint at time `elapsed`.

    `ramp` reproduces the companion listener's transmitter exactly: throttle, roll,
    pitch and yaw all sweeping 1000 -> 2000 together, wrapping back to 1000. The
    original stepped each channel by 1 per frame at 100 Hz, so one full sweep took
    ten seconds; `period` keeps that timing independent of the frame rate.

    Without `ramp`, throttle stays at zero and only the attitude axes move, so the echo
    through MSP_RC still proves the link end to end without commanding thrust.
    """
    if ramp:
        # Sawtooth over [0, 1), so the sweep peaks one step short of 2000 rather than
        # touching it -- 1999us at the default 100 Hz / 10 s, versus the original's
        # integer ramp which sat on 2000 for exactly one frame.
        phase = (elapsed % period) / period
        return {
            "roll": phase * 2.0 - 1.0,
            "pitch": phase * 2.0 - 1.0,
            "yaw": phase * 2.0 - 1.0,
            "throttle": phase,
        }
    return {
        "roll": 0.2 * math.sin(elapsed * 1.0),
        "pitch": 0.2 * math.sin(elapsed * 0.7),
        "yaw": 0.2 * math.sin(elapsed * 0.4),
        "throttle": 0.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Live Betaflight telemetry over MSP (companion_listener rewritten)."
    )
    parser.add_argument("--port", help=f"serial device (default: {DEFAULT_PORT}, or FT232 auto-detect)")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help=f"baud rate (default: {DEFAULT_BAUD})")
    parser.add_argument("--v2", action="store_true", help="send MSPv2 ($X<) instead of MSPv1")
    parser.add_argument(
        "--msp-timeout", type=float, default=0.08,
        help="per-request deadline in seconds (default: 0.08; keep it short so one dead "
             "command cannot stall the view)",
    )
    parser.add_argument(
        "--poll-rate", type=float, default=50.0,
        help="cap the poll loop at this many cycles/s (default: 50; 0 = as fast as the "
             "link allows). Uncapped polling floods the FC's MSP handler and competes "
             "with the RC stream for the same port.",
    )
    parser.add_argument(
        "--slow-divisor", type=int, default=10,
        help="poll status/analog/gps/altitude every Nth cycle (default: 10)",
    )
    parser.add_argument(
        "--listen-only", action="store_true",
        help="decode only; do not transmit RC (safe with a live FC)",
    )
    parser.add_argument(
        "--pin-throttle", action="store_true",
        help="transmit, but hold throttle at minimum and move only roll/pitch/yaw. "
             "Proves the link through the MSP_RC echo without commanding thrust.",
    )
    parser.add_argument(
        "--tx-rate", type=float, default=100.0, help="RC stream rate in Hz (default: 100)"
    )
    parser.add_argument(
        "--ramp-period", type=float, default=RAMP_PERIOD_S,
        help=f"seconds for one 1000->2000 sweep (default: {RAMP_PERIOD_S:.0f})",
    )
    parser.add_argument(
        "--max-throttle", type=float, default=1.0,
        help="hard throttle ceiling for the ramp, 0..1 (default: 1.0, the full range "
             "the companion listener used)",
    )
    args = parser.parse_args()

    _load_deps()

    if args.listen_only and args.pin_throttle:
        sys.stderr.write("--pin-throttle transmits; it cannot be combined with --listen-only.\n")
        sys.exit(2)
    if not 0.0 <= args.max_throttle <= 1.0:
        sys.stderr.write("--max-throttle must be between 0 and 1.\n")
        sys.exit(2)

    port = args.port or find_ftdi_port() or DEFAULT_PORT

    console = Console()
    console.print(f"Opening {port} @ {args.baud} baud (MSP{'v2' if args.v2 else 'v1'})")
    if args.listen_only:
        console.print("[green]listen-only[/green]: polling telemetry, transmitting no RC")
    elif args.pin_throttle:
        console.print(
            "[yellow]TRANSMITTING[/yellow]: streaming MSP_SET_RAW_RC at "
            f"{args.tx_rate:.0f} Hz. Throttle pinned to minimum; watch roll, pitch and "
            "yaw echo back in fc_rc_trpy."
        )
    else:
        ceiling = "" if args.max_throttle >= 1.0 else f" (throttle capped at {args.max_throttle:.0%})"
        console.print(
            "[bold red]TRANSMITTING[/bold red]: this sends MSP_SET_RAW_RC frames at "
            f"{args.tx_rate:.0f} Hz with throttle, roll, pitch and yaw ramping "
            f"1000->2000 continuously{ceiling}.\n"
            "         If the FC acts on them and is armed, [bold]motors will run up to "
            "full[/bold].\n"
            "         Props off. Use --listen-only to decode without transmitting."
        )

    tx_mode = (
        "listen-only" if args.listen_only else ("pin-throttle" if args.pin_throttle else "ramp")
    )

    link = msp.MSPLink(port, args.baud, timeout=args.msp_timeout, use_v2=args.v2)
    tx = None
    try:
        link.open()
    except Exception as exc:  # noqa: BLE001 - the port name is the useful part
        sys.stderr.write(f"could not open {port}: {exc}\n")
        sys.exit(2)

    try:
        # Identify once, before the live view takes over the screen. The live view runs
        # on a deliberately short deadline; give the handshake a normal one so a slow or
        # still-booting FC is not written off as absent.
        identify_timeout, link.timeout = link.timeout, max(0.5, args.msp_timeout)
        try:
            console.print(
                f"FC: {link.fc_variant()} {link.fc_version()}  "
                f"api={'.'.join(str(v) for v in link.api_version())}"
            )
        except (msp.MSPTimeout, msp.MSPError) as exc:
            console.print(f"[yellow]could not identify FC ({exc}); continuing[/yellow]")
        finally:
            link.timeout = identify_timeout

        stale = stale_msp_helpers(link)
        if stale:
            console.print(
                f"[yellow]msp.py is out of date[/yellow] (no {', '.join(stale)}); "
                "using built-in fallbacks. Copy the repo's src/msp.py over to clear this."
            )

        if not args.listen_only:
            # The transmitter neutralises the sticks if set_control() goes quiet, and we
            # only call it once per poll cycle -- so its watchdog must outlast a slow
            # poll rate, or a low --poll-rate would look like a dead controller.
            poll_period_s = 1.0 / args.poll_rate if args.poll_rate > 0 else 0.0
            tx = msp_rc.RCTransmitter(
                link,
                rate_hz=args.tx_rate,
                max_throttle=0.0 if args.pin_throttle else args.max_throttle,
                command_timeout_s=max(0.25, 3.0 * poll_period_s),
            ).start()
            tx.wait_until_streaming()

        poller = TelemetryPoller(link, slow_divisor=args.slow_divisor)
        poll_meter = RateMeter()
        tx_meter = RateMeter()

        start_time = time.monotonic()
        last_good_time = None
        since_last_good_s = 0.0
        stutter_count = 0
        stopped = False
        last_render_at = 0.0
        last_tx_frames = 0
        freq_history = []
        poll_period = 1.0 / args.poll_rate if args.poll_rate > 0 else 0.0
        next_poll = time.monotonic()

        with Live(console=console, refresh_per_second=20) as live:
            while True:
                if poll_period:
                    slack = next_poll - time.monotonic()
                    if slack > 0:
                        time.sleep(slack)
                        next_poll += poll_period
                    else:
                        # We fell behind the requested rate; resync instead of bursting.
                        next_poll = time.monotonic() + poll_period

                ok = poller.poll()
                now = time.monotonic()
                poll_meter.tick(now)

                if ok:
                    if last_good_time is not None and now - last_good_time > STUTTER_THRESHOLD_S:
                        stutter_count += 1
                    last_good_time = now
                    since_last_good_s = 0.0
                    if stopped:
                        stopped = False
                        console.print("[green]RESUMED[/green]")
                elif last_good_time is not None:
                    since_last_good_s = now - last_good_time
                    if since_last_good_s > STOP_THRESHOLD_S and not stopped:
                        stopped = True
                        console.print(
                            f"[red]STOP[/red]: no complete poll for {since_last_good_s:.3f}s"
                        )

                sent_rc = None
                if tx is not None:
                    tx.set_control(
                        **make_setpoint(
                            now - start_time, not args.pin_throttle, args.ramp_period
                        )
                    )
                    sent_rc = tx.snapshot()
                    for _ in range(tx.frames_sent - last_tx_frames):
                        tx_meter.tick(now)
                    last_tx_frames = tx.frames_sent

                if now - last_render_at < 0.1:
                    continue
                last_render_at = now

                poll_hz = poll_meter.rate(now)
                freq_history.append(poll_hz)
                if len(freq_history) > 200:
                    del freq_history[:-200]

                cycles = poller.cycles
                pct_good = 100.0 * poller.good_cycles / cycles if cycles else 0.0

                live.update(
                    format_snapshot(
                        poller.data,
                        poll_hz,
                        tx_meter.rate(now),
                        render_freq_graph(freq_history),
                        cycles,
                        pct_good,
                        poller.timeouts,
                        poller.rejects,
                        poller.malformed,
                        crc_errors_of(link),
                        stutter_count,
                        stopped,
                        since_last_good_s,
                        sent_rc,
                        tx_mode,
                    )
                )
    finally:
        if tx is not None:
            tx.close()
        link.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.stdout.write("\nStopped.\n")
