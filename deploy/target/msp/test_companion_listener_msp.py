#!/usr/bin/env python3
"""Loopback test for companion_listener_msp.py against a fake Betaflight FC on a PTY.

No hardware required:  python3 src/test_companion_listener_msp.py
Exits non-zero on any failure.

Covers the three things the MSP rewrite actually adds over msp.py: the poller's
fast/slow split and its tolerance of a field the FC refuses, the rolling rate meter,
and the table renderer against a partly-empty snapshot (an FC with no GPS).
"""
import os, pty, struct, sys, threading, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import serial  # noqa: F401
    import rich  # noqa: F401
except ImportError as exc:
    sys.stderr.write(
        f"this test needs pyserial and rich ({exc}).\n"
        "  sudo apt install python3-serial python3-rich\n"
        "  or: pip3 install pyserial rich\n"
    )
    sys.exit(1)

from msp import (
    MSPLink, MSP_API_VERSION, MSP_FC_VARIANT, MSP_FC_VERSION, MSP_BOXIDS, MSP_STATUS,
    MSP_ATTITUDE, MSP_RAW_IMU, MSP_ANALOG, MSP_RC, MSP_RAW_GPS, MSP_ALTITUDE,
    MSP_SET_RAW_RC,
)
import companion_listener_msp as cl

cl._load_deps()

master, slave = pty.openpty()
name = os.ttyname(slave)

# gps_supported=False makes the FC answer '!' to MSP_RAW_GPS, the way a build without
# GPS does. The poller must keep going and count it, not blank the view.
S = {"rc": [1500] * 8, "gps_supported": True, "set_rc_frames": 0, "mute": False}


def reply_v1(cmd, payload=b"", ok=True):
    body = bytes((len(payload), cmd)) + payload
    crc = 0
    for b in body:
        crc ^= b
    return (b"$M>" if ok else b"$M!") + body + bytes((crc,))


def parse_requests(buf):
    out = []
    while True:
        i = buf.find(b"$")
        if i < 0:
            del buf[:]; break
        if i: del buf[:i]
        if len(buf) < 3: break
        if buf[2] != 0x3C: del buf[:1]; continue
        if buf[1] == 0x4D:
            if len(buf) < 5: break
            size, cmd = buf[3], buf[4]; total = 6 + size
            if len(buf) < total: break
            payload = bytes(buf[5:5 + size])
        elif buf[1] == 0x58:
            if len(buf) < 8: break
            _f, cmd, size = struct.unpack_from("<BHH", buf, 3); total = 9 + size
            if len(buf) < total: break
            payload = bytes(buf[8:8 + size])
        else:
            del buf[:1]; continue
        del buf[:total]
        out.append((cmd, payload))
    return out


def fc():
    buf = bytearray()
    while True:
        try:
            data = os.read(master, 4096)
        except OSError:
            return
        buf.extend(data)
        for cmd, payload in parse_requests(buf):
            ok = True
            if   cmd == MSP_API_VERSION: r = bytes([0, 1, 46])
            elif cmd == MSP_FC_VARIANT:  r = b"BTFL"
            elif cmd == MSP_FC_VERSION:  r = bytes([4, 5, 1])
            elif cmd == MSP_BOXIDS:      r = bytes([0, 1, 2, 13])
            elif cmd == MSP_ATTITUDE:    r = struct.pack("<3h", -123, 45, 271)
            elif cmd == MSP_RAW_IMU:     r = struct.pack("<9h", 1, 2, 4096, 5, 6, 7, 8, 9, 10)
            elif cmd == MSP_ANALOG:      r = struct.pack("<BHHhH", 165, 350, 99, 512, 1652)
            elif cmd == MSP_RC:          r = struct.pack("<8H", *S["rc"])
            elif cmd == MSP_ALTITUDE:    r = struct.pack("<ih", 1234, -56)
            elif cmd == MSP_RAW_GPS:
                if S["gps_supported"]:
                    r = struct.pack("<BBiiHHH", 1, 9, 471234567, 82345678, 512, 350, 1801)
                else:
                    r, ok = b"", False
            elif cmd == MSP_STATUS:
                r = (struct.pack("<HHHIB", 312, 0, 0b100011, 0, 0)
                     + struct.pack("<H", 21) + struct.pack("<H", 125)
                     + bytes([0]) + bytes([26]) + struct.pack("<I", 0) + bytes([0]))
            elif cmd == MSP_SET_RAW_RC:
                n = len(payload) // 2
                S["rc"] = list(struct.unpack(f"<{n}H", payload[:n * 2]))[:8]
                S["set_rc_frames"] += 1
                r = b""
            else:
                r = b""
            if S["mute"]:      # simulates an FC that has stopped answering
                continue
            os.write(master, reply_v1(cmd, r, ok))


threading.Thread(target=fc, daemon=True).start()

fails = []


def check(label, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + label + ("  " + str(extra) if extra else ""))
    if not cond:
        fails.append(label)


# ---- pure helpers, no port needed -----------------------------------------------------
check("graph pads to width", len(cl.render_freq_graph([50.0], width=10)) == 10)
check("graph empty history is blank", cl.render_freq_graph([], width=5) == "     ")
check("graph clamps above max", cl.render_freq_graph([1e6], max_hz=100, width=1) == "#")
check("graph floors at zero", cl.render_freq_graph([0.0], width=1) == " ")

m = cl.RateMeter(window_s=1.0)
check("rate meter empty", m.rate(10.0) == 0.0)
for i in range(11):
    m.tick(10.0 + i * 0.1)
check("rate meter reads 10Hz for 0.1s spacing", abs(m.rate(11.0) - 10.0) < 0.01,
      round(m.rate(11.0), 3))
m.tick(30.0)  # long gap: everything older than the window must be dropped
check("rate meter drops stale samples", m.rate(30.0) == 0.0, m.rate(30.0))

sp = cl.make_setpoint(0.0, ramp=False)
check("pinned throttle is zero", sp["throttle"] == 0.0, sp["throttle"])
check("pinned attitude bounded", all(abs(sp[k]) <= 0.2 for k in ("roll", "pitch", "yaw")))
lo = cl.make_setpoint(0.0, ramp=True)
hi = cl.make_setpoint(cl.RAMP_PERIOD_S - 0.001, ramp=True)
check("ramp sweeps throttle 0->1", lo["throttle"] == 0.0 and hi["throttle"] > 0.99,
      (lo["throttle"], round(hi["throttle"], 3)))
check("ramp sweeps roll -1->1", lo["roll"] == -1.0 and hi["roll"] > 0.99)
check("ramp moves all four axes together",
      len({round(hi[k], 6) for k in ("roll", "pitch", "yaw")}) == 1, hi)
check("ramp wraps back to the bottom",
      cl.make_setpoint(cl.RAMP_PERIOD_S, ramp=True)["throttle"] == 0.0)
check("ramp period is honoured",
      abs(cl.make_setpoint(1.0, ramp=True, period=2.0)["throttle"] - 0.5) < 1e-9)
check("default ramp period matches the original's 10 s", cl.RAMP_PERIOD_S == 10.0)
# The sawtooth is [0, 1), so full scale is approached but never touched.
peak = max(cl.make_setpoint(cl.RAMP_PERIOD_S * i / 1000.0, ramp=True)["throttle"]
           for i in range(1000))
check("ramp peaks within one step of full", 0.998 <= peak < 1.0, round(peak, 4))

# ---- poller against the fake FC -------------------------------------------------------
with MSPLink(name, 115200, timeout=0.2) as link:
    check("fc identifies", link.fc_variant() == "BTFL", link.fc_variant())

    gps = link.raw_gps()
    check("raw_gps decodes", (gps["num_sat"], round(gps["lat_deg"], 7),
                              gps["ground_speed_m_s"], gps["ground_course_deg"])
          == (9, 47.1234567, 3.5, 180.1), gps)

    # An older msp.py has neither helper this script added. The listener must degrade,
    # not crash at render time the way it first did on the Jetson.
    class _OldLink:
        """MSPLink as it was before crc_errors / raw_gps were added."""
        def __init__(self, real):
            self.request = real.request
            self._decoder = real._decoder

    old = _OldLink(link)
    check("stale helpers detected", cl.stale_msp_helpers(old) == ["crc_errors", "raw_gps"],
          cl.stale_msp_helpers(old))
    check("stale helpers absent on a current msp.py", cl.stale_msp_helpers(link) == [],
          cl.stale_msp_helpers(link))
    check("raw_gps_via decodes without MSPLink.raw_gps",
          cl.raw_gps_via(old)["num_sat"] == 9, cl.raw_gps_via(old))
    check("crc_errors_of falls back to the private decoder",
          cl.crc_errors_of(old) == link._decoder.crc_errors)
    check("crc_errors_of degrades to 0 on an object with neither",
          cl.crc_errors_of(object()) == 0)
    p = cl.TelemetryPoller(link, slow_divisor=4)
    p.poll()
    check("fast fields on cycle 1", set(p.data) == {"attitude", "imu", "rc"}, sorted(p.data))
    check("slow fields not yet polled", "gps" not in p.data)
    for _ in range(3):
        p.poll()
    check("slow fields on cycle 4",
          {"status", "analog", "altitude", "gps"} <= set(p.data), sorted(p.data))
    check("all cycles good", p.good_cycles == 4 and p.cycles == 4, (p.good_cycles, p.cycles))
    check("no errors on a healthy link",
          (p.timeouts, p.rejects, p.malformed) == (0, 0, 0),
          (p.timeouts, p.rejects, p.malformed))
    check("attitude value", p.data["attitude"] == (-12.3, 4.5, 271.0), p.data["attitude"])
    check("altitude value", p.data["altitude"] == (12.34, -0.56), p.data["altitude"])

    # FC without GPS: rejected field must not break the cycle or the fast fields.
    S["gps_supported"] = False
    before = dict(p.data["gps"])
    for _ in range(4):
        p.poll()
    check("rejected field counted", p.rejects >= 1, p.rejects)
    check("rejected field does not fail the cycle", p.good_cycles == 8, p.good_cycles)
    check("rejected field keeps its last value", p.data["gps"] == before)
    S["gps_supported"] = True

    # Dead link: every field times out, nothing raises, cycle is not 'good'.
    good_before, cycles_before = p.good_cycles, p.cycles
    link.timeout = 0.02
    S["mute"] = True          # FC stops answering
    p.poll()
    check("dead link does not raise", p.cycles == cycles_before + 1)
    check("dead link is not a good cycle", p.good_cycles == good_before, p.good_cycles)
    check("dead link counted as timeouts", p.timeouts >= 3, p.timeouts)
    S["mute"] = False

    check("crc_errors exposed publicly", link.crc_errors == 0, link.crc_errors)

# ---- renderer -------------------------------------------------------------------------
from rich.console import Console


def render(data):
    table = cl.format_snapshot(
        data, poll_hz=42.0, tx_hz=100.0, freq_graph="####", cycles=10, pct_good=90.0,
        timeouts=1, rejects=2, malformed=0, crc_errors=0, stutter_count=3, stopped=False,
        since_last_good_s=0.012, sent_rc=[1600, 1400, 1000, 1500, 1000, 1000, 1000, 1000],
        tx_mode="rc",
    )
    console = Console(file=open(os.devnull, "w"), width=120, record=True)
    console.print(table)
    return console.export_text()


full = render({
    "attitude": (-12.3, 4.5, 271.0),
    "imu": ((1, 2, 4096), (5, 6, 7), (8, 9, 10)),
    "rc": [1500, 1490, 1050, 1510, 1000, 1000, 1000, 1000],
    "status": {"armed": True, "arming_disable_reasons": [], "sensor_flags": 0b100011,
               "system_load_pct": 21, "cycle_time_us": 312},
    "analog": {"voltage_v": 16.52, "current_a": 5.12, "mah_drawn": 350, "rssi": 99},
    "altitude": (12.34, -0.56),
    "gps": {"fix": 1, "num_sat": 9, "lat_deg": 47.1234567, "lon_deg": 8.2345678,
            "altitude_m": 512.0, "ground_speed_m_s": 3.5, "ground_course_deg": 180.1},
})
check("renders throttle from rcmap index 2", "t=1050 r=1500 p=1490 y=1510" in full, )
check("renders armed", "armed" in full and "yes" in full)
check("renders sensors", "GYRO, ACC, BARO" in full)
check("renders vbat", "16.52 V" in full)
check("renders gps", "47.1234567" in full)
check("renders sent setpoint", "t=1000 r=1600 p=1400 y=1500" in full)
check("marks ai modes unavailable", "n/a" in full)

empty = render({})
check("renders an empty snapshot without raising", "poll_hz" in empty)
check("empty snapshot marks missing gps", "no GPS" in empty)

print()
print(f"{'ALL PASS' if not fails else 'FAILURES: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
