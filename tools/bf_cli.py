"""Betaflight CLI over the MSP serial port, from the Jetson.

DO NOT RUN THIS. Team rule, 19 Sep 2026.
=========================================
Not to write, not to read, not "just a `get`". It wedges this flight
controller: measured twice out of two on 18 Sep, and again on 19 Sep the
aircraft went silent on MSP and unreachable from the transmitter, recovered by
hand. Entering the CLI also sets the `CLI` arming-disable flag, which only a
reboot clears.

Read the flight controller over MSP instead -- `bf_beginner.py --probe` and
`setup_angle_mode.py --show` cover what this was written for. Write over MSP
too: `MSP_SET_MODE_RANGE`, `MSP_PID_ADVANCED`, `MSP_SELECT_SETTING`. That is
the same path Configurator uses, with no reboot and no wedge.

If something genuinely has no MSP route -- `msp_override_channels_mask` is the
known case -- write the exact commands down and hand them to a human to run
from Configurator over the flight controller's OWN USB port. Do not run them
here, and never over the Jetson's UART.

This file is kept for its documented findings and its test double, not as a
tool to reach for.

    python3 bf_cli.py --dev /dev/ttyTHS1 --dump diff_all.txt
    python3 bf_cli.py --dev /dev/ttyTHS1 --cmd "get rc_smoothing"

Why this exists
---------------
Everything we need off the flight controller -- ``diff all`` for the rate and
throttle curves, ``msp_override_channels_mask``, the ``rc_smoothing`` settings --
normally comes from Betaflight Configurator over a USB cable to the FC. On this
setup that means a Configurator install version-matched to 4.4.3, a second
cable, and a USB passthrough fight with WSL. Each of those has failed for
somebody, and a failure costs a session.

The flight controller also accepts CLI over any serial port already speaking
MSP: send ``#`` while **disarmed** and it switches to a line-oriented text mode
and answers with a ``# `` prompt. That port is the one the Jetson is already
wired to, so this route needs no extra cable and no extra software, and it runs
over SSH.

This is not an MSP implementation. It writes ``#`` and then plain text, which is
a different mode of the same port — the software guide's "do not re-implement
MSP framing" is about the binary protocol, and none of that is here.

Safety
------
* **THIS WEDGES THE FLIGHT CONTROLLER. Budget a battery pull.** Measured twice
  out of two on 18 Sep: after a CLI session the board went silent on the UART at
  every baud rate and stayed silent. Waiting did not help; pulling the flight
  battery and plugging it back in did, both times. The ``save`` still took
  effect. **Never run this near a scored run or with a packed flight line**, and
  never assume the aircraft is usable afterwards without checking.
* **Only works while disarmed.** The FC ignores the switch otherwise.
* **It takes the MSP port down** for as long as the session lasts, so nothing
  else can poll telemetry meanwhile. Do not run this during a flight.
* **Leaving CLI requires a reboot.** ``save`` writes the EEPROM and then
  reboots; ``exit`` discards changes and reboots. Either way the FC restarts,
  which takes a few seconds and disconnects anything talking to it.
* **Writes need ``--allow-write``.** Without it this tool refuses any command
  that is not on the read-only list, so a typo cannot change a tune.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROMPT = "# "
#: Commands that cannot change anything. Anything else needs --allow-write.
READ_ONLY_PREFIXES = (
    "diff", "dump", "get", "status", "version", "tasks", "resource",
    "rxrange", "serial", "aux", "adjrange", "feature", "map", "mixer",
    "motor", "mode_color", "led", "beeper", "smix", "help",
)
#: Commands that reboot the board. Flagged so nobody is surprised.
REBOOTS = ("save", "exit", "defaults", "bl", "reboot")


class CLIError(RuntimeError):
    pass


class MockSerial:
    """A stand-in flight controller, so this tool is testable with no hardware.

    Answers the handful of commands we actually use, with realistically-shaped
    output. It is a test double, not a Betaflight emulator.
    """

    CANNED = {
        "diff all": (
            "# diff all\n"
            "# version\n"
            "# Betaflight / STM32H743 (SH74) 4.4.3 Nov 12 2023\n"
            "board_name NONEH743\n"
            "# rateprofile 0\n"
            "set rateprofile_name = RACE\n"
            "set roll_rc_rate = 55\n"
            "set pitch_rc_rate = 55\n"
            "set yaw_rc_rate = 57\n"
            "set roll_srate = 75\n"
            "set pitch_srate = 75\n"
            "set yaw_srate = 70\n"
            "set thr_mid = 54\n"
            "set thr_expo = 68\n"
            "set rates_type = BETAFLIGHT\n"
            "set msp_override_channels_mask = 15\n"
            "set rc_smoothing = ON\n"
            "set rc_smoothing_auto_factor = 30\n"
            "set rc_smoothing_setpoint_cutoff = 0\n"
            "set dshot_bidir = OFF\n"
            "set motor_poles = 14\n"
            "set blackbox_device = SPIFLASH\n"
            "set failsafe_procedure = DROP\n"
        ),
        "version": "# Betaflight / STM32H743 (SH74) 4.4.3 Nov 12 2023\n",
        "status": "MCU H743 Clock=480MHz\nArming disable flags: CLI\n",
        "get rc_smoothing": "rc_smoothing = ON\n",
    }

    #: Settings this double will answer ``get`` for, with stock Betaflight 4.4
    #: style values. Anything not listed answers like an unknown setting, which
    #: is what lets bf_beginner.py's name-discovery be tested offline.
    #:
    #: NOTE: these are a TEST DOUBLE's values, not a measurement of any drone.
    #: The canned ``diff all`` above was written from another team's published
    #: dump, and at least once it has been mistaken for our own airframe's
    #: configuration -- see the CONFLICT note in SPEC.md. Never quote numbers
    #: from this file as measurements.
    SETTINGS = {
        # Names and values as probed off the real 4.4.3 board on 18 Sep, so
        # offline runs behave the way the hardware does.
        "level_limit": "55", "angle_level_strength": "50",
        "horizon_level_strength": "50", "horizon_transition": "75",
        "thr_mid": "50", "thr_expo": "0",
        "roll_rc_rate": "100", "pitch_rc_rate": "100", "yaw_rc_rate": "100",
        "roll_srate": "70", "pitch_srate": "70", "yaw_srate": "70",
        "roll_expo": "0", "pitch_expo": "0", "yaw_expo": "0",
        "rates_type": "ACTUAL",
        "throttle_limit_type": "OFF", "throttle_limit_percent": "100",
        "rc_smoothing": "ON", "rc_smoothing_auto_factor": "30",
        "failsafe_procedure": "DROP", "small_angle": "25",
        "crash_recovery": "OFF", "acro_trainer_angle_limit": "20",
    }

    def __init__(self) -> None:
        self.buf = b""
        self.in_cli = False
        self.rebooted = False
        #: Every write the double accepted, so a tool that claims to touch only
        #: certain profiles can be checked rather than believed.
        self.writes = []
        self.saved = False

    def write(self, data: bytes) -> int:
        text = data.decode("utf-8", "replace")
        if not self.in_cli:
            if "#" in text:
                self.in_cli = True
                self.buf += b"\r\nEntering CLI Mode, type 'exit' to return, or 'help'\r\n# "
            return len(data)
        for line in text.replace("\r", "\n").split("\n"):
            cmd = line.strip()
            if not cmd:
                continue
            if cmd in REBOOTS:
                if cmd.lower() == "save":
                    self.saved = True
                self.buf += b"\r\nRebooting\r\n"
                self.rebooted = True
                self.in_cli = False
                continue
            body = self.CANNED.get(cmd)
            if body is None and cmd.lower().startswith("get "):
                key = cmd[4:].strip()
                val = self.SETTINGS.get(key)
                body = (f"{key} = {val}\n" if val is not None
                        else f"###ERROR: Invalid name\n")
            if body is None and cmd.lower().startswith("set "):
                key = cmd[4:].split("=", 1)[0].strip()
                if key in self.SETTINGS:
                    val = cmd.split("=", 1)[1].strip()
                    self.SETTINGS[key] = val
                    self.writes.append(cmd)
                    body = f"{key} set to {val}\n"
                else:
                    body = "###ERROR: Invalid name\n"
            if body is None and cmd.lower().split()[0] in ("profile", "rateprofile"):
                self.writes.append(cmd)
                body = f"{cmd}\n"
            if body is None and cmd.lower() == "dump all":
                body = "# dump all\n" + "".join(
                    f"set {k} = {v}\n" for k, v in sorted(self.SETTINGS.items())) \
                    + "\n".join(f"# filler {i}" for i in range(60)) + "\n"
            if body is None:
                body = f"###ERROR: unknown command '{cmd}'\r\n"
            self.buf += body.replace("\n", "\r\n").encode() + PROMPT.encode()
        return len(data)

    def read(self, n: int = 1) -> bytes:
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    @property
    def in_waiting(self) -> int:
        return len(self.buf)

    def close(self) -> None:
        pass


class BetaflightCLI:
    """A CLI session. Use as a context manager so it always leaves cleanly."""

    def __init__(self, dev: str | None, *, baud: int = 115200, timeout: float = 2.0,
                 allow_write: bool = False) -> None:
        self.allow_write = allow_write
        self.timeout = timeout
        self._entered = False
        if dev in (None, "", "mock"):
            self.ser = MockSerial()
            self.is_mock = True
        else:
            import serial                       # pyserial, on the Jetson image
            self.ser = serial.Serial(dev, baud, timeout=0.1)
            self.is_mock = False

    # ------------------------------------------------------------------ plumbing
    def _read_until_prompt(self, timeout: float | None = None) -> str:
        deadline = time.monotonic() + (timeout or self.timeout)
        buf = ""
        while time.monotonic() < deadline:
            chunk = self.ser.read(4096)
            if chunk:
                buf += chunk.decode("utf-8", "replace")
                if buf.rstrip(" ").endswith("#") or buf.endswith(PROMPT):
                    return buf
            else:
                time.sleep(0.01)
        return buf

    def enter(self) -> str:
        """Switch the port into CLI mode. Only works while disarmed."""
        self.ser.write(b"#")
        banner = self._read_until_prompt(timeout=3.0)
        if "CLI" not in banner and "#" not in banner:
            raise CLIError(
                "no CLI prompt. The usual causes, in order: the board is ARMED "
                "(it ignores the switch), the port is not an MSP port, or "
                "something else is holding the serial device.")
        self._entered = True
        return banner

    def command(self, cmd: str, *, timeout: float | None = None) -> str:
        if not self._entered:
            raise CLIError("not in CLI mode; call enter() first")
        head = cmd.strip().split()[0].lower() if cmd.strip() else ""
        if not self.allow_write and head not in READ_ONLY_PREFIXES:
            raise CLIError(
                f"'{cmd}' is not read-only and --allow-write was not given. "
                f"Read-only commands: {', '.join(READ_ONLY_PREFIXES[:8])}, ...")
        self.ser.write((cmd.strip() + "\r\n").encode())
        out = self._read_until_prompt(timeout)
        if "###ERROR" in out:
            raise CLIError(out.strip())
        return out

    def leave(self, *, save: bool = False, settle_s: float = 6.0) -> None:
        """Reboot out of CLI, and wait for the board to come back.

        ``save`` writes the EEPROM first. Either way the flight controller
        reboots, and while it reboots it speaks nothing at all.

        **Why the wait is long and why we drain the port.** An earlier version
        wrote ``exit``, slept 200 ms and closed the serial port. On 18 Sep that
        left the flight controller silent on every baud rate for forty minutes,
        and it took a physical battery pull to recover. In the field that is a
        lost session, or a lost scored run. Closing the port out from under a
        rebooting board is not worth the 5.8 seconds saved, so we now send the
        command, read until it goes quiet, and give it time to come back up.
        """
        if not self._entered:
            return
        try:
            self.ser.write((b"save\r\n" if save else b"exit\r\n"))
            # Drain whatever it says on the way down, rather than yanking the
            # port mid-sentence.
            deadline = time.monotonic() + settle_s
            quiet_since = None
            while time.monotonic() < deadline:
                chunk = self.ser.read(4096)
                if chunk:
                    quiet_since = None
                else:
                    quiet_since = quiet_since or time.monotonic()
                    if time.monotonic() - quiet_since > 2.0:
                        break
                time.sleep(0.05)
        finally:
            self._entered = False

    def verify_msp_returned(self, *, timeout_s: float = 12.0) -> bool:
        """After leaving CLI, check the board is answering MSP again.

        A tool that silently leaves the flight controller wedged is worse than
        one that fails loudly, because the wedge is only discovered later by
        something that matters.
        """
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).resolve().parent))
        deadline = time.monotonic() + timeout_s
        port = getattr(self.ser, "port", None)
        if self.is_mock or not port:
            return True
        try:
            self.ser.close()
        except Exception:
            pass
        while time.monotonic() < deadline:
            try:
                from mock_link import open_link      # noqa: E402
                fc = open_link(port)
                try:
                    fc.attitude()
                    return True
                finally:
                    fc.close()
            except Exception:
                time.sleep(1.0)
        return False

    def close(self) -> None:
        try:
            self.leave(save=False)
        finally:
            self.ser.close()

    def __enter__(self):
        self.enter()
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def parse_settings(dump: str) -> dict:
    """Pull ``set x = y`` pairs out of a diff/dump into a dict."""
    out = {}
    for line in dump.splitlines():
        line = line.strip()
        if line.startswith("set ") and "=" in line:
            k, v = line[4:].split("=", 1)
            out[k.strip()] = v.strip()
    return out


#: Settings worth looking at the moment a dump lands, and why.
OF_INTEREST = {
    "roll_rc_rate": "rate curve — feeds betaflight_curves.py",
    "pitch_rc_rate": "rate curve",
    "yaw_rc_rate": "rate curve",
    "roll_srate": "super-rate — the curve is flat in the middle, a linear guess is 44 % wrong",
    "pitch_srate": "super-rate",
    "yaw_srate": "super-rate",
    "rates_type": "which curve family the formulas must match",
    "thr_mid": "throttle curve — turns a stick value into an effective throttle",
    "thr_expo": "throttle curve",
    "msp_override_channels_mask": "must be 15 for all four channels to be overridable",
    "rc_smoothing": "possible ~30 ms of free latency; latency is the top sensitivity",
    "rc_smoothing_auto_factor": "auto-picks a low cutoff when RC arrives slowly over MSP",
    "rc_smoothing_setpoint_cutoff": "0 means auto",
    "dshot_bidir": "ON gives per-motor RPM telemetry — turns thrust into a measurement",
    "motor_poles": "wrong value scales every RPM reading by a constant",
    "blackbox_device": "needed for a trustworthy rate time constant",
    "failsafe_procedure": "what happens when our link dies",
}


def highlight(settings: dict) -> None:
    print()
    print("  Settings that matter to us")
    print("  " + "-" * 72)
    for key, why in OF_INTEREST.items():
        val = settings.get(key, "(not in diff — at default)")
        print(f"  {key:<32} {val:<12} {why}")
    print("  " + "-" * 72)
    mask = settings.get("msp_override_channels_mask")
    if mask is not None and mask != "15":
        print(f"  NOTE: override mask is {mask}, not 15. Not every channel is overridable.")
    if settings.get("dshot_bidir", "OFF").upper() != "ON":
        print("  NOTE: bidirectional DShot is off. Turning it on gives per-motor RPM,")
        print("        which turns the thrust coefficient from an estimate into a")
        print("        measurement. Five minutes, and it needs no extra hardware.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dev", default=None, help="serial device; omit for the mock")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--cmd", action="append", default=[], help="run a command (repeatable)")
    ap.add_argument("--dump", type=Path, help="save 'diff all' to this file")
    ap.add_argument("--allow-write", action="store_true",
                    help="permit commands that can change settings")
    ap.add_argument("--save", action="store_true",
                    help="write the EEPROM on exit (reboots the FC)")
    args = ap.parse_args()

    if args.save and not args.allow_write:
        print("  --save without --allow-write does nothing; refusing.", file=sys.stderr)
        return 2

    cli = BetaflightCLI(args.dev, baud=args.baud, allow_write=args.allow_write)
    if cli.is_mock:
        print("  (mock flight controller — pass --dev /dev/ttyTHS1 for the real one)")
    try:
        cli.enter()
        if args.dump:
            out = cli.command("diff all", timeout=6.0)
            args.dump.parent.mkdir(parents=True, exist_ok=True)
            args.dump.write_text(out)
            print(f"  wrote {len(out.splitlines())} lines to {args.dump}")
            highlight(parse_settings(out))
        for c in args.cmd:
            print(f"\n  $ {c}")
            print(cli.command(c))
    except CLIError as e:
        print(f"  CLI error: {e}", file=sys.stderr)
        return 1
    finally:
        cli.leave(save=args.save)
        cli.ser.close()
        if args.save:
            print("  saved and rebooted. Give the FC a few seconds.")
    return 0


def _self_test() -> int:
    cli = BetaflightCLI(None)
    banner = cli.enter()
    assert "CLI" in banner, banner

    dump = cli.command("diff all")
    settings = parse_settings(dump)
    assert settings["thr_mid"] == "54", settings
    assert settings["rates_type"] == "BETAFLIGHT", settings
    assert settings["msp_override_channels_mask"] == "15", settings

    # A write must be refused without the flag.
    try:
        cli.command("set thr_mid = 50")
        raise AssertionError("a write was allowed without --allow-write")
    except CLIError as e:
        assert "read-only" in str(e), e

    # ...and permitted with it.
    w = BetaflightCLI(None, allow_write=True)
    w.enter()
    try:
        w.command("set thr_mid = 50")
    except CLIError as e:
        assert "unknown command" in str(e)   # the mock does not implement set
    w.close()

    cli.leave(save=False)
    assert cli.ser.rebooted, "leaving CLI must reboot the board"
    cli.ser.close()

    print("bf_cli: all checks passed")
    print(f"  parsed {len(settings)} settings from a diff; writes refused without --allow-write")
    print(f"  dshot_bidir={settings.get('dshot_bidir')} "
          f"rc_smoothing={settings.get('rc_smoothing')} "
          f"blackbox={settings.get('blackbox_device')}")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    raise SystemExit(main())
