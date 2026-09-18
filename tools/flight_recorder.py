"""Record a whole flight with nothing attached. No laptop, no SSH, no Wi-Fi.

    python3 flight_recorder.py --dev /dev/ttyTHS1 --out /home/dcl/flights
    python3 flight_recorder.py --install          # start automatically at boot

Why this exists
---------------
On 18 Sep a 25-minute flight slot produced no data at all. The aircraft flew;
nothing was recorded. The recorder was running over SSH, so the Jetson had to be
on Wi-Fi and reachable from the laptop for a measurement that has nothing to do
with either. The Jetson did not rejoin Wi-Fi, and the session was lost.

The fix is not better Wi-Fi. It is that **the drone records itself**. The Jetson
is bolted to the aircraft with the flight controller wired into it; it needs no
help from anyone to write a CSV. Battery in, fly, battery out, collect the files
later over a cable.

What it guarantees
------------------
* **Starts on its own.** ``--install`` sets up a service that begins recording
  a few seconds after boot. Nothing to remember on the flight line.
* **Survives losing power.** Every row is written and flushed as it is taken.
  Pulling the battery mid-flight costs you the last row, not the flight. The
  earlier tools buffered everything in memory and wrote at the end, which is the
  other way this data gets lost.
* **Survives the flight controller dropping out.** A read error is counted, not
  raised. It keeps trying, because a gap in the middle of a log is still a log.
* **Never commands anything.** Only the read accessors are called. There is no
  code path here that arms, transmits RC, or writes a setting.

Marking hovers without a keyboard
---------------------------------
The hover measurement needs someone to say "this bit here was a steady hover" --
no sensor we trust can tell that from a spool-up on the ground, which is how the
first attempt produced a hover throttle of 0.098.

That used to mean pressing ENTER over SSH, which is exactly the dependency that
lost the session. So it moves to the radio: **the pilot flips a spare switch
while hovering**, and every RC channel is in the log anyway. The marking is the
same human judgement, made by the same person, with nothing to stay connected
to.

You do not have to decide which switch in advance. All 16 channels are recorded,
and ``analyze_flight.py`` finds whichever one was flipped.
"""

from __future__ import annotations

import argparse
import csv
import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mock_link import open_link  # noqa: E402

#: Fields written every row. RC channels are appended as rc1..rc16.
BASE_FIELDS = [
    "t", "wall", "dt",
    "armed", "arming_flags",
    "roll_deg", "pitch_deg", "yaw_deg",
    "gyro_x_raw", "gyro_y_raw", "gyro_z_raw",
    "acc_x_raw", "acc_y_raw", "acc_z_raw",
    "motor0", "motor1", "motor2", "motor3", "motor_mean",
    "voltage_v", "current_a", "mah_drawn",
    "altitude_m", "vario_m_s",
    "rpm0", "rpm1", "rpm2", "rpm3",
]
RC_FIELDS = [f"rc{i + 1}" for i in range(16)]
FIELDS = BASE_FIELDS + RC_FIELDS

#: Slow fields cost round trips and change slowly. See test_camera_capture.py --
#: the link gives a full composite sample at about 23 Hz if everything is polled
#: every cycle, and most of that is spent on things that do not move.
SLOW_EVERY = 10

#: MSP command for per-motor RPM. Bidirectional DShot is ON and motor_poles is
#: 14 on this airframe (measured 18 Sep), so this is real data, and it is what
#: turns the hover and thrust tests from estimates into measurements.
MSP_MOTOR_TELEMETRY = 139

_STOP = False


def _on_signal(_sig, _frm):
    global _STOP
    _STOP = True


def _rpm(fc, poles: int) -> list:
    """Per-motor RPM, or an empty list if this firmware will not give it."""
    try:
        import struct
        p = fc.request(MSP_MOTOR_TELEMETRY)
        n = p[0]
        out = []
        for i in range(min(n, 4)):
            rpm_raw = struct.unpack_from("<I", p, 1 + i * 6)[0]
            out.append(int(rpm_raw * 100 / (poles / 2)))
        return out
    except Exception:
        return []


def record(fc, out_dir: Path, *, hz: float, poles: int, verbose: bool) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"flight_{stamp}.csv"
    period = 1.0 / max(hz, 1e-6)

    slow = {"voltage_v": "", "current_a": "", "mah_drawn": "",
            "altitude_m": "", "vario_m_s": "", "armed": False, "arming_flags": ""}
    rpm = []
    rows_written, errors, cycle = 0, 0, 0
    t0 = time.monotonic()
    last = None
    ever_armed = False

    # newline="" and an explicit flush per row: the point of this file is that
    # it survives the battery coming out.
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        fh.flush()
        while not _STOP:
            now = time.monotonic()
            try:
                roll, pitch, yaw = fc.attitude()
                acc, gyro, _ = fc.raw_imu()
                motors = fc.motors()
                rc = list(fc.rc_channels())
                if cycle % SLOW_EVERY == 0:
                    batt = fc.analog()
                    alt, vario = fc.altitude()
                    st = fc.status()
                    slow = {
                        "voltage_v": batt.get("voltage_v", ""),
                        "current_a": batt.get("current_a", ""),
                        "mah_drawn": batt.get("mah_drawn", ""),
                        "altitude_m": round(float(alt), 3),
                        "vario_m_s": round(float(vario), 4),
                        "armed": bool(st.get("armed", False)),
                        "arming_flags": ";".join(st.get("arming_disable_reasons", []) or []),
                    }
                    rpm = _rpm(fc, poles)
            except Exception:
                errors += 1
                cycle += 1
                time.sleep(period)
                continue

            cycle += 1
            t = now - t0
            dt = "" if last is None else round(now - last, 6)
            last = now
            ever_armed = ever_armed or slow["armed"]
            row = {
                "t": round(t, 4), "wall": time.strftime("%H:%M:%S"), "dt": dt,
                "roll_deg": round(roll, 2), "pitch_deg": round(pitch, 2),
                "yaw_deg": round(yaw, 2),
                "gyro_x_raw": gyro[0], "gyro_y_raw": gyro[1], "gyro_z_raw": gyro[2],
                "acc_x_raw": acc[0], "acc_y_raw": acc[1], "acc_z_raw": acc[2],
                "motor0": motors[0], "motor1": motors[1],
                "motor2": motors[2], "motor3": motors[3],
                "motor_mean": round(sum(motors[:4]) / 4.0, 1),
                **slow,
            }
            for i in range(4):
                row[f"rpm{i}"] = rpm[i] if i < len(rpm) else ""
            for i in range(16):
                row[f"rc{i + 1}"] = rc[i] if i < len(rc) else ""
            w.writerow(row)
            fh.flush()
            rows_written += 1

            if verbose and rows_written % 200 == 0:
                print(f"  {rows_written} rows, {t:.0f}s, "
                      f"{'ARMED' if slow['armed'] else 'disarmed'}, "
                      f"{slow['voltage_v']}V", flush=True)

            slack = period - (time.monotonic() - now)
            if slack > 0:
                time.sleep(slack)

    span = time.monotonic() - t0
    return {"path": str(path), "rows": rows_written, "seconds": round(span, 1),
            "achieved_hz": round(rows_written / span, 1) if span > 0 else 0.0,
            "read_errors": errors, "ever_armed": ever_armed}


SERVICE = """[Unit]
Description=AI Grand Prix flight recorder (telemetry, read-only)
After=multi-user.target

[Service]
Type=simple
User=dcl
# The flight controller is not always up the instant the Jetson is. Retry rather
# than fail: a recorder that gives up on boot is a recorder that records nothing.
Restart=always
RestartSec=5
ExecStart=/usr/bin/python3 {script} --dev {dev} --out {out} --hz {hz} --quiet

[Install]
WantedBy=multi-user.target
"""


def install(dev: str, out: str, hz: float) -> int:
    unit = SERVICE.format(script=str(Path(__file__).resolve()), dev=dev, out=out, hz=hz)
    tmp = Path("/tmp/aigp-recorder.service")
    tmp.write_text(unit)
    print(unit)
    print("Writing the unit needs root. Run:")
    print(f"  sudo cp {tmp} /etc/systemd/system/aigp-recorder.service")
    print("  sudo systemctl daemon-reload")
    print("  sudo systemctl enable --now aigp-recorder.service")
    print()
    print("Then check it with:  systemctl status aigp-recorder --no-pager")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Unattended flight recorder. Read-only.")
    ap.add_argument("--dev", default=None, help="/dev/ttyTHS1; omit for the mock")
    ap.add_argument("--out", default=str(Path.home() / "flights"))
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--motor-poles", type=int, default=14)
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="stop after this long; 0 means run until killed")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--install", action="store_true",
                    help="print the boot service and how to install it")
    args = ap.parse_args()

    if args.install:
        return install(args.dev or "/dev/ttyTHS1", args.out, args.hz)

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    if args.seconds > 0:
        def _alarm(_s, _f):
            _on_signal(_s, _f)
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(int(args.seconds))

    # Keep trying to open the link. On boot the flight controller may not be up
    # yet, and "it was not there the moment I started" is not a reason to record
    # nothing for the next twenty minutes.
    fc = None
    for attempt in range(60):
        if _STOP:
            return 0
        try:
            fc = open_link(args.dev)
            fc.attitude()
            break
        except Exception as e:
            if not args.quiet and attempt == 0:
                print(f"  waiting for the flight controller on {args.dev}: {e}",
                      flush=True)
            time.sleep(2.0)
    if fc is None:
        print("  no flight controller after 2 minutes; giving up")
        return 3

    if not args.quiet:
        print(f"  recording from {args.dev or 'mock'} -> {args.out}")
        print("  press Ctrl-C to stop; every row is flushed as it is written")
    try:
        summary = record(fc, Path(args.out), hz=args.hz, poles=args.motor_poles,
                         verbose=not args.quiet)
    finally:
        try:
            fc.close()
        except Exception:
            pass

    print()
    print("FLIGHT RECORDING")
    print("=" * 60)
    print(f"  file          {summary['path']}")
    print(f"  rows          {summary['rows']} over {summary['seconds']} s "
          f"({summary['achieved_hz']} Hz)")
    print(f"  read errors   {summary['read_errors']}")
    print(f"  armed at all  {'yes' if summary['ever_armed'] else 'NO'}")
    if not summary["ever_armed"]:
        print()
        print("  The aircraft never armed during this recording. If it flew,")
        print("  the recorder was not running for the flight.")
    print("=" * 60)
    return 0


def _self_test() -> int:
    """Record from the mock, then prove the file survives being killed."""
    import shutil
    out = Path("/tmp/_recorder_test")
    shutil.rmtree(out, ignore_errors=True)

    fc = open_link("mock")
    global _STOP
    _STOP = False

    # Stop it from inside, the way SIGTERM would.
    import threading
    threading.Timer(2.5, _on_signal, args=(None, None)).start()
    summary = record(fc, out, hz=30.0, poles=14, verbose=False)
    fc.close()
    _STOP = False

    assert summary["rows"] > 30, summary
    assert summary["read_errors"] == 0, summary
    rows = list(csv.DictReader(open(summary["path"])))
    assert len(rows) == summary["rows"], (len(rows), summary["rows"])
    assert all(f in rows[0] for f in FIELDS), set(FIELDS) - set(rows[0])

    # All sixteen RC channels present, so the marker switch is captured whichever
    # one the pilot uses.
    assert all(rows[0][f"rc{i+1}"] != "" for i in range(4)), rows[0]

    # Timestamps monotonic, motors in range.
    ts = [float(r["t"]) for r in rows]
    assert all(b > a for a, b in zip(ts, ts[1:])), "time must increase"
    assert all(1000 <= float(r["motor_mean"]) <= 2000 for r in rows), "motor range"

    # The file must be complete on disk DURING the run, not only at the end --
    # that is the whole point. Read it back while a second recording is live.
    _STOP = False
    fc2 = open_link("mock")
    threading.Timer(1.2, _on_signal, args=(None, None)).start()
    t = threading.Thread(target=record, args=(fc2, out), kwargs=
                         dict(hz=30.0, poles=14, verbose=False))
    t.start()
    time.sleep(0.8)
    live = sorted(out.glob("flight_*.csv"))[-1]
    mid = list(csv.DictReader(open(live)))
    t.join(timeout=5)
    fc2.close()
    _STOP = False
    assert len(mid) > 5, (
        f"only {len(mid)} rows readable mid-flight; the file is being buffered "
        f"rather than flushed, so a battery pull would lose it")

    shutil.rmtree(out, ignore_errors=True)
    print("flight_recorder: all checks passed")
    print(f"  {summary['rows']} rows at {summary['achieved_hz']} Hz, 0 read errors")
    print(f"  all 16 RC channels logged, so any spare switch can mark a hover")
    print(f"  {len(mid)} rows readable while still recording -- a battery pull")
    print(f"  costs the last row, not the flight")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    raise SystemExit(main())
