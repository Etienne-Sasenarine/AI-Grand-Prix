"""Turn the Jetson into the drone's flight recorder.

    python3 msp_logger.py --dev /dev/ttyTHS1 --out hover.csv --seconds 30

Why this and not blackbox
-------------------------
Blackbox is better data — kilohertz, and it sees the setpoint. But it needs the
flight controller to have flash, the right settings, a USB cable and Betaflight
Configurator to pull it off, and any one of those failing on the day costs a
session. This needs a USB cable to the *Jetson*, which we already have because
that is how we log in.

One pilot hover through this gives, from a single file:

  * hover throttle, from the **motor outputs** rather than the stick
  * hover attitude bias -- the combined centre-of-gravity and thrust-axis
    offset, which adds directly to the camera tilt
  * battery sag across the pack
  * achieved body rates
  * the real telemetry rate, which sets the simulator's control decimation

So blackbox becomes a bonus rather than a dependency. The analysis tools in this
directory read the CSV this writes.

What it records
---------------
Every poll: a monotonic timestamp, attitude, raw IMU, per-motor output, the RC
channels the FC is acting on, and pack voltage. It also records **its own poll
period**, because "how fast can we actually talk to the flight controller" is
itself one of the measurements we need, and the honest way to get it is to look
at the timestamps of a real logging run rather than a synthetic benchmark.

Safety
------
This only ever *reads*. It never arms, never transmits RC, never writes a
setting. It is safe to run at any time, including during a piloted flight, which
is exactly when you want it.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mock_link import MockConfig, MockLink, open_link  # noqa: E402

FIELDS = [
    "t", "dt",
    "roll_deg", "pitch_deg", "yaw_deg",
    "gyro_x_raw", "gyro_y_raw", "gyro_z_raw",
    "acc_x_raw", "acc_y_raw", "acc_z_raw",
    "armed", "altitude_m", "vario_m_s",
    "motor0", "motor1", "motor2", "motor3",
    "rc_throttle", "rc_roll", "rc_pitch", "rc_yaw",
    "voltage_v", "current_a", "mah_drawn",
]


def record(fc, out_path: Path, *, seconds: float, hz: float, verbose: bool = True,
           driver=None) -> dict:
    """Poll the flight controller and write a CSV. Returns a summary."""
    period = 1.0 / max(hz, 1e-6)
    rows, t0, last = [], time.monotonic(), None
    errors = 0

    while True:
        now = time.monotonic()
        t = now - t0
        if t >= seconds:
            break
        if driver is not None:
            driver(fc, t)
        try:
            roll, pitch, yaw = fc.attitude()
            acc, gyro, _ = fc.raw_imu()
            motors = fc.motors()
            rc = fc.rc_channels()
            batt = fc.analog()
            alt, vario = fc.altitude()
            st = fc.status()
        except Exception:                      # a dropped frame is not fatal
            errors += 1
            time.sleep(period)
            continue

        dt = float("nan") if last is None else now - last
        last = now
        rows.append({
            "t": round(t, 6), "dt": round(dt, 6) if dt == dt else "",
            "roll_deg": round(roll, 3), "pitch_deg": round(pitch, 3),
            "yaw_deg": round(yaw, 3),
            "armed": bool(st.get("armed", False)),
            "altitude_m": round(float(alt), 3), "vario_m_s": round(float(vario), 4),
            "gyro_x_raw": gyro[0], "gyro_y_raw": gyro[1], "gyro_z_raw": gyro[2],
            "acc_x_raw": acc[0], "acc_y_raw": acc[1], "acc_z_raw": acc[2],
            "motor0": motors[0], "motor1": motors[1],
            "motor2": motors[2], "motor3": motors[3],
            "rc_throttle": rc[0] if len(rc) > 0 else "",
            "rc_roll": rc[1] if len(rc) > 1 else "",
            "rc_pitch": rc[2] if len(rc) > 2 else "",
            "rc_yaw": rc[3] if len(rc) > 3 else "",
            "voltage_v": batt.get("voltage_v", ""),
            "current_a": batt.get("current_a", ""),
            "mah_drawn": batt.get("mah_drawn", ""),
        })
        slack = period - (time.monotonic() - now)
        if slack > 0:
            time.sleep(slack)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    dts = [r["dt"] for r in rows if isinstance(r["dt"], float)]
    dts = [d for d in dts if d == d]
    achieved = (len(dts) / sum(dts)) if dts and sum(dts) > 0 else 0.0
    worst = max(dts) if dts else float("nan")
    summary = {
        "rows": len(rows), "seconds": round(rows[-1]["t"], 2) if rows else 0.0,
        "requested_hz": hz, "achieved_hz": round(achieved, 1),
        "worst_gap_ms": round(worst * 1000, 1) if worst == worst else float("nan"),
        "read_errors": errors, "path": str(out_path),
    }
    if verbose:
        print()
        print(f"  wrote {summary['rows']} rows covering {summary['seconds']} s -> {out_path}")
        print(f"  requested {hz:.0f} Hz, achieved {summary['achieved_hz']:.1f} Hz, "
              f"worst gap {summary['worst_gap_ms']:.1f} ms, {errors} read errors")
        if achieved < 0.8 * hz:
            print(f"  NOTE: well short of the requested rate. That is itself a result --")
            print(f"        it is the link rate the simulator's decimation should match.")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dev", default=None,
                    help="serial device, e.g. /dev/ttyTHS1. Omit to run against the mock")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--out", default="logs/msp_log.csv")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--hz", type=float, default=50.0,
                    help="target poll rate. Ask for more than you expect and see what you get")
    args = ap.parse_args()

    fc = open_link(args.dev, baud=args.baud)
    try:
        print(f"  {fc.fc_variant()} api {fc.api_version()} uid {fc.uid()}")
        record(fc, Path(args.out), seconds=args.seconds, hz=args.hz)
    finally:
        fc.close()
    return 0


def _self_test() -> int:
    """Record a short synthetic hover and check the file is usable."""
    cfg = MockConfig()
    fc = MockLink(cfg)

    def hover(link, t):
        link.command(cfg.hover_throttle)

    out = Path("/tmp/_msp_logger_selftest.csv")
    s = record(fc, out, seconds=1.5, hz=50.0, verbose=False, driver=hover)
    assert s["rows"] > 50, s
    assert s["read_errors"] == 0, s

    with open(out) as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0].keys()) == FIELDS, "header drifted from FIELDS"
    m = [ (int(r["motor0"]) + int(r["motor1"]) + int(r["motor2"]) + int(r["motor3"])) / 4
          for r in rows ]
    frac = (sum(m) / len(m) - 1000) / 1000.0
    assert abs(frac - cfg.hover_throttle) < 0.02, (frac, cfg.hover_throttle)
    ts = [float(r["t"]) for r in rows]
    assert all(b > a for a, b in zip(ts, ts[1:])), "timestamps must be monotonic"

    print("msp_logger: all checks passed")
    print(f"  {s['rows']} rows at {s['achieved_hz']:.0f} Hz; "
          f"mean motor output {frac:.3f} against a true hover of {cfg.hover_throttle:.3f}")
    out.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    raise SystemExit(main())
