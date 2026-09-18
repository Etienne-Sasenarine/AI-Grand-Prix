"""Turn a hover log into the numbers the simulator needs.

    python3 analyze_hover.py hover.csv

Recovers, from one 20-second pilot hover:

  * **hover throttle** -- the single most sensitive parameter in the model
  * **hover attitude bias** -- centre-of-gravity and thrust-axis offset, which
    adds directly to the effective camera tilt
  * **battery sag** -- how much thrust drifts across a pack
  * **the achieved telemetry rate** -- which sets the simulator's decimation

Two things this does differently from the obvious approach, both deliberate:

**It reads the motor outputs, not the throttle stick.** The stick is pre-curve;
the motors are what the ESCs were actually given, after the throttle curve, the
mixer and any throttle-based PID attenuation. `SPEC.md` records a live ambiguity
here -- a reported "hovers at 40 %" means two different numbers depending on
which one was read -- and this settles it by reading the unambiguous one.

**It only averages over a window where the aircraft is genuinely holding
station.** A pilot "hovering" is constantly correcting; the frames where they
are climbing, sinking or leaning are not hover frames. Averaging everything
biases the answer by however hard they were working. The gate here is a quiet
attitude and a quiet gyro, which is the same test the training code uses.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

#: A frame counts as hover only if the aircraft is this still.
MAX_TILT_DEG = 8.0
MAX_RATE_DEG_S = 20.0
#: Raw gyro scaling. See ``--gyro-scale``; the default is the Betaflight
#: convention and SPEC.md records this as an unresolved conflict.
GYRO_COUNTS_PER_DEG_S = 16.4


def load(path: Path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _f(row, key, default=float("nan")) -> float:
    v = row.get(key, "")
    if v in ("", None):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def analyse(rows: list[dict], *, gyro_scale: float = GYRO_COUNTS_PER_DEG_S) -> dict:
    if not rows:
        raise ValueError("empty log")

    quiet, all_motor, volts, times = [], [], [], []
    for r in rows:
        roll, pitch = _f(r, "roll_deg"), _f(r, "pitch_deg")
        gx, gy, gz = (_f(r, "gyro_x_raw") / gyro_scale,
                      _f(r, "gyro_y_raw") / gyro_scale,
                      _f(r, "gyro_z_raw") / gyro_scale)
        motors = [_f(r, f"motor{i}") for i in range(4)]
        if any(m != m for m in motors):
            continue
        frac = (sum(motors) / 4.0 - 1000.0) / 1000.0
        all_motor.append(frac)
        times.append(_f(r, "t"))
        v = _f(r, "voltage_v")
        if v == v:
            volts.append((_f(r, "t"), v, frac))
        tilt = math.hypot(roll, pitch)
        rate = max(abs(gx), abs(gy), abs(gz))
        if tilt <= MAX_TILT_DEG and rate <= MAX_RATE_DEG_S:
            quiet.append({"frac": frac, "roll": roll, "pitch": pitch})

    if not quiet:
        raise ValueError(
            "no frames met the hover gate (tilt <= 8 deg, rates <= 20 deg/s). "
            "Either the aircraft never settled, or the gyro scale is wrong -- "
            "try --gyro-scale 1.0 if the FC reports degrees per second already.")

    fracs = [q["frac"] for q in quiet]
    hover = statistics.median(fracs)
    spread = statistics.pstdev(fracs) if len(fracs) > 1 else 0.0

    # Battery sag: regress motor output on voltage over the quiet frames.
    sag = None
    if len(volts) > 20:
        vs = [v for _, v, _ in volts]
        fs = [f for _, _, f in volts]
        vbar, fbar = statistics.fmean(vs), statistics.fmean(fs)
        den = sum((v - vbar) ** 2 for v in vs)
        if den > 1e-9:
            slope = sum((v - vbar) * (f - fbar) for v, f in zip(vs, fs)) / den
            sag = {"slope_per_volt": slope,
                   "v_start": vs[0], "v_end": vs[-1],
                   "drift": slope * (vs[-1] - vs[0])}

    dts = [b - a for a, b in zip(times, times[1:]) if b > a]
    rate_hz = (len(dts) / sum(dts)) if dts and sum(dts) > 0 else float("nan")

    return {
        "hover_throttle": hover,
        "hover_spread": spread,
        "quiet_frames": len(quiet),
        "total_frames": len(all_motor),
        "quiet_pct": 100.0 * len(quiet) / max(len(all_motor), 1),
        "trim_roll_deg": statistics.median([q["roll"] for q in quiet]),
        "trim_pitch_deg": statistics.median([q["pitch"] for q in quiet]),
        "sag": sag,
        "telemetry_hz": rate_hz,
        # Only valid if the thrust law is linear in the motor command. It is
        # not, for a propeller -- see the note printed below.
        "tw_if_linear": 1.0 / hover if hover > 1e-6 else float("inf"),
        "tw_if_quadratic": (1.0 / hover) ** 2 if hover > 1e-6 else float("inf"),
    }


def report(a: dict) -> None:
    print()
    print("Hover analysis")
    print("=" * 66)
    print(f"  hover throttle          {a['hover_throttle']:.4f}  "
          f"(spread {a['hover_spread']:.4f} over {a['quiet_frames']} quiet frames)")
    print(f"  frames meeting the gate {a['quiet_pct']:.0f} % of {a['total_frames']}")
    print(f"  hover attitude bias     roll {a['trim_roll_deg']:+.2f} deg, "
          f"pitch {a['trim_pitch_deg']:+.2f} deg")
    print(f"  telemetry rate          {a['telemetry_hz']:.1f} Hz")
    if a["sag"]:
        s = a["sag"]
        print(f"  battery sag             {s['v_start']:.2f} -> {s['v_end']:.2f} V, "
              f"throttle drift {s['drift']:+.4f}")
    print()
    print("  Thrust-to-weight ceiling at full throttle:")
    print(f"    if thrust is LINEAR in motor command     {a['tw_if_linear']:.2f}")
    print(f"    if thrust is QUADRATIC (propellers are)  {a['tw_if_quadratic']:.2f}")
    print()
    print("  These differ by the ratio itself, and it is the most sensitive")
    print("  parameter in the model. A hover alone cannot separate them --")
    print("  fly the full-throttle climb and use analyze_climb, or read peak")
    print("  motor RPM if bidirectional DShot is on.")
    print()
    print(f"  pitch bias of {a['trim_pitch_deg']:+.2f} deg adds to the camera tilt:")
    print(f"    effective tilt = 20.00 {-a['trim_pitch_deg']:+.2f} = "
          f"{20.0 - a['trim_pitch_deg']:.2f} deg")
    print("=" * 66)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("log", type=Path)
    ap.add_argument("--gyro-scale", type=float, default=GYRO_COUNTS_PER_DEG_S,
                    help="raw gyro counts per deg/s. Use 1.0 if the FC reports deg/s")
    args = ap.parse_args()
    report(analyse(load(args.log), gyro_scale=args.gyro_scale))
    return 0


def _self_test() -> int:
    """Fly a synthetic hover with known parameters and recover them."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mock_link import MockConfig, MockLink
    from msp_logger import record

    cfg = MockConfig(hover_throttle=0.337,
                     trim_roll=math.radians(1.4), trim_pitch=math.radians(-2.6))
    fc = MockLink(cfg)
    out = Path("/tmp/_hover_selftest.csv")
    record(fc, out, seconds=4.0, hz=50.0, verbose=False,
           driver=lambda link, t: link.command(cfg.hover_throttle))

    a = analyse(load(out))
    err = abs(a["hover_throttle"] - cfg.hover_throttle)
    assert err < 0.01, f"hover throttle off by {err:.4f}: {a['hover_throttle']} vs {cfg.hover_throttle}"
    assert abs(a["trim_roll_deg"] - math.degrees(cfg.trim_roll)) < 0.5, a
    assert abs(a["trim_pitch_deg"] - math.degrees(cfg.trim_pitch)) < 0.5, a
    assert a["quiet_pct"] > 80.0, a

    print("analyze_hover: all checks passed")
    print(f"  recovered hover {a['hover_throttle']:.4f} from a true {cfg.hover_throttle:.4f} "
          f"(error {err:.4f})")
    print(f"  recovered trim  roll {a['trim_roll_deg']:+.2f} / pitch {a['trim_pitch_deg']:+.2f} deg "
          f"from a true {math.degrees(cfg.trim_roll):+.2f} / {math.degrees(cfg.trim_pitch):+.2f}")
    out.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    raise SystemExit(main())
