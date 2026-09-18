"""TEST 2 — maximum thrust, i.e. thrust-to-weight.

    python3 test_max_thrust.py --dev /dev/ttyTHS1 --hover-rpm 4820

What it settles
---------------
The simulator uses a **linear** thrust law, ``thrust = (stick/hover)*m*g``.
Propellers are **quadratic** in rotor speed. With ``THRUST_MAX 0.90`` over a
hover of ``0.255`` that is the difference between a thrust-to-weight ceiling of

    3.53  (linear)      and      12.5  (quadratic)

on the parameter every source we have ranks first or second. A 3.5x span is far
too wide to randomise usefully, so it has to be measured — and it takes one
burst of under a second.

The method, and why not the accelerometer
-----------------------------------------
Propeller thrust goes as rotor speed squared, so

    T/W  =  (rpm_max / rpm_hover) ** 2

That is a pure ratio: no mass, no thrust stand, no accelerometer. Which matters,
because **the accelerometer cannot do this job on this airframe.** Vibration
rectification biases it by roughly ``-0.46 * u^2`` g — about **-0.37 g at full
throttle**, a third of the signal, and worst exactly where the measurement is
taken. Reading peak vertical acceleration would give a confidently wrong answer.

``rpm_hover`` comes from TEST 1. Run that first, or pass ``--hover-rpm``.

If RPM telemetry is unavailable the script falls back to the ratio of motor
command values and says clearly that the result now *assumes* a thrust law
rather than measuring one — which is the very thing we are trying to settle, so
treat that output as a cross-check, not an answer.

Safety
------
**Read-only: never arms, never sends RC, never writes a setting.** The pilot
owns the aircraft throughout.

The risk here is not software, it is a **fast vertical climb indoors**. This is
the only test on the list with a real chance of hitting something. Read the
brief the program prints before flying it.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fc_rpm import erpm_to_rpm, read_rpm, thrust_to_weight  # noqa: E402
from mock_link import MockConfig, MockLink, open_link  # noqa: E402

GYRO_COUNTS_PER_DEG_S = 16.4
FIELDS = ["t", "armed", "motor_mean_us", "rpm_mean", "roll_deg", "pitch_deg",
          "az_raw", "rc_throttle_us", "voltage_v", "current_a"]


def record(fc, out: Path, *, seconds: float, hz: float, motor_poles: int,
           verbose: bool = True, driver=None) -> list[dict]:
    period, rows = 1.0 / max(hz, 1e-6), []
    t0 = time.monotonic()
    while True:
        now = time.monotonic()
        t = now - t0
        if t >= seconds:
            break
        if driver is not None:
            driver(fc, t)
        try:
            roll, pitch, _ = fc.attitude()
            acc, _gyro, _ = fc.raw_imu()
            motors = fc.motors()
            rc = fc.rc_channels()
            batt = fc.analog()
            st = fc.status()
        except Exception:
            time.sleep(period)
            continue
        rpm = read_rpm(fc)
        rpm_m = [erpm_to_rpm(r, motor_poles) for r in rpm] if rpm else []
        rows.append({
            "t": round(t, 4), "armed": bool(st.get("armed", False)),
            "motor_mean_us": round(sum(motors[:4]) / 4.0, 1),
            "rpm_mean": round(statistics.fmean(rpm_m), 1) if rpm_m else "",
            "roll_deg": round(roll, 2), "pitch_deg": round(pitch, 2),
            "az_raw": acc[2],
            "rc_throttle_us": rc[3] if len(rc) > 3 else "",
            "voltage_v": batt.get("voltage_v", ""), "current_a": batt.get("current_a", ""),
        })
        if verbose and len(rows) % max(1, int(hz * 2)) == 0:
            r = rows[-1]
            extra = f" rpm={r['rpm_mean']}" if r["rpm_mean"] != "" else ""
            print(f"  t={r['t']:6.1f}s motors={r['motor_mean_us']:6.0f}us{extra}", flush=True)
        slack = period - (time.monotonic() - now)
        if slack > 0:
            time.sleep(slack)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


def find_burst(rows: list[dict], *, key: str, min_hold_s: float = 0.15):
    """The strongest sustained peak, so a single noisy sample cannot win."""
    vals = [(r["t"], float(r[key])) for r in rows if r.get(key) not in ("", None)]
    if len(vals) < 5:
        return None
    peak = max(v for _, v in vals)
    best = None
    for i in range(len(vals)):
        if vals[i][1] < 0.97 * peak:
            continue
        j = i
        while j + 1 < len(vals) and vals[j + 1][1] >= 0.97 * peak:
            j += 1
        hold = vals[j][0] - vals[i][0]
        seg = [v for _, v in vals[i:j + 1]]
        cand = {"start": vals[i][0], "hold_s": hold,
                "value": statistics.median(seg), "samples": len(seg)}
        if best is None or cand["hold_s"] > best["hold_s"]:
            best = cand
    if best and best["hold_s"] < min_hold_s:
        best["warning"] = (f"peak held only {best['hold_s']*1000:.0f} ms — at this "
                           f"sample rate that is few points; treat as approximate")
    return best


def analyse(rows: list[dict], *, hover_rpm: float | None,
            hover_motor_us: float | None) -> dict:
    res = {"samples": len(rows), "method": None, "thrust_to_weight": None,
           "burst": None, "warning": None}
    has_rpm = any(r["rpm_mean"] != "" for r in rows)

    if has_rpm and hover_rpm:
        b = find_burst(rows, key="rpm_mean")
        if b:
            res.update(method="rpm", burst=b,
                       rpm_hover=hover_rpm, rpm_max=b["value"],
                       thrust_to_weight=thrust_to_weight(hover_rpm, b["value"]))
            res["warning"] = b.get("warning")
            return res
        res["warning"] = "RPM present but no clear burst found"

    b = find_burst(rows, key="motor_mean_us")
    if b and hover_motor_us:
        frac_max = (b["value"] - 1000.0) / 1000.0
        frac_hov = (hover_motor_us - 1000.0) / 1000.0
        res.update(method="motor_command", burst=b,
                   motor_hover_us=hover_motor_us, motor_max_us=b["value"],
                   tw_if_linear=frac_max / frac_hov if frac_hov > 0 else float("nan"),
                   tw_if_quadratic=(frac_max / frac_hov) ** 2 if frac_hov > 0 else float("nan"))
        res["warning"] = ("no RPM telemetry — this ASSUMES a thrust law instead of "
                          "measuring one, which is the question being asked. "
                          "Check dshot_bidir is ON and rerun.")
    return res


def report(res: dict) -> None:
    print()
    print("MAX THRUST RESULT")
    print("=" * 68)
    b = res.get("burst")
    if not b:
        print("  No burst detected. Was the aircraft armed and flown?")
        print("=" * 68)
        return
    print(f"  burst at t={b['start']:.1f}s, peak held {b['hold_s']*1000:.0f} ms "
          f"over {b['samples']} samples")
    if res["method"] == "rpm":
        print()
        print(f"  hover RPM  {res['rpm_hover']:.0f}     peak RPM  {res['rpm_max']:.0f}")
        print(f"  THRUST-TO-WEIGHT = (peak/hover)^2 = {res['thrust_to_weight']:.2f}")
        print()
        print("  Measured from rotor speed, so it does not assume a thrust law")
        print("  and it does not touch the accelerometer.")
        tw = res["thrust_to_weight"]
        print()
        if tw < 5.0:
            print(f"  {tw:.2f} is near the simulator's LINEAR figure of 3.53.")
            print("  Keep the linear thrust law; set hover_stick from TEST 1.")
        elif tw > 8.0:
            print(f"  {tw:.2f} is near the QUADRATIC figure of 12.5.")
            print("  The simulator's linear law is wrong -- it will understate")
            print("  available thrust badly above hover. Fix before retraining.")
        else:
            print(f"  {tw:.2f} sits between the two candidate laws (3.53 / 12.5).")
            print("  Repeat the burst; if it holds, neither pure law fits and the")
            print("  curve should be fitted from RPM against motor command.")
    else:
        print()
        print(f"  hover {res.get('motor_hover_us', 0):.0f} us   peak "
              f"{res.get('motor_max_us', 0):.0f} us")
        print(f"  T/W if the law is linear:    {res.get('tw_if_linear', float('nan')):.2f}")
        print(f"  T/W if the law is quadratic: {res.get('tw_if_quadratic', float('nan')):.2f}")
    if res.get("warning"):
        print()
        print(f"  WARNING: {res['warning']}")
    print("=" * 68)


BRIEF = """
TEST 2 - MAXIMUM THRUST                                       READ-ONLY
------------------------------------------------------------------------
This program only listens. It never arms, never sends RC, never writes a
setting. The pilot keeps full control on their own radio at all times.

THIS IS THE ONE TEST THAT CAN HIT SOMETHING. Read this properly.

The manoeuvre is a brief full-throttle climb. On an 8-inch 6S airframe that
accelerates hard, and you are indoors under a real ceiling.

BEFORE FLYING:
  * Run TEST 1 first and have the hover RPM. Without it this test cannot
    give you a thrust-to-weight at all.
  * Pick your spot: maximum clear height, nothing above, no gates or
    structure overhead, nobody underneath or downrange.
  * Agree the abort out loud: the pilot cuts to hover the instant it looks
    wrong, and does not chase it.
  * Props on and checked. Battery in. Area called clear.

THE MANOEUVRE:
  1. Stable hover, LOW - about a metre. Low start buys ceiling.
  2. Full throttle for UNDER HALF A SECOND. Count it as "one" and stop.
     Half a second is long enough; the peak is reached almost immediately.
  3. Catch it back into a hover. Land.
  4. Repeat only if the log says the peak was too brief to measure.

The script reports how long the peak was actually held, so you will know
whether the burst was long enough without guessing.
------------------------------------------------------------------------
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Maximum thrust / thrust-to-weight, read-only")
    ap.add_argument("--dev", default=None)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--hz", type=float, default=50.0,
                    help="poll as fast as the link allows; the burst is short")
    ap.add_argument("--hover-rpm", type=float, default=None,
                    help="from TEST 1. Without it, no thrust-to-weight from RPM")
    ap.add_argument("--hover-motor-us", type=float, default=None,
                    help="from TEST 1, for the fallback method")
    ap.add_argument("--motor-poles", type=int, default=14)
    ap.add_argument("--out", default="max_thrust_test.csv")
    args = ap.parse_args()

    print(BRIEF)
    if args.hover_rpm is None and args.hover_motor_us is None:
        print("  NOTE: no hover reference given. Run TEST 1 first and pass")
        print("        --hover-rpm (preferred) or --hover-motor-us.")
        print()

    fc = open_link(args.dev, baud=args.baud)
    try:
        rows = record(fc, Path(args.out), seconds=args.seconds, hz=args.hz,
                      motor_poles=args.motor_poles)
    except KeyboardInterrupt:
        print("\n  stopped by operator", flush=True)
        rows = []
    finally:
        fc.close()
    if rows:
        res = analyse(rows, hover_rpm=args.hover_rpm, hover_motor_us=args.hover_motor_us)
        report(res)
        Path(args.out).with_suffix(".json").write_text(json.dumps(res, indent=2, default=str))
        print(f"  saved {args.out} and {Path(args.out).with_suffix('.json')}")
    return 0


def _self_test() -> int:
    """Fly a synthetic hover-then-burst and recover the known thrust-to-weight."""
    cfg = MockConfig(hover_throttle=0.28)
    fc = MockLink(cfg)
    fc.arm_state(True)

    # Hover, then a 0.4 s full-throttle burst, then hover again.
    def driver(link, t):
        link.command(1.0 if 4.0 <= t < 4.4 else cfg.hover_throttle)

    out = Path("/tmp/_maxthrust.csv")
    rows = record(fc, out, seconds=8.0, hz=200.0, motor_poles=14,
                  verbose=False, driver=driver)

    # The mock has no RPM, so this exercises the fallback path -- which must
    # fire, must be labelled as assuming a law, and must not silently pretend.
    hover_us = 1000 + 1000 * cfg.hover_throttle
    res = analyse(rows, hover_rpm=None, hover_motor_us=hover_us)
    assert res["method"] == "motor_command", res["method"]
    assert res["warning"] and "ASSUMES" in res["warning"], res["warning"]
    assert res["burst"]["hold_s"] > 0.2, res["burst"]
    lin, quad = res["tw_if_linear"], res["tw_if_quadratic"]
    assert abs(lin - 1.0 / cfg.hover_throttle) < 0.25, (lin, 1.0 / cfg.hover_throttle)
    assert abs(quad - lin ** 2) < 0.5, (quad, lin)

    # A synthetic RPM trace must take the preferred path and get T/W right.
    for r in rows:
        frac = (r["motor_mean_us"] - 1000.0) / 1000.0
        r["rpm_mean"] = round(5000.0 * math.sqrt(max(frac, 0.0) / cfg.hover_throttle), 1)
    res2 = analyse(rows, hover_rpm=5000.0, hover_motor_us=hover_us)
    assert res2["method"] == "rpm", res2
    expected = 1.0 / cfg.hover_throttle          # rpm ~ sqrt(frac) => T/W = frac_max/frac_hover
    assert abs(res2["thrust_to_weight"] - expected) < 0.3, (res2["thrust_to_weight"], expected)

    print("test_max_thrust: all checks passed")
    print(f"  burst found: {res['burst']['hold_s']*1000:.0f} ms held")
    print(f"  fallback path correctly flagged: '{res['warning'][:52]}...'")
    print(f"  RPM path recovered T/W {res2['thrust_to_weight']:.2f} "
          f"against a true {expected:.2f}")
    out.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    raise SystemExit(main())
