"""TEST 1 — hover throttle.

    python3 test_hover.py --dev /dev/ttyTHS1 --mass 1.745

What it measures and why it is worth flight time
------------------------------------------------
The throttle that holds a hover is the **single most sensitive number in the
simulator**. Our own durability sweep put a 10 % error at 58 crashes per 100
gates against a baseline of 2; the sim-to-real literature ranks it second only
to mass, and a 30 % error there was total failure. Everything else on the
measurement list can be randomised over. This one cannot, usefully, because the
plausible range spans a factor of 3.5.

With per-motor RPM (bidirectional DShot is ON, measured) the same 20 seconds
also yields the **thrust coefficient** ``k_f``, since at hover the four rotors
produce exactly the aircraft's weight.

How it is different from the first attempt
------------------------------------------
The 18 Sep attempt reported a hover of 0.098 and marked it accepted. That was
the drone **sitting armed on the ground**. Three causes, all fixed here:

1. **It gated on the barometer.** Its stability test needed ``|vario| <= 0.12``,
   and vario read exactly 0.000 for all 2828 samples of the flight — so the
   gate passed on every frame and filtered nothing. *This script never reads the
   barometer.*
2. **It read the RC stick, not the motors.** The stick passes through
   Betaflight's throttle curve (``thr_mid`` 54, ``thr_expo`` 68 — measured)
   before reaching the ESCs, so it is not the number the simulator wants. *This
   script reads ``motors()``, which is post-mix, post-curve, post-TPA.*
3. **Its gyro threshold was 16x too strict.** Raw gyro is counts at 16.4 per
   degree/second on this firmware (measured, resolving an open conflict), and it
   compared raw counts against a limit of 20. *This script scales first.*

Hover is marked by a person, not a sensor
-----------------------------------------
The replacement for the dead barometer is **the operator pressing a key**. A
human watching the aircraft can tell a steady hover from a spool-up on the
ground with total reliability, in a way no sensor we currently trust can. It is
also far simpler than any automatic rule, and simpler is what survives a noisy
flight line.

The script still checks each marked window automatically — armed, level, quiet,
steady, long enough — and rejects ones that do not hold up. Marking says *where
to look*; it does not override the data.

Safety
------
**This script is read-only.** It never arms, never disarms, never sends RC, and
never writes a setting. The pilot has the aircraft on their own radio
throughout, and nothing here can take it from them.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fc_rpm import erpm_to_rpm, read_rpm, thrust_coefficient  # noqa: E402
from mock_link import MockConfig, MockLink, open_link  # noqa: E402

GYRO_COUNTS_PER_DEG_S = 16.4
#: A marked window must hold up to all of these to count.
MIN_WINDOW_S = 3.0
MAX_TILT_DEG = 10.0
MAX_RATE_DEG_S = 40.0          # a hand-flown hover is not still; 40 is generous
MAX_MOTOR_STD_US = 60.0        # steady throttle, not a climb or a descent
MIN_WINDOWS = 3

FIELDS = ["t", "marked", "armed", "roll_deg", "pitch_deg", "yaw_deg",
          "gx_dps", "gy_dps", "gz_dps",
          "motor0", "motor1", "motor2", "motor3", "motor_mean_us",
          "rpm0", "rpm1", "rpm2", "rpm3", "rpm_mean",
          "rc_throttle_us", "voltage_v", "current_a"]


class Marker:
    """Watches stdin so the operator can bracket hover periods with ENTER."""

    def __init__(self) -> None:
        self.active = False
        self.marks: list[tuple[float, float]] = []
        self._open: float | None = None
        self._stop = threading.Event()
        self._t0 = time.monotonic()

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                line = sys.stdin.readline()
            except Exception:
                return
            if not line:
                return
            now = time.monotonic() - self._t0
            if self._open is None:
                self._open = now
                self.active = True
                print(f"  [{now:6.1f}s] HOVER WINDOW OPEN  — press ENTER again to close",
                      flush=True)
            else:
                self.marks.append((self._open, now))
                print(f"  [{now:6.1f}s] window closed ({now - self._open:.1f} s). "
                      f"{len(self.marks)} marked so far.", flush=True)
                self._open = None
                self.active = False

    def close(self) -> None:
        self._stop.set()
        if self._open is not None:
            self.marks.append((self._open, time.monotonic() - self._t0))


def record(fc, out: Path, *, seconds: float, hz: float, marker: Marker,
           motor_poles: int, verbose: bool = True, driver=None) -> list[dict]:
    period, rows = 1.0 / max(hz, 1e-6), []
    t0 = time.monotonic()
    rpm_ok = None
    while True:
        now = time.monotonic()
        t = now - t0
        if t >= seconds:
            break
        if driver is not None:
            driver(fc, t)
        try:
            roll, pitch, yaw = fc.attitude()
            _acc, gyro, _ = fc.raw_imu()
            motors = fc.motors()
            rc = fc.rc_channels()
            batt = fc.analog()
            st = fc.status()
        except Exception:
            time.sleep(period)
            continue

        rpm = read_rpm(fc)
        if rpm_ok is None:
            rpm_ok = rpm is not None
            if verbose:
                print("  RPM telemetry: " + ("available" if rpm_ok else
                      "NOT available — thrust coefficient will be unavailable"), flush=True)
        rpm_m = [erpm_to_rpm(r, motor_poles) for r in rpm] if rpm else []

        mean_us = sum(motors[:4]) / 4.0
        rows.append({
            "t": round(t, 4), "marked": marker.active,
            "armed": bool(st.get("armed", False)),
            "roll_deg": round(roll, 2), "pitch_deg": round(pitch, 2), "yaw_deg": round(yaw, 2),
            "gx_dps": round(gyro[0] / GYRO_COUNTS_PER_DEG_S, 2),
            "gy_dps": round(gyro[1] / GYRO_COUNTS_PER_DEG_S, 2),
            "gz_dps": round(gyro[2] / GYRO_COUNTS_PER_DEG_S, 2),
            "motor0": motors[0], "motor1": motors[1], "motor2": motors[2], "motor3": motors[3],
            "motor_mean_us": round(mean_us, 1),
            "rpm0": rpm_m[0] if len(rpm_m) > 0 else "",
            "rpm1": rpm_m[1] if len(rpm_m) > 1 else "",
            "rpm2": rpm_m[2] if len(rpm_m) > 2 else "",
            "rpm3": rpm_m[3] if len(rpm_m) > 3 else "",
            "rpm_mean": round(statistics.fmean(rpm_m), 1) if rpm_m else "",
            "rc_throttle_us": rc[3] if len(rc) > 3 else "",
            "voltage_v": batt.get("voltage_v", ""), "current_a": batt.get("current_a", ""),
        })
        if verbose and len(rows) % max(1, int(hz * 3)) == 0:
            r = rows[-1]
            print(f"  t={r['t']:6.1f}s armed={str(r['armed']):5} motors={r['motor_mean_us']:6.0f}us "
                  f"tilt={math.hypot(r['roll_deg'], r['pitch_deg']):4.1f}deg "
                  f"{'<<< MARKED' if r['marked'] else ''}", flush=True)
        slack = period - (time.monotonic() - now)
        if slack > 0:
            time.sleep(slack)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


def check_window(rows: list[dict]) -> tuple[bool, str]:
    """Does a marked window actually look like a hover?"""
    if len(rows) < 5:
        return False, "too few samples"
    dur = rows[-1]["t"] - rows[0]["t"]
    if dur < MIN_WINDOW_S:
        return False, f"only {dur:.1f} s, need {MIN_WINDOW_S:.0f}"
    if not all(r["armed"] for r in rows):
        return False, "not armed throughout"
    tilt = max(math.hypot(r["roll_deg"], r["pitch_deg"]) for r in rows)
    if tilt > MAX_TILT_DEG:
        return False, f"tilted to {tilt:.0f} deg, limit {MAX_TILT_DEG:.0f}"
    rate = max(max(abs(r["gx_dps"]), abs(r["gy_dps"]), abs(r["gz_dps"])) for r in rows)
    if rate > MAX_RATE_DEG_S:
        return False, f"rates to {rate:.0f} deg/s, limit {MAX_RATE_DEG_S:.0f}"
    m = [r["motor_mean_us"] for r in rows]
    sd = statistics.pstdev(m) if len(m) > 1 else 0.0
    if sd > MAX_MOTOR_STD_US:
        return False, f"throttle unsteady (sd {sd:.0f} us) — climbing or sinking?"
    return True, f"{dur:.1f} s, tilt<{tilt:.0f}deg, motor sd {sd:.0f} us"


def analyse(rows: list[dict], marks, *, mass_kg: float) -> dict:
    good, rejected = [], []
    for a, b in marks:
        seg = [r for r in rows if a <= r["t"] <= b]
        ok, why = check_window(seg)
        (good if ok else rejected).append({"start": a, "dur": b - a, "why": why,
                                           "rows": seg})
    res = {"windows_marked": len(marks), "windows_accepted": len(good),
           "rejected": [{"start": r["start"], "why": r["why"]} for r in rejected],
           "hover_motor_us": None, "hover_fraction": None,
           "hover_rpm": None, "k_f": None, "voltage_v": None,
           "trim_roll_deg": None, "trim_pitch_deg": None, "accepted": False}
    if len(good) < MIN_WINDOWS:
        res["note"] = (f"only {len(good)} good windows, need {MIN_WINDOWS}. "
                       "Fly again and mark more hovers.")
        return res

    per = [statistics.median([r["motor_mean_us"] for r in g["rows"]]) for g in good]
    res["hover_motor_us"] = statistics.median(per)
    res["hover_fraction"] = (res["hover_motor_us"] - 1000.0) / 1000.0
    res["per_window_us"] = [round(x, 1) for x in per]
    res["spread_us"] = round(max(per) - min(per), 1)

    rpms = [r["rpm_mean"] for g in good for r in g["rows"] if r["rpm_mean"] != ""]
    if rpms:
        res["hover_rpm"] = statistics.median(rpms)
        res["k_f"] = thrust_coefficient(mass_kg, res["hover_rpm"])

    v = [r["voltage_v"] for g in good for r in g["rows"] if r["voltage_v"] != ""]
    if v:
        res["voltage_v"] = round(statistics.fmean([float(x) for x in v]), 2)
    res["trim_roll_deg"] = round(statistics.median(
        [r["roll_deg"] for g in good for r in g["rows"]]), 2)
    res["trim_pitch_deg"] = round(statistics.median(
        [r["pitch_deg"] for g in good for r in g["rows"]]), 2)
    res["accepted"] = True
    return res


def report(res: dict) -> None:
    print()
    print("HOVER RESULT")
    print("=" * 68)
    print(f"  windows marked {res['windows_marked']}, accepted {res['windows_accepted']}")
    for r in res["rejected"]:
        print(f"    rejected at t={r['start']:.0f}s: {r['why']}")
    if not res["accepted"]:
        print(f"\n  NOT ACCEPTED — {res.get('note')}")
        print("=" * 68)
        return
    print(f"  per window (motor us): {res['per_window_us']}  spread {res['spread_us']} us")
    print()
    print(f"  HOVER MOTOR OUTPUT   {res['hover_motor_us']:.0f} us "
          f"= {res['hover_fraction']:.3f} of the 1000-2000 range")
    print("    This is post-mix, post-curve, post-TPA -- what the ESCs saw.")
    print("    It is NOT the stick position; do not compare the two directly.")
    if res["hover_rpm"]:
        print(f"  HOVER RPM            {res['hover_rpm']:.0f} per rotor")
        print(f"  THRUST COEFFICIENT   k_f = {res['k_f']:.4e} N per RPM^2")
        print("    thrust_per_rotor = k_f * rpm^2. Feed the max-thrust test this RPM.")
    else:
        print("  RPM unavailable -- no thrust coefficient. Check dshot_bidir is ON.")
    print(f"  pack voltage         {res['voltage_v']} V  (hover throttle drifts with it)")
    print(f"  hover attitude bias  roll {res['trim_roll_deg']:+.2f} deg, "
          f"pitch {res['trim_pitch_deg']:+.2f} deg")
    print(f"    Free, and it adds to the camera tilt: effective tilt "
          f"= 20.00 {-res['trim_pitch_deg']:+.2f} = {20.0 - res['trim_pitch_deg']:.2f} deg")
    print("=" * 68)


BRIEF = """
TEST 1 - HOVER THROTTLE                                       READ-ONLY
------------------------------------------------------------------------
This program only listens. It never arms, never sends RC, never writes a
setting. The pilot keeps full control on their own radio at all times.

ON THE FIELD - before you start:
  * Area clear. Nobody near or under the aircraft, nobody downrange.
  * Pilot has a clear abort and knows where they are putting it down.
  * Second person runs this laptop, so the pilot never looks away.
  * Battery in, props on, all four checked tight.

WHAT TO DO:
  1. Pilot takes off and settles into a steady hover, roughly head height,
     holding station as still as they reasonably can.
  2. When it looks steady, press ENTER here.  <-- window opens
  3. After 5+ seconds of that hover, press ENTER again. <-- window closes
  4. Repeat at least THREE times. Land and re-hover between them if that is
     easier; separate hovers are better than one long one.
  5. Ctrl+C or wait for the timer to finish.

Three windows is the minimum because one is a number and two is a coin toss.
------------------------------------------------------------------------
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Hover throttle, read-only")
    ap.add_argument("--dev", default=None, help="/dev/ttyTHS1; omit for the mock")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--seconds", type=float, default=180.0)
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--mass", type=float, default=1.745, help="race-ready mass, kg")
    ap.add_argument("--motor-poles", type=int, default=14)
    ap.add_argument("--out", default="hover_test.csv")
    args = ap.parse_args()

    print(BRIEF)
    fc = open_link(args.dev, baud=args.baud)
    marker = Marker()
    marker.start()
    try:
        rows = record(fc, Path(args.out), seconds=args.seconds, hz=args.hz,
                      marker=marker, motor_poles=args.motor_poles)
    except KeyboardInterrupt:
        print("\n  stopped by operator", flush=True)
        rows = []
    finally:
        marker.close()
        fc.close()
    if rows:
        res = analyse(rows, marker.marks, mass_kg=args.mass)
        report(res)
        import json
        Path(args.out).with_suffix(".json").write_text(
            json.dumps({k: v for k, v in res.items() if k != "rejected"} |
                       {"rejected": res["rejected"]}, indent=2, default=str))
        print(f"  saved {args.out} and {Path(args.out).with_suffix('.json')}")
    return 0


def _self_test() -> int:
    """Fly a synthetic hover with a known throttle and recover it."""
    cfg = MockConfig(hover_throttle=0.284, trim_pitch=math.radians(-1.9))
    fc = MockLink(cfg)
    fc.arm_state(True)

    marker = Marker()
    # Mark three windows by hand rather than through stdin.
    marker.marks = [(1.0, 5.0), (6.0, 10.5), (12.0, 16.0)]

    out = Path("/tmp/_hover_test.csv")
    rows = record(fc, out, seconds=17.0, hz=60.0, marker=marker, motor_poles=14,
                  verbose=False, driver=lambda link, t: link.command(cfg.hover_throttle))
    res = analyse(rows, marker.marks, mass_kg=1.745)
    assert res["accepted"], res
    err = abs(res["hover_fraction"] - cfg.hover_throttle)
    assert err < 0.01, f"hover {res['hover_fraction']:.4f} vs true {cfg.hover_throttle:.4f}"
    assert abs(res["trim_pitch_deg"] - math.degrees(cfg.trim_pitch)) < 0.6, res

    # A window where the aircraft is NOT hovering must be rejected.
    bad = Marker()
    bad.marks = [(1.0, 5.0), (6.0, 10.5), (12.0, 16.0), (0.0, 0.5)]
    res2 = analyse(rows, bad.marks, mass_kg=1.745)
    assert len(res2["rejected"]) == 1, res2["rejected"]

    print("test_hover: all checks passed")
    print(f"  recovered hover {res['hover_fraction']:.4f} from a true "
          f"{cfg.hover_throttle:.4f} (error {err:.4f})")
    print(f"  attitude bias   pitch {res['trim_pitch_deg']:+.2f} deg from a true "
          f"{math.degrees(cfg.trim_pitch):+.2f}")
    print(f"  a too-short window was rejected: {res2['rejected'][0]['why']}")
    out.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    raise SystemExit(main())
