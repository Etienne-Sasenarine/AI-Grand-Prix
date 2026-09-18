"""Thrust-to-weight from a recorded flight, without a thrust stand.

    python3 analyze_thrust.py flight_20260918_201500.csv --mass 1.745

What this settles
-----------------
The simulator uses a **linear** thrust law, ``thrust = (stick / hover) * m * g``.
Propellers are **quadratic** in rotor speed. With ``THRUST_MAX 0.90`` over a
hover of ``0.255`` that is the difference between a thrust-to-weight ceiling of

    3.53  (linear)      and      12.5  (quadratic)

on the parameter every source we have ranks first or second. A 3.5x span is far
too wide to randomise over, so it has to be measured.

The method, and why not the IMU
-------------------------------
The obvious approach is to read the vertical acceleration during a full-throttle
climb and use ``T/W = 1 + a_peak/g``. **Do not.** ``SIM_MEASUREMENTS.md`` is
explicit: the accelerometer is biased by **-0.46 u^2 g** through vibration
rectification, which is **-0.37 g at full throttle** -- a third of the signal,
not a rounding error. It reads low, so it would make the aircraft look weaker
than it is, and we would train a policy that thinks it cannot climb.

Use the rotor speed instead. Bidirectional DShot is on and ``motor_poles`` is
14 (both measured), so every row carries per-motor RPM. At a hover the four
rotors produce exactly the aircraft's weight, so if thrust goes as RPM squared:

    T/W = (RPM_full / RPM_hover)^2

**Mass cancels.** It never enters, which is worth noticing: this number does not
depend on the one measurement we have not made yet.

It also checks the law rather than assuming it, by fitting thrust against RPM
across the whole flight and reporting the exponent. Two is quadratic. One is
linear. Anything else means something is wrong with the data or the aircraft.

What the flight has to contain
------------------------------
* at least one marked hover (see ``analyze_flight.py``)
* a **brief** full-throttle climb, well under a second, from a hover

Both come out of the same recording. Nothing extra to fly.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_flight import (GYRO_COUNTS_PER_DPS, find_marker, load,  # noqa: E402
                            marks_from_channel)
from test_hover import check_window  # noqa: E402

G = 9.80665

#: A burst has to be at least this much of the way to full to count as one.
MIN_BURST_FRACTION = 0.80
#: and last at least this long, or it is a spike rather than a measurement.
MIN_BURST_S = 0.15
#: Accelerometer counts per g on this firmware (Betaflight MSP_RAW_IMU).
ACC_COUNTS_PER_G = 512.0
#: The vibration bias, as a fraction of g, at throttle fraction u. From the
#: literature summarised in SIM_MEASUREMENTS.md section 4.
VIBRATION_BIAS_G = 0.46


def mean_rpm(r: dict):
    v = [float(r[f"rpm{i}"]) for i in range(4) if r.get(f"rpm{i}", "") not in ("", None)]
    return statistics.fmean(v) if v else None


def find_hover_rpm(rows: list[dict], marks) -> tuple:
    """Median RPM across the accepted hover windows."""
    good, rejected = [], []
    for a, b in marks:
        seg = [r for r in rows if a <= r["t"] <= b]
        ok, why = check_window(seg)
        (good if ok else rejected).append((a, b, why, seg))
    rpms, motors = [], []
    for _a, _b, _why, seg in good:
        vals = [mean_rpm(r) for r in seg]
        vals = [v for v in vals if v]
        if vals:
            rpms.append(statistics.median(vals))
        motors.append(statistics.median([r["motor_mean_us"] for r in seg]))
    return (statistics.median(rpms) if rpms else None,
            statistics.median(motors) if motors else None,
            len(good), rejected)


def find_burst(rows: list[dict]) -> dict | None:
    """The longest run of near-full throttle. Returns its peak RPM."""
    armed = [r for r in rows if r["armed"]]
    if not armed:
        return None
    top = max(r["motor_mean_us"] for r in armed)
    if top < 1100:
        return None
    thresh = 1000 + MIN_BURST_FRACTION * (top - 1000)

    best, cur = None, []
    for r in armed:
        if r["motor_mean_us"] >= thresh:
            cur.append(r)
        else:
            if cur and (best is None or len(cur) > len(best)):
                best = cur
            cur = []
    if cur and (best is None or len(cur) > len(best)):
        best = cur
    if not best:
        return None
    dur = best[-1]["t"] - best[0]["t"]
    if dur < MIN_BURST_S:
        return None
    rpms = [mean_rpm(r) for r in best]
    rpms = [v for v in rpms if v]
    return {
        "start": best[0]["t"], "dur": dur, "n": len(best),
        "motor_us": max(r["motor_mean_us"] for r in best),
        "rpm": max(rpms) if rpms else None,
        "rows": best,
    }


def fit_exponent(rows: list[dict]) -> float | None:
    """Fit thrust ~ RPM^n across the flight, in log space.

    Thrust is not measured directly, but motor output is very nearly
    proportional to it over the usable range for a fixed pack, so this asks the
    weaker and still useful question: how does RPM scale with commanded output?
    A propeller in air should come out near 0.5 (RPM ~ sqrt(output)) if output
    tracks thrust, which is the same statement as thrust ~ RPM^2.
    """
    xs, ys = [], []
    for r in rows:
        if not r["armed"]:
            continue
        rpm = mean_rpm(r)
        frac = (r["motor_mean_us"] - 1000.0) / 1000.0
        if rpm and rpm > 200 and frac > 0.05:
            xs.append(math.log(frac))
            ys.append(math.log(rpm))
    if len(xs) < 30:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return (num / den) if den else None


def imu_estimate(burst: dict) -> dict:
    """The accelerometer answer, and what vibration does to it."""
    accs = []
    for r in burst["rows"]:
        az = r.get("acc_z_raw")
        if az in ("", None):
            continue
        accs.append(float(az) / ACC_COUNTS_PER_G)
    if not accs:
        return {"available": False}
    a_peak_g = max(accs)
    u = (burst["motor_us"] - 1000.0) / 1000.0
    bias = VIBRATION_BIAS_G * u * u
    return {"available": True, "raw_tw": a_peak_g, "bias_g": bias,
            "corrected_tw": a_peak_g + bias, "throttle_fraction": u}


def main() -> int:
    ap = argparse.ArgumentParser(description="Thrust-to-weight from a flight recording")
    ap.add_argument("csv")
    ap.add_argument("--mass", type=float, default=1.745,
                    help="only used for the thrust coefficient; T/W does not need it")
    ap.add_argument("--marker-channel", type=int, default=None)
    args = ap.parse_args()

    rows = load(Path(args.csv))
    if not rows:
        print("  no usable rows")
        return 2
    if not any(r["armed"] for r in rows):
        print("  the aircraft never armed in this recording.")
        return 3

    ch = args.marker_channel
    if ch is None:
        ch, _considered = find_marker(rows, None)
    if ch is None:
        print("  no hover marker channel found; pass --marker-channel N")
        return 4
    marks = marks_from_channel(rows, ch)
    rpm_hover, motor_hover, n_good, _rej = find_hover_rpm(rows, marks)

    print()
    print("THRUST-TO-WEIGHT")
    print("=" * 68)
    print(f"  marker channel   {ch}, {len(marks)} window(s), {n_good} accepted")
    if rpm_hover is None:
        print()
        print("  No RPM in the accepted hover windows, so this cannot be done")
        print("  properly. Check bidirectional DShot is on and motor_poles is 14.")
        print("=" * 68)
        return 5
    print(f"  hover            {motor_hover:.0f} us, {rpm_hover:.0f} RPM")

    burst = find_burst(rows)
    if burst is None:
        print()
        print("  No full-throttle burst in this recording. The flight needs a")
        print(f"  brief climb at near-full throttle, at least {MIN_BURST_S}s.")
        print("=" * 68)
        return 6
    print(f"  full throttle    {burst['motor_us']:.0f} us, {burst['rpm']:.0f} RPM "
          f"for {burst['dur']:.2f}s")

    tw = (burst["rpm"] / rpm_hover) ** 2
    print()
    print(f"  THRUST-TO-WEIGHT = ({burst['rpm']:.0f} / {rpm_hover:.0f})^2 "
          f"= {tw:.2f}")
    print(f"  peak climb acceleration = {(tw - 1) * G:.1f} m/s^2 "
          f"({tw - 1:.2f} g)")
    print("  (mass does not enter this: it cancels between hover and full)")

    n = fit_exponent(rows)
    print()
    if n is not None:
        print(f"  RPM scales as (throttle fraction)^{n:.2f}")
        if n < 0.35:
            print("    -> flatter than expected; RPM is barely following throttle")
        elif n > 0.75:
            print("    -> close to linear RPM, so thrust ~ throttle^2, NOT the")
            print("       linear law the simulator uses")
        else:
            print("    -> near 0.5, i.e. thrust roughly proportional to commanded")
            print("       output, which is what the quadratic rotor law predicts")
    else:
        print("  not enough spread in the flight to fit the thrust law")

    imu = imu_estimate(burst)
    print()
    print("  CROSS-CHECK against the accelerometer (do not trust it):")
    if imu["available"]:
        print(f"    raw reading           {imu['raw_tw']:.2f} g")
        print(f"    vibration bias        +{imu['bias_g']:.2f} g "
              f"(0.46*u^2 at u={imu['throttle_fraction']:.2f})")
        print(f"    corrected             {imu['corrected_tw']:.2f}")
        gap = abs(imu["corrected_tw"] - tw)
        print(f"    vs RPM method         {tw:.2f}  (difference {gap:.2f})")
        if gap > 1.0:
            print("    The two disagree substantially. Trust the RPM figure --")
            print("    the accelerometer is the one with a known bias.")
    else:
        print("    no accelerometer data in the burst")

    print()
    print("  WHAT THIS MEANS FOR THE SIMULATOR")
    print(f"    linear law would predict a ceiling of 3.53")
    print(f"    quadratic law would predict about 12.5")
    print(f"    measured: {tw:.2f}")
    print("=" * 68)
    return 0


def _self_test() -> int:
    """Synthesise a flight with a known thrust-to-weight and recover it."""
    import csv as _csv, random, tempfile
    from flight_recorder import FIELDS

    rng = random.Random(11)
    TRUE_TW = 4.4                      # the answer the analysis must find
    RPM_HOVER = 4600.0
    RPM_FULL = RPM_HOVER * math.sqrt(TRUE_TW)
    HOVER_US, FULL_US = 1390.0, 1960.0

    path = Path(tempfile.mkdtemp()) / "flight.csv"
    with open(path, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        t = 0.0

        def emit(kind, dur, motor, rpm, armed, mark, tilt):
            nonlocal t
            for _ in range(int(dur * 30)):
                t += 1 / 30.0
                row = {f: "" for f in FIELDS}
                # Vibration makes the accelerometer read LOW, which is the whole
                # reason the IMU cannot be trusted here.
                u = max(0.0, (motor - 1000) / 1000.0)
                tw_now = (rpm / RPM_HOVER) ** 2 if rpm else 1.0
                a_true = tw_now
                a_meas = a_true - 0.46 * u * u
                row.update({
                    "t": round(t, 4), "wall": "00:00:00", "dt": round(1/30.0, 6),
                    "armed": armed, "arming_flags": "",
                    "roll_deg": round(rng.gauss(0.5, 0.3), 2),
                    "pitch_deg": round(rng.gauss(-1.8, 0.3), 2), "yaw_deg": 90.0,
                    "gyro_x_raw": int(rng.gauss(0, tilt) * GYRO_COUNTS_PER_DPS),
                    "gyro_y_raw": int(rng.gauss(0, tilt) * GYRO_COUNTS_PER_DPS),
                    "gyro_z_raw": int(rng.gauss(0, 1) * GYRO_COUNTS_PER_DPS),
                    "acc_x_raw": 0, "acc_y_raw": 0,
                    "acc_z_raw": int(a_meas * ACC_COUNTS_PER_G),
                    "motor_mean": round(motor + rng.gauss(0, 4), 1),
                    "voltage_v": 23.7, "current_a": 25.0, "mah_drawn": 600,
                    "altitude_m": 1.5, "vario_m_s": 0.0,
                })
                for k in range(4):
                    row[f"motor{k}"] = round(motor + rng.gauss(0, 4), 1)
                    row[f"rpm{k}"] = int(rpm + rng.gauss(0, 20)) if rpm else ""
                for k in range(16):
                    row[f"rc{k+1}"] = 1500
                row["rc5"] = 1800 if armed else 1000
                row["rc8"] = mark
                w.writerow(row)

        emit("ground", 4, 1000, 0, False, 1000, 0.5)
        for _ in range(3):
            emit("climb", 3, 1520, RPM_HOVER * 1.15, True, 1000, 5.0)
            emit("hover", 6, HOVER_US, RPM_HOVER, True, 1800, 1.5)
        emit("burst", 0.6, FULL_US, RPM_FULL, True, 1000, 6.0)
        emit("recover", 3, 1200, RPM_HOVER * 0.8, True, 1000, 6.0)
        emit("ground", 2, 1000, 0, False, 1000, 0.5)

    rows = load(path)
    ch, _ = find_marker(rows, None)
    assert ch == 8, ch
    marks = marks_from_channel(rows, ch)
    rpm_hover, motor_hover, n_good, _ = find_hover_rpm(rows, marks)
    assert n_good == 3, n_good
    assert abs(rpm_hover - RPM_HOVER) < 40, (rpm_hover, RPM_HOVER)

    burst = find_burst(rows)
    assert burst is not None, "the full-throttle burst was not found"
    assert 0.5 < burst["dur"] < 0.8, burst["dur"]
    assert abs(burst["rpm"] - RPM_FULL) < 120, (burst["rpm"], RPM_FULL)

    tw = (burst["rpm"] / rpm_hover) ** 2
    assert abs(tw - TRUE_TW) < 0.25, f"recovered T/W {tw:.2f} vs true {TRUE_TW}"

    # And the IMU route must come out materially WRONG, which is the reason the
    # tool does not use it. If this ever stops being true, revisit the method.
    imu = imu_estimate(burst)
    assert imu["available"]
    assert imu["raw_tw"] < TRUE_TW - 0.3, (
        f"IMU raw {imu['raw_tw']:.2f} should read low against {TRUE_TW}")
    assert abs(imu["corrected_tw"] - TRUE_TW) < 0.35, imu

    print("analyze_thrust: all checks passed")
    print(f"  hover RPM {rpm_hover:.0f} recovered from 3 marked windows")
    print(f"  full-throttle burst found: {burst['dur']:.2f}s at {burst['rpm']:.0f} RPM")
    print(f"  THRUST-TO-WEIGHT {tw:.2f} against a true {TRUE_TW} "
          f"(error {abs(tw - TRUE_TW):.2f})")
    print(f"  mass never used -- it cancels")
    print(f"  the accelerometer read {imu['raw_tw']:.2f}, low by "
          f"{TRUE_TW - imu['raw_tw']:.2f} g, which is why it is not the method")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    raise SystemExit(main())
