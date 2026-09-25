"""Get the hover throttle out of a recording made with nothing attached.

    python3 analyze_flight.py flight_20260918_201500.csv --mass 1.745
    python3 analyze_flight.py flight.csv --marker-channel 8

What this replaces
------------------
``test_hover.py`` asks the operator to press ENTER to bracket each steady hover.
That works, but it means somebody has to be holding an SSH session open to the
drone for the whole flight -- and on 18 Sep the Jetson did not rejoin Wi-Fi, so a
25-minute slot produced nothing at all.

``flight_recorder.py`` removes that dependency: the drone records itself and logs
**all sixteen RC channels**. The pilot marks a hover by flipping a spare switch,
which is the same human judgement as pressing ENTER, made by the same person, in
the one place they are definitely looking -- their own transmitter.

This reads the resulting CSV and applies **exactly the same acceptance tests**
as ``test_hover.py``: armed throughout, not tilted, not rotating, throttle
steady, long enough, and at least three separate windows. Marking says where to
look; it never overrides the data.

Finding the marker switch
-------------------------
You do not have to decide in advance which switch to use. This looks at every
aux channel and picks the one that behaves like a marker:

* it is bimodal -- a switch, not a dial being swept
* its active periods fall **inside** the armed time, because you cannot mark a
  hover before you take off
* it is not simply the arming switch, which is excluded because it tracks the
  armed flag almost perfectly

Everything it considered is printed, so a wrong guess is visible rather than
silent. ``--marker-channel`` overrides it.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_hover import analyse, report  # noqa: E402

#: Raw gyro is counts at 16.4 per degree/second on this firmware. Measured
#: 18 Sep; it resolved an open 16x conflict in SPEC.md. Getting this wrong is
#: how the first hover attempt passed a gyro gate it should have failed.
GYRO_COUNTS_PER_DPS = 16.4

#: A switch has to move at least this far to be a switch.
MIN_SWING_US = 300.0
#: and spend at least this fraction of the armed time in each state, or it is
#: something that was flipped once rather than used to mark anything.
MIN_DUTY = 0.02


def load(path: Path) -> list[dict]:
    """Read a flight_recorder CSV into the shape test_hover's checks expect."""
    rows = []
    for r in csv.DictReader(open(path)):
        try:
            rpms = [float(r[f"rpm{i}"]) for i in range(4) if r.get(f"rpm{i}", "")]
            out = {
                "t": float(r["t"]),
                "armed": str(r["armed"]).strip().lower() in ("true", "1", "yes"),
                "roll_deg": float(r["roll_deg"]),
                "pitch_deg": float(r["pitch_deg"]),
                "yaw_deg": float(r["yaw_deg"]),
                "gx_dps": float(r["gyro_x_raw"]) / GYRO_COUNTS_PER_DPS,
                "gy_dps": float(r["gyro_y_raw"]) / GYRO_COUNTS_PER_DPS,
                "gz_dps": float(r["gyro_z_raw"]) / GYRO_COUNTS_PER_DPS,
                "acc_x_raw": r.get("acc_x_raw", ""),
                "acc_y_raw": r.get("acc_y_raw", ""),
                "acc_z_raw": r.get("acc_z_raw", ""),
                "motor_mean_us": float(r["motor_mean"]),
                "rpm_mean": statistics.fmean(rpms) if rpms else "",
                "voltage_v": r.get("voltage_v", ""),
                "current_a": r.get("current_a", ""),
            }
            for i in range(4):
                out[f"motor{i}"] = float(r[f"motor{i}"])
                out[f"rpm{i}"] = r.get(f"rpm{i}", "")
            for i in range(16):
                out[f"rc{i + 1}"] = r.get(f"rc{i + 1}", "")
            rows.append(out)
        except (ValueError, KeyError):
            continue                     # a torn final row is not a reason to fail
    return rows


def _channel_stats(rows: list[dict], ch: int) -> dict | None:
    vals = [float(r[f"rc{ch}"]) for r in rows if r.get(f"rc{ch}", "") not in ("", None)]
    if len(vals) < 10:
        return None
    lo, hi = min(vals), max(vals)
    if hi - lo < MIN_SWING_US:
        return None
    mid = (lo + hi) / 2.0
    high = [v for v in vals if v > mid]
    duty = len(high) / len(vals)
    return {"ch": ch, "lo": lo, "hi": hi, "mid": mid, "duty": duty}


def find_marker(rows: list[dict], armed_span: tuple[float, float] | None) -> tuple:
    """Pick the channel the pilot used to mark hovers. Returns (ch, considered)."""
    considered, best = [], None
    for ch in range(5, 17):                      # 1-4 are the sticks
        st = _channel_stats(rows, ch)
        if st is None:
            considered.append({"ch": ch, "verdict": "does not move"})
            continue
        active = [r for r in rows
                  if r.get(f"rc{ch}", "") not in ("", None)
                  and float(r[f"rc{ch}"]) > st["mid"]]
        if not active:
            considered.append({"ch": ch, "verdict": "no active period"})
            continue
        armed_frac = sum(1 for r in active if r["armed"]) / len(active)
        # The arming switch itself: active exactly when armed. Not a marker.
        arm_like = abs(st["duty"] - (sum(1 for r in rows if r["armed"]) / len(rows))) < 0.05
        verdict = None
        if st["duty"] < MIN_DUTY or st["duty"] > 1 - MIN_DUTY:
            verdict = f"flipped once (duty {st['duty']:.0%})"
        elif armed_frac < 0.8:
            verdict = f"mostly active while disarmed ({armed_frac:.0%} armed)"
        elif arm_like:
            verdict = "tracks the armed flag; this is the arming switch"
        else:
            verdict = f"CANDIDATE (duty {st['duty']:.0%}, {armed_frac:.0%} armed)"
            if best is None or st["duty"] < best[1]["duty"]:
                best = (ch, st)
        considered.append({"ch": ch, "verdict": verdict})
    return (best[0] if best else None), considered


def marks_from_channel(rows: list[dict], ch: int) -> list:
    """Turn a switch channel into (start, end) windows."""
    st = _channel_stats(rows, ch)
    if st is None:
        return []
    marks, open_at, prev_t = [], None, None
    for r in rows:
        v = r.get(f"rc{ch}", "")
        if v in ("", None):
            continue
        high = float(v) > st["mid"]
        if high and open_at is None:
            open_at = r["t"]
        elif not high and open_at is not None:
            # Close at the LAST high sample, not the first low one. The sample
            # where the switch goes low already belongs to whatever came next --
            # often the landing, where the aircraft is disarmed -- and including
            # it makes an otherwise good window fail "not armed throughout".
            marks.append((open_at, prev_t if prev_t is not None else r["t"]))
            open_at = None
        if high:
            prev_t = r["t"]
    if open_at is not None:
        marks.append((open_at, rows[-1]["t"]))
    return marks


def main() -> int:
    ap = argparse.ArgumentParser(description="Hover throttle from an unattended recording")
    ap.add_argument("csv", help="a flight_recorder.py CSV")
    ap.add_argument("--mass", type=float, default=1.745, help="race-ready mass, kg")
    ap.add_argument("--marker-channel", type=int, default=None,
                    help="RC channel the pilot flipped to mark hovers (1-16)")
    args = ap.parse_args()

    rows = load(Path(args.csv))
    if not rows:
        print("  no usable rows in that file")
        return 2

    armed = [r for r in rows if r["armed"]]
    print()
    print("FLIGHT RECORDING")
    print("=" * 68)
    print(f"  file            {args.csv}")
    print(f"  rows            {len(rows)} over {rows[-1]['t'] - rows[0]['t']:.0f} s")
    print(f"  armed for       {len(armed)} rows "
          f"({len(armed) / len(rows):.0%} of the recording)")
    if not armed:
        print()
        print("  THE AIRCRAFT NEVER ARMED IN THIS RECORDING.")
        print("  If it flew, the recorder was not running at the time.")
        print("=" * 68)
        return 3
    if any(r["rpm_mean"] != "" for r in armed):
        print("  RPM telemetry   present (hover gives the thrust coefficient too)")
    else:
        print("  RPM telemetry   ABSENT -- hover throttle only, no thrust coefficient")

    ch = args.marker_channel
    if ch is None:
        ch, considered = find_marker(rows, None)
        print()
        print("  MARKER SWITCH SEARCH")
        for c in considered:
            flag = "->" if c["ch"] == ch else "  "
            print(f"   {flag} ch{c['ch']:<3d} {c['verdict']}")
        if ch is None:
            print()
            print("  No channel looks like a hover marker. Either nobody flipped a")
            print("  switch, or it is one of the ones dismissed above -- pass")
            print("  --marker-channel N to force it.")
            print("=" * 68)
            return 4
    marks = marks_from_channel(rows, ch)
    print()
    print(f"  using channel {ch}: {len(marks)} marked window(s)")
    print("=" * 68)

    res = analyse(rows, marks, mass_kg=args.mass)
    report(res)
    return 0 if res.get("accepted") else 1


def _self_test() -> int:
    """Synthesise a flight, mark it on a spare channel, and recover the hover."""
    import tempfile, math, random
    rng = random.Random(7)
    true_hover_us = 1387.0            # a hover the analysis must find without being told
    path = Path(tempfile.mkdtemp()) / "flight.csv"

    from flight_recorder import FIELDS
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        t = 0.0
        # 5 s on the ground, then three hovers with climbs between them.
        phases = ([("ground", 5.0)] +
                  [p for _ in range(3)
                   for p in (("climb", 4.0), ("hover", 6.0))] +
                  [("ground", 3.0)])
        for kind, dur in phases:
            n = int(dur * 30)
            for i in range(n):
                t += 1 / 30.0
                if kind == "ground":
                    motor, armed, mark, tilt = 1000.0, False, 1000, 0.0
                elif kind == "climb":
                    motor, armed, mark, tilt = 1600.0, True, 1000, 6.0
                else:
                    motor, armed, mark, tilt = true_hover_us, True, 1800, 1.5
                row = {f: "" for f in FIELDS}
                row.update({
                    "t": round(t, 4), "wall": "00:00:00", "dt": round(1 / 30.0, 6),
                    "armed": armed, "arming_flags": "",
                    "roll_deg": round(rng.gauss(0.8, 0.3), 2),
                    "pitch_deg": round(rng.gauss(-2.1, 0.3), 2),
                    "yaw_deg": 180.0,
                    "gyro_x_raw": int(rng.gauss(0, 3) * 16.4),
                    "gyro_y_raw": int(rng.gauss(0, 3) * 16.4),
                    "gyro_z_raw": int(rng.gauss(0, 2) * 16.4),
                    "acc_x_raw": 0, "acc_y_raw": 0, "acc_z_raw": 512,
                    "motor_mean": round(motor + rng.gauss(0, 6), 1),
                    "voltage_v": 23.9, "current_a": 20.0, "mah_drawn": 500,
                    "altitude_m": 1.5, "vario_m_s": 0.0,
                })
                for k in range(4):
                    row[f"motor{k}"] = round(motor + rng.gauss(0, 6), 1)
                    row[f"rpm{k}"] = int(4800 * (motor - 1000) / 387.0)
                for k in range(16):
                    row[f"rc{k+1}"] = 1500
                row["rc5"] = 1800 if armed else 1000        # the ARM switch
                row["rc8"] = mark                            # the hover marker
                w.writerow(row)

    rows = load(path)
    assert len(rows) > 500, len(rows)

    ch, considered = find_marker(rows, None)
    assert ch == 8, f"picked channel {ch}, expected 8; {considered}"
    # and it must have dismissed the arming switch for the right reason
    arm_verdict = [c["verdict"] for c in considered if c["ch"] == 5][0]
    assert "arming switch" in arm_verdict or "armed" in arm_verdict, arm_verdict

    marks = marks_from_channel(rows, 8)
    assert len(marks) == 3, f"{len(marks)} windows, expected 3"

    res = analyse(rows, marks, mass_kg=1.745)
    assert res["accepted"], res
    err = abs(res["hover_motor_us"] - true_hover_us)
    assert err < 5.0, f"recovered {res['hover_motor_us']:.1f} us vs true {true_hover_us}"

    print("analyze_flight: all checks passed")
    print(f"  found the marker on channel {ch} without being told")
    print(f"  dismissed the arming switch on ch5 correctly")
    print(f"  recovered hover {res['hover_motor_us']:.1f} us against a true "
          f"{true_hover_us:.0f} us ({err:.1f} us error)")
    print(f"  hover fraction {res['hover_fraction']:.3f}, from motors not sticks")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    raise SystemExit(main())
