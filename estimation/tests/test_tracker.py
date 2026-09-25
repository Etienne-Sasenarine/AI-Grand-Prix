"""Stress the gate tracker across conditions, and find where it breaks.

A tracker that works on a clean synthetic flight proves very little. These
cases are chosen to break it: detections dropping out, corners missing, false
positives from pylons and signage, a short-range detector, no pose estimate at
all, a noisy pose, pipeline latency, and a sloppy flight path that arrives at
gates well off-centre.

Scoring per case, over several seeds:
  counted    crossings declared (22 is right: 10 gates, the double counted
             twice, two laps)
  wrong      bookkeeping errors: the k-th declared crossing was not
             ``sequence[k]``. This is the number that must be zero
  lag        largest gap, in crossings, between the tracker's belief and the
             truth at the instant an event fired. The vision fallback fires
             shortly *after* the crossing, so a lag of 1 is expected, not a
             fault
  lost %     fraction of frames with no confident association
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import geometry as G  # noqa: E402
import gate_tracker as T  # noqa: E402
import sim as S  # noqa: E402

EXPECTED = 22
SEEDS = (1, 2, 3, 4, 5)


def case(name, *, detector=None, flight=None, give_pose=True, pose_noise_m=0.0,
         camera=None, seeds=SEEDS):
    counted, wrong, lost, steps, reassoc, order_ok, lag = [], [], [], [], [], True, []
    confident_frac = []
    for sd in seeds:
        tr = T.GateTracker(sequence=G.race_sequence(laps=2),
                           camera=camera or G.Camera.training())
        d = detector or S.DetectorModel()
        f = flight or S.FlightModel()
        f.seed = sd
        r = S.run_tracker_sim(tr, detector=d, flight=f, camera=camera or tr.camera,
                              seed=sd, give_pose=give_pose, pose_noise_m=pose_noise_m)
        confident_frac.append(r.confident_frames / max(1, r.steps))
        counted.append(r.counted)
        wrong.append(r.wrong_gate)
        lost.append(r.lost_frames)
        steps.append(r.steps)
        reassoc.append(r.reassociations)
        lag.append(r.lag_index)
        order_ok = order_ok and r.correct_order
    lost_pct = 100.0 * sum(lost) / max(1, sum(steps))
    return {
        "name": name,
        "counted_min": min(counted), "counted_max": max(counted),
        "wrong": sum(wrong), "lost_pct": lost_pct, "lag": max(lag),
        "reassoc": sum(reassoc) / len(seeds), "order_ok": order_ok,
        "confident": 100.0 * sum(confident_frac) / len(seeds),
    }


def main() -> int:
    cases = []

    cases.append(case("reference (clean detector, good pose)"))

    cases.append(case("heavy dropout 25 %",
                      detector=S.DetectorModel(dropout=0.25, corner_dropout=0.10)))

    cases.append(case("severe dropout 50 %",
                      detector=S.DetectorModel(dropout=0.50, corner_dropout=0.20)))

    cases.append(case("noisy corners 8 px",
                      detector=S.DetectorModel(pixel_noise_px=8.0, noise_growth_per_m=0.5)))

    cases.append(case("false positives 5 /s",
                      detector=S.DetectorModel(false_positives_per_s=5.0)))

    cases.append(case("false positives 20 /s",
                      detector=S.DetectorModel(false_positives_per_s=20.0)))

    cases.append(case("short range 6 m",
                      detector=S.DetectorModel(max_range_m=6.0)))

    cases.append(case("no back-of-gate detection",
                      detector=S.DetectorModel(behind_gate_visible=False)))

    cases.append(case("detection latency 6 frames (100 ms)",
                      detector=S.DetectorModel(latency_frames=6)))

    cases.append(case("pose noise 0.30 m", pose_noise_m=0.30))
    cases.append(case("pose noise 1.00 m", pose_noise_m=1.00))

    cases.append(case("sloppy flight (0.35 m lateral, 0.25 m vertical)",
                      flight=S.FlightModel(lateral_error_m=0.35, vertical_error_m=0.25)))

    cases.append(case("fast flight 9 m/s",
                      flight=S.FlightModel(speed=9.0, lag_s=0.25)))

    cases.append(case("real camera (fx 425, 12 cm forward)",
                      camera=G.Camera.real()))

    # These two are expected to be unreliable, and the point of the test is
    # that the tracker *says so*. See the module docstring in gate_tracker.
    cases.append(case("NO POSE (degraded; must report not-confident)", give_pose=False))

    cases.append(case("no pose + 25 % dropout (degraded)", give_pose=False,
                      detector=S.DetectorModel(dropout=0.25, corner_dropout=0.10)))

    cases.append(case("everything at once",
                      detector=S.DetectorModel(dropout=0.25, corner_dropout=0.10,
                                               pixel_noise_px=6.0, noise_growth_per_m=0.4,
                                               false_positives_per_s=8.0,
                                               max_range_m=9.0, latency_frames=4),
                      flight=S.FlightModel(lateral_error_m=0.30, vertical_error_m=0.20),
                      pose_noise_m=0.30, camera=G.Camera.real()))

    print()
    print(f"{'case':<46} {'counted':>9} {'wrong':>6} {'lag':>4} {'lost%':>7} {'conf%':>7}")
    print("-" * 94)
    failures = []
    for c in cases:
        span = (f"{c['counted_min']}" if c["counted_min"] == c["counted_max"]
                else f"{c['counted_min']}-{c['counted_max']}")
        degraded = "degraded" in c["name"] or "NO POSE" in c["name"]
        if degraded:
            # Success here means honestly reporting low confidence.
            ok = c["confident"] < 1.0
        else:
            ok = (c["counted_min"] == c["counted_max"] == EXPECTED
                  and c["wrong"] == 0 and c["lag"] <= 1)
        flag = "" if ok else "  <-- "
        print(f"{c['name']:<46} {span:>9} {c['wrong']:>6} {c['lag']:>4} {c['lost_pct']:>6.1f}% "
              f"{c['confident']:>6.1f}%{flag}")
        if flag:
            failures.append(c["name"])

    print("-" * 94)
    print(f"expected {EXPECTED} crossings per run (10 gates, double gate twice, two laps)")
    if failures:
        print(f"\n{len(failures)} case(s) not perfect:")
        for f in failures:
            print(f"  - {f}")
    else:
        print("\nall pose-based cases: exactly 22 crossings, correct, lag <= 1")
        print("degraded (no-pose) cases: correctly reported as not confident")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
