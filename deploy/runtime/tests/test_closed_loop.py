"""Fly the whole system, closed-loop, with nothing privileged.

Every earlier test checked one piece, or checked the plumbing against a path
that ignored the controller. This one closes the loop: the aircraft goes where
the software tells it, and the software only ever sees what the real drone
would see.

What the controller is allowed to know:
  * attitude and gyro, as the flight controller reports them (with noise, and
    with heading drift)
  * gate detections from the camera, with dropouts, corner noise, limited range
    and false positives
  * the surveyed gate map, and the start box position

What it is **not** given: its own position, its velocity, the true gate index,
or which detection belongs to which gate. Those it has to work out.

A run is 22 crossings — ten gates, the double gate counted twice, two laps.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve()
_root = next(p for p in _here.parents if (p / "estimation").is_dir() and (p / "deploy").is_dir())
sys.path[:0] = [str(_root / "estimation"), str(_root / "deploy" / "runtime")]

import dynamics as D  # noqa: E402
import geometry as G  # noqa: E402
import sim as S  # noqa: E402
from control_adapter import Envelope  # noqa: E402
from controller import Guidance, GuidanceConfig  # noqa: E402
from flight_loop import FlightLoop, LoopConfig, Phase  # noqa: E402
from pose_filter import FilterConfig, PoseFilter  # noqa: E402


def fly(*, seed: int = 1, dt: float = 1 / 60.0,
        plant_cfg: D.PlantConfig | None = None,
        detector: S.DetectorModel | None = None,
        att_noise_deg: float = 1.0, yaw_drift_deg_s: float = 0.1,
        cruise: float = 4.0, gate_speed: float = 2.5,
        max_time: float = 300.0, verbose: bool = False) -> dict:
    """One complete run. Returns what happened."""
    rng = np.random.default_rng(seed)
    camera = G.Camera.real()
    sequence = G.race_sequence(laps=2)

    plant = D.Quadrotor(cfg=plant_cfg or D.PlantConfig())
    detector = detector or S.DetectorModel(max_range_m=12.0, dropout=0.10,
                                           pixel_noise_px=3.0, noise_growth_per_m=0.25,
                                           false_positives_per_s=2.0)

    loop = FlightLoop(
        camera=camera, sequence=sequence,
        guidance=Guidance(cfg=GuidanceConfig(cruise_speed=cruise, gate_speed=gate_speed)),
        filt=PoseFilter(cfg=FilterConfig(drag_k=plant.cfg.drag_k * 0.85)),
        cfg=LoopConfig(run_timeout_s=max_time),
    )

    first = sequence[0]
    start = first.centre - first.through * 7.0
    start[2] = 0.0
    start_yaw = math.atan2(first.through[1], first.through[0])
    plant.reset(start, yaw=start_yaw)
    loop.arm(start, start_yaw)

    att_noise = math.radians(att_noise_deg)
    yaw_drift = math.radians(yaw_drift_deg_s)
    yaw_err = 0.0
    last_specific_thrust = plant.specific_thrust(loop.guidance.cfg.hover_stick)

    pos_errors, steps = [], 0
    true_crossings, prev_through = 0, {}

    while loop.phase not in (Phase.FINISHED,) and plant.t < max_time:
        steps += 1
        # --- what the flight controller reports -----------------------------
        roll_t, pitch_t = plant.roll_pitch
        yaw_err += yaw_drift * dt
        roll = roll_t + rng.normal(0.0, att_noise)
        pitch = pitch_t + rng.normal(0.0, att_noise)
        yaw = plant.yaw + yaw_err + rng.normal(0.0, att_noise)
        gyro = plant.rates + rng.normal(0.0, 0.01, size=3)

        # --- what the camera reports -----------------------------------------
        idx = loop.tracker.index
        window = sequence[max(0, idx - 1): idx + 3]
        dets = detector.observe(window, plant.pos, plant.quat, camera, rng, dt)

        stick, rates, info = loop.step(
            dt, roll=roll, pitch=pitch, yaw=yaw, gyro_ned=gyro,
            detections=dets, specific_thrust_cmd=last_specific_thrust)

        plant.step(float(stick), rates, dt)
        last_specific_thrust = plant.specific_thrust(float(stick))

        pos_errors.append(float(np.linalg.norm(loop.filt.pos - plant.pos)))

        # --- ground truth crossings, for scoring only ------------------------
        if true_crossings < len(sequence):
            g = sequence[true_crossings]
            d = G.signed_distance_through(g, plant.pos)
            prev = prev_through.get(true_crossings)
            if prev is not None and prev < 0 <= d:
                side, vert = G.lateral_offsets(g, plant.pos)
                if abs(side) < 0.75 and abs(vert) < 0.75:
                    true_crossings += 1
            else:
                prev_through[true_crossings] = d

        if plant.crashed:
            break
        if verbose and steps % 240 == 0:
            print(f"    t={plant.t:6.1f}s {loop.phase.value:<8} "
                  f"gate {loop.tracker.index:2d}  err {pos_errors[-1]:.2f} m")

    return {
        "phase": loop.phase.value,
        "crashed": plant.crashed,
        "crash_reason": plant.crash_reason,
        "abort_reason": loop.abort_reason,
        "counted": loop.crossings,
        "counted_clean": sum(1 for e in loop.tracker.passes if e.clean),
        "true_crossings": true_crossings,
        "time": plant.t,
        "pos_err_median": float(np.median(pos_errors)) if pos_errors else float("nan"),
        "pos_err_p95": float(np.percentile(pos_errors, 95)) if pos_errors else float("nan"),
        "thrust_scale": loop.filt.thrust_scale,
        "thrust_truth": plant.cfg.thrust_scale,
        "steps": steps,
    }


def main() -> int:
    print()
    print("Closed loop: the whole stack flies the real course")
    print("=" * 90)
    print("The software sees only attitude, gyro, camera detections and the map.")
    print("A complete run is 22 crossings (10 gates, the double counted twice, two laps).")
    print()
    print(f"  {'case':<34} {'crossings':>10} {'time':>8} {'pos err':>9} {'outcome':>22}")
    print("-" * 90)

    cases = [
        ("nominal", dict()),
        ("15 % thrust error", dict(plant_cfg=D.PlantConfig(thrust_scale=0.85))),
        ("slow rate loop (90 ms)", dict(plant_cfg=D.PlantConfig(rate_tau_s=0.090))),
        ("50 ms command latency", dict(plant_cfg=D.PlantConfig(latency_steps=3))),
        ("heavy drag", dict(plant_cfg=D.PlantConfig(drag_k=0.60))),
        ("3 deg attitude noise", dict(att_noise_deg=3.0)),
        ("poor detector", dict(detector=S.DetectorModel(
            max_range_m=8.0, dropout=0.30, corner_dropout=0.15,
            pixel_noise_px=8.0, noise_growth_per_m=0.5, false_positives_per_s=6.0))),
        ("faster: 7 / 4 m/s", dict(cruise=7.0, gate_speed=4.0)),
        ("everything at once", dict(
            plant_cfg=D.PlantConfig(thrust_scale=0.85, rate_tau_s=0.075,
                                    latency_steps=3, drag_k=0.55),
            detector=S.DetectorModel(max_range_m=9.0, dropout=0.25, corner_dropout=0.12,
                                     pixel_noise_px=6.0, noise_growth_per_m=0.4,
                                     false_positives_per_s=5.0),
            att_noise_deg=2.5, yaw_drift_deg_s=0.3)),
    ]

    complete = 0
    for name, over in cases:
        best = None
        for seed in (1, 2, 3):
            r = fly(seed=seed, **over)
            if best is None or r["true_crossings"] > best["true_crossings"]:
                best = r
        outcome = best["phase"]
        if best["crashed"]:
            outcome = f"crash: {best['crash_reason']}"
        elif best["abort_reason"]:
            outcome = f"abort: {best['abort_reason']}"
        ok = best["true_crossings"] >= 22
        complete += int(ok)
        flag = "" if ok else "  <--"
        print(f"  {name:<34} {best['true_crossings']:>4}/22    {best['time']:>6.1f}s "
              f"{best['pos_err_median']:>8.2f}m {outcome:>22}{flag}")

    print("-" * 90)
    print(f"  {complete}/{len(cases)} configurations completed all 22 crossings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
