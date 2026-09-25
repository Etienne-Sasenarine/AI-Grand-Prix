"""Does dead reckoning plus gate fixes actually keep us located?

This is the experiment that decides whether the PnP-plus-filter approach is
sufficient, or whether we need something much larger (VIO). The bar is set by
the course: **the nearest two gates are 7.54 m apart**, so as long as our
position uncertainty stays well inside that, gate association is unambiguous
and the gate index is reliable.

The test flies the real course and feeds the filter what the drone will
actually have, with realistic errors injected:

  * thrust scale wrong by a fixed amount (the filter must learn it)
  * drag coefficient wrong (the filter never learns this one)
  * attitude noise on roll, pitch and yaw
  * yaw bias, i.e. gyro heading drift
  * gate fixes only when a gate is genuinely in view, with PnP-grade noise

Two configurations are compared:
  DEAD RECKONING ONLY  - no gate fixes at all. Shows how fast it falls apart.
  WITH GATE FIXES      - the proposed system.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve()
_root = next(p for p in _here.parents if (p / "estimation").is_dir() and (p / "deploy").is_dir())
sys.path[:0] = [str(_root / "estimation"), str(_root / "deploy" / "runtime")]

import geometry as G  # noqa: E402
import sim as S  # noqa: E402
from pose_filter import G_ACCEL, FilterConfig, PoseFilter  # noqa: E402
from pnp import camera_position_world_from_attitude, solve_gate_pose  # noqa: E402


def run_case(*, use_fixes: bool, thrust_err: float, drag_true: float, drag_model: float,
             att_noise_deg: float, yaw_drift_deg_s: float, corner_noise_px: float,
             detector_range: float, speed: float, seed: int, dt: float = 1 / 60.0):
    camera = G.Camera.real()
    sequence = G.race_sequence(laps=2)
    flight = S.FlightModel(speed=speed, seed=seed)
    detector = S.DetectorModel(max_range_m=detector_range, dropout=0.10,
                               pixel_noise_px=corner_noise_px,
                               noise_growth_per_m=0.2 * corner_noise_px)
    rng = np.random.default_rng(seed)

    cfg = FilterConfig(drag_k=drag_model)
    filt = PoseFilter(cfg=cfg)

    att_noise = math.radians(att_noise_deg)
    yaw_drift = math.radians(yaw_drift_deg_s)

    prev_vel = None
    errors, sigmas, fix_errors = [], [], []
    started = False
    yaw_err = 0.0

    for t, pos, quat, vel, gyro, true_i in flight.run(sequence, dt=dt):
        if not started:
            filt.initialise(pos, vel)      # known start box
            started = True
            prev_vel = vel.copy()
            continue

        # --- what the flight controller reports -------------------------
        roll, pitch = G.roll_pitch_from_quat(quat)
        yaw = math.atan2(2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
                         1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2))
        yaw_err += yaw_drift * dt
        roll_m = roll + rng.normal(0.0, att_noise)
        pitch_m = pitch + rng.normal(0.0, att_noise)
        yaw_m = yaw + yaw_err + rng.normal(0.0, att_noise)

        # --- the thrust the autopilot is commanding ---------------------
        accel_true = (vel - prev_vel) / dt
        prev_vel = vel.copy()
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        up = np.array([cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr])
        required = accel_true + np.array([0.0, 0.0, G_ACCEL]) + drag_true * vel
        specific_thrust = float(np.dot(up, required))
        # The drone delivers less (or more) than commanded; the filter must
        # discover that factor for itself.
        commanded = specific_thrust / thrust_err

        filt.predict(dt, roll=roll_m, pitch=pitch_m, yaw=yaw_m, specific_thrust=commanded)

        # --- gate fixes --------------------------------------------------
        if use_fixes:
            window = sequence[max(0, true_i - 1): true_i + 3]
            for g in window:
                d = float(np.linalg.norm(g.centre - pos))
                if d > detector_range or d < 0.8:
                    continue
                uv, vis = camera.project(g.corners_world(), pos, quat)
                if vis.sum() < 4 or rng.random() < 0.10:
                    continue
                sigma = corner_noise_px + 0.2 * corner_noise_px * d
                uv_n = uv + rng.normal(0.0, sigma, size=uv.shape)
                pose = solve_gate_pose(uv_n, vis, camera, roll=roll_m, pitch=pitch_m)
                if pose is None or pose.reprojection_px > 25.0:
                    continue
                quat_m = G.quat_from_euler(roll_m, pitch_m, yaw_m)
                fix = camera_position_world_from_attitude(pose, g, quat_m, camera)
                fix_errors.append(float(np.linalg.norm(fix - pos)))
                filt.update_gate_fix(fix, range_m=pose.distance, gate=g)
                # Heading correction: where the gate actually sits in the body
                # frame, from PnP's (well-conditioned) translation.
                r_cb = G.camera_body_rotation(camera.tilt_up_deg)
                p_flu = G.flu_to_ned(r_cb.T @ pose.t_cam)
                filt.update_yaw_from_bearing(g, math.atan2(p_flu[1], p_flu[0]))
                break   # one fix per frame, as the real tracker would

        errors.append(float(np.linalg.norm(filt.pos - pos)))
        sigmas.append(filt.position_sigma)

    errors = np.array(errors)
    return {
        "median": float(np.median(errors)),
        "p95": float(np.percentile(errors, 95)),
        "max": float(errors.max()),
        "fixes": filt.n_fixes,
        "rejected": filt.n_rejected,
        "thrust_scale": filt.thrust_scale,
        "thrust_truth": thrust_err,
        "fix_err_median": float(np.median(fix_errors)) if fix_errors else float("nan"),
        "frames": len(errors),
    }


BASE = dict(thrust_err=0.90, drag_true=0.35, drag_model=0.30, att_noise_deg=1.0,
            yaw_drift_deg_s=0.5, corner_noise_px=3.0, detector_range=12.0,
            speed=5.0, seed=1)


def main() -> int:
    print()
    print("Can we stay located with gate fixes and no VIO?")
    print("=" * 84)
    print("The bar: the two nearest gates on this course are 7.54 m apart.")
    print()
    print(f"  {'case':<40} {'median':>8} {'p95':>8} {'max':>8} {'fixes':>7} {'rej':>5}")
    print("-" * 84)

    cases = [
        ("dead reckoning only, no gate fixes", dict(use_fixes=False)),
        ("with gate fixes (baseline)", dict(use_fixes=True)),
        ("+ 20 % thrust error", dict(use_fixes=True, thrust_err=0.80)),
        ("+ drag model 2x wrong", dict(use_fixes=True, drag_model=0.15, drag_true=0.45)),
        ("+ 3 deg attitude noise", dict(use_fixes=True, att_noise_deg=3.0)),
        ("+ 3 deg/s yaw drift", dict(use_fixes=True, yaw_drift_deg_s=3.0)),
        ("+ 10 px corner noise", dict(use_fixes=True, corner_noise_px=10.0)),
        ("+ short detector range 7 m", dict(use_fixes=True, detector_range=7.0)),
        ("+ fast flight 12 m/s", dict(use_fixes=True, speed=12.0)),
        ("everything bad at once", dict(use_fixes=True, thrust_err=0.80, drag_model=0.15,
                                        drag_true=0.45, att_noise_deg=3.0,
                                        yaw_drift_deg_s=3.0, corner_noise_px=10.0,
                                        detector_range=8.0, speed=9.0)),
    ]

    worst = 0.0
    for name, over in cases:
        kw = dict(BASE)
        kw.update(over)
        use_fixes = kw.pop("use_fixes")
        r = run_case(use_fixes=use_fixes, **kw)
        flag = ""
        if use_fixes:
            worst = max(worst, r["p95"])
            if r["p95"] > 7.54 / 2:
                flag = "  <-- exceeds half the gate separation"
        print(f"  {name:<40} {r['median']:>7.2f}m {r['p95']:>7.2f}m {r['max']:>7.2f}m "
              f"{r['fixes']:>7} {r['rejected']:>5}{flag}")
        if use_fixes and name == "with gate fixes (baseline)":
            print(f"  {'':<40} thrust scale learned {r['thrust_scale']:.3f} "
                  f"(truth {r['thrust_truth']:.3f}); single-fix error "
                  f"{r['fix_err_median']:.2f} m")

    print("-" * 84)
    print(f"  worst p95 with fixes: {worst:.2f} m against a 7.54 m gate separation "
          f"({7.54 / max(worst, 1e-9):.1f}x margin)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
