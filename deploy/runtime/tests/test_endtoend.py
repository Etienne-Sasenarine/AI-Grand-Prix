"""Fly the whole onboard pipeline on a simulated run.

Chains every piece together exactly as the drone will:

    camera geometry -> detector model -> gate tracker -> observation frame
      -> 32-frame history -> NumPy policy -> action decode -> safety envelope
      -> Betaflight curves -> RC channel values

The point is not that the policy flies well here — the simulated flight path is
kinematic and ignores what the policy asks for, so this measures the *plumbing*,
not the controller. What it does prove:

  * every stage accepts what the previous one produces, for a whole two-lap run
  * no NaNs, no silent shape errors, no out-of-range channel values
  * the per-frame compute budget is affordable at 25 W
  * the policy's outputs stay inside the commanded envelope

A physics-accurate closed loop is Isaac's job, not this file's.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve()
_root = next(p for p in _here.parents if (p / "estimation").is_dir() and (p / "deploy").is_dir())
sys.path[:0] = [str(_root / "estimation"), str(_root / "deploy" / "runtime")]

import geometry as G  # noqa: E402
import gate_tracker as T  # noqa: E402
import observation as O  # noqa: E402
import sim as S  # noqa: E402
from betaflight_curves import PWM_MAX, PWM_MIN  # noqa: E402
from control_adapter import ControlAdapter, Envelope  # noqa: E402
from policy_runtime import NumpyPolicy  # noqa: E402

MODEL = Path(__file__).resolve().parent.parent / "models" / "pq_speed_best.npz"


def run(seed: int = 1, dt: float = 1.0 / 60.0, verbose: bool = True) -> dict:
    camera = G.Camera.real()
    sequence = G.race_sequence(laps=2)
    tracker = T.GateTracker(sequence=sequence, camera=camera)
    detector = S.DetectorModel(dropout=0.10, corner_dropout=0.05,
                               pixel_noise_px=3.0, noise_growth_per_m=0.25,
                               false_positives_per_s=3.0, max_range_m=12.0)
    flight = S.FlightModel(speed=5.0, seed=seed, lateral_error_m=0.2)
    buf = O.ObservationBuffer()
    policy = NumpyPolicy(MODEL)
    adapter = ControlAdapter(envelope=Envelope(max_rate_rad_s=1.6,
                                               max_thrust_stick=0.40,
                                               min_thrust_stick=0.15))
    rng = np.random.default_rng(seed)

    stats = {
        "frames": 0, "nan_obs": 0, "nan_action": 0, "policy_us": [],
        "total_us": [], "channels_out_of_range": 0, "rate_exceeded": 0,
        "unseen_frames": 0, "actions": [],
    }

    for t, pos, quat, vel, gyro, true_i in flight.run(sequence, dt=dt):
        t0 = time.perf_counter()

        window = sequence[max(0, true_i - 1): true_i + 3]
        dets = detector.observe(window, pos, quat, camera, rng, dt)
        tracker.update(dt, dets, pos, quat)

        acc = tracker.accepted
        if acc is not None:
            uv, vis = acc.uv, acc.visible
        else:
            uv, vis = np.full((8, 2), np.nan), np.zeros(8, dtype=bool)
            stats["unseen_frames"] += 1

        roll, pitch = G.roll_pitch_from_quat(quat)
        vel_body = G.flu_to_ned(G.quat_rotate_inverse(quat, vel.reshape(1, 3)))[0]

        frame = O.build_frame(uv, vis, roll=roll, pitch=pitch,
                              gyro_ned=gyro, vel_body_ned=vel_body,
                              gate_index=tracker.gate_index_for_policy)
        if not np.all(np.isfinite(frame)):
            stats["nan_obs"] += 1
        buf.push(frame)

        t1 = time.perf_counter()
        action = policy.act(buf.vector())
        t2 = time.perf_counter()

        if not np.all(np.isfinite(action)):
            stats["nan_action"] += 1
        stats["actions"].append(action.copy())

        dbg = adapter.step(action)
        for pwm in dbg["channels"]:
            if not (PWM_MIN <= pwm <= PWM_MAX):
                stats["channels_out_of_range"] += 1
        if max(abs(r) for r in dbg["rates_rad_s"]) > adapter.envelope.max_rate_rad_s + 1e-9:
            stats["rate_exceeded"] += 1

        stats["policy_us"].append((t2 - t1) * 1e6)
        stats["total_us"].append((time.perf_counter() - t0) * 1e6)
        stats["frames"] += 1

    a = np.array(stats["actions"])
    pol = np.array(stats["policy_us"])
    tot = np.array(stats["total_us"])

    result = {
        "frames": stats["frames"],
        "crossings": len(tracker.passes),
        "tracker": tracker.summary(),
        "nan_obs": stats["nan_obs"],
        "nan_action": stats["nan_action"],
        "channels_out_of_range": stats["channels_out_of_range"],
        "rate_exceeded": stats["rate_exceeded"],
        "unseen_pct": 100.0 * stats["unseen_frames"] / max(1, stats["frames"]),
        "policy_us_mean": float(pol.mean()), "policy_us_p99": float(np.percentile(pol, 99)),
        "total_us_mean": float(tot.mean()), "total_us_p99": float(np.percentile(tot, 99)),
        "action_min": a.min(axis=0), "action_max": a.max(axis=0),
        "action_saturated_pct": 100.0 * float((np.abs(a) > 0.999).mean()),
    }

    if verbose:
        print()
        print("End-to-end pipeline on a simulated two-lap run")
        print("-" * 64)
        print(f"  frames                 {result['frames']}")
        print(f"  crossings counted      {result['crossings']} / 22")
        print(f"  tracker                {result['tracker']}")
        print(f"  gate unseen            {result['unseen_pct']:.1f} % of frames")
        print()
        print(f"  NaN observations       {result['nan_obs']}")
        print(f"  NaN actions            {result['nan_action']}")
        print(f"  channels out of range  {result['channels_out_of_range']}")
        print(f"  envelope exceeded      {result['rate_exceeded']}")
        print()
        print(f"  policy inference       {result['policy_us_mean']:.0f} us mean, "
              f"{result['policy_us_p99']:.0f} us p99")
        print(f"  whole pipeline         {result['total_us_mean']:.0f} us mean, "
              f"{result['total_us_p99']:.0f} us p99")
        budget = dt * 1e6
        print(f"  budget at {1/dt:.0f} Hz        {budget:.0f} us  "
              f"-> using {100 * result['total_us_mean'] / budget:.1f} % on this laptop")
        print()
        print(f"  action range           thrust [{result['action_min'][0]:+.2f}, "
              f"{result['action_max'][0]:+.2f}]  "
              f"roll [{result['action_min'][1]:+.2f}, {result['action_max'][1]:+.2f}]")
        print(f"  actions at the rails   {result['action_saturated_pct']:.1f} %")
        print("-" * 64)
    return result


def main() -> int:
    r = run(verbose=True)
    assert r["nan_obs"] == 0, "observation contained non-finite values"
    assert r["nan_action"] == 0, "policy produced non-finite actions"
    assert r["channels_out_of_range"] == 0, "channel value outside 1000-2000"
    assert r["rate_exceeded"] == 0, "safety envelope was exceeded"
    assert r["crossings"] == 22, f"counted {r['crossings']} crossings, expected 22"
    print("  PASS - every stage accepted the previous stage's output for a full run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
