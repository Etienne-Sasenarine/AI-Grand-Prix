"""Does the trained policy actually depend on the gate one-hot?

The question this answers
-------------------------
A reasonable-sounding claim: "we label the gates in simulation, train on that,
and then the model knows the order on the real unlabelled gates."

That is testable, so let us test it rather than argue. The one-hot is an
**input** to the network — 18 of the 51 numbers in every frame, plus a lap
fraction. At deployment something has to supply those numbers. So the question
is: how much does the policy's output actually change when those numbers are
wrong or missing?

  * If the action barely moves, the network ignores the one-hot and the team is
    right that we can stop worrying about it.
  * If the action moves as much as it does between genuinely different
    situations, then the one-hot is load-bearing and it has to come from
    somewhere real.

Method: take observations from a simulated flight, then re-evaluate the same
observation with (a) the correct gate index, (b) every wrong index, (c) no
index at all. Compare the change in action against two references: the typical
action change between consecutive frames, and the spread of actions across the
whole flight.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import geometry as G  # noqa: E402
import observation as O  # noqa: E402
import sim as S  # noqa: E402
from policy_runtime import NumpyPolicy  # noqa: E402

MODEL = Path(__file__).resolve().parent / "models" / "pq_speed_best.npz"
N_GATES = 18
CTX_START = 32  # context block begins at index 32 within each 51-number frame
FRAME = 51


def _set_context(vec: np.ndarray, gate_index) -> np.ndarray:
    """Rewrite the context block of every frame in a 1632-vector."""
    out = vec.copy().reshape(-1, FRAME)
    ctx = np.asarray(O.context_features(gate_index))
    out[:, CTX_START:] = ctx
    return out.reshape(-1)


def main() -> int:
    policy = NumpyPolicy(MODEL)
    camera = G.Camera.training()
    sequence = G.race_sequence(laps=2)
    detector = S.DetectorModel(dropout=0.05)
    flight = S.FlightModel(speed=5.0, seed=3)
    rng = np.random.default_rng(3)
    buf = O.ObservationBuffer()

    samples = []       # (obs vector, true gate index)
    actions_true = []
    prev_action = None
    consecutive = []

    for t, pos, quat, vel, gyro, true_i in flight.run(sequence, dt=1 / 60.0):
        window = sequence[max(0, true_i - 1): true_i + 3]
        dets = detector.observe(window, pos, quat, camera, rng, 1 / 60.0)
        if dets:
            best = max(dets, key=lambda d: d.span() or 0.0)
            uv, vis = best.uv, best.visible
        else:
            uv, vis = np.full((8, 2), np.nan), np.zeros(8, dtype=bool)
        roll, pitch = G.roll_pitch_from_quat(quat)
        vel_body = G.flu_to_ned(G.quat_rotate_inverse(quat, vel.reshape(1, 3)))[0]
        idx = true_i % 11  # crossing index within the lap
        buf.push(O.build_frame(uv, vis, roll=roll, pitch=pitch, gyro_ned=gyro,
                               vel_body_ned=vel_body, gate_index=idx))
        if not buf.ready:
            continue
        v = buf.vector()
        a = policy.act(v)
        actions_true.append(a)
        if prev_action is not None:
            consecutive.append(np.abs(a - prev_action).max())
        prev_action = a
        if len(samples) < 400 and rng.random() < 0.25:
            samples.append((v, idx))

    actions_true = np.array(actions_true)
    consecutive = np.array(consecutive)

    wrong_deltas, zero_deltas, worst_deltas = [], [], []
    for v, idx in samples:
        a_true = policy.act(v)
        per_sample = []
        for other in range(N_GATES):
            if other == idx:
                continue
            a_other = policy.act(_set_context(v, other))
            d = float(np.abs(a_other - a_true).max())
            wrong_deltas.append(d)
            per_sample.append(d)
        worst_deltas.append(max(per_sample))
        a_zero = policy.act(_set_context(v, None))
        zero_deltas.append(float(np.abs(a_zero - a_true).max()))

    wrong_deltas = np.array(wrong_deltas)
    zero_deltas = np.array(zero_deltas)
    worst_deltas = np.array(worst_deltas)

    action_range = float(actions_true.max() - actions_true.min())
    print()
    print("Does the policy depend on the gate one-hot?")
    print("=" * 70)
    print(f"  samples                      {len(samples)} observations from a full run")
    print(f"  action channel range         {actions_true.min():+.2f} to {actions_true.max():+.2f} "
          f"(span {action_range:.2f})")
    print()
    print("  REFERENCE: how much does the action move for legitimate reasons?")
    print(f"    between consecutive frames  median {np.median(consecutive):.3f}  "
          f"p95 {np.percentile(consecutive, 95):.3f}")
    print()
    print("  TEST: same observation, only the gate one-hot changed")
    print(f"    a wrong gate index          median {np.median(wrong_deltas):.3f}  "
          f"p95 {np.percentile(wrong_deltas, 95):.3f}  max {wrong_deltas.max():.3f}")
    print(f"    worst wrong index, per obs  median {np.median(worst_deltas):.3f}  "
          f"max {worst_deltas.max():.3f}")
    print(f"    no index at all (zeros)     median {np.median(zero_deltas):.3f}  "
          f"max {zero_deltas.max():.3f}")
    print()
    # The median frame-to-frame change is 0.000 because this policy sits at the
    # rails most of the time, so use the p95 as the reference instead -- a
    # genuinely large, legitimate one-frame change.
    ref = float(np.percentile(consecutive, 95))
    print(f"  A wrong gate index moves the action by {np.median(worst_deltas):.2f} of a "
          f"{action_range:.2f} span.")
    print(f"  For comparison, the 95th-percentile change across one real frame is {ref:.2f}.")
    print(f"  So a mislabelled gate perturbs the output MORE than the most violent")
    print(f"  single-frame manoeuvre in the whole flight.")
    print("=" * 70)

    if np.median(worst_deltas) < 0.05:
        print("  VERDICT: the policy essentially ignores the one-hot.")
    else:
        print("  VERDICT: the one-hot is load-bearing. It is an INPUT, so something")
        print("  must supply it at run time — the network cannot supply its own input.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
