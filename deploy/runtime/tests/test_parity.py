"""Cross-check our NumPy geometry against the torch code the policy trained on.

If these disagree, the network sees something different in flight from what it
saw in training, and no amount of tuning will fix it. This is the single most
important test in the package.

Run it from the laptop (needs torch). It is not needed on the drone.
"""

from __future__ import annotations

import math
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import geometry as G  # noqa: E402

def _training_fx(d: Path) -> float | None:
    """Read FX out of a candidate's aigp_obs.py without importing it.

    Importing the wrong copy would poison sys.modules for the rest of the run,
    and the value is a plain module-level constant, so a read is enough.
    """
    try:
        src = (d / "utils" / "aigp_obs.py").read_text(encoding="utf-8")
    except OSError:
        return None
    m = re.search(r"^FX\s*=\s*([0-9.]+)", src, re.MULTILINE)
    return float(m.group(1)) if m else None


def _find_training_dir() -> Path:
    """Locate the isaac_drone_racer copy the checkpoint was actually trained on.

    Two things vary. The path: on the team laptop flight/ sits at the workspace
    root beside a clone of AI_GP, while in-repo it sits at pq/flight/ and the
    training code is two levels up. And the camera: the repo's main branch
    carries the corrected PQ camera (FX 423.6) while the checkpoint in
    flight/models was trained against the virtual-qualifier camera (FX 320).

    Comparing against the wrong camera does not error, it just reports a ~100 px
    parity failure that looks like a geometry bug. So select on FX rather than
    on path order, and say plainly what was found if nothing matches.
    """
    here = Path(__file__).resolve()
    candidates = [c for c in (
        Path(os.environ["AIGP_TRAIN_DIR"]) if os.environ.get("AIGP_TRAIN_DIR") else None,
        here.parents[1] / "AI_GP" / "isaac_drone_racer",   # laptop workspace
        here.parents[2] / "isaac_drone_racer",             # in-repo, under pq/
        here.parents[1] / "isaac_drone_racer",
    ) if c is not None]

    found = [(c, _training_fx(c)) for c in candidates]
    for c, fx in found:
        if fx is not None and abs(fx - G.TRAINING_FX) < 1e-6:
            return c

    seen = ", ".join(f"{c} (FX={fx})" for c, fx in found if fx is not None)
    raise SystemExit(
        f"no isaac_drone_racer copy matches the checkpoint's camera "
        f"(TRAINING_FX={G.TRAINING_FX}).\n"
        f"Found: {seen or 'none'}\n"
        "Set AIGP_TRAIN_DIR to the copy the checkpoint was trained with, or "
        "retrain against the corrected camera and update geometry.TRAINING_FX."
    )


_TRAIN = _find_training_dir()


def _load_training_module():
    sys.path.insert(0, str(_TRAIN))
    import utils.aigp_obs as m  # noqa: E402
    return m


def test_projection_parity(n_cases: int = 400, seed: int = 0) -> tuple[float, int]:
    """Project random points from random drone poses, both ways, and compare."""
    import torch

    m = _load_training_module()
    cam = G.Camera.training()
    rng = np.random.default_rng(seed)

    max_uv_err = 0.0
    vis_mismatch = 0
    checked = 0

    for _ in range(n_cases):
        drone_pos = rng.uniform(-30, 30, size=3)
        roll = rng.uniform(-0.6, 0.6)
        pitch = rng.uniform(-0.6, 0.6)
        yaw = rng.uniform(-math.pi, math.pi)
        quat = G.quat_from_euler(roll, pitch, yaw)

        gate_pos = drone_pos + rng.uniform(-15, 15, size=3)
        gate_yaw = rng.uniform(-math.pi, math.pi)

        ours_pts = G.gate_corners_world(gate_pos, gate_yaw)

        gq = G.quat_from_euler(0.0, 0.0, gate_yaw)
        theirs_pts = m.gate_keypoints_world(
            torch.tensor(gate_pos, dtype=torch.float64).unsqueeze(0),
            torch.tensor(gq, dtype=torch.float64).unsqueeze(0),
        )[0].numpy()
        err_pts = np.abs(ours_pts - theirs_pts).max()
        # The training module stores KEYPOINT_OBJECT_POINTS as float32, so a
        # residual of ~1e-8 m on metre-scale values is float32 epsilon, not a
        # disagreement. Anything above 1e-6 would be a real difference.
        assert err_pts < 1e-6, f"corner world positions differ by {err_pts}"

        ours_uv, ours_vis = cam.project(ours_pts, drone_pos, quat)
        theirs_uv, theirs_vis = m.project_points_aigp_camera(
            torch.tensor(theirs_pts, dtype=torch.float64).unsqueeze(0),
            torch.tensor(drone_pos, dtype=torch.float64).unsqueeze(0),
            torch.tensor(quat, dtype=torch.float64).unsqueeze(0),
        )
        theirs_uv = theirs_uv[0].numpy()
        theirs_vis = theirs_vis[0].numpy()

        vis_mismatch += int((ours_vis != theirs_vis).sum())
        both = ours_vis & theirs_vis
        if both.any():
            max_uv_err = max(max_uv_err, float(np.abs(ours_uv[both] - theirs_uv[both]).max()))
            checked += int(both.sum())

    return max_uv_err, vis_mismatch


def test_attitude_parity(n_cases: int = 500, seed: int = 7) -> tuple[float, float]:
    """Roll and pitch must be derived the same way as in training.

    The policy reads these in slots 24 and 25. A sign error here tells it the
    aircraft is leaning the opposite way, which is unrecoverable in flight and
    invisible on the ground.
    """
    import torch

    m = _load_training_module()
    rng = np.random.default_rng(seed)
    worst_roll = worst_pitch = 0.0
    for _ in range(n_cases):
        roll = rng.uniform(-1.2, 1.2)
        pitch = rng.uniform(-1.2, 1.2)
        yaw = rng.uniform(-math.pi, math.pi)
        q = G.quat_from_euler(roll, pitch, yaw)
        mine_r, mine_p = G.roll_pitch_from_quat(q)
        t_r, t_p = m.attitude_roll_pitch_ned(torch.tensor(q, dtype=torch.float64).unsqueeze(0))
        worst_roll = max(worst_roll, abs(mine_r - float(t_r[0])))
        worst_pitch = max(worst_pitch, abs(mine_p - float(t_p[0])))
    return worst_roll, worst_pitch


def test_observation_parity(n_cases: int = 200, seed: int = 1) -> float:
    """Build a full 51-number frame both ways and compare element by element."""
    import torch

    m = _load_training_module()
    # race_obs.py lives at the repo root, beside isaac_drone_racer,
    # wherever _find_training_dir() landed.
    sys.path.insert(0, str(_TRAIN.parent))
    import race_obs  # noqa: E402

    import observation as O

    cam = G.Camera.training()
    rng = np.random.default_rng(seed)
    worst = 0.0

    for _ in range(n_cases):
        drone_pos = rng.uniform(-20, 20, size=3)
        roll, pitch = rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5)
        yaw = rng.uniform(-math.pi, math.pi)
        quat = G.quat_from_euler(roll, pitch, yaw)
        gyro = rng.uniform(-10, 10, size=3)
        vel = rng.uniform(-25, 25, size=3)
        gate_idx = int(rng.integers(0, 18))

        gate_pos = drone_pos + rng.uniform(-12, 12, size=3)
        gate_yaw = rng.uniform(-math.pi, math.pi)
        pts = G.gate_corners_world(gate_pos, gate_yaw)
        uv, vis = cam.project(pts, drone_pos, quat)

        ours = O.build_frame(uv, vis, roll=roll, pitch=pitch,
                             gyro_ned=gyro, vel_body_ned=vel, gate_index=gate_idx)

        kp = [(float(uv[i, 0]), float(uv[i, 1])) if vis[i] else (float("nan"), float("nan"))
              for i in range(8)]
        theirs = race_obs.build_observation(
            kp, roll=roll, pitch=pitch,
            gx=float(gyro[0]), gy=float(gyro[1]), gz=float(gyro[2]),
            vx=float(vel[0]), vy=float(vel[1]), vz=float(vel[2]),
            with_velocity=True, gate_index=gate_idx,
        )
        assert len(ours) == len(theirs) == 51, (len(ours), len(theirs))
        worst = max(worst, float(np.abs(np.asarray(ours) - np.asarray(theirs)).max()))

    return worst


def test_history_parity(seed: int = 2) -> float:
    """The 32-frame stack must flatten in the same order."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "AI_GP"))
    import race_obs  # noqa: E402

    import observation as O

    rng = np.random.default_rng(seed)
    worst = 0.0
    for n_rows in (1, 5, 31, 32, 40):
        rows = [list(rng.uniform(-1, 1, size=51)) for _ in range(n_rows)]
        theirs = np.asarray(race_obs.stack_history(rows, 32), dtype=np.float64).reshape(-1)

        buf = O.ObservationBuffer(history=32)
        for r in rows:
            buf.push(np.asarray(r))
        ours = buf.vector()

        assert ours.shape == theirs.shape == (1632,), (ours.shape, theirs.shape)
        worst = max(worst, float(np.abs(ours - theirs).max()))
    return worst


def main() -> int:
    print("Parity against the training code")
    print("-" * 64)

    uv_err, vis_mismatch = test_projection_parity()
    print(f"  projection    max pixel error {uv_err:.3e}, visibility mismatches {vis_mismatch}")
    # Sub-thousandth-of-a-pixel: float32 object points propagated through the
    # projection. A real convention error would be tens of pixels.
    assert uv_err < 1e-3, uv_err
    assert vis_mismatch == 0, vis_mismatch

    r_err, p_err = test_attitude_parity()
    print(f"  attitude      roll {r_err:.3e} rad, pitch {p_err:.3e} rad  (500 random attitudes)")
    assert r_err < 1e-9, r_err
    assert p_err < 1e-9, p_err

    obs_err = test_observation_parity()
    print(f"  observation   max element error {obs_err:.3e}  (51 numbers x 200 cases)")
    assert obs_err < 1e-6, obs_err

    hist_err = test_history_parity()
    print(f"  history stack max element error {hist_err:.3e}  (1632 numbers)")
    assert hist_err < 1e-12, hist_err

    print("-" * 64)
    print("  PASS - what the network sees in flight matches what it saw in training")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
