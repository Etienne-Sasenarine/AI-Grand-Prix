# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

"""Contract tests for the AI_GP observation port (no Isaac Sim required)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Repo root on path when pytest is launched from elsewhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.aigp_obs import (
    OFF_FRAME_MARGIN,
    DEFAULT_MASS_KG,
    FEATURE_DIM,
    FEATURE_DIM_CTX,
    FEATURE_NAMES,
    FRAME_H,
    FRAME_W,
    KEYPOINT_COUNT,
    KEYPOINT_OBJECT_POINTS,
    NOT_SEEN,
    N_GATES,
    VISUAL_END,
    align_keypoints_to_view,
    attitude_roll_pitch_ned,
    build_aigp_observation,
    context_features,
    feature_dim,
    flu_to_ned,
    gate_center_score,
    gate_keypoints_world,
    gate_visibility_frac,
    heading_alignment,
    pack_observation,
    project_points_aigp_camera,
)


def _identity_quat(n: int = 1) -> torch.Tensor:
    q = torch.zeros(n, 4)
    q[:, 0] = 1.0
    return q


def test_calibrated_intrinsics_are_1080p_scaled():
    from utils.aigp_obs import CX, CY, FX, FY

    assert abs(FX - 1302.941063 * (640.0 / 1920.0)) < 1e-6
    assert abs(FY - 1303.105865 * (640.0 / 1920.0)) < 1e-6
    assert abs(CX - 952.287144 * (640.0 / 1920.0)) < 1e-6
    assert abs(CY - 529.035788 * (360.0 / 1080.0)) < 1e-6


def test_feature_dim_matches_names():
    assert len(FEATURE_NAMES) == FEATURE_DIM
    assert feature_dim() == FEATURE_DIM
    assert feature_dim(with_context=True) == FEATURE_DIM_CTX
    assert feature_dim(with_context=True) == FEATURE_DIM + N_GATES + 1


def test_gate_keypoints_centered_at_origin_yaw0():
    pos = torch.zeros(1, 3)
    quat = _identity_quat(1)
    kps = gate_keypoints_world(pos, quat)
    assert kps.shape == (1, 8, 3)
    # yaw=0, through=+X: right=(0,-1,0), down=(0,0,-1).
    # Outer TL = (-1.35, -1.35, 0) → world (0, +1.35, +1.35) = up+left looking +X.
    assert torch.allclose(kps[0, 0], torch.tensor([0.0, 1.35, 1.35]), atol=1e-5)
    # Outer TR = (+1.35, -1.35, 0) → world (0, -1.35, +1.35) = up+right.
    assert torch.allclose(kps[0, 1], torch.tensor([0.0, -1.35, 1.35]), atol=1e-5)
    # Rings stay coplanar on x=0 for yaw=0.
    assert torch.allclose(kps[0, :, 0], torch.zeros(8), atol=1e-5)


def test_keypoints_keep_view_order_from_behind_g6():
    """Looking against through (official G6 after G5) must not reverse ids."""
    gate_pos = torch.zeros(1, 3)
    gate_quat = _identity_quat(1)
    raw = gate_keypoints_world(gate_pos, gate_quat)
    along = align_keypoints_to_view(raw, gate_pos, gate_quat, torch.tensor([[-4.0, 0.0, 0.0]]))
    against = align_keypoints_to_view(raw, gate_pos, gate_quat, torch.tensor([[4.0, 0.0, 0.0]]))
    assert torch.allclose(along, raw)
    # From the back, object TR (world -Y) is the visual TL.
    assert torch.allclose(against[0, 0], raw[0, 1])
    assert torch.allclose(against[0, 1], raw[0, 0])
    assert torch.allclose(against[0, 4], raw[0, 5])


def _pitched_quat(deg):
    """Body quaternion for a nose-down pitch, Isaac FLU (rotation about +y)."""
    half = math.radians(deg) / 2.0
    return torch.tensor([[math.cos(half), 0.0, math.sin(half), 0.0]])


def test_level_approach_clips_the_bottom_of_the_gate():
    """Not a defect -- the consequence of a real lens and a real mount.

    The camera is tilted 20 degrees up so that it looks level when the aircraft
    is flying nose-down, which is the attitude a race is flown in. Sitting
    level, the horizon sits low in the image: at fy = 434 the tilt pushes the
    optical centre down by 158 px in a 360 px frame, so a gate at the
    aircraft's own altitude lands near the bottom and its lower corners fall
    outside the image at 10 m.

    This matters on the start pad, where the aircraft is level and waiting for
    a gate box before it arms -- not in flight. The old fx = 320 camera had a
    90 degree field of view and did not clip here, which is one reason
    checkpoints trained against it do not transfer.
    """
    kps = gate_keypoints_world(torch.zeros(1, 3), _identity_quat(1))
    uv, visible = project_points_aigp_camera(kps, torch.tensor([[-10.0, 0.0, 0.0]]), _identity_quat(1))

    v = uv[0, :, 1]
    assert float(v.max()) > FRAME_H, "expected the lower corners to clip"
    # They are still inside the off-frame margin, so they read as seen. A
    # YOLO-pose model does predict keypoints just past the edge; the margin is
    # what lets the simulator say so too.
    assert bool(visible.all())
    assert float(v.max()) <= FRAME_H * (1.0 + OFF_FRAME_MARGIN)


def test_racing_attitude_frames_the_whole_gate():
    """The attitude that actually matters. A few degrees nose-down is enough to
    bring every corner inside the image."""
    kps = gate_keypoints_world(torch.zeros(1, 3), _identity_quat(1))
    drone_pos = torch.tensor([[-10.0, 0.0, 0.0]])

    for pitch_deg in (5.0, 15.0, 25.0):
        uv, visible = project_points_aigp_camera(kps, drone_pos, _pitched_quat(pitch_deg))
        u, v = uv[0, :, 0], uv[0, :, 1]
        assert bool(visible.all()), f"pitch {pitch_deg}: {visible}"
        assert bool(((u >= 0) & (u <= FRAME_W)).all()), f"pitch {pitch_deg} u={u}"
        assert bool(((v >= 0) & (v <= FRAME_H)).all()), f"pitch {pitch_deg} v={v}"

    # Outer ring must still project larger than the flyable opening.
    uv, _ = project_points_aigp_camera(kps, drone_pos, _pitched_quat(15.0))
    outer_span = float(uv[0, :4, 0].max() - uv[0, :4, 0].min())
    inner_span = float(uv[0, 4:, 0].max() - uv[0, 4:, 0].min())
    assert outer_span > inner_span > 0.0


def test_side_gate_distortion_stays_finite():
    """A gate 10 m left used to overflow Brown-Conrady to [-inf, nan]."""
    drone_pos = torch.zeros(1, 3)
    drone_quat = _identity_quat(1)
    pts = torch.tensor([[[0.0, 10.0, 0.0]]])
    uv, visible = project_points_aigp_camera(pts, drone_pos, drone_quat)
    assert torch.isfinite(uv).all(), uv
    score = gate_center_score(
        drone_pos, drone_quat, torch.tensor([[0.0, 10.0, 0.0]]), drone_quat
    )
    assert torch.isfinite(score).all()
    assert not bool(visible.any())


def test_behind_camera_is_unseen():
    drone_pos = torch.tensor([[10.0, 0.0, 0.0]])  # past the gate, looking +X
    drone_quat = _identity_quat(1)
    kps = gate_keypoints_world(torch.zeros(1, 3), _identity_quat(1))
    uv, visible = project_points_aigp_camera(kps, drone_pos, drone_quat)
    assert not bool(visible.any())


def test_gate_center_score_is_loose_not_binary():
    ident = _identity_quat(1)
    gate_pos = torch.zeros(1, 3)
    # Camera is pitched 20° up, so a same-height approach is not image-centre.
    # Drop the drone so the optical axis hits the gate, then shift it sideways.
    z = -10.0 * math.tan(math.radians(20.0))
    on_axis = gate_center_score(torch.tensor([[-10.0, 0.0, z]]), ident, gate_pos, ident)
    offset = gate_center_score(torch.tensor([[-10.0, 2.5, z]]), ident, gate_pos, ident)
    behind = gate_center_score(torch.tensor([[10.0, 0.0, 0.0]]), ident, gate_pos, ident)
    assert float(on_axis) > 0.7
    assert 0.0 < float(offset) < float(on_axis)
    assert float(behind) == 0.0


def test_pack_observation_unseen_sentinel():
    uv = torch.zeros(1, KEYPOINT_COUNT, 2)
    visible = torch.zeros(1, KEYPOINT_COUNT, dtype=torch.bool)
    visible[0, 0] = True
    uv[0, 0] = torch.tensor([320.0, 180.0])
    roll = torch.zeros(1)
    pitch = torch.zeros(1)
    gyro = torch.zeros(1, 3)
    obs = pack_observation(uv, visible, roll, pitch, gyro)
    assert obs.shape == (1, FEATURE_DIM)
    assert obs[0, 0].item() == pytest.approx(0.5)
    assert obs[0, 1].item() == pytest.approx(0.5)
    assert obs[0, 16].item() == 1.0
    for i in range(1, KEYPOINT_COUNT):
        assert obs[0, 2 * i].item() == NOT_SEEN
        assert obs[0, 2 * i + 1].item() == NOT_SEEN
        assert obs[0, 16 + i].item() == 0.0


def test_gyro_is_clipped():
    uv = torch.zeros(1, KEYPOINT_COUNT, 2)
    visible = torch.ones(1, KEYPOINT_COUNT, dtype=torch.bool)
    obs = pack_observation(
        uv,
        visible,
        torch.zeros(1),
        torch.zeros(1),
        torch.tensor([[1000.0, -1000.0, 0.5]]),
    )
    assert abs(obs[0, 26].item()) <= 8.0
    assert abs(obs[0, 27].item()) <= 8.0


def test_context_one_hot():
    ctx = context_features(torch.tensor([3, 17]))
    assert ctx.shape == (2, N_GATES + 1)
    assert ctx[0, 3].item() == 1.0
    assert ctx[0, :N_GATES].sum().item() == 1.0
    assert ctx[1, 17].item() == 1.0
    assert ctx[1, N_GATES].item() == pytest.approx(1.0)


def test_level_attitude_is_zero():
    roll, pitch = attitude_roll_pitch_ned(_identity_quat(1))
    assert abs(roll.item()) < 1e-5
    assert abs(pitch.item()) < 1e-5


def test_flu_to_ned_signs():
    v = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.allclose(flu_to_ned(v), torch.tensor([[1.0, -2.0, -3.0]]))


def test_build_aigp_observation_shape_with_context():
    n = 4
    obs = build_aigp_observation(
        drone_pos_w=torch.tensor([[-8.0, 0.0, 0.0]]).repeat(n, 1),
        drone_quat_w=_identity_quat(n),
        drone_ang_vel_b_flu=torch.zeros(n, 3),
        gate_pos_w=torch.zeros(n, 3),
        gate_quat_w=_identity_quat(n),
        gate_index=torch.arange(n) % 7,
        with_velocity=False,
        with_context=True,
    )
    assert obs.shape == (n, FEATURE_DIM_CTX)
    assert obs[:, :VISUAL_END].isfinite().all()
    # Active-gate one-hot occupies indices STATE_END .. STATE_END+N_GATES
    from utils.aigp_obs import STATE_END

    onehots = obs[:, STATE_END : STATE_END + N_GATES]
    assert torch.allclose(onehots.sum(dim=-1), torch.ones(n))


def test_commanded_velocity_matches_aigp_integrator():
    """Torch integrator must track AI_GP BodyVelocityIntegrator on the same inputs."""
    aigp_root = Path(__file__).resolve().parents[2] / "AI_GP"
    if not (aigp_root / "ekf" / "commanded_accel.py").is_file():
        pytest.skip("AI_GP commanded_accel not found")
    sys.path.insert(0, str(aigp_root))
    from ekf.commanded_accel import BodyVelocityIntegrator  # noqa: WPS433

    from utils.aigp_obs import G, CommandedBodyVelocity

    mass = DEFAULT_MASS_KG
    hover_n = mass * G
    # Dimensionless collective 1.2× hover ↔ Newtons = 1.2 * hover_n
    thr_frac = 1.2
    thr_n = thr_frac * hover_n
    roll, pitch = 0.1, -0.05
    omega = np.array([0.2, -0.1, 0.05], dtype=np.float64)
    dt = 0.02

    ref = BodyVelocityIntegrator(hover_trim=0.255)
    # Drive AI_GP with dimensionless thrust relative to its hover_trim.
    for _ in range(25):
        ref.step(dt, thr_frac * 0.255, roll, pitch, omega, hover_trim=0.255)

    bat = CommandedBodyVelocity(1, torch.device("cpu"), hover_trim=hover_n)
    for _ in range(25):
        bat.step(
            dt,
            torch.tensor([thr_n]),
            torch.tensor([roll]),
            torch.tensor([pitch]),
            omega_ned=torch.tensor(omega, dtype=torch.float32).unsqueeze(0),
        )

    assert torch.allclose(
        bat.v[0],
        torch.tensor(ref.v, dtype=torch.float32),
        atol=1e-4,
        rtol=1e-4,
    )


def test_level_hover_command_keeps_velocity_near_zero():
    from utils.aigp_obs import G, CommandedBodyVelocity

    hover = DEFAULT_MASS_KG * G
    bat = CommandedBodyVelocity(2, torch.device("cpu"), hover_trim=hover)
    for _ in range(50):
        bat.step(
            0.01,
            torch.full((2,), hover),
            torch.zeros(2),
            torch.zeros(2),
            omega_ned=torch.zeros(2, 3),
        )
    assert torch.allclose(bat.v, torch.zeros(2, 3), atol=1e-5)


def test_pack_observation_accepts_commanded_velocity():
    uv = torch.zeros(1, KEYPOINT_COUNT, 2)
    visible = torch.ones(1, KEYPOINT_COUNT, dtype=torch.bool)
    obs = pack_observation(
        uv,
        visible,
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros(1, 3),
        velocity_body_ned=torch.tensor([[1.0, -2.0, 0.5]]),
        with_velocity=True,
        with_context=False,
    )
    assert obs.shape == (1, FEATURE_DIM + 3)
    assert obs[0, 29].item() == pytest.approx(1.0)
    assert obs[0, 30].item() == pytest.approx(-2.0)
    assert obs[0, 31].item() == pytest.approx(0.5)

def test_object_points_match_aigp_rings():
    assert KEYPOINT_OBJECT_POINTS.shape == (8, 3)
    half_out = 2.7 / 2.0
    half_in = 1.5 / 2.0
    assert KEYPOINT_OBJECT_POINTS[0, 0].item() == pytest.approx(-half_out)
    assert KEYPOINT_OBJECT_POINTS[4, 0].item() == pytest.approx(-half_in)
    # Clockwise from TL: index 2 is outer BR.
    assert KEYPOINT_OBJECT_POINTS[2, 0].item() == pytest.approx(half_out)
    assert KEYPOINT_OBJECT_POINTS[2, 1].item() == pytest.approx(half_out)


def test_matches_aigp_build_observation_when_available():
    """Cross-check packing against AI_GP/race_obs.build_observation if present."""
    aigp_root = Path(__file__).resolve().parents[2] / "AI_GP"
    if not (aigp_root / "race_obs.py").is_file():
        pytest.skip("AI_GP/race_obs.py not found")
    sys.path.insert(0, str(aigp_root))
    import race_obs  # noqa: WPS433

    pts = [[100.0 + 10 * i, 50.0 + 5 * i] for i in range(8)]
    conf = [1.0] * 8
    ref = race_obs.build_observation(
        pts,
        conf,
        roll=0.1,
        pitch=-0.2,
        gx=0.3,
        gy=-0.4,
        gz=0.5,
        gate_index=2,
    )
    uv = torch.tensor(pts, dtype=torch.float32).view(1, 8, 2)
    visible = torch.ones(1, 8, dtype=torch.bool)
    got = pack_observation(
        uv,
        visible,
        torch.tensor([0.1]),
        torch.tensor([-0.2]),
        torch.tensor([[0.3, -0.4, 0.5]]),
        gate_index=torch.tensor([2]),
        with_context=True,
    )
    assert got.shape[1] == len(ref)
    assert torch.allclose(got[0], torch.tensor(ref, dtype=torch.float32), atol=1e-5)


# --- camera frame rate -------------------------------------------------------


def test_camera_latch_holds_frames_at_30hz():
    """Vision updates at 30 Hz even though the control loop steps at 100 Hz."""
    from utils.aigp_obs import CameraFrameLatch

    dt = 0.01  # 100 Hz control
    latch = CameraFrameLatch(1, "cpu", num_keypoints=2, rate_hz=30.0)

    captures = []
    for step in range(100):
        # A distinct frame every step, so a change in output means a capture.
        uv = torch.full((1, 2, 2), float(step))
        vis = torch.ones(1, 2, dtype=torch.bool)
        out, _ = latch.step(dt, uv, vis)
        if out[0, 0, 0].item() == float(step):
            captures.append(step)

    # 1 s of control steps must yield 30 frames, not 33 (stride 3) or 25 (stride 4).
    assert len(captures) == 30, captures
    # The non-integer 100/30 ratio must show up as a mixed 3/4 step cadence.
    strides = {captures[i + 1] - captures[i] for i in range(len(captures) - 1)}
    assert strides == {3, 4}, strides


def test_camera_latch_replays_stale_frame_between_captures():
    """Between frames the policy sees the previous keypoints, not fresh ones."""
    from utils.aigp_obs import CameraFrameLatch

    latch = CameraFrameLatch(1, "cpu", num_keypoints=1, rate_hz=30.0)
    first = torch.full((1, 1, 2), 5.0)
    out, _ = latch.step(0.01, first, torch.ones(1, 1, dtype=torch.bool))
    assert out[0, 0, 0].item() == 5.0

    # Next control step: new geometry arrives but no new camera frame is due.
    out, _ = latch.step(0.01, torch.full((1, 1, 2), 99.0), torch.ones(1, 1, dtype=torch.bool))
    assert out[0, 0, 0].item() == 5.0, "stale frame should be replayed"


def test_camera_latch_reset_captures_immediately():
    """A reset env must not serve the previous episode's frame."""
    from utils.aigp_obs import CameraFrameLatch

    latch = CameraFrameLatch(2, "cpu", num_keypoints=1, rate_hz=30.0)
    latch.step(0.01, torch.full((2, 1, 2), 1.0), torch.ones(2, 1, dtype=torch.bool))
    latch.reset(torch.tensor([0]))
    out, vis = latch.step(0.01, torch.full((2, 1, 2), 7.0), torch.ones(2, 1, dtype=torch.bool))
    assert out[0, 0, 0].item() == 7.0, "reset env should capture at once"
    assert out[1, 0, 0].item() == 1.0, "untouched env should keep its held frame"


def test_camera_latch_at_60hz_control_is_exactly_every_other_step():
    """60 Hz control divides evenly into the 30 Hz stream, so the stride is 2."""
    from utils.aigp_obs import CameraFrameLatch

    dt = 1.0 / 120 * 2  # 120 Hz physics, decimation 2 -> 60 Hz policy
    latch = CameraFrameLatch(1, "cpu", num_keypoints=1, rate_hz=30.0)

    captures = []
    for step in range(60):
        uv = torch.full((1, 1, 2), float(step))
        out, _ = latch.step(dt, uv, torch.ones(1, 1, dtype=torch.bool))
        if out[0, 0, 0].item() == float(step):
            captures.append(step)

    assert len(captures) == 30, captures
    strides = {captures[i + 1] - captures[i] for i in range(len(captures) - 1)}
    assert strides == {2}, strides


def _yaw_quat(yaw: float) -> torch.Tensor:
    q = torch.zeros(1, 4)
    q[0, 0] = math.cos(yaw * 0.5)
    q[0, 3] = math.sin(yaw * 0.5)
    return q


def test_heading_alignment_is_one_when_yawed_at_the_gate():
    pos = torch.zeros(1, 3)
    gate = torch.tensor([[10.0, 0.0, 2.0]])
    on = heading_alignment(pos, _identity_quat(1), gate)
    off = heading_alignment(pos, _identity_quat(1), torch.tensor([[0.0, 10.0, 2.0]]))
    turned = heading_alignment(pos, _yaw_quat(math.pi / 2), torch.tensor([[0.0, 10.0, 2.0]]))
    assert float(on) > 0.95
    assert float(off) < 0.15
    assert float(turned) > 0.95


def test_gate_visibility_drops_when_not_looking():
    drone = torch.tensor([[-10.0, 0.0, 2.0]])
    gate = torch.tensor([[0.0, 0.0, 2.0]])
    gq = _identity_quat(1)
    looking = gate_visibility_frac(drone, _identity_quat(1), gate, gq)
    away = gate_visibility_frac(drone, _yaw_quat(math.pi), gate, gq)
    assert float(looking) > 0.9
    assert float(away) < 0.1


def test_current_race_obs_frame_reads_last_history_slot():
    from utils.aigp_obs import FEATURE_DIM_VEL_CTX
    from utils.keypoint_overlay import HISTORY_FLAT, current_race_obs_frame

    hist = np.zeros(HISTORY_FLAT, dtype=np.float32)
    hist[-FEATURE_DIM_VEL_CTX] = 0.42
    assert current_race_obs_frame(hist)[0] == pytest.approx(0.42)
    with_look = np.concatenate([hist, np.ones(7, dtype=np.float32)])
    assert current_race_obs_frame(with_look)[0] == pytest.approx(0.42)


def test_vel_align_straight_vs_sideways():
    from utils.vel_align import vel_align_from_vel

    n = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    straight = vel_align_from_vel(torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]]), n)
    assert float(straight[0]) == pytest.approx(1.0)
    assert float(straight[1]) == pytest.approx(0.0)
    backward = vel_align_from_vel(torch.tensor([[-5.0, 0.0, 0.0]]), n[:1])
    assert float(backward) == pytest.approx(0.0)
    stopped = vel_align_from_vel(torch.zeros(1, 3), n[:1])
    assert float(stopped) == pytest.approx(0.0)
    # Sparse: weight 3.0 * max(0, v · n̂). 5 m/s straight → +15; sideways → 0.
    n0 = torch.tensor([1.0, 0.0, 0.0])
    assert float(torch.tensor([5.0, 0.0, 0.0]).dot(n0).clamp(min=0.0) * 3.0) == pytest.approx(15.0)
    assert float(torch.tensor([0.0, 5.0, 0.0]).dot(n0).clamp(min=0.0) * 3.0) == pytest.approx(0.0)


def test_cam_passage_cone_vs_gate_normal():
    from utils.vel_align import cam_cone_score, cam_horizontal_align

    n = torch.tensor([[1.0, 0.0, 0.0]])
    on = cam_horizontal_align(_identity_quat(1), n)
    side = cam_horizontal_align(_yaw_quat(math.pi / 2), n)
    twenty = cam_horizontal_align(_yaw_quat(math.radians(20.0)), n)
    forty = cam_horizontal_align(_yaw_quat(math.radians(40.0)), n)
    assert float(on) == pytest.approx(1.0, abs=1e-5)
    assert float(side) == pytest.approx(0.0, abs=1e-5)
    assert float(cam_cone_score(on, psi_deg=40.0)) == pytest.approx(1.0)
    assert float(cam_cone_score(twenty, psi_deg=40.0)) > 0.7
    assert float(cam_cone_score(forty, psi_deg=40.0)) == pytest.approx(0.0, abs=0.02)
    assert float(cam_cone_score(side, psi_deg=40.0)) == pytest.approx(0.0)


def test_center_passage_bullseye_vs_rim():
    from utils.vel_align import center_passage_score

    n = torch.tensor([[1.0, 0.0, 0.0]])
    gate = torch.zeros(1, 3)
    mid = center_passage_score(torch.zeros(1, 3), gate, n, half_size=0.75)
    rim = center_passage_score(torch.tensor([[0.0, 0.75, 0.0]]), gate, n, half_size=0.75)
    half = center_passage_score(torch.tensor([[0.0, 0.375, 0.0]]), gate, n, half_size=0.75)
    along = center_passage_score(torch.tensor([[0.4, 0.0, 0.0]]), gate, n, half_size=0.75)
    assert float(mid) == pytest.approx(1.0)
    assert float(rim) == pytest.approx(0.0)
    assert float(half) == pytest.approx(0.5)
    # Offset along the normal does not count: still the opening centre.
    assert float(along) == pytest.approx(1.0)


def test_run_in_progress_pulls_behind_gate_from_exit_side():
    """After gate 11 the short hook is west to the run-in, not around the hall."""
    from utils.vel_align import run_in_progress_from_pos

    # Gate 1: (11.31, 23.43), n̂ = +X. Waypoint is 3 m west.
    gate = torch.tensor([[11.31, 23.43, 2.0]])
    n = torch.tensor([[1.0, 0.0, 0.0]])
    # Exit of gate 11, east and a bit north of gate 1.
    prev = torch.tensor([[18.68, 27.96, 2.0]])
    west = torch.tensor([[17.68, 27.96, 2.0]])
    south = torch.tensor([[18.68, 26.96, 2.0]])
    behind = torch.tensor([[8.31, 23.43, 2.0]])

    cut = run_in_progress_from_pos(prev, west, gate, n)
    long_way = run_in_progress_from_pos(prev, south, gate, n)
    already_behind = run_in_progress_from_pos(behind, behind + torch.tensor([[0.2, 0.0, 0.0]]), gate, n)

    assert float(cut) > 0.9
    assert float(cut) > float(long_way)
    # Already on the approach side: ordinary progress owns that leg.
    assert float(already_behind) == pytest.approx(0.0)


def test_progress_aims_at_run_in_from_the_exit_side():
    """Main progress must not pull into the face after the 11→1 wrap."""
    from utils.vel_align import progress_to_aim

    gate = torch.tensor([[11.31, 23.43, 2.0]])
    n = torch.tensor([[1.0, 0.0, 0.0]])
    prev = torch.tensor([[18.68, 27.96, 2.0]])
    west = torch.tensor([[17.68, 27.96, 2.0]])
    south = torch.tensor([[18.68, 26.96, 2.0]])
    behind = torch.tensor([[8.31, 23.43, 2.0]])
    through = torch.tensor([[9.31, 23.43, 2.0]])

    assert float(progress_to_aim(prev, west, gate, n)) > float(progress_to_aim(prev, south, gate, n))
    # Behind the opening: aim is the centre, so flying +X through it pays.
    assert float(progress_to_aim(behind, through, gate, n)) == pytest.approx(1.0, abs=1e-5)


def test_annotate_rgb_paints_visible_corner():
    from utils.aigp_obs import FEATURE_DIM_VEL_CTX, NOT_SEEN
    from utils.keypoint_overlay import annotate_rgb

    obs = np.full(FEATURE_DIM_VEL_CTX, float(NOT_SEEN), dtype=np.float32)
    obs[0], obs[1], obs[16] = 0.5, 0.5, 1.0
    rgb = np.zeros((360, 640, 3), dtype=np.uint8)
    out = annotate_rgb(rgb, obs)
    assert out.shape == rgb.shape
    assert out.sum() > 0
