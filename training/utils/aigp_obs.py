# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

"""AI Grand Prix observation contract for Isaac Drone Racer.

Mirrors ``AI_GP/race_obs.py`` so an RL policy here sees the same channels the
HG-DAgger student flies with on the VQ simulator:

  * 8 gate keypoints as normalised pixels in [0, 1] (outer 2.7 m, inner 1.5 m)
  * per-keypoint visibility flags (unseen = NOT_SEEN / vis=0)
  * gravity-referenced roll, pitch
  * body rates (NED) gx, gy, gz — gyro only; never IMU linear accelerometer
  * commanded body velocity (thrust + attitude + drag), not privileged twist
  * optional 18-gate course context

Keypoints are produced by projecting the active gate's known 3D corners through
the AI_GP pinhole camera (spec §3.8), not by rendering + YOLO. That is the same
privileged-corner training path used by Xing et al. (CoRL 2024): the vector
layout stays valid when a real detector replaces the projector at deployment.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

# ---- layout (must match AI_GP/race_obs.py) ---------------------------------
FRAME_W = 640.0
FRAME_H = 360.0
# On-site ChArUco at 1920x1080, scaled to the 640x360 policy frame (×1/3).
# HFOV = 2 atan((1920/2)/1302.941) ≈ 72.8 deg.
_CALIB_SCALE = FRAME_W / 1920.0
FX = 1302.941063 * _CALIB_SCALE
FY = 1303.105865 * _CALIB_SCALE
CX = 952.287144 * _CALIB_SCALE
CY = 529.035788 * _CALIB_SCALE
# OpenCV Brown-Conrady (k1, k2, p1, p2, k3). Applied in normalised coords.
DIST_K = (0.047494607, -0.182658343, 0.153277219)
DIST_P = (0.001303410, 0.000583514)
# tan(55°)^2 ≈ 2.0 — already past the ChArUco domain; beyond this the
# k3 r^6 term is not a real lens, it is float overflow.
R2_DIST_MAX = 2.0
CAMERA_TILT_UP_DEG = 20.0

GATE_OUTER_M = 2.7
GATE_INNER_M = 1.5
KEYPOINT_COUNT = 8
# Policy action channels: thrust, roll rate, pitch rate, yaw rate.
ACTION_DIM = 4
NOT_SEEN = -1.0
OFF_FRAME_MARGIN = 0.15
GYRO_CLIP = 8.0
VEL_CLIP = 20.0
# Vision stream rate, VADR-TS-001 section 4.6. Independent of the control rate.
CAMERA_RATE_HZ = 30.0
N_GATES = 18  # AI_GP course; pad unused slots with zeros on shorter tracks

CORNER_CHANNELS = tuple(f"{axis}{i}" for i in range(KEYPOINT_COUNT) for axis in ("u", "v"))
VIS_CHANNELS = tuple(f"vis{i}" for i in range(KEYPOINT_COUNT))
STATE_CHANNELS = ("roll", "pitch", "gx", "gy", "gz")
VEL_CHANNELS = ("vx", "vy", "vz")
CONTEXT_CHANNELS = tuple(f"gate{i}" for i in range(N_GATES)) + ("gate_frac",)

FEATURE_NAMES = CORNER_CHANNELS + VIS_CHANNELS + STATE_CHANNELS
FEATURE_DIM = len(FEATURE_NAMES)
FEATURE_NAMES_VEL = FEATURE_NAMES + VEL_CHANNELS
FEATURE_DIM_VEL = len(FEATURE_NAMES_VEL)
FEATURE_NAMES_CTX = FEATURE_NAMES + CONTEXT_CHANNELS
FEATURE_DIM_CTX = len(FEATURE_NAMES_CTX)
FEATURE_NAMES_VEL_CTX = FEATURE_NAMES_VEL + CONTEXT_CHANNELS
FEATURE_DIM_VEL_CTX = len(FEATURE_NAMES_VEL_CTX)

VISUAL_END = 2 * KEYPOINT_COUNT + KEYPOINT_COUNT  # 24
STATE_END = VISUAL_END + len(STATE_CHANNELS)  # 29


def feature_dim(with_context: bool = False, with_velocity: bool = False) -> int:
    n = FEATURE_DIM
    if with_velocity:
        n += len(VEL_CHANNELS)
    if with_context:
        n += len(CONTEXT_CHANNELS)
    return n


# Object points in the AI_GP gate frame: X=right, Y=down, Z=through.
# Each ring clockwise from top-left (outer 0-3, inner 4-7).
_HALF_OUT = GATE_OUTER_M / 2.0
_HALF_IN = GATE_INNER_M / 2.0
KEYPOINT_OBJECT_POINTS = torch.tensor(
    [
        [-_HALF_OUT, -_HALF_OUT, 0.0],
        [+_HALF_OUT, -_HALF_OUT, 0.0],
        [+_HALF_OUT, +_HALF_OUT, 0.0],
        [-_HALF_OUT, +_HALF_OUT, 0.0],
        [-_HALF_IN, -_HALF_IN, 0.0],
        [+_HALF_IN, -_HALF_IN, 0.0],
        [+_HALF_IN, +_HALF_IN, 0.0],
        [-_HALF_IN, +_HALF_IN, 0.0],
    ],
    dtype=torch.float32,
)
# Left-right swap of both rings. Used when the camera looks against the
# gate through-axis so id 0 stays image top-left as *seen*, not the
# object-frame TL (which is mirrored from the back). Official G6 is the
# usual case: flown west after G5, through points east.
KEYPOINT_FLIP_LR = (1, 0, 3, 2, 5, 4, 7, 6)


def _camera_body_rotation(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """NED body -> camera-optical rotation (AI_GP ``camera_model.R_CB``)."""
    t = math.radians(CAMERA_TILT_UP_DEG)
    st, ct = math.sin(t), math.cos(t)
    return torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [st, 0.0, ct],
            [ct, 0.0, -st],
        ],
        device=device,
        dtype=dtype,
    )


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate world vectors ``v`` into the frame of unit quaternion ``q`` (wxyz)."""
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q.unbind(-1)
    # Equivalent to R(q)^T @ v for Isaac wxyz quaternions.
    q_inv = torch.stack([w, -x, -y, -z], dim=-1)
    return quat_rotate(q_inv, v)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors ``v`` by unit quaternion ``q`` (wxyz)."""
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1e-8)
    w = q[..., 0]
    u = q[..., 1:]
    uv = torch.cross(u, v, dim=-1)
    uuv = torch.cross(u, uv, dim=-1)
    return v + 2.0 * (w.unsqueeze(-1) * uv + uuv)


def flu_to_ned(v_flu: torch.Tensor) -> torch.Tensor:
    """Isaac body FLU (x fwd, y left, z up) -> MAVLink body NED (x fwd, y right, z down)."""
    return torch.stack([v_flu[..., 0], -v_flu[..., 1], -v_flu[..., 2]], dim=-1)


def gate_keypoints_world(
    gate_pos_w: torch.Tensor,
    gate_quat_w: torch.Tensor,
    object_points: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """World positions of the eight gate corners.

    Gate facing follows the yaw used by ``GateTargetingCommand``: the through-axis
    is the horizontal yaw direction. AI_GP gate axes map as
    ``offset = X*right + Y*down + Z*through``.

    Args:
        gate_pos_w: (N, 3) gate centre in world.
        gate_quat_w: (N, 4) gate orientation wxyz.
        object_points: (8, 3) optional override of KEYPOINT_OBJECT_POINTS.

    Returns:
        (N, 8, 3) world positions.
    """
    if object_points is None:
        object_points = KEYPOINT_OBJECT_POINTS.to(device=gate_pos_w.device, dtype=gate_pos_w.dtype)
    else:
        object_points = object_points.to(device=gate_pos_w.device, dtype=gate_pos_w.dtype)

    # Yaw from quaternion (Z-up world), matching commands.py gate-normal logic.
    w, x, y, z = gate_quat_w.unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    # Looking along through: right = through × world_up, down = -world_up.
    through = torch.stack([cy, sy, torch.zeros_like(cy)], dim=-1)  # (N, 3)
    right = torch.stack([sy, -cy, torch.zeros_like(cy)], dim=-1)
    down = torch.zeros_like(through)
    down[..., 2] = -1.0

    # (N, 8, 3) = X*right + Y*down + Z*through
    ox = object_points[:, 0].view(1, -1, 1)
    oy = object_points[:, 1].view(1, -1, 1)
    oz = object_points[:, 2].view(1, -1, 1)
    offset = ox * right.unsqueeze(1) + oy * down.unsqueeze(1) + oz * through.unsqueeze(1)
    return gate_pos_w.unsqueeze(1) + offset


def align_keypoints_to_view(
    kps_w: torch.Tensor,
    gate_pos_w: torch.Tensor,
    gate_quat_w: torch.Tensor,
    drone_pos_w: torch.Tensor,
) -> torch.Tensor:
    """Keep keypoint ids in camera-view order, never object-frame reversed.

    Object-frame TL/TR is defined looking *along* through. From the back
    (G6 after the split-S, any against-through approach) that pair is
    left-right swapped on the image. Swap both rings so the policy always
    sees id 0 as the top-left corner in the picture.
    """
    yaw = yaw_wxyz(gate_quat_w)
    through = torch.stack(
        [torch.cos(yaw), torch.sin(yaw), torch.zeros_like(gate_pos_w[:, 0])],
        dim=-1,
    )
    against = ((drone_pos_w - gate_pos_w) * through).sum(dim=-1) > 0.0
    flip = torch.tensor(KEYPOINT_FLIP_LR, device=kps_w.device, dtype=torch.long)
    return torch.where(against.view(-1, 1, 1), kps_w[:, flip], kps_w)


def project_points_aigp_camera(
    points_w: torch.Tensor,
    drone_pos_w: torch.Tensor,
    drone_quat_w: torch.Tensor,
    *,
    frame_w: float = FRAME_W,
    frame_h: float = FRAME_H,
    off_frame_margin: float = OFF_FRAME_MARGIN,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project world points through the AI_GP pinhole camera on the drone.

    Args:
        points_w: (N, K, 3)
        drone_pos_w: (N, 3)
        drone_quat_w: (N, 4) wxyz, Isaac FLU body.

    Returns:
        uv: (N, K, 2) pixel coordinates (undefined where not visible)
        visible: (N, K) bool
    """
    n, k, _ = points_w.shape
    device, dtype = points_w.device, points_w.dtype
    rel_w = points_w - drone_pos_w.unsqueeze(1)
    # World -> Isaac body FLU.
    flat = rel_w.reshape(n * k, 3)
    quat = drone_quat_w.unsqueeze(1).expand(-1, k, -1).reshape(n * k, 4)
    p_flu = quat_rotate_inverse(quat, flat).view(n, k, 3)
    p_ned = flu_to_ned(p_flu)

    r_cb = _camera_body_rotation(device, dtype)
    p_cam = torch.matmul(p_ned, r_cb.T)  # (N, K, 3)

    z = p_cam[..., 2]
    # Avoid div-by-zero; behind-camera points marked unseen below.
    z_safe = torch.where(z.abs() < 1e-6, torch.full_like(z, 1e-6), z)
    x = p_cam[..., 0] / z_safe
    y = p_cam[..., 1] / z_safe
    # Brown-Conrady is only valid near the calibration domain. Unclamped r^6
    # overflows float32 on a gate beside the camera (10 m left is enough),
    # and 0 * inf on a principal axis becomes NaN. That NaN used to reach
    # ``gate_visible`` / PPO and poison the whole 4096-env batch.
    r2 = (x * x + y * y).clamp(max=R2_DIST_MAX)
    r4 = r2 * r2
    r6 = r4 * r2
    radial = 1.0 + DIST_K[0] * r2 + DIST_K[1] * r4 + DIST_K[2] * r6
    x_d = x * radial + 2.0 * DIST_P[0] * x * y + DIST_P[1] * (r2 + 2.0 * x * x)
    y_d = y * radial + DIST_P[0] * (r2 + 2.0 * y * y) + 2.0 * DIST_P[1] * x * y
    u = FX * x_d + CX
    v = FY * y_d + CY
    u = torch.nan_to_num(u, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
    v = torch.nan_to_num(v, nan=0.0, posinf=1.0e6, neginf=-1.0e6)

    mx = off_frame_margin * frame_w
    my = off_frame_margin * frame_h
    in_front = z > 1e-4
    in_frame = (u >= -mx) & (u <= frame_w + mx) & (v >= -my) & (v <= frame_h + my)
    visible = in_front & in_frame
    uv = torch.stack([u, v], dim=-1)
    return uv, visible


def yaw_wxyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """World yaw (Z-up) from an Isaac wxyz quaternion."""
    w, x, y, z = quat_wxyz.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def heading_alignment(
    drone_pos_w: torch.Tensor,
    drone_quat_w: torch.Tensor,
    gate_pos_w: torch.Tensor,
    std: float = 0.35,
) -> torch.Tensor:
    """1 at yaw-on-bearing, ~0 when the gate is 60°+ off the nose (horizontal)."""
    yaw = yaw_wxyz(drone_quat_w)
    delta = gate_pos_w - drone_pos_w
    bearing = torch.atan2(delta[:, 1], delta[:, 0])
    err = wrap_pi(bearing - yaw).abs()
    return torch.exp(-err / std)


def gate_visibility_frac(
    drone_pos_w: torch.Tensor,
    drone_quat_w: torch.Tensor,
    gate_pos_w: torch.Tensor,
    gate_quat_w: torch.Tensor,
) -> torch.Tensor:
    """Fraction of the 8 gate corners inside the AI_GP camera frame."""
    pts = align_keypoints_to_view(
        gate_keypoints_world(gate_pos_w, gate_quat_w),
        gate_pos_w,
        gate_quat_w,
        drone_pos_w,
    )
    _, visible = project_points_aigp_camera(pts, drone_pos_w, drone_quat_w)
    return visible.to(dtype=drone_pos_w.dtype).mean(dim=-1)


def gate_center_score(
    drone_pos_w: torch.Tensor,
    drone_quat_w: torch.Tensor,
    gate_pos_w: torch.Tensor,
    gate_quat_w: torch.Tensor,
    *,
    std: float = 0.55,
    frame_w: float = FRAME_W,
    frame_h: float = FRAME_H,
) -> torch.Tensor:
    """Loose in-frame score: visibility × how near the opening is to image centre.

    Not binary and not a bullseye. ``std=0.55`` (normalised pixels) so dead
    centre is 1, halfway to a frame edge is ~0.63, and a corner of the image
    is ~0.27. No visible corners → 0. Meant as a small nudge, not a stare farm.
    """
    pts = align_keypoints_to_view(
        gate_keypoints_world(gate_pos_w, gate_quat_w),
        gate_pos_w,
        gate_quat_w,
        drone_pos_w,
    )
    uv, visible = project_points_aigp_camera(pts, drone_pos_w, drone_quat_w)
    vis = visible.to(dtype=drone_pos_w.dtype)
    n_vis = vis.sum(dim=-1)
    vis_frac = n_vis / float(KEYPOINT_COUNT)
    scale = uv.new_tensor([frame_w, frame_h])
    uv_n = uv / scale
    mean_uv = (uv_n * vis.unsqueeze(-1)).sum(dim=1) / n_vis.clamp_min(1.0).unsqueeze(-1)
    offset = (mean_uv - 0.5).norm(dim=-1)
    loose = torch.exp(-offset / max(float(std), 1e-6))
    score = torch.where(n_vis > 0, vis_frac * loose, torch.zeros_like(vis_frac))
    return torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)


def attitude_roll_pitch_ned(
    drone_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gravity-referenced roll, pitch in the NED body frame (radians)."""
    # Gravity direction in world (Z-up): -Z.
    g_w = torch.zeros(drone_quat_w.shape[0], 3, device=drone_quat_w.device, dtype=drone_quat_w.dtype)
    g_w[:, 2] = -1.0
    g_flu = quat_rotate_inverse(drone_quat_w, g_w)
    g = flu_to_ned(g_flu)
    # Standard NED AHRS: roll about x, pitch about y.
    roll = torch.atan2(g[:, 1], g[:, 2])
    pitch = torch.atan2(-g[:, 0], torch.sqrt(g[:, 1] * g[:, 1] + g[:, 2] * g[:, 2]).clamp_min(1e-8))
    return roll, pitch


def context_features(gate_index: torch.Tensor, n_gates: int = N_GATES) -> torch.Tensor:
    """One-hot active gate + fractional lap progress. Shape (N, n_gates+1)."""
    n = gate_index.shape[0]
    out = torch.zeros(n, n_gates + 1, device=gate_index.device, dtype=torch.float32)
    idx = gate_index.long().clamp(0, n_gates - 1)
    out[torch.arange(n, device=gate_index.device), idx] = 1.0
    out[:, n_gates] = idx.float() / float(max(1, n_gates - 1))
    return out


G = 9.80665
# AI_GP default linear-drag coeffs in FRD (a = k ⊙ v). Negative = opposes motion.
DEFAULT_DRAG_BODY = (-0.50, -0.50, -0.15)
# 5" drone mass used by ControlActionCfg thrust scaling (hover = m·g Newtons).
DEFAULT_MASS_KG = 1.745


def gravity_in_body_ned(roll: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
    """Gravity (NED +z down) in FRD from roll/pitch. Yaw does not enter."""
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    # [-G sin(p), G sin(r) cos(p), G cos(r) cos(p)]
    return torch.stack([-G * sp, G * sr * cp, G * cr * cp], dim=-1)


def specific_thrust(thrust: torch.Tensor, hover_trim: torch.Tensor | float) -> torch.Tensor:
    """T/m in m/s^2 from collective (or Newtons) and the 1 g hover calibration."""
    trim = hover_trim if torch.is_tensor(hover_trim) else torch.as_tensor(hover_trim, device=thrust.device, dtype=thrust.dtype)
    trim = trim.clamp_min(1e-6)
    return (thrust / trim) * G


class KeypointDelayLine:
    """Hand the policy the gate it saw a few control steps ago.

    ``CameraFrameLatch`` models when a frame is *captured*. This models when the
    result of that frame becomes *available*, which is a separate and larger
    quantity: sensor readout, the copy to the Orin, YOLO-pose inference, and the
    keypoints finally reaching the observation. None of it is instantaneous, and
    the simulator previously assumed all of it was.

    Applied after the latch, not before. The camera samples the world at its own
    rate and *then* the pipeline takes time to deliver the sample, so the delay
    acts on the held frame. Delaying first and latching afterwards -- which the
    evaluation harness does -- samples an already-shifted signal and lands on
    different frames whenever the delay is an odd number of control steps.

    The delay is per environment. A single global value is something a policy
    can learn exactly and quietly depend on; the real number moves with scene
    complexity, how many gates are in view, and whatever else the Orin is doing.

    On reset the buffer is backfilled with the current frame rather than zeroed.
    A fresh episode has no older frame to show, and handing it blank keypoints
    would teach the policy that every episode begins blind -- an artefact of the
    simulator, not of the aircraft, which has been sitting there looking at the
    same gate.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        *,
        max_delay: int,
        num_keypoints: int = KEYPOINT_COUNT,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if max_delay < 0:
            raise ValueError(f"max_delay must be non-negative, got {max_delay}")
        self.num_envs = int(num_envs)
        self.device = device
        self.dtype = dtype
        self.max_delay = int(max_delay)
        depth = self.max_delay + 1
        self.uv = torch.zeros(self.num_envs, depth, num_keypoints, 2, device=device, dtype=dtype)
        self.visible = torch.zeros(
            self.num_envs, depth, num_keypoints, device=device, dtype=torch.bool
        )
        self.delay = torch.zeros(self.num_envs, device=device, dtype=torch.long)
        self._head = 0
        self._rows = torch.arange(self.num_envs, device=device)
        # Environments whose history must be refilled with the next frame they
        # see. Deferred rather than done eagerly because the backfill needs the
        # current projection, which the caller has not computed yet when it
        # learns an episode has reset.
        self._pending_reset: Optional[torch.Tensor] = None

    def resample(self, env_ids: torch.Tensor, lo: int, hi: int) -> None:
        """Draw a fresh delay for these environments, in control steps."""
        if not len(env_ids):
            return
        self.delay[env_ids] = torch.randint(
            int(lo), int(hi) + 1, (len(env_ids),), device=self.device
        )

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        """Mark these environments for a backfill on the next :meth:`step`.

        The history is filled with the frame the aircraft can actually see at
        that moment, not with zeros. A fresh episode has no older frame to
        offer, and blank keypoints would teach the policy that every episode
        opens blind -- an artefact of the simulator rather than of the aircraft,
        which has been sitting on the pad looking at the same gate.
        """
        if env_ids is None:
            env_ids = self._rows
        if not len(env_ids):
            return
        if self._pending_reset is None:
            self._pending_reset = env_ids
        else:
            self._pending_reset = torch.unique(torch.cat([self._pending_reset, env_ids]))

    def step(
        self, uv_px: torch.Tensor, visible: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Record this step's frame and return the one each env is due.

        A delay of zero returns exactly what was passed in, so switching the
        delay off costs nothing and reproduces the old behaviour bit for bit.
        """
        if self._pending_reset is not None:
            ids = self._pending_reset
            self.uv[ids] = uv_px[ids].unsqueeze(1).to(self.dtype)
            self.visible[ids] = visible[ids].unsqueeze(1)
            self._pending_reset = None
        if self.max_delay == 0:
            return uv_px, visible
        depth = self.uv.shape[1]
        self._head = (self._head + 1) % depth
        self.uv[:, self._head] = uv_px.to(self.dtype)
        self.visible[:, self._head] = visible
        read = (self._head - self.delay) % depth
        return self.uv[self._rows, read], self.visible[self._rows, read]


class CommandedBodyVelocity:
    """Batched FRD body-velocity integrator (AI_GP ``BodyVelocityIntegrator``).

    Never touches the accelerometer. World/body accel is commanded physics:

        a = g_body − (T/m) e_z + k ⊙ v − ω × v

    with T/m = (thrust / hover_trim) · g. Integrate in FRD so yaw is unused.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        *,
        hover_trim: float = DEFAULT_MASS_KG * G,
        k_body: tuple[float, float, float] = DEFAULT_DRAG_BODY,
        max_dt: float = 0.05,
        dtype: torch.dtype = torch.float32,
    ):
        self.num_envs = num_envs
        self.device = device
        self.dtype = dtype
        self.hover_trim = float(hover_trim)
        self.max_dt = float(max_dt)
        self.k_body = torch.tensor(k_body, device=device, dtype=dtype)
        self.v = torch.zeros(num_envs, 3, device=device, dtype=dtype)

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        if env_ids is None:
            self.v.zero_()
        else:
            self.v[env_ids] = 0.0

    def step(
        self,
        dt: float | torch.Tensor,
        thrust: torch.Tensor,
        roll: torch.Tensor,
        pitch: torch.Tensor,
        omega_ned: Optional[torch.Tensor] = None,
        hover_trim: Optional[float | torch.Tensor] = None,
    ) -> torch.Tensor:
        """Advance one control step. Returns body FRD velocity (N, 3)."""
        trim = self.hover_trim if hover_trim is None else hover_trim
        if torch.is_tensor(dt):
            step = dt.to(device=self.device, dtype=self.dtype).clamp(0.0, self.max_dt)
            if step.ndim == 0:
                step = step.expand(self.num_envs)
            step = step.unsqueeze(-1)
        else:
            step_f = 0.0 if (not math.isfinite(float(dt)) or float(dt) <= 0.0) else min(float(dt), self.max_dt)
            step = torch.full((self.num_envs, 1), step_f, device=self.device, dtype=self.dtype)

        thr = thrust.to(device=self.device, dtype=self.dtype).reshape(self.num_envs)
        if not torch.is_tensor(trim):
            trim_t = torch.full_like(thr, float(trim))
        else:
            trim_t = trim.to(device=self.device, dtype=self.dtype).reshape(self.num_envs)
        # Missing / zero thrust → hover (1 g, no net vertical accel when level).
        thr = torch.where(torch.isfinite(thr) & (thr.abs() >= 1e-9), thr, trim_t)

        g_body = gravity_in_body_ned(roll, pitch)
        tm = specific_thrust(thr, trim_t)
        body_down = torch.zeros_like(g_body)
        body_down[:, 2] = 1.0
        a = g_body - tm.unsqueeze(-1) * body_down + self.k_body * self.v
        if omega_ned is not None:
            w = omega_ned.to(device=self.device, dtype=self.dtype)
            a = a - torch.cross(w, self.v, dim=-1)
        v = self.v + a * step
        # Bound the *state*, not just the value handed to the observation. A
        # tumbling drone drives this dead-reckoning integrator divergent, and
        # once v overflows the omega x v term evaluates inf x inf -> NaN, which
        # then propagates into the observation and poisons the network for the
        # rest of the run. Clamping here keeps a blown-up episode local to that
        # episode.
        self.v = torch.nan_to_num(v, nan=0.0, posinf=VEL_CLIP, neginf=-VEL_CLIP).clamp(
            -VEL_CLIP, VEL_CLIP
        )
        return self.v


class CameraFrameLatch:
    """Hold gate keypoints at the camera frame rate, not the control rate.

    The control loop runs at 100 Hz but the vision stream is 30 Hz, so a new
    frame -- and therefore a new set of keypoints and visibility flags -- only
    arrives every ~33 ms. Without this the keypoints get re-projected every
    control step, which hands the policy vision 3.3x fresher than the real
    vehicle can deliver and lets it learn a servo loop that will not exist at
    deployment. Latching instead forces it to bridge the steps between frames on
    gyro and attitude alone.

    100/30 is not an integer, so the accumulator carries its remainder across
    frames rather than resetting to zero. That gives a 3, 3, 4 step cadence
    averaging exactly 30 Hz, instead of the 33.3 Hz (every 3 steps) or 25 Hz
    (every 4 steps) that a fixed stride would produce.

    Only the camera channels belong here. Attitude, body rates and commanded
    velocity come off the IMU and the controller, so they stay at 100 Hz.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        *,
        num_keypoints: int = KEYPOINT_COUNT,
        rate_hz: float = CAMERA_RATE_HZ,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if rate_hz <= 0.0:
            raise ValueError(f"camera rate must be positive, got {rate_hz}")
        self.num_envs = int(num_envs)
        self.device = device
        self.dtype = dtype
        self.rate_hz = float(rate_hz)
        self.period = 1.0 / self.rate_hz
        self.uv = torch.zeros(self.num_envs, num_keypoints, 2, device=device, dtype=dtype)
        self.visible = torch.zeros(self.num_envs, num_keypoints, device=device, dtype=torch.bool)
        self._elapsed = torch.zeros(self.num_envs, device=device, dtype=dtype)
        # Forcing the first capture with a flag rather than by starting the timer
        # already overdue keeps the phase clean. Starting it overdue leaves a
        # leftover of one control step after the first capture, which fires a
        # second capture immediately and puts the stream permanently out of phase.
        self._needs_capture = torch.ones(self.num_envs, device=device, dtype=torch.bool)

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        """Drop the held frame so the next step captures a fresh one."""
        if env_ids is None:
            self._needs_capture.fill_(True)
            self.visible.zero_()
            return
        self._needs_capture[env_ids] = True
        self.visible[env_ids] = False

    def step(
        self, dt: float, uv_px: torch.Tensor, visible: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance by one control step, capturing a frame only when one is due.

        Args:
            dt: control step in seconds.
            uv_px: (N, K, 2) freshly projected pixels, used only on capture steps.
            visible: (N, K) fresh visibility, used only on capture steps.

        Returns:
            The currently held (uv_px, visible), which is the new frame on a
            capture step and the previous frame otherwise.
        """
        self._elapsed += float(dt)
        forced = self._needs_capture
        natural = self._elapsed >= self.period - 1e-9
        due = forced | natural
        if bool(due.any()):
            self.uv[due] = uv_px[due].to(self.dtype)
            self.visible[due] = visible[due]
            # A scheduled capture carries its remainder so the average rate stays
            # exact; a forced one restarts the clock from this frame.
            self._elapsed[natural & ~forced] -= self.period
            self._elapsed[forced] = 0.0
            self._needs_capture[forced] = False
        return self.uv, self.visible


def pack_observation(
    uv_px: torch.Tensor,
    visible: torch.Tensor,
    roll: torch.Tensor,
    pitch: torch.Tensor,
    gyro_ned: torch.Tensor,
    *,
    velocity_body_ned: Optional[torch.Tensor] = None,
    gate_index: Optional[torch.Tensor] = None,
    with_velocity: bool = False,
    with_context: bool = False,
    actions: Optional[torch.Tensor] = None,
    frame_w: float = FRAME_W,
    frame_h: float = FRAME_H,
) -> torch.Tensor:
    """Assemble the AI_GP observation vector. Shape (N, feature_dim).

    ``actions`` appends the four raw policy channels to the end of the frame,
    making it observation ``v2`` (55 wide) instead of ``v1`` (51). They go last
    so v2 is a strict append and the first 51 channels still mean exactly what
    a v1 reader expects.
    """
    n = uv_px.shape[0]
    device = uv_px.device
    dtype = torch.float32

    u_n = (uv_px[..., 0] / frame_w).clamp(0.0, 1.0)
    v_n = (uv_px[..., 1] / frame_h).clamp(0.0, 1.0)
    # Interleave (u, v) per keypoint, then replace unseen with NOT_SEEN.
    corners = torch.stack([u_n, v_n], dim=-1)  # (N, 8, 2)
    corners = torch.where(visible.unsqueeze(-1), corners, torch.full_like(corners, NOT_SEEN))
    corners = corners.reshape(n, KEYPOINT_COUNT * 2)
    vis = visible.to(dtype=dtype)

    gyro = gyro_ned.clamp(-GYRO_CLIP, GYRO_CLIP)
    state = torch.stack([roll, pitch, gyro[:, 0], gyro[:, 1], gyro[:, 2]], dim=-1).to(dtype)

    parts = [corners.to(dtype), vis, state]
    if with_velocity:
        if velocity_body_ned is None:
            velocity_body_ned = torch.zeros(n, 3, device=device, dtype=dtype)
        parts.append(velocity_body_ned.clamp(-VEL_CLIP, VEL_CLIP).to(dtype))
    if with_context:
        if gate_index is None:
            gate_index = torch.zeros(n, device=device, dtype=torch.long)
        parts.append(context_features(gate_index))
    if actions is not None:
        # What the policy *issued*, not what the plant received. The aircraft
        # knows what it sent the instant it sends it; whether that command has
        # landed yet is exactly the thing it cannot observe, and handing it over
        # would put the delay back in the policy's hands as free information.
        parts.append(actions[:, :ACTION_DIM].clamp(-1.0, 1.0).to(dtype))
    obs = torch.cat(parts, dim=-1)
    # Last line of defence. clamp() passes NaN through unchanged, so a single
    # non-finite pixel would otherwise reach the network, and one NaN gradient
    # destroys the weights permanently -- the failure is silent and unrecoverable
    # rather than merely a bad episode.
    return torch.nan_to_num(obs, nan=NOT_SEEN, posinf=1.0, neginf=NOT_SEEN)


def build_aigp_observation(
    drone_pos_w: torch.Tensor,
    drone_quat_w: torch.Tensor,
    drone_ang_vel_b_flu: torch.Tensor,
    gate_pos_w: torch.Tensor,
    gate_quat_w: torch.Tensor,
    gate_index: torch.Tensor,
    *,
    velocity_body_ned: Optional[torch.Tensor] = None,
    with_velocity: bool = False,
    with_context: bool = True,
    camera_latch: Optional["CameraFrameLatch"] = None,
    keypoint_delay: Optional["KeypointDelayLine"] = None,
    actions: Optional[torch.Tensor] = None,
    dt: float = 0.0,
) -> torch.Tensor:
    """Full AI_GP observation from Isaac scene state. Shape (N, D).

    Velocity, when requested, must be *commanded* body velocity (thrust +
    attitude + drag), never IMU-linear-accel integration or privileged sim twist.

    Pass ``camera_latch`` (with ``dt`` set to the control step) to rate-limit the
    keypoints to the 30 Hz vision stream. Without it they refresh every control
    step, which is faster than any real camera.

    Pass ``keypoint_delay`` to additionally hold the result for the time the
    perception pipeline takes to produce it. Order matters and is not
    interchangeable: the camera samples the world at its own rate, and only
    then does inference take time to deliver that sample, so the delay is
    applied to the *latched* frame.

    Pass ``actions`` -- the four raw channels the policy last emitted, in
    ``[-1, 1]`` -- to append them to the frame. That is observation ``v2``,
    and it is what lets a policy account for a command it has issued but not
    yet seen take effect. Omit it for ``v1``.
    """
    kps_w = align_keypoints_to_view(
        gate_keypoints_world(gate_pos_w, gate_quat_w),
        gate_pos_w,
        gate_quat_w,
        drone_pos_w,
    )
    uv, visible = project_points_aigp_camera(kps_w, drone_pos_w, drone_quat_w)
    if camera_latch is not None:
        uv, visible = camera_latch.step(dt, uv, visible)
    if keypoint_delay is not None:
        uv, visible = keypoint_delay.step(uv, visible)
    roll, pitch = attitude_roll_pitch_ned(drone_quat_w)
    gyro_ned = flu_to_ned(drone_ang_vel_b_flu)
    return pack_observation(
        uv,
        visible,
        roll,
        pitch,
        gyro_ned,
        velocity_body_ned=velocity_body_ned,
        gate_index=gate_index,
        with_velocity=with_velocity,
        with_context=with_context,
        actions=actions,
    )
