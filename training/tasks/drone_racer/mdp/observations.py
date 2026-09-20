# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from utils.aigp_obs import DEFAULT_MASS_KG, build_aigp_observation
from utils.logger import log

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def root_lin_vel_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root linear velocity in the body frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel = asset.data.root_lin_vel_b
    log(env, ["vx", "vy", "vz"], lin_vel)
    return lin_vel


def root_ang_vel_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root angular velocity in the body frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel = asset.data.root_ang_vel_b
    log(env, ["wx", "wy", "wz"], ang_vel)
    return ang_vel


def root_quat_w(
    env: ManagerBasedRLEnv, make_quat_unique: bool = False, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Asset root orientation (w, x, y, z) in the environment frame."""

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    quat = asset.data.root_quat_w
    log(env, ["qw", "qx", "qy", "qz"], quat)
    return math_utils.quat_unique(quat) if make_quat_unique else quat


def root_rotmat_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root orientation (3x3 flattened rotation matrix) in the world frame."""
    asset: RigidObject = env.scene[asset_cfg.name]

    quat = asset.data.root_quat_w
    rotmat = math_utils.matrix_from_quat(quat)
    flat_rotmat = rotmat.view(-1, 9)
    log(env, ["r11", "r12", "r13", "r21", "r22", "r23", "r31", "r32", "r33"], flat_rotmat)
    return flat_rotmat


def root_pos_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root position in the world frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    position = asset.data.root_pos_w
    log(env, ["px", "py", "pz"], position)
    return position


def root_pose_g(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Asset root position in the gate frame."""
    asset: RigidObject = env.scene[asset_cfg.name]

    gate_pose_w = env.command_manager.get_term(command_name).command  # (num_envs, 7)
    drone_pose_w = asset.data.root_state_w[:, :7]  # (num_envs, 7)

    # Extract positions and quaternions
    gate_pos_w = gate_pose_w[:, :3]
    gate_quat_w = gate_pose_w[:, 3:7]
    drone_pos_w = drone_pose_w[:, :3]
    drone_quat_w = drone_pose_w[:, 3:7]

    # Compute drone pose in gate frame
    # Inverse gate quaternion
    gate_quat_w_inv = math_utils.quat_inv(gate_quat_w)

    # Position of drone in gate frame
    rel_pos = drone_pos_w - gate_pos_w
    drone_pos_g = math_utils.quat_rotate(gate_quat_w_inv, rel_pos)

    # Orientation of drone in gate frame
    drone_quat_g = math_utils.quat_mul(gate_quat_w_inv, drone_quat_w)

    # Concatenate position and quaternion
    position = torch.cat([drone_pos_g, drone_quat_g], dim=-1)

    return position


def next_gate_pose_g(
    env: ManagerBasedRLEnv,
    command_name: str,
    scale: float = 30.0,
) -> torch.Tensor:
    """Pose of the gate *after* the target, in the target gate's frame.

    The AI_GP keypoint channel projects only the current target gate, so
    approaching gate 8 carries no information that a 42 deg turn follows it.
    Measured crashes concentrate hard on exactly those corners, which a policy
    cannot anticipate from an input it does not have.

    Expressed relative to the target gate rather than to the drone, this is a
    static property of the course -- where gate i+1 sits relative to gate i --
    so it needs a track map but no localisation, and is available on hardware
    from a course description or a practice lap. Position is divided by
    ``scale`` to keep it in roughly the same range as the rest of the vector.
    """
    gate_pose_w = env.command_manager.get_term(command_name).command  # (num_envs, 7)
    next_gate_pose_w = env.command_manager.get_term(command_name).lookahead_gate  # (num_envs, 7)

    # Extract positions and quaternions
    gate_pos_w = gate_pose_w[:, :3]
    gate_quat_w = gate_pose_w[:, 3:7]
    next_gate_pos_w = next_gate_pose_w[:, :3]
    next_gate_quat_w = next_gate_pose_w[:, 3:7]

    # Compute drone pose in gate frame
    # Inverse gate quaternion
    gate_quat_w_inv = math_utils.quat_inv(gate_quat_w)

    # Position of drone in gate frame
    rel_pos = next_gate_pos_w - gate_pos_w
    next_gate_pos_g = math_utils.quat_rotate(gate_quat_w_inv, rel_pos) / scale

    # Orientation of drone in gate frame
    next_gate_quat_g = math_utils.quat_mul(gate_quat_w_inv, next_gate_quat_w)

    # Concatenate position and quaternion
    position = torch.cat([next_gate_pos_g, next_gate_quat_g], dim=-1)

    return torch.nan_to_num(position, nan=0.0, posinf=0.0, neginf=0.0)


def target_pos_b(
    env: ManagerBasedRLEnv,
    command_name: str | None = None,
    target_pos: list | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Position of target in body frame."""

    asset: RigidObject = env.scene[asset_cfg.name]

    if target_pos is None:
        target_pos = env.command_manager.get_term(command_name).command[:, :3]
        target_pos_tensor = target_pos[:, :3]
    else:
        target_pos_tensor = (
            torch.tensor(target_pos, dtype=torch.float32, device=asset.device).repeat(env.num_envs, 1)
            + env.scene.env_origins
        )

    pos_b, _ = math_utils.subtract_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, target_pos_tensor)

    return pos_b


def aigp_race_observation(
    env: ManagerBasedRLEnv,
    command_name: str = "target",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    action_name: str = "control_action",
    with_context: bool = True,
    with_velocity: bool = True,
    mass_kg: float = DEFAULT_MASS_KG,
    drag_body: tuple[float, float, float] = (-0.50, -0.50, -0.15),
    camera_rate_hz: float = 30.0,
    use_visual_gate_counter: bool | None = None,
) -> torch.Tensor:
    """AI_GP / ``race_obs`` state vector for the policy.

    Layout (default ``with_context=True``, ``with_velocity=True`` → 51-D):
      corners(16) + visibility(8) + roll/pitch/gyro(5) + cmd_vel(3)
      + gate one-hot(18) + gate_frac(1)

    Keypoints are geometric projections of the active gate through the AI_GP
    camera model. Gyro is used for body rates; the IMU *accelerometer is never
    read*. Velocity channels are commanded body velocity from thrust + attitude
    + drag (AI_GP ``BodyVelocityIntegrator``), not sim ground-truth twist.

    The observation is deliberately mixed-rate. Keypoints and visibility are
    held at ``camera_rate_hz`` (30 Hz vision stream, VADR-TS-001 4.6) while
    attitude, body rates and commanded velocity update every control step at
    100 Hz. Set ``camera_rate_hz`` to 0 to refresh vision every step instead,
    which is faster than the real camera and only useful for ablations.

    Training keeps the privileged ``next_gate_idx`` one-hot. For play / a real
    bird set ``use_visual_gate_counter`` or ``VISUAL_GATE_COUNTER=1``: keypoints
    and the one-hot then follow an onboard pass counter (opening grows, through,
    leaves frame) instead of the sim's plane-crossing oracle.
    """
    from utils.aigp_obs import (
        G,
        CameraFrameLatch,
        CommandedBodyVelocity,
        attitude_roll_pitch_ned,
        flu_to_ned,
    )

    robot: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(command_name)
    if use_visual_gate_counter is None:
        raw = os.environ.get("VISUAL_GATE_COUNTER", "0").strip().lower()
        use_visual_gate_counter = raw in ("1", "true", "yes")
    visual_counter = _maybe_visual_counter(env, cmd, robot.device, bool(use_visual_gate_counter))
    if visual_counter is not None:
        idx = visual_counter.index
        rows = torch.arange(env.num_envs, device=robot.device)
        gate_pos_w = cmd.track.data.object_com_pos_w[rows, idx]
        gate_quat_w = cmd.track.data.object_quat_w[rows, idx]
        gate_index = idx
    else:
        gate_pose_w = cmd.command  # (N, 7) pos + quat
        gate_pos_w = gate_pose_w[:, :3]
        gate_quat_w = gate_pose_w[:, 3:7]
        gate_index = cmd.next_gate_idx

    roll, pitch = attitude_roll_pitch_ned(robot.data.root_quat_w)
    gyro_ned = flu_to_ned(robot.data.root_ang_vel_b)

    # Fresh episodes must not inherit a held camera frame or prior dead-reckoning
    # state. The buffer is absent while the observation manager probes term
    # dimensions, hence the getattr.
    episode_len = getattr(env, "episode_length_buf", None)
    reset_ids = None
    if episode_len is not None:
        reset_ids = (episode_len == 0).nonzero(as_tuple=False).flatten()
        if not len(reset_ids):
            reset_ids = None

    camera_latch = None
    if camera_rate_hz and camera_rate_hz > 0.0:
        camera_latch = getattr(env, "_aigp_cam_latch", None)
        if (
            camera_latch is None
            or camera_latch.num_envs != env.num_envs
            or camera_latch.device != robot.device
            or abs(camera_latch.rate_hz - float(camera_rate_hz)) > 1e-6
        ):
            camera_latch = CameraFrameLatch(
                env.num_envs, robot.device, rate_hz=float(camera_rate_hz)
            )
            env._aigp_cam_latch = camera_latch
        if reset_ids is not None:
            camera_latch.reset(reset_ids)

    velocity_body = None
    if with_velocity:
        hover_trim_n = float(mass_kg) * G
        integ = getattr(env, "_aigp_cmd_vel", None)
        if (
            integ is None
            or integ.num_envs != env.num_envs
            or integ.device != robot.device
            or abs(integ.hover_trim - hover_trim_n) > 1e-6
        ):
            integ = CommandedBodyVelocity(
                env.num_envs,
                robot.device,
                hover_trim=hover_trim_n,
                k_body=tuple(drag_body),
            )
            env._aigp_cmd_vel = integ

        if reset_ids is not None:
            integ.reset(reset_ids)

        # Applied thrust in Newtons (ControlAction collective, body +Z).
        action_term = None
        if getattr(env, "action_manager", None) is not None:
            try:
                action_term = env.action_manager.get_term(action_name)
            except (KeyError, ValueError, AttributeError):
                action_term = None
        if action_term is not None and hasattr(action_term, "processed_actions"):
            thrust_n = action_term.processed_actions[:, 0]
        else:
            thrust_n = torch.full((env.num_envs,), hover_trim_n, device=robot.device)

        velocity_body = integ.step(
            float(env.step_dt),
            thrust_n,
            roll,
            pitch,
            omega_ned=gyro_ned,
            hover_trim=hover_trim_n,
        )
        log(env, ["cmd_vx", "cmd_vy", "cmd_vz"], velocity_body)

    obs = build_aigp_observation(
        drone_pos_w=robot.data.root_pos_w,
        drone_quat_w=robot.data.root_quat_w,
        drone_ang_vel_b_flu=robot.data.root_ang_vel_b,
        gate_pos_w=gate_pos_w,
        gate_quat_w=gate_quat_w,
        gate_index=gate_index,
        velocity_body_ned=velocity_body,
        with_velocity=with_velocity,
        with_context=with_context,
        camera_latch=camera_latch,
        dt=float(env.step_dt),
    )

    if visual_counter is not None:
        uv = obs[:, :16].reshape(env.num_envs, 8, 2)
        vis = obs[:, 16:24] > 0.5
        passed = visual_counter.step(uv, vis, dt=float(env.step_dt))
        if camera_latch is not None and bool(passed.any()):
            camera_latch.reset(passed.nonzero(as_tuple=False).flatten())
        agree = (visual_counter.index == cmd.next_gate_idx.long()).to(obs.dtype)
        log(env, ["visual_gate"], visual_counter.index.to(obs.dtype).unsqueeze(-1))
        log(env, ["visual_gate_agree"], agree.unsqueeze(-1))
        extras_log = env.extras.setdefault("log", {})
        extras_log["visual_gate_agree"] = agree.mean()
        extras_log["visual_gate_pass"] = passed.to(obs.dtype).mean()

    log(env, ["aigp_roll", "aigp_pitch"], obs[:, 24:26])
    log(env, ["aigp_gx", "aigp_gy", "aigp_gz"], obs[:, 26:29])
    n_vis = obs[:, 16:24].sum(dim=-1, keepdim=True)
    log(env, ["aigp_n_vis"], n_vis)
    return obs


def _maybe_visual_counter(env, cmd, device, enabled: bool):
    """Attach / reset the onboard pass counter. None when privileged index is used."""
    if not enabled:
        return None
    from utils.gate_counter import TorchVisualGateCounter

    n_course = int(getattr(cmd, "num_gates", 11) or 11)
    counter = getattr(env, "_visual_gate_counter", None)
    if (
        counter is None
        or getattr(counter, "num_envs", None) != env.num_envs
        or getattr(counter, "n_course", None) != n_course
        or torch.device(getattr(counter, "device", device)) != torch.device(device)
    ):
        counter = TorchVisualGateCounter(
            env.num_envs,
            device,
            start_idx=cmd.next_gate_idx,
            n_course=n_course,
            loop=bool(getattr(cmd.cfg, "loop", True)),
        )
        env._visual_gate_counter = counter

    episode_len = getattr(env, "episode_length_buf", None)
    if episode_len is not None:
        reset_ids = (episode_len == 0).nonzero(as_tuple=False).flatten()
        if len(reset_ids):
            counter.reset(reset_ids, start_idx=cmd.next_gate_idx[reset_ids])
    return counter
