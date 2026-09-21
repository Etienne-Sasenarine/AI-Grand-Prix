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

from contract.observation import NOT_SEEN
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


def _keypoint_dropout(env, obs: torch.Tensor) -> torch.Tensor:
    """Drop corners the way the real detector does: in bursts, not at random.

    In simulation the corners are projected analytically and are always
    detected, so the policy has never seen one blink out. On the aircraft they
    do, constantly. Measured at a real gate on 2026-09-21 over 1345 frames,
    a corner's visibility flips between consecutive frames **4.2%** of the
    time, and the runner rejects any frame carrying fewer than four corners --
    clearing its entire six-frame history when it does.

    The distribution matters more than the rate. Failures arrive in bursts, not
    independently: an independent-frames model predicts the policy would be
    ready 9% of the time, and it was measured at 58%, because the bad frames
    clump together and leave long clean stretches between them. Training
    against independent dropout would therefore teach the wrong thing -- it
    would present a hostile world with no usable runs in it.

    So each corner carries a two-state Markov chain per environment,
    parameterised by where it settles and how sticky it is:

        p(visible -> dropped) = (1 - rho) * q
        p(dropped -> visible) = (1 - rho) * (1 - q)

    which gives a steady-state dropped fraction of ``q`` and a flip rate of
    ``2 q (1 - q) (1 - rho)``.

    The flip rate alone does not pin ``q`` -- it constrains only the product --
    so ``q`` was measured separately. At the 3 m hover pose the geometry offers
    four corners, and of the frames where a gate was found, 49% carried four
    and 50% carried three: about half a corner missing out of four, so
    **q ~ 0.13**. With rho = 0.81 that reproduces the measured 4.2% flip rate.

    Only corners the geometry already put in frame are dropped. A corner
    outside the frame is not a detector failure and the simulator models it
    already, so dropping it again would double-count.
    """
    q = float(os.environ.get("AIGP_KP_DROP", "0.0"))
    if q <= 0.0:
        return obs
    rho = float(os.environ.get("AIGP_KP_STICKY", "0.81"))
    vis = obs[:, 16:24]
    state = getattr(env, "_kp_dropped", None)
    if state is None or state.shape != vis.shape or state.device != vis.device:
        state = torch.rand(vis.shape, device=vis.device) < q
        env._kp_dropped = state

    p_drop = (1.0 - rho) * q
    p_recover = (1.0 - rho) * (1.0 - q)
    roll = torch.rand(vis.shape, device=vis.device)
    state = torch.where(state, roll >= p_recover, roll < p_drop)
    env._kp_dropped = state

    # Episodes that just reset start from a fresh draw rather than inheriting
    # the previous flight's burst.
    episode_len = getattr(env, "episode_length_buf", None)
    if episode_len is not None:
        fresh = (episode_len == 0)
        if bool(fresh.any()):
            state[fresh] = torch.rand(
                (int(fresh.sum()), vis.shape[1]), device=vis.device) < q
            env._kp_dropped = state

    keep = (~state).to(obs.dtype)
    obs[:, 16:24] = vis * keep
    # A corner that was not detected must carry the contract's absent
    # sentinel, NOT_SEEN = -1.0, and not zero.
    #
    # Zero is a *valid on-screen position* -- the top-left of the normalised
    # frame. Writing it makes a dropped corner teleport to the origin instead
    # of disappearing, which is a confidently wrong position rather than a
    # missing one, and strictly worse than the failure being modelled. The
    # first version of this did exactly that and collapsed racing from 12.80
    # gates per episode to 0.13.
    dropped_uv = state.repeat_interleave(2, dim=-1)
    obs[:, :16] = torch.where(
        dropped_uv, torch.full_like(obs[:, :16], NOT_SEEN), obs[:, :16]
    )
    return obs


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
    vision_delay_steps_range: tuple[int, int] | None = None,
    with_actions: bool = False,
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
        KeypointDelayLine,
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

    # Perception latency: how long after the shutter the keypoints exist.
    # Built after the latch because it wraps the *latched* frame -- the camera
    # samples at its own rate and only then does inference take time.
    keypoint_delay = None
    if vision_delay_steps_range is not None:
        lo_d, hi_d = vision_delay_steps_range
        keypoint_delay = getattr(env, "_aigp_vis_delay", None)
        if (
            keypoint_delay is None
            or keypoint_delay.num_envs != env.num_envs
            or keypoint_delay.device != robot.device
            or keypoint_delay.max_delay != int(hi_d)
        ):
            keypoint_delay = KeypointDelayLine(
                env.num_envs, robot.device, max_delay=int(hi_d)
            )
            keypoint_delay.resample(
                torch.arange(env.num_envs, device=robot.device), lo_d, hi_d
            )
            env._aigp_vis_delay = keypoint_delay
        if reset_ids is not None:
            keypoint_delay.resample(reset_ids, lo_d, hi_d)
            keypoint_delay.reset(reset_ids)

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

    # Observation v2: the four channels the policy last emitted.
    #
    # Deliberately ``issued_actions`` and not ``raw_actions``. The aircraft
    # knows what it sent the instant it sends it; whether that command has
    # reached the flight controller yet is exactly what it cannot observe.
    # Feeding back the delayed command would return the delay to the policy as
    # free information and undo the point of modelling it.
    issued_actions = None
    if with_actions:
        term = None
        if getattr(env, "action_manager", None) is not None:
            try:
                term = env.action_manager.get_term(action_name)
            except (KeyError, ValueError, AttributeError):
                term = None
        if term is not None and hasattr(term, "issued_actions"):
            issued_actions = term.issued_actions
        else:
            # The observation manager probes term dimensions before the action
            # manager exists. Zeros keep the width right during that probe.
            issued_actions = torch.zeros(env.num_envs, 4, device=robot.device)

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
        keypoint_delay=keypoint_delay,
        actions=issued_actions,
        dt=float(env.step_dt),
    )

    # Detector dropout, before anything downstream reads the corners.
    obs = _keypoint_dropout(env, obs)

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
