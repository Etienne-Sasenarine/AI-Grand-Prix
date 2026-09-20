# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from utils.vel_align import (
    approach_aim_pos,
    cam_cone_score,
    progress_to_aim,
    run_in_progress_from_pos,
    vel_align_from_vel,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def pos_error_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_pos: list | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize asset pos from its target pos using L2 squared kernel."""

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    if target_pos is None:
        target_pos = env.command_manager.get_term(command_name).command
        target_pos_tensor = target_pos[:, :3]
    else:
        target_pos_tensor = (
            torch.tensor(target_pos, dtype=torch.float32, device=asset.device).repeat(env.num_envs, 1)
            + env.scene.env_origins
        )

    # Compute sum of squared errors
    return torch.sum(torch.square(asset.data.root_pos_w - target_pos_tensor), dim=1)


def pos_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str | None = None,
    target_pos: list | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize asset pos from its target pos using L2 squared kernel."""

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    if target_pos is None:
        target_pos = env.command_manager.get_term(command_name).command
        target_pos_tensor = target_pos[:, :3]
    else:
        target_pos_tensor = (
            torch.tensor(target_pos, dtype=torch.float32, device=asset.device).repeat(env.num_envs, 1)
            + env.scene.env_origins
        )

    distance = torch.norm(asset.data.root_pos_w - target_pos_tensor, dim=1)
    return 1 - torch.tanh(distance / std)


def progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize asset pos from its target pos using L2 squared kernel."""

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    cmd = env.command_manager.get_term(command_name)
    n_hat = gate_normal_w(cmd.command[:, 3:7])
    return progress_to_aim(
        cmd.previous_pos,
        asset.data.root_pos_w,
        cmd.command[:, :3],
        n_hat,
    )


def run_in_progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    run_in_m: float = 3.0,
    only_exit_side: bool = True,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense metres closed toward the 3 m approach waypoint, exit-side only.

    See ``run_in_progress_from_pos``. Same run-in distance as play spawn.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(command_name)
    n_hat = gate_normal_w(cmd.command[:, 3:7])
    return run_in_progress_from_pos(
        cmd.previous_pos,
        asset.data.root_pos_w,
        cmd.command[:, :3],
        n_hat,
        run_in_m=run_in_m,
        only_exit_side=only_exit_side,
    )


def gate_proximity(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 8.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense reward for sitting near the target gate's centre.

    ``progress`` only pays for *changes* in distance, so a drone that stalls or
    circles earns nothing from it and the sparse ``gate_passed`` bonus is
    unreachable until it already flies well. This gives a standing gradient that
    points at the centre of the opening from anywhere on the approach, which is
    what an early, mostly-random policy needs to get its first gate.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(command_name)
    n_hat = gate_normal_w(cmd.command[:, 3:7])
    aim = approach_aim_pos(asset.data.root_pos_w, cmd.command[:, :3], n_hat)
    distance = torch.norm(asset.data.root_pos_w - aim, dim=1)
    return torch.exp(-distance / std)


def gate_passed(
    env: ManagerBasedRLEnv,
    command_name: str | None = None,
) -> torch.Tensor:
    """+1 on a clean pass through the target opening.

    Misses used to share this term as −1. At weight 400 that is the same
    magnitude as a success, so after a map change the cheapest policy is
    “never cross the plane”: zero from this term beats a likely miss.
    A miss still ends the episode via ``gate_missed`` + ``terminating``.
    """
    return env.command_manager.get_term(command_name).gate_passed.to(dtype=torch.float32)


def gate_pass_speed(
    env: ManagerBasedRLEnv,
    command_name: str,
    ref_speed: float = 12.0,
    max_scale: float = 1.5,
) -> torch.Tensor:
    """Bonus on gate pass, scaled by how fast the opening was crossed.

    ``gate_passed`` pays a flat bonus regardless of speed, and ``progress``
    telescopes to the same total over a leg however long it takes, so nothing in
    the reward distinguishes a fast lap from a slow one beyond discounting. This
    pays out only on the crossing step, proportional to the speed along the gate
    normal, so the incentive is to carry speed *through* the gate rather than to
    fly fast anywhere.

    Returns 1.0 at ``ref_speed``, capped at ``max_scale`` so an overspeed
    approach cannot outbid actually staying on the course.
    """
    speed = env.command_manager.get_term(command_name).gate_pass_speed
    return (speed / ref_speed).clamp(0.0, max_scale)


def gate_normal_w(gate_quat_w: torch.Tensor) -> torch.Tensor:
    """World-XY gate normal ``[cos θ, sin θ, 0]`` from an Isaac wxyz quaternion."""
    _, _, yaw = math_utils.euler_xyz_from_quat(gate_quat_w)
    zero = torch.zeros_like(yaw)
    return torch.stack([torch.cos(yaw), torch.sin(yaw), zero], dim=-1)


def vel_align_gate(
    env: ManagerBasedRLEnv,
    command_name: str,
    min_speed: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Fly along the gate normal, or toward the run-in when already in front."""
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(command_name)
    n_hat = gate_normal_w(cmd.command[:, 3:7])
    pos = asset.data.root_pos_w
    gate = cmd.command[:, :3]
    on_exit = ((pos - gate) * n_hat).sum(dim=-1) > 0.0
    to_run_in = torch.nn.functional.normalize(
        approach_aim_pos(pos, gate, n_hat) - pos, dim=-1, eps=1e-6
    )
    direction = torch.where(on_exit.unsqueeze(-1), to_run_in, n_hat)
    return vel_align_from_vel(asset.data.root_lin_vel_w, direction, min_speed=min_speed)


def vel_gate_passage(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Sparse crossing bonus: ``max(0, v · n̂)`` in m/s, else 0.

    Weight 3.0 → a 5 m/s straight pass is +15 on top of ``gate_passed``.
    A sideways tumble (v ⟂ n) is 0.
    """
    return env.command_manager.get_term(command_name).gate_pass_speed.clamp(min=0.0)


def center_gate_passage(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Sparse centering: linear in plane offset, ``4 * (1 - r_⊥ / 0.75)``.

    Dead-center +4, 0.375 m off +2, rim +0. Same scale as ``cam_gate_passage``.
    """
    return env.command_manager.get_term(command_name).gate_pass_center


def cam_gate_passage(
    env: ManagerBasedRLEnv,
    command_name: str,
    psi_deg: float = 40.0,
) -> torch.Tensor:
    """Sparse look-through: 1 if the FPV axis is on ``n̂``, 0 at ``ψ`` or outside.

    Horizontal camera bearing vs the gate normal, same ``n̂`` as velocity.
    Smaller than ``vel_gate_passage`` (weight 4 → +4 dead-center, 0 at 40°).
    """
    align = env.command_manager.get_term(command_name).gate_pass_cam_align
    return cam_cone_score(align, psi_deg=psi_deg)


def time_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Constant 1.0 so the RewTerm weight is the per-step cost."""
    return torch.ones(env.num_envs, device=env.device, dtype=torch.float32)


def heading_to_gate(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Pay for yawing the nose onto the gate bearing (horizontal only).

    ``lookat_next_gate`` is 3D body-X alignment: pitch or roll can fake it, and
    the weight was too small to beat ``progress``. This is the turn the camera
    actually needs.
    """
    from utils.aigp_obs import heading_alignment

    asset: RigidObject = env.scene[asset_cfg.name]
    gate_pos = env.command_manager.get_term(command_name).command[:, :3]
    return heading_alignment(asset.data.root_pos_w, asset.data.root_quat_w, gate_pos, std=std)


def gate_visible(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.55,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Loose in-frame score: corner fraction × proximity to image centre.

    Not a binary lock and not a bullseye. ``std`` is in normalised pixels
    (see ``gate_center_score``); keep the RewTerm weight tiny so this cannot
    beat ``gate_passed``.
    """
    from utils.aigp_obs import gate_center_score

    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(command_name)
    pose = cmd.command
    return gate_center_score(
        asset.data.root_pos_w,
        asset.data.root_quat_w,
        pose[:, :3],
        pose[:, 3:7],
        std=std,
    )


def lookat_next_gate(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for looking at the next gate."""

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    drone_pos = asset.data.root_pos_w
    drone_att = asset.data.root_quat_w
    next_gate_pos = env.command_manager.get_term(command_name).command[:, :3]

    vec_to_gate = next_gate_pos - drone_pos
    vec_to_gate = math_utils.normalize(vec_to_gate)

    x_axis = torch.tensor([1.0, 0.0, 0.0], device=asset.device).expand(env.num_envs, 3)
    drone_x_axis = math_utils.quat_apply(drone_att, x_axis)
    drone_x_axis = math_utils.normalize(drone_x_axis)

    dot = (drone_x_axis * vec_to_gate).sum(dim=1).clamp(-1.0, 1.0)
    angle = torch.acos(dot)
    return torch.nan_to_num(torch.exp(-angle / std), nan=0.0, posinf=0.0, neginf=0.0)


def low_gate_dive(
    env: ManagerBasedRLEnv,
    command_name: str,
    drop_m: float = 0.6,
    range_min: float = 2.0,
    range_max: float = 18.0,
    pitch_ref: float = 0.35,
    sink_ref: float = 3.0,
    hover_n: float = 5.96,
    action_name: str = "control_action",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Pay for the dive mechanic on a low gate: nose down, cut thrust, sink.

    Active only when the target opening is at least ``drop_m`` below the drone
    and the horizontal range is an approach (not already past / on top of it).
    Each cue is 0..1; the mean is the term so a weight of 1.0 is a nudge
    against ``progress`` (tens of points per metre) rather than a takeover.
    """
    from utils.aigp_obs import attitude_roll_pitch_ned

    asset: RigidObject = env.scene[asset_cfg.name]
    gate_pos = env.command_manager.get_term(command_name).command[:, :3]
    pos = asset.data.root_pos_w
    delta = gate_pos - pos
    height_above = pos[:, 2] - gate_pos[:, 2]
    horiz = torch.linalg.norm(delta[:, :2], dim=1)
    active = (height_above > drop_m) & (horiz > range_min) & (horiz < range_max)

    _, pitch = attitude_roll_pitch_ned(asset.data.root_quat_w)
    # NED: negative pitch is nose-down / tilt forward.
    nose_down = (-pitch / pitch_ref).clamp(0.0, 1.0)

    thrust = torch.full((env.num_envs,), hover_n, device=pos.device, dtype=pos.dtype)
    if getattr(env, "action_manager", None) is not None:
        try:
            action_term = env.action_manager.get_term(action_name)
        except (KeyError, ValueError, AttributeError):
            action_term = None
        if action_term is not None and hasattr(action_term, "processed_actions"):
            thrust = action_term.processed_actions[:, 0].to(dtype=pos.dtype)
    cut_thrust = ((hover_n - thrust) / hover_n).clamp(0.0, 1.0)

    # World -Z is down in this Z-up stage.
    sink = (-asset.data.root_lin_vel_w[:, 2] / sink_ref).clamp(0.0, 1.0)
    score = (nose_down + cut_thrust + sink) / 3.0
    return torch.where(active, score, torch.zeros_like(score))


def ang_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b), dim=1)


# The qualifier is scored on completing two laps, not on lap time -- and at the
# time of writing exactly one team has managed two full laps. So speed is not
# the objective; it is a cost paid in precision.
#
# Measured on the first racing policy trained against the corrected plant: it
# flew at 14.0 m/s median and 21.3 m/s at p95, and 73% of its crashes were gate
# strikes at that speed, one metre from the gate centre. Two laps of this
# 106.2 m course inside the 40 s episode needs 5.3 m/s average. The policy was
# flying roughly three times faster than the task requires and spending the
# difference on hitting gate frames.
#
# Slowing down helps three ways at once, and the third is the one that is easy
# to miss: the modelled vision delay of 1-4 control steps is 0.24-0.94 m of
# stale keypoints at 14 m/s, and half that at 7. Latency hurts in metres, not
# milliseconds, so halving speed halves the effective sensor staleness.
DEFAULT_SPEED_CAP_MPS = 8.0


def over_speed(
    env: ManagerBasedRLEnv,
    cap_mps: float = DEFAULT_SPEED_CAP_MPS,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared excess speed above ``cap_mps``, zero below it.

    Squared rather than linear so the penalty is gentle at the cap and steep
    well beyond it: a policy drifting to 9 m/s is barely discouraged, one
    charging at 20 m/s is strongly so. A linear penalty would tax the useful
    range as hard as the dangerous one.

    Deliberately *not* the post-policy ``utils.speed_cap`` limiter. That brakes
    after the policy has already committed, which trains a policy that fights
    its own limiter and oscillates at the boundary. This makes the speed part
    of what the policy is optimising, so it learns to fly at a pace it can
    actually hold a line at.

    Cap the *whole* velocity rather than the forward component: a 15 m/s
    sideways drift through a gate opening is no more survivable than a 15 m/s
    charge at its frame.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    speed = torch.norm(asset.data.root_lin_vel_w, dim=-1)
    return torch.square((speed - cap_mps).clamp(min=0.0))
