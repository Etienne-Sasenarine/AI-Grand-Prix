"""Rewards for gate-relative station keeping.

The task is to sit still a fixed distance in front of a gate, centred on its
through-axis, with the gate framed by the camera. Two things it buys:

**A cage deployment test.** It exercises the whole real chain -- camera, YOLO
keypoints, observation assembly, policy, MSP, flight controller -- against a
position reference the aircraft can actually sense. A *blind* hover cannot:
``orin-setup/xy_hold.py`` says it outright, "IMU acceleration alone cannot
detect constant horizontal drift or anchor a location", and the barometer
collapses under prop load. Without an exteroceptive anchor a hover policy holds
attitude and drifts into a wall, which tests nothing.

**A warm-start seed for racing.** The observation is the *same* vector the
racing policy uses, byte for byte, so the weights transfer. The hard part of
racing early on is not the racing -- it is that a policy which cannot hold
altitude never reaches gate 1, so the sparse gate reward never fires and there
is nothing to climb. Learning thrust trim and attitude first, against the same
randomised plant, gives that policy something to start from.

Note what this shares with racing and what it does not. Scene, observation,
action and plant are identical and reused unchanged. Only the reward and the
termination conditions differ: hold the point rather than pass through it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from tasks.drone_racer.mdp.rewards import gate_normal_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def hold_point(
    env: ManagerBasedRLEnv,
    command_name: str,
    standoff_m: float = 2.0,
) -> torch.Tensor:
    """Where the aircraft is supposed to sit: ``standoff_m`` in front of the gate.

    "In front" means on the approach side of the opening, along the gate's own
    through-axis, at the height of its centre. Picked at 2 m because that is
    close enough for the gate to fill a useful part of a 72.8 degree frame --
    the policy needs keypoints it can actually resolve -- while leaving room to
    drift without touching anything.
    """
    cmd = env.command_manager.get_term(command_name)
    gate_pos = cmd.command[:, :3]
    n_hat = gate_normal_w(cmd.command[:, 3:7])
    return gate_pos - standoff_m * n_hat


def station_keep(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.75,
    standoff_m: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gaussian on distance to the hold point. The task's whole objective.

    Gaussian rather than the racing task's ``exp(-d/std)`` because this is a
    *holding* reward, not an approach gradient. The exponential's long tail pays
    almost as well at 2 m as at 1 m, which is fine when the point is to draw a
    policy in from across a hall, and wrong when the point is to be precisely
    somewhere. The square falls away sharply enough that the last few tens of
    centimetres are worth working for.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    target = hold_point(env, command_name, standoff_m)
    distance = torch.norm(asset.data.root_pos_w - target, dim=1)
    return torch.exp(-((distance / std) ** 2))


def stillness(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise speed, in m/s squared.

    ``station_keep`` alone is satisfied by orbiting the hold point at speed, or
    by oscillating through it -- both sit near the target on average and neither
    is a hover. Pairing position with a velocity penalty is what makes the
    optimum *stationary* rather than merely *nearby*.

    Privileged world velocity is fine here. Rewards may read anything the
    simulator knows; only the observation is held to what the aircraft can
    sense.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_lin_vel_w), dim=1)


def upright(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward keeping the body's up-axis vertical, 1 level and 0 on its side.

    A quad holding station does tilt -- that is how it cancels drift -- so this
    is deliberately weak. Its job is to rule out the degenerate solutions where
    the aircraft holds position while inverted or knife-edge, which the position
    and velocity terms alone would happily accept.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    # Body +Z (FLU up) expressed in world, third component.
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return up_z.clamp(min=0.0)


def settled(
    env: ManagerBasedRLEnv,
    command_name: str,
    radius_m: float = 0.30,
    speed_m_s: float = 0.30,
    standoff_m: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """1.0 while genuinely parked: inside ``radius_m`` *and* under ``speed_m_s``.

    The dense terms shape the approach; this one names the goal. Without a term
    that only pays when both conditions hold at once, "close but drifting" and
    "slow but displaced" each collect most of the available reward and the
    policy has little reason to close the gap between them.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    target = hold_point(env, command_name, standoff_m)
    near = torch.norm(asset.data.root_pos_w - target, dim=1) < radius_m
    slow = torch.norm(asset.data.root_lin_vel_w, dim=1) < speed_m_s
    return (near & slow).to(dtype=torch.float32)


def approach(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 3.0,
    standoff_m: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Broad pull toward the hold point: ``exp(-d / std)``.

    Exists because ``station_keep`` has no reach. A Gaussian with a 0.75 m
    width returns ``exp(-178)`` at ten metres -- indistinguishable from zero in
    float32 -- so a policy that starts anywhere but on top of the hold point
    has no gradient to follow and learns nothing.

    The first fix for that was to move the *spawn* onto the hold point, which
    worked and was the wrong lever: it produced a policy that only knows how to
    hold from somewhere it is already holding. In the cage the aircraft is
    handed over from wherever it happens to be, not from 2.0 m out on the gate
    normal.

    So the reward gets the reach instead, and the spawn widens to anywhere the
    gate is visible. Exponential rather than Gaussian on purpose: its tail
    decays slowly enough to still be a usable gradient at five or six metres,
    which is the whole point. Pairing a broad shaping term with a narrow
    precision term is the standard shape for approach-then-hold, and keeps
    ``station_keep`` and ``settled`` meaning what they meant -- this term pays
    for getting there, they pay for staying.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    target = hold_point(env, command_name, standoff_m=standoff_m)
    distance = torch.norm(asset.data.root_pos_w - target, dim=1)
    return torch.exp(-distance / std)
