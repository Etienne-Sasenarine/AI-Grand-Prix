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
from isaaclab.assets import Articulation, RigidObject

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_after_prev_gate(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    gate_pose: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg_name: str = "robot",
    heading_quat: torch.Tensor | None = None,
):
    """Reset the asset right after a random gate.

    ``gate_pose`` fixes where the drone appears. ``heading_quat``, when given,
    fixes which way it faces; the random pose_range is then applied on top of
    that. Without it the heading falls back to the asset's default orientation,
    which is world +x regardless of where the course actually goes -- fine on a
    track that runs down one axis, but on a turning course it spawns the drone
    facing away from its target gate and the camera sees nothing to fly toward.
    """

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg_name]

    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    gate_pos = gate_pose[env_ids, :3]
    gate_quat = gate_pose[env_ids, 3:7]
    offset = torch.tensor([1.0, 0.0, 0.0], device=asset.device).expand(len(env_ids), 3)
    offset_world = math_utils.quat_apply(gate_quat, offset)
    pos_after_prev_gate = gate_pos + offset_world

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + pos_after_prev_gate + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    base_quat = root_states[:, 3:7] if heading_quat is None else heading_quat[env_ids]
    orientations = math_utils.quat_mul(base_quat, orientations_delta)

    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


# Estimated for the 1.745 kg competition aircraft: a 5-inch airframe carrying a
# Jetson Orin NX and its carrier. The URDF shipped 0.003 / 0.003 / 0.006, which
# came in with the upstream vendoring paired with ``mass 0.5`` and was never
# revisited when the mass tripled -- the inherited numbers imply a radius of
# gyration of 4.1 cm, small for a machine with motors 11 cm out.
#
# Modelling the airframe as four arm assemblies at 45 degrees plus a central
# box (battery + Orin + plates), across plausible arm masses and frame sizes:
#
#     light arms, tall central stack   0.0034  0.0053  0.0062
#     heavy arms, compact stack        0.0047  0.0060  0.0094
#     light arms, 7-inch frame         0.0050  0.0068  0.0093
#
# The roll estimate spans 1.46x, so these are not precise. They are, however,
# better centred than the inherited values, and pitch is the one that was most
# wrong: a racing quad's central stack is markedly longer front-to-back than it
# is wide, so Iyy should sit well above Ixx rather than equal to it.
#
# Not randomised. Published sensitivity studies put inertia low on the list --
# SimpleFlight measured a +30% inertia offset at 1.5x tracking error where a
# +30% mass offset crashed outright, and reports that randomising a roughly
# known inertia mostly adds learning difficulty. ``scale_range`` is here so a
# band can be switched on in one line if the A/B says otherwise.
BODY_INERTIA_DIAG = (0.0040, 0.0055, 0.0075)


def set_body_inertia(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    inertia_diag: tuple[float, float, float] = BODY_INERTIA_DIAG,
    asset_cfg_name: str = "robot",
    body_name: str = "body",
    scale_range: tuple[float, float] | None = None,
):
    """Write the body's diagonal inertia straight into PhysX.

    ``MassPropertiesCfg`` can override mass at spawn but carries no inertia
    field, and the drone loads from a binary USD whose authored
    ``diagonalInertia`` survives a mass override. Setting it here makes the
    number explicit and independent of what the asset happens to contain, so
    the plant is the same whoever rebuilt the USD and whenever.

    ``scale_range`` multiplies all three axes by one per-env factor when given,
    keeping the airframe's shape while varying how hard it is to spin.

    Run this at ``startup``. PhysX inertia is not a per-step quantity, and the
    tensor API wants CPU tensors, which is why this mirrors Isaac Lab's own
    ``randomize_rigid_body_mass`` rather than writing on the sim device.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg_name]
    view = asset.root_physx_view

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    env_ids = env_ids.cpu()

    body_ids = asset.find_bodies(body_name)[0]

    # (num_instances, num_bodies, 9): each body's 3x3 inertia, row-major.
    inertias = view.get_inertias().clone()
    ixx, iyy, izz = inertia_diag

    if scale_range is None:
        scale = torch.ones(len(env_ids), 1)
    else:
        lo, hi = scale_range
        scale = torch.empty(len(env_ids), 1).uniform_(lo, hi)

    diag = torch.tensor([[ixx, iyy, izz]], dtype=inertias.dtype) * scale
    for body_id in body_ids:
        # Flat positions of the 3x3 diagonal; off-diagonal terms stay zero for
        # a body whose principal axes are the body axes.
        for slot, axis in enumerate((0, 4, 8)):
            inertias[env_ids, body_id, axis] = diag[:, slot]

    view.set_inertias(inertias, env_ids)
