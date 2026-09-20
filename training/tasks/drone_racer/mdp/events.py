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

# The aircraft on the scale: all-up competition weight, **props included**.
# Must equal ``contract.plant.MASS_KG``; a test parses this literal to make
# sure it still does.
#
# Props therefore come out of this number, not on top of it -- the airframe
# link gets 1.745 minus the four props, and the five links sum to 1.745. The
# other reading gives a 1.765 kg aircraft and is the obvious way for someone
# to "fix" this later, so: confirmed with the person who weighed it.
AIRFRAME_MASS_KG = 1.745

# Mass of one propeller, and its inertia about its own centre.
#
# The USD defines five rigid bodies -- ``body`` plus four spinning ``propN``
# links -- and ``MassPropertiesCfg(mass=...)`` applies its value to *every one
# of them*. Spawning with ``mass=1.745`` therefore built a 8.725 kg aircraft
# carrying thrust sized for 1.745 kg: a thrust-to-weight of 0.2, which cannot
# leave the ground. Nothing in the stack noticed, because every part of it was
# individually correct.
#
# 5 g is a tri-blade 5-inch prop, weighed as a class rather than this one. The
# exact figure barely matters: the props are 1.1% of the airframe and the body
# mass is derived from the total, so an error here moves mass between links
# without changing what the aircraft weighs.
#
# Inertia is a thin planar body of that mass and 0.127 m span: I_spin = mL^2/12,
# and the two perpendicular axes are half that. Which flat axis PhysX calls the
# spin axis depends on how the joint was authored; at 7e-6 against the body's
# 4e-3 the distinction is below anything the policy could feel.
PROP_MASS_KG = 0.005
PROP_INERTIA_DIAG = (3.4e-6, 3.4e-6, 6.7e-6)


def set_body_mass(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    total_mass_kg: float = AIRFRAME_MASS_KG,
    prop_mass_kg: float = PROP_MASS_KG,
    asset_cfg_name: str = "robot",
    body_name: str = "body",
    prop_name_expr: str = "prop.*",
):
    """Distribute the airframe's mass across its links, in PhysX.

    ``MassPropertiesCfg`` cannot express this: it carries one number and hands
    it to every body it spawns. So the spawn config sets no mass at all and the
    split is made here, where the links can be told apart.

    The central body is given whatever is left after the props, so the total is
    ``total_mass_kg`` by construction rather than by three numbers happening to
    add up. Mass is the one plant parameter that must be *right* rather than
    randomised -- a +30% error is unrecoverable on real hardware even with
    domain randomisation -- so it is worth the total being arithmetically
    guaranteed instead of maintained by hand.

    Run at ``startup``, and before the inertia event: PhysX's tensor API keeps
    mass and inertia independent, but should a future version rescale inertia
    with mass, writing inertia second means the explicit value wins.

    Raises ``ValueError`` if the name patterns do not account for every rigid
    body, since a link nobody matched would silently keep its spawn mass and
    the total would not be the total.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg_name]
    view = asset.root_physx_view

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    env_ids = env_ids.cpu()

    body_ids = asset.find_bodies(body_name)[0]
    prop_ids = asset.find_bodies(prop_name_expr)[0]

    # (num_instances, num_bodies)
    masses = view.get_masses().clone()
    num_bodies = masses.shape[1]
    matched = set(body_ids) | set(prop_ids)
    if len(matched) != num_bodies:
        raise ValueError(
            f"mass split covers {sorted(matched)} of {num_bodies} bodies in "
            f"'{asset_cfg_name}'; every rigid body must be named by "
            f"body_name={body_name!r} or prop_name_expr={prop_name_expr!r}, "
            "or the airframe will not weigh what the contract says it does"
        )
    if len(body_ids) != 1:
        raise ValueError(f"body_name={body_name!r} matched {len(body_ids)} bodies, expected 1")

    central_mass = total_mass_kg - prop_mass_kg * len(prop_ids)
    if central_mass <= 0.0:
        raise ValueError(
            f"{len(prop_ids)} props at {prop_mass_kg} kg leave "
            f"{central_mass:.4f} kg for the airframe"
        )

    for prop_id in prop_ids:
        masses[env_ids, prop_id] = prop_mass_kg
    masses[env_ids, body_ids[0]] = central_mass

    view.set_masses(masses, env_ids)


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

    # Write all nine entries, not just the diagonal. The products of inertia
    # are what is left over from the USD's own tensor, and the authored asset
    # carries 4.7e-4 in them -- 12% of the roll moment we are setting, which is
    # a genuinely different airframe: it couples roll into pitch on every
    # input. A body whose principal axes are its body axes has none, and this
    # one is modelled as exactly that, so they belong at zero by statement
    # rather than by whatever the exporter happened to emit.
    tensor = torch.zeros(len(env_ids), 9, dtype=inertias.dtype)
    for slot, axis in enumerate((0, 4, 8)):
        tensor[:, axis] = diag[:, slot]

    for body_id in body_ids:
        inertias[env_ids, body_id] = tensor

    view.set_inertias(inertias, env_ids)
