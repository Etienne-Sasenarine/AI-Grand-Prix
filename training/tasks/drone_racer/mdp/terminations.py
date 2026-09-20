# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gate_missed(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Terminate when the drone crosses the target gate plane outside the opening.

    Crossing the plane without being inside the 1.5 m cube is already flagged as
    ``gate_missed`` (and pays the ``gate_passed`` miss penalty). Without this
    term the episode keeps going with that same gate still as the target, so a
    fast high-then-low miss just flies on until collision, flyaway, or timeout.
    """
    return env.command_manager.get_term(command_name).gate_missed


def course_finished(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Terminate once the final gate of a point-to-point course has been passed.

    Register this with ``time_out=True``. Finishing is a success, and the
    ``is_terminated`` reward penalises only non-timeout terminations, so routing
    it through the truncation buffer keeps the crash penalty off a clean run.
    """
    return env.command_manager.get_term(command_name).course_completed


def flyaway(
    env: ManagerBasedRLEnv,
    distance: float,
    command_name: str | None = None,
    target_pos: list | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the asset's is too far away from the target position."""

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    if target_pos is None:
        target_pos = env.command_manager.get_term(command_name).command[:, :3]
        target_pos_tensor = target_pos[:, :3]
    else:
        target_pos_tensor = (
            torch.tensor(target_pos, dtype=torch.float32, device=asset.device).repeat(env.num_envs, 1)
            + env.scene.env_origins
        )

    # Compute distance
    distance_tensor = torch.linalg.norm(asset.data.root_pos_w - target_pos_tensor, dim=1)
    return distance_tensor > distance


def nonfinite_state(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reset envs whose pose left the reals — one NaN would poison PPO."""
    asset: RigidObject = env.scene[asset_cfg.name]
    pos_bad = ~torch.isfinite(asset.data.root_pos_w).all(dim=-1)
    quat_bad = ~torch.isfinite(asset.data.root_quat_w).all(dim=-1)
    return pos_bad | quat_bad
