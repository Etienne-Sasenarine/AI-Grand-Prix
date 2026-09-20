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
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from dynamics.rate_control import (
    DEFAULT_MASS_KG,
    PLANT_HOVER_RANGE,
    PLANT_TWR_RANGE,
    RATE_LIMIT,
    THRUST_HOVER,
    THRUST_MAX,
    THRUST_MIN,
    decode_aigp_action,
    plant_thrust_scale,
    quad_share_for_twr,
    rate_moments,
    stick_to_newtons,
)
from utils.logger import log

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ControlAction(ActionTerm):
    """AI_GP ``SET_ATTITUDE_TARGET`` plant: collective thrust + NED body rates.

    Policy action is ``[thrust, roll_rate, pitch_rate, yaw_rate]`` in ``[-1, 1]``.
    ``process_actions`` decodes that to the same physical units ``PolicyPlanner``
    emits. A 120 Hz PD rate loop in ``apply_actions`` turns the rate command
    into a body wrench; collective is applied along Isaac body +Z (FLU up).

    ``processed_actions[:, 0]`` is thrust in Newtons so observation / reward
    terms that read the last wrench stay valid.

    The thrust channel is computed **twice**, and the split is the point:

    * ``_cmd_thrust_n`` / ``processed_actions[:, 0]`` -- the straight line the
      *client* assumes, on the client's own ``hover_thrust``. This is what the
      commanded-velocity observation is built from, because that is what the
      real client computes from the stick it just sent. It must not know
      anything about this env's plant.
    * ``_applied_thrust_n`` -- what the *aircraft* actually produces: this
      episode's hover point and thrust-curve shape, drawn at reset from
      ``cfg.plant_hover_range`` and ``cfg.plant_twr_range``.

    The deployed client has one baked ``HOVER_THRUST``, one assumed curve, and
    no way to know which aircraft, which pack, or which set of props it is
    flying. A policy trained against a single exact plant learns a trim and a
    throttle feel it will not have. Feeding the plant numbers into the
    observation instead of the client's numbers would hand the policy its true
    thrust authority for free -- privileged state that disappears the moment it
    flies.
    """

    cfg: ControlActionCfg

    def __init__(self, cfg: ControlActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        self.cfg = cfg

        self._robot: Articulation = env.scene[self.cfg.asset_name]
        self._body_id = self._robot.find_bodies("body")[0]

        self._elapsed_time = torch.zeros(self.num_envs, 1, device=self.device)
        self._raw_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._cmd_rates_ned = torch.zeros(self.num_envs, 3, device=self.device)
        self._cmd_thrust_n = torch.zeros(self.num_envs, device=self.device)
        self._cmd_stick = torch.zeros(self.num_envs, device=self.device)
        self._applied_thrust_n = torch.zeros(self.num_envs, device=self.device)
        # This episode's plant, drawn at reset. Defaults are the client's own
        # assumption, i.e. no randomisation until reset says otherwise.
        self._plant_hover = torch.full(
            (self.num_envs,), float(self.cfg.hover_thrust), device=self.device
        )
        self._plant_quad_share = torch.zeros(self.num_envs, device=self.device)
        self._plant_twr = torch.full(
            (self.num_envs,),
            float(self.cfg.max_thrust) / float(self.cfg.hover_thrust),
            device=self.device,
        )
        # Thrust bias at hover: what this env gives for the client's hover stick.
        self._thrust_scale = torch.ones(self.num_envs, device=self.device)

        dtype = self._raw_actions.dtype
        self._rate_kp = torch.tensor(self.cfg.rate_kp, device=self.device, dtype=dtype)
        self._rate_kd = torch.tensor(self.cfg.rate_kd, device=self.device, dtype=dtype)
        self._moment_limit = torch.tensor(self.cfg.moment_limit, device=self.device, dtype=dtype)

    @property
    def action_dim(self) -> int:
        return 4

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def has_debug_vis_implementation(self) -> bool:
        return False

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        thrust_stick, rates_ned = decode_aigp_action(
            self._raw_actions,
            thrust_min=self.cfg.min_thrust,
            thrust_hover=self.cfg.hover_thrust,
            thrust_max=self.cfg.max_thrust,
            rate_limit=self.cfg.rate_limit,
        )
        # Nominal: the linear map the client believes, on the client's own hover
        # constant. This is what the commanded-velocity observation is built
        # from, so it must not know anything about this env's plant.
        thrust_n = stick_to_newtons(
            thrust_stick,
            mass_kg=self.cfg.mass_kg,
            hover_stick=self.cfg.hover_thrust,
        )
        # Plant: this aircraft's real curve, on this episode's hover point.
        self._applied_thrust_n[:] = stick_to_newtons(
            thrust_stick,
            mass_kg=self.cfg.mass_kg,
            hover_stick=self._plant_hover,
            quad_share=self._plant_quad_share,
        )
        self._cmd_stick[:] = thrust_stick
        self._cmd_thrust_n[:] = thrust_n
        self._cmd_rates_ned[:] = rates_ned
        self._processed_actions[:, 0] = thrust_n
        self._processed_actions[:, 1:] = rates_ned

        log(self._env, ["a_thr", "a_p", "a_q", "a_r"], self._raw_actions)
        log(self._env, ["thr_n", "p_cmd", "q_cmd", "r_cmd"], self._processed_actions)

    def apply_actions(self):
        omega_flu = self._robot.data.root_ang_vel_b
        moment = rate_moments(
            self._cmd_rates_ned,
            omega_flu,
            self._rate_kp,
            self._rate_kd,
            self._moment_limit,
        )
        self._thrust[:, 0, 2] = self._applied_thrust_n
        self._moment[:, 0, :] = moment
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

        self._elapsed_time += self._env.physics_dt
        log(self._env, ["time"], self._elapsed_time)
        log(self._env, ["thr_scale"], self._thrust_scale.unsqueeze(-1))
        log(self._env, ["thr_n_applied"], self._applied_thrust_n.unsqueeze(-1))
        log(self._env, ["plant_twr"], self._plant_twr.unsqueeze(-1))

    def reset(self, env_ids):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._cmd_rates_ned[env_ids] = 0.0
        self._cmd_thrust_n[env_ids] = 0.0
        self._cmd_stick[env_ids] = 0.0
        self._applied_thrust_n[env_ids] = 0.0
        self._elapsed_time[env_ids] = 0.0

        n = len(env_ids)
        dtype = self._plant_hover.dtype
        if self.cfg.plant_hover_range is not None:
            lo, hi = self.cfg.plant_hover_range
            self._plant_hover[env_ids] = torch.empty(
                n, device=self.device, dtype=dtype
            ).uniform_(lo, hi)
        else:
            self._plant_hover[env_ids] = float(self.cfg.hover_thrust)
        if self.cfg.plant_twr_range is not None:
            lo, hi = self.cfg.plant_twr_range
            twr = torch.empty(n, device=self.device, dtype=dtype).uniform_(lo, hi)
            self._plant_twr[env_ids] = twr
            # Shape follows from the authority, given where *this* env hovers --
            # a drained pack needs a different curve to reach the same rail.
            # The clamp keeps the curve physical (convex, at most pure s^2) if
            # the configured range ever undershoots a hover point; with the
            # derived PLANT_TWR_RANGE floor it never binds.
            self._plant_quad_share[env_ids] = quad_share_for_twr(
                twr, stick=self.cfg.max_thrust, hover_stick=self._plant_hover[env_ids]
            ).clamp(0.0, 1.0)
        else:
            self._plant_twr[env_ids] = float(self.cfg.max_thrust) / float(self.cfg.hover_thrust)
            self._plant_quad_share[env_ids] = 0.0
        # Diagnostic only: thrust bias at the client's hover stick. The curve is
        # pinned to 1 g at ``_plant_hover``, so this is where the trim lands.
        self._thrust_scale[env_ids] = plant_thrust_scale(
            self._plant_hover[env_ids], nominal_hover=self.cfg.hover_thrust
        )

        self._robot.reset(env_ids)
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


@configclass
class ControlActionCfg(ActionTermCfg):
    """See :class:`ControlAction`."""

    class_type: type[ActionTerm] = ControlAction
    asset_name: str = "robot"
    mass_kg: float = DEFAULT_MASS_KG
    """5\" airframe mass. Hover stick maps to ``mass_kg · g`` Newtons."""
    hover_thrust: float = THRUST_HOVER
    """AI_GP hover stick (``config.HOVER_THRUST``). Action 0 on the thrust channel.

    This is the client's *assumption*, not the aircraft. It is one number on the
    wire, identical every flight; it must stay fixed across envs or the policy
    is learning against a contract it cannot be deployed under.
    """
    plant_hover_range: tuple[float, float] | None = PLANT_HOVER_RANGE
    """Episode's true hover stick, drawn uniform. ``None`` pins the plant to
    ``hover_thrust`` exactly, which is the pre-randomisation behaviour and what
    every checkpoint through ``pq_speed_best`` trained under."""
    plant_twr_range: tuple[float, float] | None = PLANT_TWR_RANGE
    """Episode's thrust-to-weight at the ``max_thrust`` rail, drawn uniform and
    converted to a curve shape by ``quad_share_for_twr``.

    ``None`` gives every env the client's straight line -- what the keeper
    trained under, and equivalent to pinning this to 4.0."""
    min_thrust: float = THRUST_MIN
    max_thrust: float = THRUST_MAX
    rate_limit: float = RATE_LIMIT
    """NED rate at action ±1, matching ``race_obs.ACTION_RANGES`` (±3.2 rad/s)."""
    rate_kp: tuple[float, float, float] = (0.08, 0.08, 0.08)
    """PD P gains on FLU rate error, N·m / (rad/s). Yaw matches roll/pitch."""
    rate_kd: tuple[float, float, float] = (0.003, 0.003, 0.003)
    """PD D gains on measured FLU rate, N·m / (rad/s)."""
    moment_limit: tuple[float, float, float] = (0.30, 0.30, 0.20)
    """Saturate body moments (N·m). Yaw is usable, not a leftover motor-mix leftover."""
    # Kept so old Hydra overrides (`use_motor_model=False`) still parse.
    use_motor_model: bool = False
