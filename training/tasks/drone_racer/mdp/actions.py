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
    ACTION_DELAY_STEPS_RANGE,
    DEFAULT_MASS_KG,
    PLANT_HOVER_RANGE,
    PLANT_TWR_RANGE,
    RATE_LIMIT,
    RATE_TAU_S_RANGE,
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

        # --- latency -------------------------------------------------------
        # Raw actions are held in a ring buffer and read back this env's own
        # delay later. A ring rather than a list so the cost does not grow with
        # the delay, and per-env rather than global so the policy cannot learn
        # one specific lag and rely on it.
        lo, hi = self.cfg.action_delay_steps_range or (0, 0)
        self._max_action_delay = int(hi)
        self._action_queue = torch.zeros(
            self.num_envs, self._max_action_delay + 1, 4, device=self.device
        )
        self._queue_head = 0
        self._action_delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._env_rows = torch.arange(self.num_envs, device=self.device)
        # Filtered rate setpoint: the flight controller's 15 Hz smoothing.
        self._rate_filtered = torch.zeros(self.num_envs, 3, device=self.device)
        # Ones, not zeros: reset overwrites this, but it must never be a zero
        # if anything steps before the first reset.
        self._rate_tau = torch.ones(self.num_envs, 1, device=self.device)
        # Per-step blend factor, precomputed at reset because tau is constant
        # within an episode. The exact discretisation of a first-order lag is
        # ``1 - exp(-dt/tau)``, not ``dt/tau``: physics runs at 120 Hz and the
        # measured FC smoothing is ~10.6 ms, so dt/tau is around 0.8 and Euler
        # is 44% wrong there. At the fast end of the band Euler clamps to 1.0
        # and models no smoothing at all -- the opposite of what is intended.
        self._rate_alpha = torch.ones(self.num_envs, 1, device=self.device)
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

    def _delayed(self, actions: torch.Tensor) -> torch.Tensor:
        """This env's command from ``action_delay`` policy steps ago.

        The newest action goes in at the head; each env reads back from its own
        offset behind it. A delay of zero reads the head and is exactly the old
        instantaneous behaviour, so this costs nothing when latency is off.
        """
        if self._max_action_delay == 0:
            return actions
        self._queue_head = (self._queue_head + 1) % self._action_queue.shape[1]
        self._action_queue[:, self._queue_head] = actions
        read = (self._queue_head - self._action_delay) % self._action_queue.shape[1]
        return self._action_queue[self._env_rows, read]

    def process_actions(self, actions: torch.Tensor):
        # What the policy just asked for is not what the aircraft is doing yet.
        # ``_raw_actions`` deliberately holds the *delayed* command, because it
        # is what the rest of the term and the reward read as "the action".
        actions = self._delayed(actions)
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

        # The flight controller does not track a rate setpoint that changes
        # faster than its own smoothing filter -- ours measures
        # rc_smoothing_setpoint_cutoff = 15 Hz. Modelled as a first-order lag at
        # the physics rate, which is where the real inner loop runs too. This is
        # a bandwidth limit, not just a delay: a step command arrives rounded
        # off rather than merely late.
        cmd_rates = self._cmd_rates_ned
        if self.cfg.rate_tau_s_range is not None:
            self._rate_filtered += self._rate_alpha * (cmd_rates - self._rate_filtered)
            cmd_rates = self._rate_filtered

        moment = rate_moments(
            cmd_rates,
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
        log(self._env, ["act_delay"], self._action_delay.unsqueeze(-1).to(self._thrust_scale.dtype))
        log(self._env, ["rate_tau"], self._rate_tau)

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

        # --- latency -------------------------------------------------------
        # Clear the whole queue for these envs, not just the head. Otherwise a
        # fresh episode spends its first few steps executing commands the
        # *previous* episode issued -- which, with 4096 envs resetting at
        # staggered times, is a steady trickle of nonsense the policy cannot
        # explain and will try to learn around.
        self._action_queue[env_ids] = 0.0
        self._rate_filtered[env_ids] = 0.0
        if self.cfg.action_delay_steps_range is not None:
            lo, hi = self.cfg.action_delay_steps_range
            self._action_delay[env_ids] = torch.randint(
                int(lo), int(hi) + 1, (n,), device=self.device
            )
        if self.cfg.rate_tau_s_range is not None:
            lo, hi = self.cfg.rate_tau_s_range
            self._rate_tau[env_ids, 0] = torch.empty(
                n, device=self.device, dtype=dtype
            ).uniform_(lo, hi)
        else:
            self._rate_tau[env_ids, 0] = 1.0
        self._rate_alpha[env_ids, 0] = 1.0 - torch.exp(
            -float(self._env.physics_dt) / self._rate_tau[env_ids, 0]
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
    action_delay_steps_range: tuple[int, int] | None = ACTION_DELAY_STEPS_RANGE
    """Policy steps between the policy deciding and the aircraft acting, drawn
    uniform per episode. One step is 1/60 s.

    ``None`` applies commands instantly, which is what every checkpoint through
    ``pq_speed_best`` trained under and is the single largest thing the plant
    was getting wrong: 50 ms of delay took our own stress runs from 2.6 to 55
    crashes per 100 gates."""
    rate_tau_s_range: tuple[float, float] | None = RATE_TAU_S_RANGE
    """First-order lag on the rate setpoint, seconds, drawn uniform per episode.
    Models ``rc_smoothing_setpoint_cutoff`` (15 Hz measured on our FC, about
    10.6 ms) plus motor lag. ``None`` tracks the setpoint perfectly."""
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
