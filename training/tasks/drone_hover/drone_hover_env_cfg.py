"""Station-keeping environment: hold position in front of a gate.

Subclasses the racing config rather than restating it. Scene, observation,
action, plant, events and the command term are inherited **unchanged** --
that is not laziness, it is the requirement. A policy trained here is only
useful as a racing seed if it saw the same vector, and only useful as a
deployment test if it exercised the same perception path. Anything restated
here is something that could drift.

What changes is the objective. Racing pays for passing through the opening and
for closing distance; this pays for being parked in front of it and still.
"""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from tasks.drone_racer.drone_racer_env_cfg import (
    DroneRacerEnvCfg,
    DroneRacerEnvCfg_PLAY,
)

from . import mdp

# Metres in front of the gate opening to hold. Close enough that the gate fills
# a useful part of a 72.8 degree frame, far enough to drift without contact.
STANDOFF_M = 2.0


@configclass
class HoverRewardsCfg:
    """Hold the point, be still, stay upright, keep the gate in frame.

    Weights are set so a perfectly parked episode earns roughly ten times what
    a crashed one loses, and so no single shaping term can outbid actually
    arriving. ``station_keep`` is the objective; everything else either rules
    out a degenerate way of satisfying it or keeps the perception honest.
    """

    # Broad pull toward the hold point, with reach. station_keep alone has
    # none -- a 0.75 m Gaussian is numerically zero past a few metres -- so
    # without this a policy spawned anywhere but on the point has no gradient
    # and the spawn has to be narrowed to compensate. Widening the reward
    # instead is what lets the spawn scatter across the gate's visible region,
    # which is how the aircraft will actually be handed over in a cage.
    approach = RewTerm(
        func=mdp.approach,
        weight=0.6,
        params={"command_name": "target", "std": 3.0, "standoff_m": STANDOFF_M},
    )
    # The objective. Gaussian on distance to the hold point.
    station_keep = RewTerm(
        func=mdp.station_keep,
        weight=2.0,
        params={"command_name": "target", "std": 0.75, "standoff_m": STANDOFF_M},
    )
    # Names the goal rather than shaping toward it: pays only when near *and*
    # slow, so "close but orbiting" cannot collect most of the reward.
    settled = RewTerm(
        func=mdp.settled,
        weight=3.0,
        params={
            "command_name": "target",
            "radius_m": 0.30,
            "speed_m_s": 0.30,
            "standoff_m": STANDOFF_M,
        },
    )
    # Without this, orbiting the hold point at speed scores as well as hovering.
    # Squared m/s, so 1 m/s costs 0.4 per step against station_keep's max 2.0.
    stillness = RewTerm(func=mdp.stillness, weight=-0.4)
    # Weak: a quad holding station genuinely does tilt to cancel drift. This
    # only rules out holding position while inverted.
    upright = RewTerm(func=mdp.upright, weight=0.3)
    # Rate damping, same term and weight the racing task uses.
    ang_vel_l2 = RewTerm(func=mdp.ang_vel_l2, weight=-0.0001)
    # The gate must stay framed, or the observation the policy is learning to
    # use goes blind and the cage test proves nothing about perception.
    gate_visible = RewTerm(
        func=mdp.gate_visible,
        weight=0.3,
        params={"command_name": "target", "std": 0.55},
    )
    lookat = RewTerm(
        func=mdp.lookat_next_gate,
        weight=0.2,
        params={"command_name": "target", "std": 0.5},
    )
    # A crash must dominate. At 2400 steps a flawless episode earns roughly
    # +13000, so -100 is a real but not paralysing penalty -- the racing task
    # learned the hard way that a termination cost 100x every dense term
    # collapses the policy onto "do nothing".
    terminating = RewTerm(func=mdp.is_terminated, weight=-100.0)


@configclass
class HoverTerminationsCfg:
    """End on a crash, on drifting away, or on a broken state.

    Deliberately missing the racing terminations: there is no gate to miss and
    no course to finish. ``flyaway`` is tightened well below the racing value --
    a station-keeping policy that has wandered 8 m has already failed, and
    letting the episode run teaches it that failure is survivable.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    collision = DoneTerm(
        func=mdp.illegal_contact,
        # 0.01 N, the same threshold the racing task uses -- a prop brushing a
        # gate is a crash, and a higher bar lets contact go unnoticed.
        params={"sensor_cfg": SceneEntityCfg("collision_sensor"), "threshold": 0.01},
    )
    flyaway = DoneTerm(
        func=mdp.flyaway,
        params={"distance": 8.0, "command_name": "target"},
    )
    nonfinite = DoneTerm(func=mdp.nonfinite_state)


@configclass
class DroneHoverEnvCfg(DroneRacerEnvCfg):
    """Racing environment, station-keeping objective."""

    rewards: HoverRewardsCfg = HoverRewardsCfg()
    terminations: HoverTerminationsCfg = HoverTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # Shorter than racing's 40 s. Holding station is a stationary problem:
        # an episode either settles in the first few seconds or does not, and
        # shorter episodes mean more resets per wall-clock hour, which is where
        # the plant randomisation actually gets sampled.
        self.episode_length_s = 15.0

        # Start on the hold point, not after the previous gate.
        #
        # Inheriting the racing spawn looked like the conservative choice --
        # same initial-state distribution, so the policy transfers -- and was
        # wrong in a way that only running it showed. Racing puts the aircraft
        # one metre past the gate it just passed, a full gate-spacing from the
        # gate it is now targeting. ``station_keep`` is a Gaussian with a 0.75 m
        # width, which at that range returns zero: no gradient, nothing to
        # learn from. ``flyaway`` at 8 m then ended those episodes on step one,
        # 40 of 64 of them, before the policy had acted at all.
        #
        # Both the reward and the termination radius are right for a station
        # keeper. The spawn was the thing that did not belong, so it is the
        # thing that changes. Gate choice stays random and the jitter stays on:
        # the aircraft still sees every gate, and still has to correct a small
        # offset rather than starting perfect.
        self.commands.target.start_at_run_in = True
        self.commands.target.start_run_in_m = STANDOFF_M
        # Scatter the spawn across the region where the gate is in frame,
        # rather than dropping the aircraft on the point it is meant to hold.
        #
        # Spawning on the hold point was the first fix for station_keep having
        # no reach, and it worked while producing a policy that only knows how
        # to hold from somewhere it is already holding. The cage hands the
        # aircraft over from wherever it happens to be.
        #
        # Applied before the aim heading is computed, so the gate stays in
        # frame from every sampled position -- see spawn_scatter_m. The box is
        # asymmetric on purpose: wide across and along the approach, tighter
        # vertically, because the gate opening is 1.35 m off the floor and a
        # symmetric box would put a fifth of the spawns underground.
        self.commands.target.spawn_scatter_m = (2.5, 2.5, 0.9)


@configclass
class DroneHoverEnvCfg_PLAY(DroneRacerEnvCfg_PLAY):
    """Single-aircraft view for watching it hold, with the camera enabled."""

    rewards: HoverRewardsCfg = HoverRewardsCfg()
    terminations: HoverTerminationsCfg = HoverTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 60.0
