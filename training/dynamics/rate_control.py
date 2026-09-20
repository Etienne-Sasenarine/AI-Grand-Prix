# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

"""AI_GP / VADR-TS-001 action contract: collective thrust + NED body rates.

Policy actions are in ``[-1, 1]`` (Isaac PPO). They decode to the same four
channels ``PolicyPlanner`` sends on ``SET_ATTITUDE_TARGET``:

  * thrust — dimensionless stick in ``[min, max]``, hover at 0
  * roll / pitch / yaw rates — NED rad/s, ``±rate_limit`` at the rails

``flu_to_ned`` is an involution (same map both ways): Isaac body is FLU,
MAVLink body is NED.
"""

from __future__ import annotations

import torch

# Match AI_GP ``config.MIN_THRUST`` / ``HOVER_THRUST`` / ``MAX_THRUST`` and
# ``race_obs.ACTION_RANGES`` rate envelope.
THRUST_MIN = 0.05
# Centre of the measured hover band below. Was 0.255, a pad estimate; John's
# stick-hover runs on the radio put the real thing at 0.21..0.24, so zero action
# was commanding ~13% more than weight. Must stay equal to AI_GP
# ``config.HOVER_THRUST`` -- it is the one number both ends assume.
THRUST_HOVER = 0.225
THRUST_MAX = 0.90
# What the bird actually hovers at. John's stick-hover runs span 0.21..0.24 --
# unit to unit, prop wear, the day's mass -- but those were flown on healthy
# packs and that is not the whole story. Thrust goes as V^2 at fixed duty, so a
# 6S pack falling 4.2 -> 3.5 V/cell loses ~31% of its thrust and the stick that
# holds a hover climbs to ~0.28. The upper end is the end of a pack, not a
# different aircraft. The deployed client cannot know where in that range it is
# -- ``config.HOVER_THRUST`` is one baked constant on the wire -- so training
# randomises the plant across the whole span. See ``plant_thrust_scale``.
PLANT_HOVER_RANGE = (0.21, 0.29)
# Fraction of full-stick thrust carried by the s^2 term; see ``stick_to_newtons``.
# 0.68 is a pooled fit to four 6S / 5-inch thrust-stand datasets (MiniQuadTestBench
# x2, DroneHiTech, Tyto Robotics), which agree on thrust ~ stick^1.6 with R^2 > 0.99.
PLANT_QUAD_SHARE = 0.68
# Thrust-to-weight at the THRUST_MAX rail, which is how the plant is actually
# randomised. Shape and authority are the same parameter here -- pin the curve
# at hover and its steepness *is* the top-end thrust -- so we dial the end a
# human can check against the build instead of the abstract one.
#
# The ceiling is what the measured 5-inch thrust-stand shape implies. We have
# never had this airframe on a stand and it weighs 1.745 kg, so rather than pick
# a point we train across the whole span up to it.
#
# The floor is *derived*, not chosen. A curve cannot be weaker than the straight
# line through its own hover point without going concave, and a concave thrust
# curve is not a thing a motor does. The lowest TWR every plant in
# PLANT_HOVER_RANGE can reach with a physical (convex) curve is therefore the
# straight-line value at the *lowest* hover point -- an aircraft that hovers at
# 0.21 already pulls 0.90/0.21 at the rail with no curvature at all.
PLANT_TWR_MAX = 8.0
PLANT_TWR_RANGE = (THRUST_MAX / PLANT_HOVER_RANGE[0], PLANT_TWR_MAX)
RATE_LIMIT = 3.2
G = 9.80665

# --- latency ----------------------------------------------------------------
#
# The simulator used to apply a command the instant the policy produced it. The
# aircraft does not. Between the policy deciding and the airframe responding
# there is an MSP frame, the flight controller's own loop, and the ESCs.
#
# This is not a small correction. Our own sweep took crashes from 2.6 to 55 per
# 100 gates at 50 ms, and Sun et al. measured the same cliff independently on
# different hardware -- 0% crashes at 10 ms, 2.7% at 30 ms, 68% at 50 ms.
# Agilicious measured Betaflight command-to-actuation at 40 ms with a load cell.
#
# Expressed in policy steps, which are 1/60 s. One step is 16.7 ms.
ACTION_DELAY_STEPS_RANGE = (0, 2)

# First-order lag on the rate setpoint, seconds.
#
# Measured, not guessed: this flight controller runs
# ``rc_smoothing_setpoint_cutoff = 15 Hz, fixed``. That is a bandwidth limit as
# much as a delay -- the FC simply cannot follow a rate command that changes
# faster than that -- and a first-order fit gives tau = 1/(2*pi*15) = 10.6 ms.
#
# Narrow band rather than a point: the cutoff is a known number, but where the
# rest of the FC's filtering lands on top of it is not, and a real airframe's
# motors add their own lag on the order of 30-40 ms for a 5-inch quad.
RATE_TAU_S_RANGE = (0.008, 0.045)
DEFAULT_MASS_KG = 1.745


def flu_to_ned(v_flu: torch.Tensor) -> torch.Tensor:
    """Isaac body FLU (x fwd, y left, z up) → MAVLink NED (x fwd, y right, z down)."""
    return torch.stack([v_flu[..., 0], -v_flu[..., 1], -v_flu[..., 2]], dim=-1)


ned_to_flu = flu_to_ned


def decode_aigp_action(
    raw: torch.Tensor,
    *,
    thrust_min: float = THRUST_MIN,
    thrust_hover: float = THRUST_HOVER,
    thrust_max: float = THRUST_MAX,
    rate_limit: float = RATE_LIMIT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map policy ``[-1, 1]`` to (thrust stick, NED rates).

    Zero action is hover + zero rates. Positive thrust channel climbs toward
    ``thrust_max``; negative sinks toward ``thrust_min``.
    """
    a = raw.clamp(-1.0, 1.0)
    x = a[..., 0]
    up = thrust_hover + x * (thrust_max - thrust_hover)
    down = thrust_hover + x * (thrust_hover - thrust_min)
    thrust = torch.where(x >= 0.0, up, down).clamp(thrust_min, thrust_max)
    rates_ned = a[..., 1:4] * rate_limit
    return thrust, rates_ned


def stick_to_newtons(
    thrust_stick: torch.Tensor,
    *,
    mass_kg: float = DEFAULT_MASS_KG,
    hover_stick: float | torch.Tensor = THRUST_HOVER,
    g: float = G,
    quad_share: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    """AI_GP stick → body collective in Newtons. Hover stick holds ``m·g``.

    ``quad_share`` is how much of the full-stick thrust the ``s^2`` term carries:

        T(s) ∝ quad_share · s² + (1 − quad_share) · s

    ``0.0`` is the straight line the *client* assumes and every checkpoint
    through ``pq_speed_best`` trained against. It is right at the hover point by
    construction and wrong everywhere else, because the real chain is
    throttle → duty → RPM → thrust: thrust goes as RPM² (propeller), but
    duty → RPM is markedly sublinear (ESC and motor), and the two compose to
    roughly ``stick^1.6``. Four 6S thrust-stand datasets put a quarter stick at
    ~12% of max thrust, half at ~35%, three-quarters at ~68% -- not 25/50/75.

    Pinned at the same hover point, the straight line reads ~22% high at
    ``s = 0.10`` and ~49% *low* at ``THRUST_MAX``. That error has a direction
    that matters: the real aircraft answers a punch-out with far more thrust
    than the linear plant ever gave, so a policy trained on ``quad_share=0``
    overshoots on hardware rather than falling short.

    A parabola is not the true shape. The measured local exponent runs
    1.53 → 1.62 → 1.33 across the stick -- the real curve steepens through the
    middle, then flattens near full throttle as the motor nears its no-load RPM
    -- and a parabola through the origin has no inflection to match that. It
    reads about six points low at three-quarter stick. We accept that: it is
    one interpretable parameter, it pins hover exactly, it stays monotonic, and
    it cuts the rail error from ~49% to under 10%. A better-fitting form would
    be false precision until this airframe has been on a thrust stand, and the
    ``PLANT_TWR_RANGE`` we randomise over is wider than the residual.

    ``hover_stick`` and ``quad_share`` may be per-env tensors.
    """
    num = quad_share * thrust_stick**2 + (1.0 - quad_share) * thrust_stick
    den = quad_share * hover_stick**2 + (1.0 - quad_share) * hover_stick
    return (num / den) * (mass_kg * g)


def quad_share_for_twr(
    twr: torch.Tensor | float,
    *,
    stick: float = THRUST_MAX,
    hover_stick: float | torch.Tensor = THRUST_HOVER,
) -> torch.Tensor | float:
    """Curve shape that puts ``twr`` times weight at ``stick``.

    ``stick_to_newtons`` is pinned to 1 g at ``hover_stick``, so its steepness
    and its top-end authority are one and the same number. Thrust-to-weight at
    the rail is the readable end of that: it can be checked against the motors
    and props on the bench, where ``quad_share`` cannot.

    Writing ``T(s) ∝ s + q·s·(s − 1)`` and asking for ``T(stick)/T(hover) = twr``:

        q = (twr·h − s) / (s·(s − 1) − twr·h·(h − 1))

    ``twr = stick / hover_stick`` returns exactly ``q = 0``, the straight line
    the client assumes. This is an exact inversion and is **not** clamped: ask
    for less thrust than the straight line already gives at that hover point and
    you get a negative ``q``, meaning a concave curve, which no motor produces.
    Callers that sample ``twr`` must keep it at or above ``stick / hover_stick``
    for every hover point they draw -- which is how ``PLANT_TWR_RANGE`` gets its
    floor -- or clamp the result into ``[0, 1]`` themselves.
    """
    s, h = stick, hover_stick
    return (twr * h - s) / (s * (s - 1.0) - twr * h * (h - 1.0))


def plant_thrust_scale(
    plant_hover: torch.Tensor,
    *,
    nominal_hover: float = THRUST_HOVER,
) -> torch.Tensor:
    """Thrust the aircraft *gives* over thrust the client *assumed*.

    ``decode_aigp_action`` and ``stick_to_newtons`` both use the one hover
    constant the client ships, so they cancel: a zero thrust action is exactly
    ``m·g`` whatever that constant is. Randomising it therefore changes nothing
    at hover -- only the thrust-to-weight at the ±1 rails.

    The real gap is that the constant is a *guess about this aircraft*. A bird
    whose true hover stick is ``plant_hover`` answers the client's hover stick
    with ``nominal_hover / plant_hover`` times the Newtons the client expected:
    hand 0.255 to a bird that hovers at 0.21 and it climbs at 0.21 g. That
    ratio is what has to vary across envs, and it has to be applied to the
    force only -- never to ``processed_actions``, which feeds the observation
    and must keep saying what the client believes it commanded.
    """
    return nominal_hover / plant_hover


def rate_moments(
    omega_des_ned: torch.Tensor,
    omega_flu: torch.Tensor,
    kp: torch.Tensor,
    kd: torch.Tensor,
    moment_limit: torch.Tensor,
) -> torch.Tensor:
    """PD body-rate loop. Commands are NED; gyro and moments are Isaac FLU."""
    omega_des_flu = ned_to_flu(omega_des_ned)
    err = omega_des_flu - omega_flu
    moment = kp * err - kd * omega_flu
    return moment.clamp(-moment_limit, moment_limit)


# ---------------------------------------------------------------------------
# Inner rate loop: gains derived from the inertia, not pinned beside it
# ---------------------------------------------------------------------------
#
# ``rate_moments`` is a first-order loop -- moment = kp*err - kd*omega -- so
# closing it around a rigid body gives
#
#     I * omega_dot + (kp + kd) * omega = kp * omega_des
#
# whose time constant is  tau = I / (kp + kd).  **The response scales with
# inertia.**  Gains that are right for one airframe are wrong for a heavier-to-
# spin one by exactly the inertia ratio.
#
# That is not hypothetical. The gains below were inherited from the 0.5 kg
# upstream aircraft, whose inertia was (0.003, 0.003, 0.006), and stayed
# untouched when this airframe's inertia was corrected to (0.004, 0.0055,
# 0.0075). The pairing broke silently:
#
#     axis    tau before    tau after    change
#     roll      36.1 ms      48.2 ms      +33%
#     pitch     36.1 ms      66.3 ms      +84%
#     yaw       72.3 ms      90.4 ms      +25%
#
# A policy commanding body rates through an 84%-slower pitch axis reaches gates
# and cannot thread them, which is what the first corrected-plant run did.
#
# So the gains are *derived* here rather than written down next to the inertia
# and trusted to stay in step. Change the inertia and the gains follow.
INERTIA_DIAG = (0.0040, 0.0055, 0.0075)

# Target closed-loop time constant per axis.
#
# These restore the response the gains were originally designed for, which is
# the defensible choice: it removes a regression without inventing a number we
# have not measured. Yaw is slower than roll and pitch on purpose -- that
# asymmetry is real on a quad, where yaw authority comes from prop drag rather
# than thrust differential.
#
# Two things bound this from below and are worth stating:
#
#   * Physics runs at 120 Hz (dt = 8.33 ms). A first-order loop needs tau
#     comfortably above dt or the discretisation stops representing it; below
#     roughly 2-3 dt it is numerically marginal. That puts a floor near 25 ms
#     on anything this simulator can honestly claim.
#   * A real Betaflight rate loop on a 5-inch quad settles far faster than
#     this -- single-digit milliseconds. We cannot model that at 120 Hz, so the
#     simulated aircraft is slower to respond than the real one, in a direction
#     that makes the policy *more* conservative rather than less. Closing that
#     gap properly needs a bench measurement: step a rate command with props
#     off, log the gyro, fit the time constant. Until then this is a documented
#     approximation, not a claim.
RATE_TAU_TARGET_S = (0.036, 0.036, 0.072)

# kd as a fraction of kp, preserved from the original tuning (0.003 / 0.08).
# kd damps measured rate rather than error, so it trades overshoot against
# steady-state tracking; keeping the ratio keeps that character while the
# magnitudes follow the inertia.
RATE_KD_KP_RATIO = 0.0375

# Peak angular acceleration the original tuning allowed, rad/s^2, per axis:
# the old moment limits (0.30, 0.30, 0.20) over the old inertia. Holding
# *acceleration* fixed rather than torque is what keeps the aircraft feeling
# the same when its inertia changes.
RATE_ALPHA_MAX = (100.0, 100.0, 100.0 / 3.0)


def rate_gains_for_inertia(
    inertia_diag: tuple[float, float, float] = INERTIA_DIAG,
    tau_s: tuple[float, float, float] = RATE_TAU_TARGET_S,
    kd_kp_ratio: float = RATE_KD_KP_RATIO,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Per-axis ``(kp, kd)`` giving each axis its target time constant.

    From ``tau = I / (kp + kd)`` and ``kd = ratio * kp``:

        kp = I / (tau * (1 + ratio))
        kd = ratio * kp
    """
    kp, kd = [], []
    for inertia, tau in zip(inertia_diag, tau_s):
        total = inertia / tau
        k_p = total / (1.0 + kd_kp_ratio)
        kp.append(k_p)
        kd.append(kd_kp_ratio * k_p)
    return tuple(kp), tuple(kd)


def moment_limits_for_inertia(
    inertia_diag: tuple[float, float, float] = INERTIA_DIAG,
    alpha_max: tuple[float, float, float] = RATE_ALPHA_MAX,
) -> tuple[float, float, float]:
    """Torque ceiling giving each axis ``alpha_max`` angular acceleration.

    Scaling the ceiling with inertia is the point: a fixed torque limit on a
    heavier-to-spin airframe is a quietly weaker aircraft.
    """
    return tuple(i * a for i, a in zip(inertia_diag, alpha_max))


RATE_KP, RATE_KD = rate_gains_for_inertia()
MOMENT_LIMIT = moment_limits_for_inertia()
