"""The aircraft, and what the policy's four numbers mean physically.

Split into two halves that are easy to confuse and must not be:

**Contract** -- what the client assumes and sends on the wire. One baked hover
constant, one rate limit, one straight-line thrust map. These are part of the
observation contract because the commanded-velocity channels are derived from
them, so changing one changes what the vector means. They are hashed.

**Plant** -- what the aircraft actually is. Measured where possible, estimated
where not, and *randomised* where it genuinely varies flight to flight. None of
it is hashed: the policy is supposed to see a spread.

The distinction is the whole trick. The client cannot know which aircraft,
which pack or which set of props it is flying -- it has one constant for all of
them. So the simulator holds the constant fixed and moves the aircraft
underneath it, which is the situation the policy will actually face.
"""

from __future__ import annotations

G = 9.80665

# ---------------------------------------------------------------------------
# Contract: what the client assumes. Hashed.
# ---------------------------------------------------------------------------

ACTION_NAMES: tuple[str, ...] = ("thrust", "roll_rate", "pitch_rate", "yaw_rate")

# Dimensionless throttle stick. Zero policy action is THRUST_HOVER.
THRUST_MIN = 0.05
# Was 0.255, a pad estimate. John's stick-hover runs on the radio put the real
# aircraft at 0.21-0.24, so zero action had been commanding about 13% over
# weight. Must equal AI_GP ``config.HOVER_THRUST`` -- it is the one number both
# ends assume, and the wire carries no way to negotiate it.
THRUST_HOVER = 0.225
THRUST_MAX = 0.90

# NED body rates at action +-1, rad/s. Matches ``race_obs.ACTION_RANGES``.
RATE_LIMIT = 3.2

ACTION_RANGES: dict[str, tuple[float, float]] = {
    "thrust": (THRUST_MIN, THRUST_MAX),
    "roll_rate": (-RATE_LIMIT, RATE_LIMIT),
    "pitch_rate": (-RATE_LIMIT, RATE_LIMIT),
    "yaw_rate": (-RATE_LIMIT, RATE_LIMIT),
}

# The imitation policy does not share the thrust rail above.
#
# ``race_obs.ACTION_RANGES`` discretises actions into bins for behaviour
# cloning, and bins thrust over ``(0.05, 0.70)`` -- the range a human actually
# flew -- so the bins spend their resolution where the demonstrations live
# instead of on stick positions nobody used. The rate envelope is shared and
# must stay shared; the thrust ceiling is deliberately not.
#
# Consequence worth stating plainly: a BC-seeded policy has a hard thrust
# ceiling at 0.70 stick. Under the old straight-line plant that was 3.1 times
# weight. Under the measured curve it is 5.2 -- so correcting the thrust map
# did not merely make the top of the stick stronger, it widened the gap between
# what the imitation policy can ask for and what the PPO policy can. Warm-
# starting one from the other crosses that boundary.
BC_THRUST_BIN_RANGE = (THRUST_MIN, 0.70)

# ---------------------------------------------------------------------------
# Plant: what the aircraft is. Not hashed.
# ---------------------------------------------------------------------------

# Scale reading, confirmed. This is the one parameter that must be right rather
# than randomised: SimpleFlight measured a +30% mass error as unrecoverable on
# real hardware *even with* domain randomisation, and separately found that
# randomising a well-calibrated mass mostly adds learning difficulty.
MASS_KG = 1.745

# Estimated, not measured. Four arm assemblies at 45 degrees plus a central box
# (battery + Orin NX + plates); see ``env`` for the derivation. The roll figure
# spans 1.46x across plausible mass distributions, so treat it as a centred
# guess. Pitch sits above roll because a racing quad's stack is longer
# front-to-back than it is wide.
INERTIA_DIAG = (0.0040, 0.0055, 0.0075)

# --- randomised per episode -------------------------------------------------

# Where the aircraft actually hovers. John's runs span 0.21-0.24 unit to unit
# on healthy packs; the top of this range is a *drained* pack, not a different
# aircraft. Thrust goes as V^2 at fixed duty, so a 6S pack falling 4.2 to
# 3.5 V/cell loses about 31% of its thrust and the stick that holds a hover
# climbs to roughly 0.28.
PLANT_HOVER_RANGE = (0.21, 0.29)

# Thrust-to-weight at the THRUST_MAX rail.
#
# The client's straight line is wrong away from hover: the real chain is
# throttle -> duty -> RPM -> thrust, and while thrust does go as RPM^2, duty to
# RPM is markedly sublinear, so the two compose to about stick^1.6. Four 6S
# thrust-stand datasets agree that quarter stick is ~12% of max thrust, half
# ~35%, three-quarters ~68% -- not 25/50/75.
#
# The floor is derived, not chosen: a curve weaker than the straight line
# through its own hover point is concave, which no motor is. The ceiling is
# what the measured 5-inch shape implies. This airframe has never been on a
# thrust stand and weighs 1.745 kg, so rather than pick a point between them we
# train across the span.
PLANT_TWR_MAX = 8.0
PLANT_TWR_RANGE = (THRUST_MAX / PLANT_HOVER_RANGE[0], PLANT_TWR_MAX)

# Pooled fit to those four datasets, as a fraction of full-stick thrust carried
# by the s^2 term. Kept for reference; the plant samples TWR and derives this.
MEASURED_QUAD_SHARE = 0.68
