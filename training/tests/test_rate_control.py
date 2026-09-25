"""Thrust + NED rate decode, no Isaac Sim."""

import torch

from dynamics.rate_control import (
    DEFAULT_MASS_KG,
    G,
    PLANT_HOVER_RANGE,
    PLANT_QUAD_SHARE,
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


def test_zero_action_is_hover_and_zero_rates():
    raw = torch.zeros(2, 4)
    stick, rates = decode_aigp_action(raw)
    assert torch.allclose(stick, torch.full((2,), THRUST_HOVER))
    assert torch.allclose(rates, torch.zeros(2, 3))


def test_thrust_rails_match_aigp_band():
    lo = torch.tensor([[-1.0, 0.0, 0.0, 0.0]])
    hi = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    stick_lo, _ = decode_aigp_action(lo)
    stick_hi, _ = decode_aigp_action(hi)
    assert abs(float(stick_lo) - THRUST_MIN) < 1e-6
    assert abs(float(stick_hi) - THRUST_MAX) < 1e-6


def test_rate_rails_are_action_ranges():
    raw = torch.tensor([[0.0, 1.0, -1.0, 0.5]])
    _, rates = decode_aigp_action(raw)
    assert abs(float(rates[0, 0]) - RATE_LIMIT) < 1e-6
    assert abs(float(rates[0, 1]) + RATE_LIMIT) < 1e-6
    assert abs(float(rates[0, 2]) - 0.5 * RATE_LIMIT) < 1e-6


def test_hover_stick_is_weight():
    from dynamics.rate_control import DEFAULT_MASS_KG

    n = stick_to_newtons(torch.tensor([THRUST_HOVER]), mass_kg=DEFAULT_MASS_KG)
    assert abs(float(n) - DEFAULT_MASS_KG * 9.80665) < 1e-5


def test_rate_loop_rolls_toward_ned_command():
    # +NED roll, gyro at rest → +FLU Mx (same axis).
    des = torch.tensor([[1.0, 0.0, 0.0]])
    gyro = torch.zeros(1, 3)
    kp = torch.tensor([0.08, 0.08, 0.03])
    kd = torch.tensor([0.0, 0.0, 0.0])
    limit = torch.tensor([0.30, 0.30, 0.08])
    m = rate_moments(des, gyro, kp, kd, limit)
    assert float(m[0, 0]) > 0.0
    assert abs(float(m[0, 1])) < 1e-8
    assert abs(float(m[0, 2])) < 1e-8


def test_ned_pitch_flips_to_flu_moment():
    # +NED pitch (y right) → −FLU My (y left).
    des = torch.tensor([[0.0, 1.0, 0.0]])
    gyro = torch.zeros(1, 3)
    kp = torch.tensor([0.08, 0.08, 0.03])
    kd = torch.zeros(3)
    limit = torch.tensor([0.30, 0.30, 0.08])
    m = rate_moments(des, gyro, kp, kd, limit)
    assert float(m[0, 1]) < 0.0


def test_hover_constant_alone_cannot_randomise_the_plant():
    """The decode and the Newton conversion cancel: this is why the obvious
    domain randomisation (sample ``cfg.hover_thrust`` per env) does nothing at
    hover. Zero action is m*g for every candidate hover stick."""
    raw = torch.zeros(1, 4)
    for hover in (0.21, 0.225, 0.24, 0.255):
        stick, _ = decode_aigp_action(raw, thrust_hover=hover)
        newtons = stick_to_newtons(stick, hover_stick=hover)
        assert abs(float(newtons) - DEFAULT_MASS_KG * G) < 1e-5


def test_plant_scale_is_one_when_the_client_guessed_right():
    scale = plant_thrust_scale(torch.tensor([THRUST_HOVER]), nominal_hover=THRUST_HOVER)
    assert abs(float(scale) - 1.0) < 1e-6


def test_a_bird_that_hovers_low_climbs_on_the_clients_hover_stick():
    """Bird hovers at 0.21, client sends its own 0.225: more thrust than weight."""
    raw = torch.zeros(1, 4)
    stick, _ = decode_aigp_action(raw, thrust_hover=THRUST_HOVER)
    nominal_n = stick_to_newtons(stick, hover_stick=THRUST_HOVER)
    scale = plant_thrust_scale(torch.tensor([0.21]), nominal_hover=THRUST_HOVER)
    applied_n = nominal_n * scale
    weight_n = DEFAULT_MASS_KG * G
    assert abs(float(applied_n / weight_n) - THRUST_HOVER / 0.21) < 1e-5
    # ...and a bird that hovers high sinks on the same stick.
    heavy = plant_thrust_scale(torch.tensor([0.30]), nominal_hover=THRUST_HOVER)
    assert float(nominal_n * heavy) < weight_n


def test_randomisation_is_centred_on_the_decode_constant():
    """The decode constant sits inside the measured band, so zero action is
    over-thrusted in some envs and under-thrusted in others. If it drifts
    outside the band every env leans the same way and the policy just learns a
    fixed trim, wasting half the thrust channel."""
    lo, hi = PLANT_HOVER_RANGE
    assert lo < THRUST_HOVER < hi
    scales = plant_thrust_scale(torch.tensor([lo, hi]), nominal_hover=THRUST_HOVER)
    assert float(scales[0]) > 1.0 > float(scales[1])



# --- thrust curve -----------------------------------------------------------
#
# The client assumes thrust is a straight line through the hover point. The
# real chain is throttle -> duty -> RPM -> thrust: thrust goes as RPM^2, but
# duty -> RPM is markedly sublinear, and the two compose to about stick^1.6.
# ``quad_share`` is how much of the full-stick thrust the s^2 term carries.


def test_quad_share_zero_is_exactly_the_linear_map():
    """The default must reproduce the old plant bit for bit, so every existing
    checkpoint keeps the physics it was trained against."""
    stick = torch.linspace(0.05, 0.90, 25)
    linear = (stick / THRUST_HOVER) * (DEFAULT_MASS_KG * G)
    assert torch.allclose(stick_to_newtons(stick, quad_share=0.0), linear, atol=1e-5)


def test_every_curve_shape_is_pinned_to_one_g_at_hover():
    """``quad_share`` changes the shape, never the hover point. Whatever the
    curve, the hover stick must hold exactly the aircraft's weight."""
    weight_n = DEFAULT_MASS_KG * G
    for share in (0.0, 0.25, 0.55, 0.68, 0.80, 1.0):
        n = stick_to_newtons(torch.tensor([THRUST_HOVER]), quad_share=share)
        assert abs(float(n) - weight_n) < 1e-4, f"quad_share={share}"


def test_quad_share_one_is_pure_rpm_squared():
    """The textbook limit: thrust proportional to stick^2."""
    stick = torch.tensor([2.0 * THRUST_HOVER])
    n = stick_to_newtons(stick, quad_share=1.0)
    assert abs(float(n) / (DEFAULT_MASS_KG * G) - 4.0) < 1e-4


# Pooled from four 6S / 5-inch thrust-stand datasets: fraction of full-stick
# thrust at quarter, half and three-quarter stick. A straight line would
# predict 0.25 / 0.50 / 0.75.
MEASURED_THRUST_FRACTION = ((0.25, 0.1225), (0.50, 0.354), (0.75, 0.683))


def _thrust_fraction(stick, quad_share=None):
    """Thrust at ``stick`` as a fraction of thrust at full stick."""
    share = PLANT_QUAD_SHARE if quad_share is None else quad_share
    at = stick_to_newtons(torch.tensor([stick]), quad_share=share)
    full = stick_to_newtons(torch.tensor([1.0]), quad_share=share)
    return float(at / full)


def test_measured_shape_beats_a_straight_line_everywhere():
    """The whole reason the curve exists. At every measured point the quadratic
    must land closer to the thrust stands than the client's straight line."""
    for stick, measured in MEASURED_THRUST_FRACTION:
        curved = abs(_thrust_fraction(stick) - measured)
        straight = abs(stick - measured)
        assert curved < straight, f"stick {stick}: curve {curved:.3f} vs line {straight:.3f}"


def test_measured_shape_is_close_at_the_bottom_of_the_stick():
    """Quarter and half stick are where a race actually lives, and the
    quadratic tracks the thrust stands there to within a few percent."""
    for stick, measured in MEASURED_THRUST_FRACTION[:2]:
        assert abs(_thrust_fraction(stick) - measured) < 0.03, f"stick {stick}"


def test_known_residual_a_parabola_flattens_too_late():
    """Documented limitation, asserted so it stays documented.

    The measured local exponent runs 1.53 -> 1.62 -> 1.33 across the stick: the
    real curve steepens through the middle and then *flattens* near full
    throttle as the motor approaches its no-load RPM and current limits bite. A
    parabola through the origin has no inflection and cannot follow that, so it
    reads low at three-quarter stick by about six points.

    We keep the parabola anyway. It is one interpretable parameter, it pins the
    hover point exactly, it stays monotonic, and it cuts the error at the rail
    from the straight line's ~49% to under 10%. Fitting a third parameter to
    other people's 5-inch aircraft would be false precision when ours weighs
    1.745 kg and has never been on a thrust stand -- and the spread we randomise
    ``quad_share`` over is far wider than this residual anyway.
    """
    stick, measured = MEASURED_THRUST_FRACTION[2]
    residual = _thrust_fraction(stick) - measured
    assert -0.08 < residual < -0.04, f"residual moved to {residual:+.3f}"


def test_curve_is_monotonic_across_the_usable_band():
    """A non-monotonic thrust map would make the policy's thrust channel
    ambiguous. Every admissible quad_share must stay increasing."""
    stick = torch.linspace(THRUST_MIN, THRUST_MAX, 200)
    for share in (0.0, PLANT_QUAD_SHARE, 1.0):
        n = stick_to_newtons(stick, quad_share=share)
        assert bool((n.diff() > 0).all()), f"quad_share={share} is not monotonic"


def test_linear_map_understates_thrust_at_the_top_of_the_stick():
    """The error that matters. Pinned at the same hover point, the straight
    line delivers about half the real thrust at THRUST_MAX -- so an aircraft
    answers a punch-out harder than the linear plant ever did, and a policy
    trained on it overshoots rather than falling short."""
    rail = torch.tensor([THRUST_MAX])
    linear = float(stick_to_newtons(rail, quad_share=0.0))
    real = float(stick_to_newtons(rail, quad_share=PLANT_QUAD_SHARE))
    assert 0.45 < linear / real < 0.56


def test_linear_map_overstates_thrust_near_idle():
    """Below hover the error flips sign: the straight line promises thrust the
    aircraft does not make, so a policy learns to sink faster than it can."""
    idle = torch.tensor([0.10])
    linear = float(stick_to_newtons(idle, quad_share=0.0))
    real = float(stick_to_newtons(idle, quad_share=PLANT_QUAD_SHARE))
    assert 1.15 < linear / real < 1.30


def test_hover_and_share_broadcast_per_env():
    """Both plant parameters are drawn per environment, so they arrive as
    tensors and must broadcast against a per-env stick."""
    stick = torch.full((4,), THRUST_HOVER)
    hover = torch.tensor([0.21, 0.24, 0.26, 0.29])
    share = torch.tensor([0.55, 0.62, 0.74, 0.80])
    n = stick_to_newtons(stick, hover_stick=hover, quad_share=share)
    assert n.shape == (4,)
    weight_n = DEFAULT_MASS_KG * G
    # An aircraft that hovers *below* the client's constant is over-thrusted
    # when handed it; one that hovers above it is under-thrusted. Only the
    # first of these four sits below 0.225.
    assert float(n[0]) > weight_n
    assert all(float(n[i]) < weight_n for i in (1, 2, 3))
    # Monotone in the plant's hover point: the higher it hovers, the less it
    # gives back for the same stick.
    assert float(n[1]) > float(n[2]) > float(n[3])


def test_sag_end_of_pack_costs_real_thrust_at_the_hover_stick():
    """Top of PLANT_HOVER_RANGE is a drained pack, not a different aircraft.
    The client still sends 0.225 and gets visibly less than weight for it."""
    _, drained = PLANT_HOVER_RANGE
    n = stick_to_newtons(
        torch.tensor([THRUST_HOVER]), hover_stick=drained, quad_share=PLANT_QUAD_SHARE
    )
    deficit = 1.0 - float(n) / (DEFAULT_MASS_KG * G)
    assert 0.15 < deficit < 0.40



# --- thrust authority -------------------------------------------------------
#
# quad_share is how the curve is *implemented*; thrust-to-weight at the rail is
# how it is *chosen*. They are the same degree of freedom -- pinning the curve
# at hover means its steepness is its top-end authority -- but only one of them
# can be checked against the motors bolted to the aircraft.


def _twr_at_rail(quad_share, hover_stick=THRUST_HOVER):
    n = stick_to_newtons(
        torch.tensor([THRUST_MAX]), hover_stick=hover_stick, quad_share=quad_share
    )
    return float(n) / (DEFAULT_MASS_KG * G)


def test_twr_round_trips_through_the_curve():
    """The inversion has to be exact, because the plant is sampled in TWR and
    applied in quad_share."""
    for twr in (4.0, 5.0, 6.5, 8.0):
        q = quad_share_for_twr(twr)
        assert abs(_twr_at_rail(q) - twr) < 1e-4, f"twr {twr} -> q {q}"


def test_a_straight_line_inverts_to_no_curvature():
    """The anchor the whole parameterisation hangs on."""
    for hover in (0.21, THRUST_HOVER, 0.29):
        q = quad_share_for_twr(THRUST_MAX / hover, hover_stick=hover)
        assert abs(float(q)) < 1e-9, f"hover {hover} -> q {float(q)}"


def test_measured_thrust_stand_shape_sits_at_the_top_of_the_range():
    """The ceiling is the 5-inch measurement, not an arbitrary number."""
    assert abs(_twr_at_rail(PLANT_QUAD_SHARE) - PLANT_TWR_RANGE[1]) < 0.2


def test_range_floor_is_the_weakest_physical_curve_not_a_guess():
    """A curve weaker than the straight line through its own hover point is
    concave, which no motor does. The floor must therefore be the straight-line
    TWR at the *lowest* hover point, so every plant in PLANT_HOVER_RANGE can
    reach every TWR in PLANT_TWR_RANGE convexly."""
    assert PLANT_TWR_RANGE[0] == THRUST_MAX / PLANT_HOVER_RANGE[0]


def test_no_plant_in_range_ever_needs_a_concave_curve():
    """The regression this file previously missed by only ever solving at the
    nominal hover point: with hover at 0.21 and a TWR floor of 4.0, the solved
    quad_share came out at -0.105."""
    lo_t, hi_t = PLANT_TWR_RANGE
    # The floor is exactly the straight line at the lowest hover point, so that
    # corner solves to q = 0 and lands either side of it. torch.linspace is
    # float32, so the tolerance is float32-scale, not double-scale.
    eps = 1e-6
    for hover in torch.linspace(*PLANT_HOVER_RANGE, 9):
        for twr in torch.linspace(lo_t, hi_t, 9):
            q = float(quad_share_for_twr(twr, hover_stick=float(hover)))
            assert -eps <= q <= 1.0, f"hover {float(hover):.3f} twr {float(twr):.2f} -> q {q:.3f}"


def test_more_authority_is_a_steeper_curve():
    """Monotone in TWR at every hover point in the range."""
    lo, hi = PLANT_TWR_RANGE
    for hover in (PLANT_HOVER_RANGE[0], THRUST_HOVER, PLANT_HOVER_RANGE[1]):
        shares = [
            float(quad_share_for_twr(t, hover_stick=hover)) for t in torch.linspace(lo, hi, 12)
        ]
        assert all(b > a for a, b in zip(shares, shares[1:])), f"hover {hover}"


def test_authority_is_solved_against_this_env_own_hover_point():
    """A drained pack hovers higher, so reaching the same TWR at the rail needs
    a different curve. Solving against the nominal hover instead of the env's
    would quietly mis-set the authority on exactly the envs that matter most."""
    drained = PLANT_HOVER_RANGE[1]
    q = quad_share_for_twr(6.0, hover_stick=drained)
    assert abs(_twr_at_rail(q, hover_stick=drained) - 6.0) < 1e-4
    # Solving against the wrong hover point is visibly wrong, not a rounding slip.
    q_wrong = quad_share_for_twr(6.0, hover_stick=THRUST_HOVER)
    assert abs(_twr_at_rail(q_wrong, hover_stick=drained) - 6.0) > 0.3


def test_range_spans_a_believable_aircraft():
    """Sanity rail on the whole scheme: total static thrust at the top of the
    range must stay in the region a real racing quad occupies."""
    lo, hi = PLANT_TWR_RANGE
    assert 4.0 <= lo < hi <= 10.0
    heaviest_pull_kgf = hi * DEFAULT_MASS_KG
    assert heaviest_pull_kgf / 4.0 < 5.0, "more than 5 kgf per motor is not a quad"
