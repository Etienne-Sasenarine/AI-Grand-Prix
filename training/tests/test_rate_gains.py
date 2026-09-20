"""The rate loop must stay matched to the airframe it controls.

This file exists because the pairing broke once and nothing noticed. The gains
came from the 0.5 kg upstream aircraft with inertia (0.003, 0.003, 0.006), and
survived untouched when this airframe's inertia was corrected to (0.004,
0.0055, 0.0075). Since ``tau = I / (kp + kd)``, that silently made the pitch
axis 84% slower to respond -- and a policy commanding body rates through a
sluggish pitch axis reaches gates and cannot thread them.

The gains are now derived from the inertia rather than written beside it, so
the failure mode is structurally impossible. These tests hold that property
down: that the derivation is correct, that it tracks a changed inertia, and
that nobody has quietly reintroduced a hardcoded constant.
"""

from __future__ import annotations

import pytest

from contract import plant
from dynamics import rate_control as rc


def tau_of(inertia, kp, kd):
    """Closed-loop time constant of I*w' + (kp + kd)*w = kp*w_des."""
    return inertia / (kp + kd)


# --- the derivation ---------------------------------------------------------


def test_each_axis_hits_its_target_time_constant():
    kp, kd = rc.rate_gains_for_inertia()
    for axis, (inertia, target) in enumerate(
        zip(rc.INERTIA_DIAG, rc.RATE_TAU_TARGET_S)
    ):
        assert tau_of(inertia, kp[axis], kd[axis]) == pytest.approx(target, rel=1e-9), axis


def test_gains_scale_with_inertia():
    """The property the old code lacked: double the inertia, double the gains.

    Without this, correcting an airframe's mass distribution quietly changes
    how fast it responds -- which is exactly what happened."""
    base_kp, base_kd = rc.rate_gains_for_inertia((0.004, 0.0055, 0.0075))
    heavy_kp, heavy_kd = rc.rate_gains_for_inertia((0.008, 0.011, 0.015))
    for axis in range(3):
        assert heavy_kp[axis] == pytest.approx(2.0 * base_kp[axis])
        assert heavy_kd[axis] == pytest.approx(2.0 * base_kd[axis])


def test_time_constant_is_invariant_to_inertia():
    """The whole point: a different airframe responds the same way."""
    for inertia in ((0.003, 0.003, 0.006), (0.004, 0.0055, 0.0075), (0.01, 0.02, 0.03)):
        kp, kd = rc.rate_gains_for_inertia(inertia)
        for axis in range(3):
            assert tau_of(inertia[axis], kp[axis], kd[axis]) == pytest.approx(
                rc.RATE_TAU_TARGET_S[axis], rel=1e-9
            )


def test_kd_keeps_its_ratio_to_kp():
    kp, kd = rc.rate_gains_for_inertia()
    for axis in range(3):
        assert kd[axis] / kp[axis] == pytest.approx(rc.RATE_KD_KP_RATIO)


# --- moment limits ----------------------------------------------------------


def test_moment_limit_holds_angular_acceleration_not_torque():
    """A fixed torque ceiling on a heavier-to-spin airframe is a weaker
    aircraft. Scaling with inertia keeps the authority the tuning intended."""
    limits = rc.moment_limits_for_inertia()
    for axis in range(3):
        alpha = limits[axis] / rc.INERTIA_DIAG[axis]
        assert alpha == pytest.approx(rc.RATE_ALPHA_MAX[axis], rel=1e-9)


def test_moment_limits_scale_with_inertia():
    base = rc.moment_limits_for_inertia((0.004, 0.0055, 0.0075))
    heavy = rc.moment_limits_for_inertia((0.008, 0.011, 0.015))
    for axis in range(3):
        assert heavy[axis] == pytest.approx(2.0 * base[axis])


# --- the pairing that actually broke ----------------------------------------


def test_rate_control_inertia_matches_the_contract():
    """One airframe, one inertia. If these part company the gains are derived
    from a body the simulator is not using."""
    assert tuple(rc.INERTIA_DIAG) == tuple(plant.INERTIA_DIAG)


def test_inertia_matches_the_event_that_writes_it():
    """And the event that pushes inertia into PhysX must agree too, or the
    gains are tuned for a body that never reaches the simulation."""
    import ast
    from pathlib import Path

    source = (
        Path(__file__).resolve().parent.parent
        / "tasks" / "drone_racer" / "mdp" / "events.py"
    ).read_text()
    written = None
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "BODY_INERTIA_DIAG" for t in node.targets
        ):
            written = ast.literal_eval(node.value)
    assert written is not None
    assert tuple(written) == tuple(rc.INERTIA_DIAG)


def test_action_config_uses_the_derived_gains_not_literals():
    """Guards against someone pasting numbers back in. A literal here would
    recreate the exact bug this module exists to prevent."""
    import ast
    from pathlib import Path

    source = (
        Path(__file__).resolve().parent.parent
        / "tasks" / "drone_racer" / "mdp" / "actions.py"
    ).read_text()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id in ("rate_kp", "rate_kd", "moment_limit"):
                assert isinstance(node.value, ast.Name), (
                    f"{node.target.id} must reference the derived constant, "
                    "not a literal tuple"
                )


# --- sanity against the simulator it runs in --------------------------------


def test_time_constants_are_representable_at_the_physics_rate():
    """Physics runs at 120 Hz. A first-order loop with tau near dt is not
    being simulated, it is being aliased -- so the targets must stay clear of
    the step size by a real margin."""
    dt = 1.0 / 120.0
    for axis, tau in enumerate(rc.RATE_TAU_TARGET_S):
        assert tau > 2.5 * dt, f"axis {axis}: tau {tau * 1000:.1f} ms too close to dt"


def test_yaw_is_slower_than_roll_and_pitch():
    """Real quads yaw with prop drag, not thrust differential, so yaw
    authority genuinely lags. Equal time constants would be the suspicious
    result, not this."""
    roll, pitch, yaw = rc.RATE_TAU_TARGET_S
    assert yaw > roll and yaw > pitch


def test_gains_are_larger_than_the_upstream_aircraft_needed():
    """The regression in one assertion: this airframe is heavier to spin than
    the 0.5 kg drone the old gains came from, so its gains must be larger."""
    old_kp, _ = rc.rate_gains_for_inertia((0.003, 0.003, 0.006))
    for axis in range(3):
        assert rc.RATE_KP[axis] > old_kp[axis]
