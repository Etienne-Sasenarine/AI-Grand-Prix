"""Action delay and rate-setpoint smoothing, without Isaac Sim.

``ControlAction`` needs a live env, so these exercise the two mechanisms
directly against the same tensor operations the term performs. That is enough
to catch what actually goes wrong with a delay line: reading the wrong slot,
the ring wrapping incorrectly, and a reset leaking the previous episode's
commands into the next one.

That last is not hypothetical. The eval harness this was ported from keeps a
plain list and never clears it on reset, which is harmless for a single scripted
evaluation and quietly wrong across 4096 environments resetting at staggered
times -- a steady trickle of commands the policy never issued and cannot
explain.
"""

from __future__ import annotations

import pytest
import torch

from dynamics.rate_control import ACTION_DELAY_STEPS_RANGE, RATE_TAU_S_RANGE

CONTROL_HZ = 60.0
PHYSICS_DT = 1.0 / 120.0


class DelayLine:
    """The ring buffer from ``ControlAction._delayed``, standalone."""

    def __init__(self, num_envs: int, max_delay: int):
        self.buffer = torch.zeros(num_envs, max_delay + 1, 4)
        self.head = 0
        self.delay = torch.zeros(num_envs, dtype=torch.long)
        self.rows = torch.arange(num_envs)
        self.max_delay = max_delay

    def push(self, actions: torch.Tensor) -> torch.Tensor:
        if self.max_delay == 0:
            return actions
        self.head = (self.head + 1) % self.buffer.shape[1]
        self.buffer[:, self.head] = actions
        read = (self.head - self.delay) % self.buffer.shape[1]
        return self.buffer[self.rows, read]

    def reset(self, env_ids: torch.Tensor) -> None:
        self.buffer[env_ids] = 0.0


# --- delay line -------------------------------------------------------------


def test_zero_delay_is_the_old_instantaneous_behaviour():
    """The rollback path has to be exact, not approximately exact."""
    line = DelayLine(4, max_delay=0)
    for value in (0.3, -0.7, 1.0):
        actions = torch.full((4, 4), value)
        assert torch.equal(line.push(actions), actions)


def test_a_command_comes_back_after_exactly_its_delay():
    line = DelayLine(1, max_delay=3)
    line.delay[0] = 2
    marks = [torch.full((1, 4), float(i)) for i in range(6)]

    out = [float(line.push(m)[0, 0]) for m in marks]
    # Steps 0 and 1 predate any command, so the queue's zeros come back.
    assert out[:2] == [0.0, 0.0]
    # From then on each step returns the command issued two steps earlier.
    assert out[2:] == [0.0, 1.0, 2.0, 3.0]


def test_each_env_gets_its_own_delay():
    """Per-env, so the policy cannot learn one specific lag and lean on it."""
    line = DelayLine(3, max_delay=3)
    line.delay = torch.tensor([0, 1, 3])
    for i in range(6):
        out = line.push(torch.full((3, 4), float(i)))
    assert float(out[0, 0]) == 5.0  # no delay: the command just issued
    assert float(out[1, 0]) == 4.0
    assert float(out[2, 0]) == 2.0


def test_the_ring_wraps_without_losing_alignment():
    """Run well past the buffer length; a modulo error shows up here."""
    line = DelayLine(1, max_delay=2)
    line.delay[0] = 2
    out = []
    for i in range(20):
        out.append(float(line.push(torch.full((1, 4), float(i)))[0, 0]))
    assert out[2:] == [float(i) for i in range(18)]


def test_reset_does_not_leak_the_previous_episode():
    """The bug the eval harness has. A fresh episode must not begin by
    executing commands from the one before it."""
    line = DelayLine(2, max_delay=3)
    line.delay = torch.tensor([3, 3])

    for i in range(1, 5):
        line.push(torch.full((2, 4), float(i)))

    line.reset(torch.tensor([0]))          # env 0 starts a new episode
    out = line.push(torch.full((2, 4), 99.0))

    assert float(out[0, 0]) == 0.0, "env 0 replayed a command from its last episode"
    assert float(out[1, 0]) != 0.0, "env 1 was not reset and should still be delayed"


def test_resetting_one_env_leaves_the_others_alone():
    line = DelayLine(4, max_delay=2)
    line.delay = torch.full((4,), 2, dtype=torch.long)
    for i in range(1, 4):
        line.push(torch.full((4, 4), float(i)))
    before = line.buffer.clone()
    line.reset(torch.tensor([2]))
    assert torch.equal(line.buffer[2], torch.zeros_like(line.buffer[2]))
    for env in (0, 1, 3):
        assert torch.equal(line.buffer[env], before[env])


# --- rate smoothing ---------------------------------------------------------


def _step_filter(state, command, tau, dt=PHYSICS_DT):
    """Exact discretisation of a first-order lag.

    Not ``dt / tau``. Physics runs at 120 Hz and the measured FC smoothing is
    ~10.6 ms, so dt/tau lands near 0.8 where the Euler form is 44% wrong -- and
    at the fast end of the band it clamps to 1.0 and models no lag at all.
    """
    import math

    alpha = 1.0 - math.exp(-dt / tau)
    return state + alpha * (command - state)


def test_filter_converges_to_a_held_command():
    state, tau = 0.0, 0.0106
    for _ in range(200):
        state = _step_filter(state, 3.2, tau)
    assert state == pytest.approx(3.2, rel=1e-3)


def test_filter_reaches_63_percent_in_one_time_constant():
    """What a first-order lag means. If this drifts, the tau we quote from the
    FC's 15 Hz cutoff no longer describes what the sim does."""
    # A tau that is a clean multiple of the physics step, so exactly one time
    # constant elapses and the 63.2% is not blurred by rounding.
    steps = 10
    tau = steps * PHYSICS_DT
    state = 0.0
    for _ in range(steps):
        state = _step_filter(state, 1.0, tau)
    assert state == pytest.approx(0.632, abs=0.005)


def test_exact_discretisation_is_used_not_the_euler_form():
    """Guards the bug this file found. At the fast end of the band dt/tau > 1,
    where Euler clamps to 1.0 -- instantaneous tracking, i.e. no filter."""
    import math

    tau = RATE_TAU_S_RANGE[0]
    euler = min(1.0, PHYSICS_DT / tau)
    exact = 1.0 - math.exp(-PHYSICS_DT / tau)
    assert euler == pytest.approx(1.0), "the fast end really does saturate Euler"
    assert exact < 0.7, "the exact form still leaves meaningful lag"
    one_step = _step_filter(0.0, 1.0, tau)
    assert one_step == pytest.approx(exact, abs=1e-9)


def test_the_measured_fc_cutoff_is_inside_the_configured_band():
    """rc_smoothing_setpoint_cutoff is 15 Hz fixed on this flight controller,
    which is tau = 1/(2*pi*15). The band has to contain it or we are modelling
    an aircraft we do not own."""
    import math

    measured_tau = 1.0 / (2.0 * math.pi * 15.0)
    lo, hi = RATE_TAU_S_RANGE
    assert lo <= measured_tau <= hi, f"{measured_tau:.4f} outside {RATE_TAU_S_RANGE}"


def test_a_slower_filter_lags_further_behind():
    fast = slow = 0.0
    for _ in range(12):
        fast = _step_filter(fast, 1.0, RATE_TAU_S_RANGE[0])
        slow = _step_filter(slow, 1.0, RATE_TAU_S_RANGE[1])
    assert fast > slow


def test_filter_attenuates_a_command_that_reverses_every_step():
    """The point of calling it a bandwidth limit rather than a delay: content
    faster than the cutoff does not arrive late, it arrives smaller."""
    state, tau = 0.0, RATE_TAU_S_RANGE[1]
    seen = []
    for i in range(60):
        state = _step_filter(state, 3.2 if i % 2 else -3.2, tau)
        seen.append(abs(state))
    assert max(seen[20:]) < 3.2 * 0.5


# --- ranges -----------------------------------------------------------------


def test_delay_range_brackets_the_measured_command_latency():
    """Agilicious measured Betaflight command-to-actuation at about 40 ms, and
    our own sweep put the cliff at 50 ms. At 60 Hz the range must be able to
    express something in that neighbourhood."""
    lo, hi = ACTION_DELAY_STEPS_RANGE
    assert lo == 0, "zero must stay reachable so the old plant is in-distribution"
    worst_ms = hi * 1000.0 / CONTROL_HZ
    assert 25.0 <= worst_ms <= 60.0, f"worst case {worst_ms:.0f} ms"


def test_rate_tau_band_is_plausible_for_a_five_inch_quad():
    """Published motor time constants for 5-inch airframes cluster at 30-40 ms;
    the FC's own smoothing is about 10 ms. The band should span that, and not
    stray into territory belonging to a much larger aircraft."""
    lo, hi = RATE_TAU_S_RANGE
    assert 0.005 <= lo < hi <= 0.06
