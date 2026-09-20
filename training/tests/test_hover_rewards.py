"""Station-keeping reward maths, with Isaac Sim stubbed out.

The env wiring needs a live simulator, but the arithmetic does not, and the
arithmetic is where a reward quietly means the wrong thing. A sign error in
``upright`` or a hold point computed on the wrong side of the gate produces a
policy that trains happily to do something useless, and nothing raises.
"""

from __future__ import annotations

import math
import sys
import types

import pytest
import torch


from tests._isaaclab_stub import ensure, load_module

ensure()

# drone_hover.mdp.rewards imports gate_normal_w from the racing rewards, whose
# own imports the stub satisfies. Load it under the name the hover module will
# look for, so the import resolves here rather than dragging in the package.
_racer = load_module("tasks/drone_racer/mdp/rewards.py", "tasks.drone_racer.mdp.rewards")
_pkg = types.ModuleType("tasks.drone_racer.mdp")
_pkg.rewards = _racer
sys.modules.setdefault("tasks", types.ModuleType("tasks"))
sys.modules.setdefault("tasks.drone_racer", types.ModuleType("tasks.drone_racer"))
sys.modules["tasks.drone_racer.mdp"] = _pkg
hover = load_module("tasks/drone_hover/mdp/rewards.py", "_hover_rewards")


N = 3


class FakeCmd:
    def __init__(self, gate_pos, gate_quat):
        self.command = torch.cat([gate_pos, gate_quat], dim=-1)


class FakeAsset:
    def __init__(self, pos, vel=None, quat=None):
        self.data = types.SimpleNamespace(
            root_pos_w=pos,
            root_lin_vel_w=torch.zeros(len(pos), 3) if vel is None else vel,
            root_quat_w=torch.tensor([[1.0, 0, 0, 0]] * len(pos)) if quat is None else quat,
        )


class FakeEnv:
    """Minimal stand-in exposing scene[...] and command_manager.get_term(...)."""

    def __init__(self, asset, cmd):
        self._asset, self._cmd = asset, cmd
        self.scene = self
        self.command_manager = types.SimpleNamespace(get_term=lambda _n: self._cmd)

    def __getitem__(self, _key):
        return self._asset


def _env(drone_pos, gate_pos=None, vel=None, quat=None, yaw=0.0):
    n = len(drone_pos)
    gate_pos = torch.zeros(n, 3) if gate_pos is None else gate_pos
    half = yaw / 2.0
    gate_quat = torch.tensor([[math.cos(half), 0.0, 0.0, math.sin(half)]] * n)
    return FakeEnv(FakeAsset(drone_pos, vel, quat), FakeCmd(gate_pos, gate_quat))


# --- hold point -------------------------------------------------------------


def test_hold_point_sits_in_front_of_the_gate():
    """A gate facing +x must put its hold point at negative x -- the approach
    side. Getting this backwards parks the aircraft behind the gate, which
    trains perfectly and tests nothing."""
    env = _env(torch.zeros(N, 3))
    point = hover.hold_point(env, "target", standoff_m=2.0)
    assert torch.allclose(point[:, 0], torch.full((N,), -2.0), atol=1e-5)
    assert torch.allclose(point[:, 1:], torch.zeros(N, 2), atol=1e-5)


def test_hold_point_follows_gate_yaw():
    """Rotate the gate 90 degrees and the hold point must rotate with it."""
    env = _env(torch.zeros(N, 3), yaw=math.pi / 2)
    point = hover.hold_point(env, "target", standoff_m=2.0)
    assert torch.allclose(point[:, 1], torch.full((N,), -2.0), atol=1e-4)
    assert torch.allclose(point[:, 0], torch.zeros(N), atol=1e-4)


def test_standoff_sets_the_distance():
    for standoff in (1.0, 2.0, 3.5):
        env = _env(torch.zeros(N, 3))
        point = hover.hold_point(env, "target", standoff_m=standoff)
        assert torch.allclose(torch.norm(point, dim=1), torch.full((N,), standoff), atol=1e-5)


# --- station_keep -----------------------------------------------------------


def test_station_keep_peaks_at_the_hold_point():
    env = _env(torch.tensor([[-2.0, 0.0, 0.0]]))
    assert float(hover.station_keep(env, "target")) == pytest.approx(1.0, abs=1e-5)


def test_station_keep_falls_off_with_distance():
    positions = torch.tensor([[-2.0, 0.0, 0.0], [-2.0, 0.5, 0.0], [-2.0, 2.0, 0.0]])
    env = _env(positions)
    r = hover.station_keep(env, "target", std=0.75)
    assert float(r[0]) > float(r[1]) > float(r[2])
    assert float(r[2]) < 0.001, "two metres off should be worth almost nothing"


def test_gaussian_not_exponential_falloff():
    """Deliberate difference from the racing task's exp(-d/std). A Gaussian
    falls away sharply enough that the last tens of centimetres are worth
    working for; the exponential's tail pays nearly as well at 2 m as at 1 m,
    which is right for an approach gradient and wrong for holding a point."""
    env = _env(torch.tensor([[-2.0, 0.75, 0.0]]))
    gaussian = float(hover.station_keep(env, "target", std=0.75))
    assert gaussian == pytest.approx(math.exp(-1.0), abs=1e-4)
    assert gaussian < math.exp(-1.0) + 1e-6


# --- stillness --------------------------------------------------------------


def test_stillness_is_zero_when_stationary():
    env = _env(torch.zeros(N, 3), vel=torch.zeros(N, 3))
    assert torch.allclose(hover.stillness(env), torch.zeros(N))


def test_stillness_grows_with_the_square_of_speed():
    env = _env(torch.zeros(2, 3), vel=torch.tensor([[1.0, 0, 0], [2.0, 0, 0]]))
    r = hover.stillness(env)
    assert float(r[0]) == pytest.approx(1.0)
    assert float(r[1]) == pytest.approx(4.0), "must be squared, not linear"


def test_stillness_counts_every_axis():
    env = _env(torch.zeros(1, 3), vel=torch.tensor([[1.0, 1.0, 1.0]]))
    assert float(hover.stillness(env)) == pytest.approx(3.0)


# --- upright ----------------------------------------------------------------


def test_upright_is_one_when_level():
    env = _env(torch.zeros(1, 3), quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert float(hover.upright(env)) == pytest.approx(1.0)


def test_upright_is_zero_on_its_side():
    half = math.pi / 4  # 90 degree roll
    env = _env(torch.zeros(1, 3), quat=torch.tensor([[math.cos(half), math.sin(half), 0.0, 0.0]]))
    assert float(hover.upright(env)) == pytest.approx(0.0, abs=1e-6)


def test_upright_never_rewards_being_inverted():
    """Clamped at zero: upside down must be worth nothing, not negative --
    a negative here would make crashing inverted *better* than crashing level,
    which is not a distinction worth teaching."""
    env = _env(torch.zeros(1, 3), quat=torch.tensor([[0.0, 1.0, 0.0, 0.0]]))  # 180 roll
    assert float(hover.upright(env)) == pytest.approx(0.0)
    assert float(hover.upright(env)) >= 0.0


def test_upright_is_unaffected_by_yaw():
    """Heading is the lookat term's business, not this one's."""
    half = math.pi / 4
    env = _env(torch.zeros(1, 3), quat=torch.tensor([[math.cos(half), 0.0, 0.0, math.sin(half)]]))
    assert float(hover.upright(env)) == pytest.approx(1.0, abs=1e-6)


# --- settled ----------------------------------------------------------------


def test_settled_needs_near_and_slow_together():
    """The whole point of the term. Near-but-fast and slow-but-far each collect
    plenty from the dense terms; only both at once is a hover."""
    pos = torch.tensor([
        [-2.0, 0.0, 0.0],   # on point, still      -> settled
        [-2.0, 0.0, 0.0],   # on point, moving     -> not
        [-2.0, 1.0, 0.0],   # displaced, still     -> not
        [-2.0, 1.0, 0.0],   # displaced, moving    -> not
    ])
    vel = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [0.0, 0, 0], [1.0, 0, 0]])
    env = _env(pos, vel=vel)
    r = hover.settled(env, "target")
    assert r.tolist() == [1.0, 0.0, 0.0, 0.0]


def test_settled_thresholds_are_respected():
    pos = torch.tensor([[-2.0, 0.25, 0.0], [-2.0, 0.35, 0.0]])
    vel = torch.tensor([[0.25, 0, 0], [0.25, 0, 0]])
    env = _env(pos, vel=vel)
    r = hover.settled(env, "target", radius_m=0.30, speed_m_s=0.30)
    assert r.tolist() == [1.0, 0.0]


def test_settled_is_a_clean_indicator():
    env = _env(torch.tensor([[-2.0, 0.0, 0.0]]), vel=torch.zeros(1, 3))
    r = hover.settled(env, "target")
    assert r.dtype == torch.float32
    assert float(r) in (0.0, 1.0)
