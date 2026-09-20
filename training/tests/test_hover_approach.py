"""The approach term must have the reach that ``station_keep`` does not.

The hover task originally spawned the aircraft a full gate-spacing from its
hold point, where a 0.75 m Gaussian returns numerically zero -- no gradient,
nothing to learn, and 40 of 64 envs terminated on step one. The first fix moved
the spawn onto the hold point, which worked and produced a policy that only
knows how to hold from somewhere it is already holding.

``approach`` is the second fix and the right one: give the reward reach so the
spawn can scatter across the region where the gate is visible, which is how the
aircraft gets handed over in a cage. These tests hold the property that makes
that safe -- that the term is still meaningfully non-zero at the far edge of
the spawn box.
"""

from __future__ import annotations

import math
import sys
import types

import pytest
import torch

from tests._isaaclab_stub import ensure, load_module

ensure()

_racer = load_module("tasks/drone_racer/mdp/rewards.py", "tasks.drone_racer.mdp.rewards")
_pkg = types.ModuleType("tasks.drone_racer.mdp")
_pkg.rewards = _racer
sys.modules.setdefault("tasks", types.ModuleType("tasks"))
sys.modules.setdefault("tasks.drone_racer", types.ModuleType("tasks.drone_racer"))
sys.modules["tasks.drone_racer.mdp"] = _pkg
hover = load_module("tasks/drone_hover/mdp/rewards.py", "_hover_approach_rewards")


class FakeCmd:
    def __init__(self, gate_pos, gate_quat):
        self.command = torch.cat([gate_pos, gate_quat], dim=-1)


class FakeAsset:
    def __init__(self, pos):
        self.data = types.SimpleNamespace(
            root_pos_w=pos,
            root_lin_vel_w=torch.zeros(len(pos), 3),
            root_quat_w=torch.tensor([[1.0, 0, 0, 0]] * len(pos)),
        )


class FakeEnv:
    def __init__(self, asset, cmd):
        self._asset, self._cmd = asset, cmd
        self.scene = self
        self.command_manager = types.SimpleNamespace(get_term=lambda _n: self._cmd)

    def __getitem__(self, _key):
        return self._asset


def _env(drone_pos):
    n = len(drone_pos)
    gate_pos = torch.zeros(n, 3)
    gate_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n)
    return FakeEnv(FakeAsset(drone_pos), FakeCmd(gate_pos, gate_quat))


# The hold point for a gate at the origin facing +x is (-2, 0, 0).
HOLD = torch.tensor([-2.0, 0.0, 0.0])


def at(offset):
    return (HOLD + torch.tensor(offset)).unsqueeze(0)


def test_peaks_at_the_hold_point():
    assert float(hover.approach(_env(at([0.0, 0, 0])), "target")) == pytest.approx(1.0)


def test_decays_monotonically_with_distance():
    prev = 2.0
    for d in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0):
        v = float(hover.approach(_env(at([0.0, d, 0])), "target"))
        assert v < prev
        prev = v


def test_still_has_usable_gradient_at_the_spawn_box_edge():
    """The property the whole change rests on.

    The hover spawn scatters by up to (2.5, 2.5, 0.9) m, so the far corner is
    about 3.6 m from the hold point. ``station_keep`` there is exp(-23) -- zero
    in float32, no gradient, nothing to learn. ``approach`` must not be."""
    corner = math.sqrt(2.5**2 + 2.5**2 + 0.9**2)
    a = float(hover.approach(_env(at([2.5, 2.5, 0.9])), "target"))
    s = float(hover.station_keep(_env(at([2.5, 2.5, 0.9])), "target"))
    assert a > 0.25, f"approach {a:.4f} too weak at {corner:.2f} m"
    assert s < 1e-8, f"station_keep {s:.2e} should be numerically dead at {corner:.2f} m"


def test_gradient_survives_well_past_the_box():
    """A policy blown off station should still be pulled back, not abandoned."""
    for d in (4.0, 6.0, 8.0):
        assert float(hover.approach(_env(at([0.0, d, 0])), "target")) > 0.02


def test_is_exponential_not_gaussian():
    """Deliberate: a Gaussian's tail dies quadratically in the exponent and is
    useless at range, which is the failure this term exists to fix."""
    std = 3.0
    v = float(hover.approach(_env(at([0.0, 3.0, 0])), "target", std=std))
    assert v == pytest.approx(math.exp(-1.0), abs=1e-4)


def test_std_widens_the_reach():
    far = at([0.0, 6.0, 0])
    narrow = float(hover.approach(_env(far), "target", std=1.5))
    wide = float(hover.approach(_env(far), "target", std=6.0))
    assert wide > narrow


def test_does_not_outrank_settling():
    """approach pays for getting there; station_keep and settled pay for
    staying. If the broad term could outscore them the policy would be content
    to loiter nearby, which is the degenerate solution this task already
    guards against elsewhere."""
    APPROACH_W, STATION_W, SETTLED_W = 0.6, 2.0, 3.0
    on_point = at([0.0, 0, 0])
    approach_max = APPROACH_W * float(hover.approach(_env(on_point), "target"))
    hold_max = STATION_W * float(hover.station_keep(_env(on_point), "target"))
    assert approach_max < hold_max
    assert approach_max < SETTLED_W
