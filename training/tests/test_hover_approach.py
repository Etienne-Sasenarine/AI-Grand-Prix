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


def test_every_hover_reward_is_exported_from_the_package():
    """The config reaches these as ``mdp.<name>``, and this list is written by
    hand -- so a new reward function can exist, be tested, and still crash the
    environment at import with ``module has no attribute``.

    That is exactly what happened to ``approach``: the tests above load
    ``rewards.py`` by file path to avoid needing a full Isaac install, which
    means they never touch ``__init__.py`` and cannot see a missing export.
    This test reads the export list as text, for the same reason.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "tasks" / "drone_hover" / "mdp"

    exported: set[str] = set()
    for node in ast.parse((root / "__init__.py").read_text()).body:
        if isinstance(node, ast.ImportFrom) and node.module == "rewards":
            exported |= {a.name for a in node.names}

    defined = {
        node.name
        for node in ast.parse((root / "rewards.py").read_text()).body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    }

    missing = defined - exported
    assert not missing, (
        f"defined in rewards.py but not exported from mdp/__init__.py: "
        f"{sorted(missing)} -- the env config will fail at import"
    )


def test_floor_start_fraction_is_a_mixture_not_a_switch():
    """Racing must keep practising in-motion starts.

    The race begins at G1 from a standstill, which training had shown the
    policy at exactly one gate in eleven -- hence a 9.5-gate policy failing
    every episode at the real start. The fix is to add standing starts, not to
    replace the moving ones: a policy trained only from rest stops learning to
    carry speed between gates, which is the other ten elevenths of the race.
    """
    import ast
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "tasks" / "drone_racer"
           / "drone_racer_env_cfg.py").read_text()
    frac = None
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Attribute)
                and node.targets[0].attr == "floor_start_fraction"):
            frac = ast.literal_eval(node.value)
    assert frac is not None, "racing never sets floor_start_fraction"
    assert 0.0 < frac < 1.0, f"{frac} must be a mixture, not all-or-nothing"


def test_racing_floor_spawn_clears_the_ground():
    """The collision termination fires at 0.01 N, so a spawn resting on the
    floor ends its episode before the policy has acted -- the same trap the
    hover task had to avoid."""
    import ast
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "tasks" / "drone_racer"
           / "drone_racer_env_cfg.py").read_text()
    rng = None
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Attribute)
                and node.targets[0].attr == "spawn_z_range"):
            rng = ast.literal_eval(node.value)
    assert rng is not None
    assert rng[0] > 0.05, f"floor of {rng[0]} m is too close to the ground"
    assert rng[0] < rng[1]


def _racer_cfg_assign(name):
    import ast
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "tasks" / "drone_racer"
           / "drone_racer_env_cfg.py").read_text()
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Attribute)
                and node.targets[0].attr == name):
            return ast.literal_eval(node.value)
    return None


def test_race_start_sits_between_g10_and_g1():
    """The pad is on the G10->G1 leg, not somewhere off the course.

    Pinned because the coordinate was read off a map by eye: a typo that put
    it inside a wall or behind a gate would train a start that cannot exist,
    and nothing else in the suite looks at where it is.
    """
    import math
    xy = _racer_cfg_assign("race_start_xy")
    assert xy is not None, "racing never sets race_start_xy"
    G10, G1 = (3.96, 16.76), (3.66, 28.65)
    d10, d1 = math.dist(xy, G10), math.dist(xy, G1)
    assert d10 < d1, "the pad should be nearer G10 than G1 -- it is behind the start gate"
    assert 2.0 < d10 < 8.0, f"{d10:.1f} m past G10 is not on that leg"
    assert 5.0 < d1 < 12.0, f"{d1:.1f} m to G1 is not a plausible run-in"
    # Lateral offset from the straight G10->G1 line.
    t = ((xy[0]-G10[0])*(G1[0]-G10[0]) + (xy[1]-G10[1])*(G1[1]-G10[1])) / math.dist(G10, G1)**2
    proj = (G10[0] + t*(G1[0]-G10[0]), G10[1] + t*(G1[1]-G10[1]))
    assert math.dist(xy, proj) < 1.5, "the pad is well off the G10-G1 line"


def test_race_start_targets_the_official_first_gate():
    """Isaac index 6 is official G1. The organizer table says gates are flown
    in numerical order starting at Gate 1, so the start gate is not a choice."""
    assert _racer_cfg_assign("race_start_idx") == 6


def test_start_distribution_leaves_most_episodes_racing():
    """The start happens once per run; the other twenty-one gates happen after
    it. A policy trained mostly on starts optimises the rarest part of the
    race, which is the overfitting this split exists to avoid."""
    race = _racer_cfg_assign("race_start_fraction") or 0.0
    floor = _racer_cfg_assign("floor_start_fraction") or 0.0
    assert race > 0.0, "the competition start is never trained"
    assert floor > 0.0, "close-range starts are only trained at one gate"
    assert race + floor < 0.5, (
        f"{race + floor:.0%} of episodes are starts; most should be racing")


def test_hover_does_not_inherit_racing_start_distribution():
    """Hover subclasses the racing config, so racing's *future* changes arrive
    here uninvited.

    That is not hypothetical. Adding the competition start pad to racing put
    20% of hover episodes at that pad targeting G1 -- hover derives its hold
    point from the target gate, so those episodes were asked to station-keep
    from a racing start. Episode length fell from 898 steps to 26 and nothing
    failed loudly; the task just stopped working.
    """
    import ast
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "tasks" / "drone_hover"
           / "drone_hover_env_cfg.py").read_text()
    cleared = {}
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Attribute)
                and node.targets[0].attr in ("race_start_fraction",
                                             "floor_start_fraction",
                                             "race_start_xy")):
            cleared[node.targets[0].attr] = ast.literal_eval(node.value)

    assert cleared.get("race_start_fraction") == 0.0, (
        "hover must clear race_start_fraction; it inherits racing's value otherwise")
    assert cleared.get("floor_start_fraction") == 0.0, (
        "hover must clear floor_start_fraction")
    assert cleared.get("race_start_xy") is None, (
        "hover must clear race_start_xy")
