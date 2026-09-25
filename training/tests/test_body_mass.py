"""Mass distributed across the airframe's links, with isaaclab stubbed out.

This file exists because of a bug that trained for a week without anyone
noticing. ``MassPropertiesCfg(mass=1.745)`` looks exactly like "the aircraft
weighs 1.745 kg" and is in fact "every rigid body weighs 1.745 kg" -- which,
for a USD with an airframe and four prop links, is a 8.725 kg drone carrying
thrust sized for a fifth of that. Thrust-to-weight 0.2. It cannot take off, and
every run since the mass was changed trained a policy for a brick.

Nothing caught it because every piece was individually right: the contract said
1.745, the thrust curve was built for 1.745, the asset config said 1.745. Only
the product of the config and the asset's link count was wrong, and no test
looked at that product. These do.
"""

from __future__ import annotations

import types

import pytest
import torch


from tests._isaaclab_stub import ensure

ensure()


def _load_events_module():
    """Load ``mdp/events.py`` by path, bypassing ``tasks/__init__.py``.

    Same reason as ``test_body_inertia.py``: the package pulls in
    ``isaaclab_tasks`` and needs a full Isaac install, and this check is too
    important to skip on a laptop.
    """
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "tasks" / "drone_racer" / "mdp" / "events.py"
    spec = importlib.util.spec_from_file_location("_mass_events_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_events = _load_events_module()
set_body_mass = _events.set_body_mass
AIRFRAME_MASS_KG = _events.AIRFRAME_MASS_KG
PROP_MASS_KG = _events.PROP_MASS_KG
PROP_INERTIA_DIAG = _events.PROP_INERTIA_DIAG

NUM_ENVS = 6
BODY_ID = 0
PROP_IDS = [1, 2, 3, 4]
NUM_BODIES = 5

# What the broken spawn produced: one plausible number, on every link.
SPAWN_MASS = 1.745


class FakeView:
    def __init__(self, num_bodies=NUM_BODIES):
        self.masses = torch.full((NUM_ENVS, num_bodies), SPAWN_MASS)
        self.written_indices = None

    def get_masses(self):
        return self.masses

    def set_masses(self, values, indices):
        self.masses = values
        self.written_indices = indices


class FakeAsset:
    def __init__(self, num_bodies=NUM_BODIES, prop_ids=None):
        self.root_physx_view = FakeView(num_bodies)
        self._prop_ids = PROP_IDS if prop_ids is None else prop_ids

    def find_bodies(self, name):
        """Stand in for Isaac Lab's regex body lookup.

        Only the two patterns this event uses need to resolve; anything else is
        a typo in the test rather than a case worth supporting.
        """
        if name == "body":
            return [BODY_ID], [name]
        if name == "prop.*":
            return list(self._prop_ids), [f"prop{i}" for i in self._prop_ids]
        raise AssertionError(f"unexpected body pattern {name!r}")


class FakeEnv:
    def __init__(self, num_bodies=NUM_BODIES, prop_ids=None):
        self._asset = FakeAsset(num_bodies, prop_ids)
        self.scene = self

    @property
    def num_envs(self):
        return NUM_ENVS

    def __getitem__(self, key):
        return self._asset


@pytest.fixture
def env():
    return FakeEnv()


# --- the total --------------------------------------------------------------


def test_total_mass_is_what_the_aircraft_weighs(env):
    """The check the old code would have failed: 8.725 vs 1.745."""
    set_body_mass(env, None)
    total = float(env._asset.root_physx_view.masses[0].sum())
    assert total == pytest.approx(AIRFRAME_MASS_KG, abs=1e-6)


def test_props_come_out_of_the_total_not_on_top_of_it(env):
    """1.745 kg is all-up competition weight with props fitted, so the five
    links sum to it and the airframe link is *lighter* than it. Adding the
    props to the total instead would give a 1.765 kg aircraft -- a small error,
    but the plausible-looking way for someone to misread this later."""
    set_body_mass(env, None)
    masses = env._asset.root_physx_view.masses[0]
    assert float(masses[BODY_ID]) < AIRFRAME_MASS_KG
    assert float(masses.sum()) == pytest.approx(AIRFRAME_MASS_KG, abs=1e-6)


def test_total_holds_for_any_prop_mass(env):
    """The airframe takes whatever the props leave, so the total is
    arithmetic rather than three numbers maintained by hand. Revising the prop
    estimate must never change what the aircraft weighs."""
    for prop_mass in (0.0, 0.005, 0.02, 0.05):
        e = FakeEnv()
        set_body_mass(e, None, prop_mass_kg=prop_mass)
        total = float(e._asset.root_physx_view.masses[0].sum())
        assert total == pytest.approx(AIRFRAME_MASS_KG, abs=1e-6), prop_mass


def test_every_env_gets_the_same_total(env):
    """Mass is the one plant parameter that is not randomised."""
    set_body_mass(env, None)
    totals = env._asset.root_physx_view.masses.sum(dim=1)
    assert torch.allclose(totals, torch.full_like(totals, AIRFRAME_MASS_KG), atol=1e-6)


# --- the distribution -------------------------------------------------------


def test_mass_lands_on_the_airframe_not_the_props(env):
    set_body_mass(env, None)
    masses = env._asset.root_physx_view.masses[0]
    assert float(masses[BODY_ID]) == pytest.approx(AIRFRAME_MASS_KG - 4 * PROP_MASS_KG)
    for prop_id in PROP_IDS:
        assert float(masses[prop_id]) == pytest.approx(PROP_MASS_KG)


def test_a_correct_total_on_the_wrong_links_is_still_caught(env):
    """Guards the check, not the code. The total alone cannot distinguish a
    real airframe from one whose entire mass sits on a single propeller, so
    anything asserting only the sum is not testing this function."""
    set_body_mass(env, None)
    masses = env._asset.root_physx_view.masses[0]
    assert float(masses[BODY_ID]) > 0.9 * AIRFRAME_MASS_KG


def test_only_named_envs_are_written(env):
    ids = torch.tensor([1, 4])
    set_body_mass(env, ids)
    masses = env._asset.root_physx_view.masses[:, BODY_ID]
    for touched in (1, 4):
        assert float(masses[touched]) == pytest.approx(AIRFRAME_MASS_KG - 4 * PROP_MASS_KG)
    for untouched in (0, 2, 3, 5):
        assert float(masses[untouched]) == pytest.approx(SPAWN_MASS)


# --- the guard rails --------------------------------------------------------


def test_an_unmatched_link_is_an_error():
    """A sixth rigid body nobody named would keep its spawn mass and quietly
    break the total. That must be loud: it is exactly how the original bug
    stayed invisible -- a link nobody was thinking about carrying mass nobody
    intended."""
    env = FakeEnv(num_bodies=6)
    with pytest.raises(ValueError, match="every rigid body"):
        set_body_mass(env, None)


def test_props_too_heavy_to_leave_an_airframe_is_an_error():
    env = FakeEnv()
    with pytest.raises(ValueError, match="leave"):
        set_body_mass(env, None, prop_mass_kg=1.0)


def test_more_than_one_central_body_is_an_error():
    env = FakeEnv(num_bodies=5, prop_ids=[1, 2, 3])

    def two_bodies(name):
        if name == "body":
            return [0, 4], [name, name]
        return [1, 2, 3], ["prop1", "prop2", "prop3"]

    env._asset.find_bodies = two_bodies
    with pytest.raises(ValueError, match="expected 1"):
        set_body_mass(env, None)


# --- the estimates ----------------------------------------------------------


def test_prop_mass_is_a_propeller_not_an_aircraft(env):
    """The bug in one assertion."""
    assert 0.001 < PROP_MASS_KG < 0.05
    assert PROP_MASS_KG < 0.05 * AIRFRAME_MASS_KG


def test_prop_inertia_is_a_thin_planar_body():
    """Spin axis resists twice what the flat axes do -- the perpendicular axis
    theorem for anything flat. A prop that failed this would be modelled as a
    sphere."""
    ixx, iyy, izz = PROP_INERTIA_DIAG
    assert ixx == pytest.approx(iyy)
    assert izz == pytest.approx(2.0 * ixx, rel=0.05)


def test_prop_inertia_matches_its_mass_and_span():
    """I_spin = mL^2/12 for a 0.127 m span. If the prop mass is ever revised
    and this is not, the props keep a different aircraft's inertia -- which is
    the precise shape of the bug this file is about."""
    span_m = 0.127
    expected = PROP_MASS_KG * span_m**2 / 12.0
    assert PROP_INERTIA_DIAG[2] == pytest.approx(expected, rel=0.05)


def test_prop_inertia_is_negligible_beside_the_airframe():
    """States the reason the estimate is allowed to be rough."""
    assert PROP_INERTIA_DIAG[2] < 0.01 * min(_events.BODY_INERTIA_DIAG)
