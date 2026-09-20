"""Inertia written into PhysX, with isaaclab stubbed out.

The tensor call itself needs a live sim, but the arithmetic around it -- which
of the nine flattened entries are the diagonal, how the per-env scale
broadcasts, which envs get touched -- does not, and that is where a silent
error would sit. A wrong slot writes a product of inertia instead of a
principal moment and the airframe quietly becomes a different shape.
"""

import sys
import types

import pytest
import torch


from tests._isaaclab_stub import ensure

ensure()


def _load_events_module():
    """Load ``mdp/events.py`` by path.

    Importing it normally would run ``tasks/__init__.py``, which pulls in
    ``isaaclab_tasks`` and needs a full Isaac install. The module itself has no
    relative imports, so loading the file directly is enough.
    """
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "tasks" / "drone_racer" / "mdp" / "events.py"
    spec = importlib.util.spec_from_file_location("_events_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_events = _load_events_module()
BODY_INERTIA_DIAG = _events.BODY_INERTIA_DIAG
set_body_inertia = _events.set_body_inertia

NUM_ENVS = 6
NUM_BODIES = 3
BODY_ID = 1


class FakeView:
    def __init__(self):
        # Start from a recognisably wrong plant so writes are visible.
        self.inertias = torch.zeros(NUM_ENVS, NUM_BODIES, 9)
        self.inertias[..., 0] = 0.003
        self.inertias[..., 4] = 0.003
        self.inertias[..., 8] = 0.006
        self.written_indices = None

    def get_inertias(self):
        return self.inertias

    def set_inertias(self, values, indices):
        self.inertias = values
        self.written_indices = indices


class FakeAsset:
    def __init__(self):
        self.root_physx_view = FakeView()

    def find_bodies(self, name):
        return [BODY_ID], [name]


class FakeEnv:
    def __init__(self):
        self.scene = types.SimpleNamespace(num_envs=NUM_ENVS, __getitem__=None)
        self._asset = FakeAsset()
        self.scene = self

    @property
    def num_envs(self):
        return NUM_ENVS

    def __getitem__(self, key):
        return self._asset


@pytest.fixture
def env():
    return FakeEnv()


def test_diagonal_lands_in_slots_zero_four_eight(env):
    set_body_inertia(env, None, inertia_diag=(0.1, 0.2, 0.3))
    row = env._asset.root_physx_view.inertias[0, BODY_ID]
    assert torch.allclose(row[[0, 4, 8]], torch.tensor([0.1, 0.2, 0.3]))


def test_products_of_inertia_are_cleared(env):
    """The authored USD carries 4.7e-4 in its off-diagonal terms -- 12% of the
    roll moment. Writing only the diagonal leaves them, and a tensor with
    products of inertia is a differently shaped aircraft: it couples roll into
    pitch on every input. The fake seeds them non-zero, because a fixture that
    starts at zero cannot tell "cleared" from "never touched"."""
    view = env._asset.root_physx_view
    for slot in (1, 2, 3, 5, 6, 7):
        view.inertias[:, BODY_ID, slot] = 4.7e-4

    set_body_inertia(env, None, inertia_diag=(0.1, 0.2, 0.3))

    row = view.inertias[0, BODY_ID]
    assert all(float(row[i]) == 0.0 for i in (1, 2, 3, 5, 6, 7))
    assert torch.allclose(row[[0, 4, 8]], torch.tensor([0.1, 0.2, 0.3]))


def test_clearing_products_does_not_reach_other_bodies(env):
    """Zeroing is per-body: a prop's tensor is not the airframe's business."""
    view = env._asset.root_physx_view
    view.inertias[:, 0, 1] = 4.7e-4

    set_body_inertia(env, None, inertia_diag=(0.1, 0.2, 0.3))

    assert float(view.inertias[0, 0, 1]) == pytest.approx(4.7e-4)


def test_other_bodies_are_untouched(env):
    before = env._asset.root_physx_view.inertias[:, 0].clone()
    set_body_inertia(env, None, inertia_diag=(0.1, 0.2, 0.3))
    after = env._asset.root_physx_view.inertias[:, 0]
    assert torch.allclose(before, after)


def test_only_named_envs_are_written(env):
    ids = torch.tensor([1, 4])
    set_body_inertia(env, ids, inertia_diag=(0.1, 0.2, 0.3))
    written = env._asset.root_physx_view.inertias[:, BODY_ID, 0]
    assert float(written[1]) == pytest.approx(0.1)
    assert float(written[4]) == pytest.approx(0.1)
    for untouched in (0, 2, 3, 5):
        assert float(written[untouched]) == pytest.approx(0.003)


def test_scale_range_varies_envs_but_keeps_the_airframe_shape(env):
    torch.manual_seed(0)
    set_body_inertia(env, None, inertia_diag=(0.1, 0.2, 0.4), scale_range=(0.7, 1.3))
    diag = env._asset.root_physx_view.inertias[:, BODY_ID][:, [0, 4, 8]]
    # Every env is scaled, so no two need agree...
    assert diag[:, 0].std() > 0.0
    # ...but the ratios between axes are a property of the airframe, not the draw.
    ratios = diag[:, 2] / diag[:, 0]
    assert torch.allclose(ratios, torch.full_like(ratios, 4.0), atol=1e-5)
    assert bool(((diag[:, 0] >= 0.07 - 1e-6) & (diag[:, 0] <= 0.13 + 1e-6)).all())


def test_no_scale_range_is_deterministic(env):
    set_body_inertia(env, None, inertia_diag=(0.1, 0.2, 0.3))
    diag = env._asset.root_physx_view.inertias[:, BODY_ID][:, [0, 4, 8]]
    assert torch.allclose(diag, diag[0].expand_as(diag))


def test_default_estimate_is_the_shape_a_racing_quad_has(env):
    ixx, iyy, izz = BODY_INERTIA_DIAG
    # Central stack is longer front-to-back than wide, so pitch resists more
    # than roll; yaw is roughly twice roll on a 5-inch X-quad.
    assert iyy > ixx
    assert 1.6 < izz / ixx < 2.2
    # And it must be heavier to spin than the 0.5 kg aircraft it inherited from.
    assert ixx > 0.003
