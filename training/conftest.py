"""Let the suite run on a laptop as well as on the training box.

Some tests need a full Isaac Sim install -- anything that touches
``isaaclab_tasks``, the env registration, or a live PhysX view. Most do not:
the contract, the thrust curve, the observation maths and the rate loop are all
plain torch and are the tests you most want to be able to run while editing.

Rather than let collection blow up on a machine without Isaac, the modules that
need it are skipped with a reason. ``pytest -q`` then means "everything that can
be checked here", and on the GPU box it means everything.

Be aware of what that hides: a green run on a laptop has not exercised the env,
the plant, or any PhysX call. Before a training run that costs money, run the
suite somewhere Isaac is installed and confirm nothing skipped that matters.
"""

from __future__ import annotations

import importlib.util

import pytest

# Modules that cannot even be imported without Isaac Sim present.
# Paths are relative to this file, which is the repo root.
_NEEDS_ISAAC = (
    "tests/test_dynamics.py",
    "tests/test_gate_counter_torch.py",
    "tests/test_miss_and_dive.py",
    "tests/test_speed_cap.py",
)


def _isaac_available() -> bool:
    for module in ("isaaclab", "isaaclab_tasks"):
        if importlib.util.find_spec(module) is None:
            return False
    return True


ISAAC = _isaac_available()

collect_ignore = [] if ISAAC else list(_NEEDS_ISAAC)


def pytest_configure(config):
    config.addinivalue_line("markers", "isaac: requires a full Isaac Sim install")


def pytest_collection_modifyitems(config, items):
    if ISAAC:
        return
    skip = pytest.mark.skip(reason="needs Isaac Sim; run on the training box")
    for item in items:
        if "isaac" in item.keywords:
            item.add_marker(skip)


def pytest_report_header(config):
    where = "with Isaac Sim" if ISAAC else "WITHOUT Isaac Sim (env tests skipped)"
    return f"aigp-sim: running {where}"
