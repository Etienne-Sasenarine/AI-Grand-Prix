"""Where the flight client lives, for the few places that import from it.

The flight repo (``Code-Red-Cables/AI_GP``, checked out as ``ai-grand-prix``)
owns ``gate_counter.py``, ``race_obs.py`` and ``ekf/commanded_accel.py``.
This repo borrows them only to test that the torch twins agree with the code
the Jetson actually runs, and for the optional visual gate counter.

Resolution order: ``$AIGP_FLIGHT_REPO``, then ``~/dev/ai-grand-prix``, then a
sibling directory named ``ai-grand-prix`` or ``AI_GP``. Nothing here imports
anything; it only says where to look, so importing it can never fail.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()


def candidates() -> tuple[Path, ...]:
    env = os.environ.get("AIGP_FLIGHT_REPO")
    paths = [Path(env).expanduser()] if env else []
    paths += [
        Path.home() / "dev" / "ai-grand-prix",
        _HERE.parents[2] / "ai-grand-prix",
        _HERE.parents[2] / "AI_GP",
    ]
    return tuple(paths)


def find(marker: str = "race_obs.py") -> Path | None:
    """The first candidate containing ``marker``, or ``None``."""
    for path in candidates():
        if (path / marker).is_file():
            return path
    return None


def add_to_sys_path(marker: str = "race_obs.py") -> Path | None:
    """Put the flight repo first on ``sys.path`` if it can be found."""
    repo = find(marker)
    if repo is not None and str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo
