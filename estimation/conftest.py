"""Source-root shim for the estimation suite.

The onboard flight code was written as a flat set of modules that import each
other by bare name (``import geometry``, ``from pose_filter import ...``). When
the repository was consolidated it was split into ``estimation/`` (the VIO:
gate-detection PnP + pose filter) and ``deploy/runtime/`` (the control loop),
which are tightly coupled: the estimation tests exercise ``sim`` (in
deploy/runtime) and the runtime imports ``pose_filter``/``pnp`` (here).

Rather than rewrite every import, both directories are treated as source roots
for the test run. They contain no clashing module names. Note this is scoped to
``estimation/`` on purpose: ``deploy/runtime`` ships a top-level ``dynamics.py``
that would shadow the ``training/dynamics`` package, so these paths must never
be added while the training suite runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT / "estimation", _ROOT / "deploy" / "runtime"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
