"""Source-root shim for the deploy/runtime suite.

See ``estimation/conftest.py`` for the rationale. The onboard flight modules
import each other by bare name; after the consolidation they live in
``deploy/runtime/`` (control loop) and ``estimation/`` (VIO), which import each
other. Both are added as source roots for the test run.

Scoped to ``deploy/`` deliberately: ``deploy/runtime/dynamics.py`` would shadow
the ``training/dynamics`` package, so these paths must not leak into the
training suite. Run the suites separately.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT / "deploy" / "runtime", _ROOT / "estimation"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
