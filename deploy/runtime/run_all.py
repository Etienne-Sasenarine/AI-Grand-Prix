"""Run every test in the package. One command, one verdict.

    python run_all.py

The parity and end-to-end tests need torch (laptop only). Everything else is
pure NumPy and runs on the drone, which is the point — the onboard code has no
framework dependency at all.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

SUITES = [
    ("observation contract", "observation.py", False),
    ("Betaflight curves", "betaflight_curves.py", False),
    ("control adapter", "control_adapter.py", False),
    ("gate pose (PnP)", "pnp.py", False),
    ("quadrotor plant", "dynamics.py", False),
    ("guidance controller", "controller.py", False),
    ("geometry vs training code", "test_parity.py", True),
    ("gate tracker stress suite", "test_tracker.py", True),
    ("pose filter drift study", "test_pose_filter.py", True),
    ("crossing evidence", "test_crossing_evidence.py", False),
    ("policy context dependence", "test_context_dependence.py", True),
    ("end-to-end pipeline", "test_endtoend.py", True),
    ("closed loop (full system)", "test_closed_loop.py", True),
]


def main() -> int:
    python = sys.executable
    results = []
    for name, script, needs_torch in SUITES:
        t0 = time.perf_counter()
        proc = subprocess.run([python, str(HERE / script)], capture_output=True, text=True)
        dt = time.perf_counter() - t0
        ok = proc.returncode == 0
        results.append((name, ok, dt, proc))
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name:<32} {dt:6.1f} s")
        if not ok:
            print(proc.stdout[-3000:])
            print(proc.stderr[-3000:])

    print()
    passed = sum(1 for _, ok, _, _ in results if ok)
    print(f"{passed}/{len(results)} suites passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
