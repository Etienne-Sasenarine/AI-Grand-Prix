"""Run every field tool's self-test. One command, one verdict.

    python3 run_all.py

Every tool here is testable with no drone, no flight controller and no
``~/target/``, because each one runs against ``mock_link.py`` — a small flight
model with **known** parameters. The analysis tools are therefore checked
against a ground truth they were never told, which is the only way to know a
parser is doing arithmetic rather than merely not crashing.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

SUITES = [
    ("mock flight controller", "mock_link.py", []),
    ("MSP telemetry logger", "msp_logger.py", ["--self-test"]),
    ("hover analysis", "analyze_hover.py", ["--self-test"]),
    ("rate step analysis", "analyze_rates.py", ["--self-test"]),
    ("Betaflight CLI bridge", "bf_cli.py", ["--self-test"]),
    ("beginner flight setup", "bf_beginner.py", ["--self-test"]),
    ("RPM telemetry helper", "fc_rpm.py", []),
    ("TEST 1 hover throttle", "test_hover.py", ["--self-test"]),
    ("TEST 2 max thrust", "test_max_thrust.py", ["--self-test"]),
    ("TEST 3 camera capture", "test_camera_capture.py", ["--self-test"]),
]


def main() -> int:
    results = []
    for name, script, args in SUITES:
        t0 = time.perf_counter()
        proc = subprocess.run([sys.executable, str(HERE / script), *args],
                              capture_output=True, text=True)
        dt = time.perf_counter() - t0
        # Exit 2 means "could not run here", which is NOT a pass. A self-test
        # that silently skips and reports success is worse than no test: it is
        # a green light for code nothing has exercised.
        code = proc.returncode
        status = "PASS" if code == 0 else ("SKIP" if code == 2 else "FAIL")
        results.append((name, status))
        print(f"[{status}] {name:<28} {dt:5.1f} s")
        if status == "SKIP":
            tail = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
            if tail:
                print(f"         reason: {tail[-1].strip()}")
        if status == "FAIL":
            print(proc.stdout[-2000:])
            print(proc.stderr[-2000:])

    print()
    passed = sum(1 for _, st in results if st == "PASS")
    skipped = sum(1 for _, st in results if st == "SKIP")
    failed = sum(1 for _, st in results if st == "FAIL")
    print(f"{passed} passed, {skipped} skipped, {failed} failed "
          f"(of {len(results)})")
    if skipped:
        print()
        print("SKIPPED tools are untested here. They must be run on the drone")
        print("before anyone relies on them:")
        for name, st in results:
            if st == "SKIP":
                print(f"  - {name}")
    if not failed and not skipped:
        print()
        print("All of these run against the mock. None has touched hardware.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
