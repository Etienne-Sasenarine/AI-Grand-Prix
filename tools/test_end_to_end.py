"""Fly a mission, record it, analyse it. The whole chain, no shortcuts.

    python3 test_end_to_end.py

Why a separate test
-------------------
Every tool here has its own self-test, and each one passes. That is not the same
as the chain working. The 18 Sep session failed at a join: the recorder was fine,
the analysis was fine, and the two were never run against each other on real
output.

So this drives a simulated aircraft through an actual mission, has the **real
recorder** write a **real CSV**, and hands that file to the **real analysers** --
no synthesised rows, no hand-built dictionaries. If any pair of these stops
fitting together, this fails.

It checks the numbers come back right, not merely that nothing crashed:

* hover throttle, against the simulated aircraft's own ``hover_throttle``
* thrust-to-weight, which for a linear-thrust model must equal
  ``1 / hover_throttle`` -- and the analysis computes it from **RPM** by a
  completely different route, so agreement is real evidence
* the marker switch, found unaided among sixteen channels
* ANGLE mode assignment, end to end over MSP

None of these numbers is given to the code under test.
"""

from __future__ import annotations

import csv
import math
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import flight_recorder  # noqa: E402
from analyze_flight import find_marker, load, marks_from_channel  # noqa: E402
from analyze_thrust import find_burst, find_hover_rpm  # noqa: E402
from mock_link import MockConfig, MockLink, open_link  # noqa: E402
from setup_angle_mode import BOX_ANGLE, assign, find_angle, read_ranges  # noqa: E402
from test_hover import analyse  # noqa: E402

#: The mission, as (label, seconds, throttle, marker-switch-high).
MISSION = [
    ("on the pad",   2.0, 0.00, False),
    ("spool up",     1.5, 0.20, False),
    ("climb",        2.0, 0.45, False),
    ("hover 1",      6.0, None, True),
    ("reposition",   2.0, 0.42, False),
    ("hover 2",      6.0, None, True),
    ("reposition",   2.0, 0.40, False),
    ("hover 3",      6.0, None, True),
    ("settle",       1.5, None, False),
    ("FULL THROTTLE", 0.7, 0.98, False),
    ("recover",      2.5, 0.22, False),
    ("land",         2.0, 0.00, False),
]

MARKER_CH = 11          # a spare channel, chosen to be one the tool must find


def fly(fc: MockLink, done: threading.Event) -> None:
    """Drive the simulated aircraft through the mission in real time."""
    hover = fc.cfg.hover_throttle
    for label, dur, thr, mark in MISSION:
        throttle = hover if thr is None else thr
        fc.arm_state(throttle > 0.05)
        fc.command(throttle)
        fc.aux[MARKER_CH] = 1900 if mark else 1100
        t0 = time.monotonic()
        while time.monotonic() - t0 < dur:
            fc.command(throttle)          # keep it fed, as a transmitter would
            time.sleep(0.02)
    fc.arm_state(False)
    fc.command(0.0)
    done.set()


def main() -> int:
    cfg = MockConfig()
    true_hover = cfg.hover_throttle
    true_tw = 1.0 / true_hover          # linear thrust law: full / hover
    out_dir = Path(tempfile.mkdtemp())

    print()
    print("END-TO-END: fly -> record -> analyse")
    print("=" * 68)
    print(f"  simulated aircraft hovers at {true_hover:.3f} throttle")
    print(f"  so thrust-to-weight must come out at {true_tw:.2f}")
    print(f"  marker switch on RC channel {MARKER_CH}; the analysis is not told")
    total = sum(d for _l, d, _t, _m in MISSION)
    print(f"  mission is {total:.0f}s: 3 marked hovers and one full-throttle burst")
    print()

    fc = open_link("mock")
    done = threading.Event()

    # The recorder stops itself when the mission finishes.
    def stopper():
        done.wait(timeout=total + 15)
        flight_recorder._STOP = True

    flight_recorder._STOP = False
    threading.Thread(target=stopper, daemon=True).start()
    threading.Thread(target=fly, args=(fc, done), daemon=True).start()

    t0 = time.monotonic()
    summary = flight_recorder.record(fc, out_dir, hz=30.0, poles=14, verbose=False)
    flight_recorder._STOP = False
    fc.close()
    print(f"  recorded {summary['rows']} rows in {time.monotonic() - t0:.0f}s "
          f"-> {Path(summary['path']).name}")
    assert summary["read_errors"] == 0, summary
    assert summary["ever_armed"], "the mission never armed the aircraft"

    # --- from here on, only the real analysis code, on the real file ---
    rows = load(Path(summary["path"]))
    assert len(rows) > 200, len(rows)
    print(f"  loaded {len(rows)} rows back from disk")

    ch, considered = find_marker(rows, None)
    assert ch == MARKER_CH, (
        f"marker search picked channel {ch}, the pilot used {MARKER_CH}\n" +
        "\n".join(f"    ch{c['ch']}: {c['verdict']}" for c in considered))
    print(f"  marker switch found unaided: channel {ch}")

    marks = marks_from_channel(rows, ch)
    assert len(marks) == 3, f"{len(marks)} marked windows, expected 3"
    print(f"  {len(marks)} marked windows")

    res = analyse(rows, marks, mass_kg=1.745)
    assert res["accepted"], res
    got_hover = res["hover_fraction"]
    err = abs(got_hover - true_hover)
    print(f"  HOVER  measured {got_hover:.3f} vs true {true_hover:.3f} "
          f"(error {err:.4f})")
    assert err < 0.02, f"hover off by {err:.4f}"

    rpm_hover, motor_hover, n_good, _rej = find_hover_rpm(rows, marks)
    assert rpm_hover, "no RPM recovered from the hovers"
    burst = find_burst(rows)
    assert burst is not None, "the full-throttle burst was not found"
    assert burst["rpm"], "no RPM in the burst"
    tw = (burst["rpm"] / rpm_hover) ** 2
    tw_err = abs(tw - true_tw)
    print(f"  THRUST-TO-WEIGHT {tw:.2f} vs true {true_tw:.2f} "
          f"(error {tw_err:.2f}), from RPM, mass never used")
    assert tw_err < 0.35, f"T/W off by {tw_err:.2f}"

    # The two routes are independent: one reads motor microseconds during the
    # hovers, the other reads rotor speed at two different throttles. They must
    # agree, or one of them is wrong.
    implied = 1.0 / got_hover
    print(f"  cross-check: 1/hover = {implied:.2f} against T/W {tw:.2f}")
    assert abs(implied - tw) < 0.4, (implied, tw)

    # --- ANGLE assignment, over MSP, on a fresh link ---
    fc2 = open_link("mock")
    assert find_angle(read_ranges(fc2)) is None
    code = assign(fc2, aux_channel=1, low_us=1600, high_us=2100, dry_run=False)
    assert code == 0, code
    a = find_angle(read_ranges(fc2))
    assert a and a[2] == 1, a
    assert fc2.eeprom_writes == 1
    fc2.close()
    print(f"  ANGLE assigned over MSP and verified by read-back")

    shutil.rmtree(out_dir, ignore_errors=True)
    print()
    print("  ALL STAGES PASSED — the chain holds together on real output")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
