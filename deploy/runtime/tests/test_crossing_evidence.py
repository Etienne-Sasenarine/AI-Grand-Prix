"""A declared crossing needs two independent witnesses.

The plane-crossing test in ``GateTracker`` asks "did my position estimate pass
through the opening". That is self-assessed: it is answered from the estimate,
which is exactly the thing that is wrong when it is wrong. Measured before this
existed, on a run where the estimate had drifted 8.5 m:

    declared 8 crossings, 6 actually happened
    t=36.53  "gate 7 crossed, clean, 0.12 m off centre"

Every declared pass looked textbook — under 0.15 m off centre — and two of them
never happened.

The second witness is apparent size. To pass through a gate you must first have
been close enough to fill the view, and how big a gate looks owes nothing to
where you think you are. PnP's range is that measurement, and the flight loop
feeds it back through ``note_fix``.

This file checks three things:

  1. a normal approach *is* corroborated, so the check does not simply refuse
     everything and look safe while scoring nothing;
  2. a crossing declared with no close-range sighting is *not* corroborated;
  3. a tracker that is never told about ranges does not invent a verdict.

Points 1 and 3 matter as much as 2. A safety check that fires constantly gets
turned off, and one that fires when it has no evidence is just noise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import geometry as G  # noqa: E402
import gate_tracker as T  # noqa: E402


def _detection_of(gate, pos, quat, camera):
    uv, vis = camera.project(gate.corners_world(), pos, quat)
    if not np.asarray(vis, dtype=bool).any():
        return None
    return T.Detection(uv=uv, visible=vis, confidence=1.0)


def _fly_through(tracker, camera, *, report_range: bool, closest: float,
                 dt: float = 1 / 60.0, speed: float = 4.0):
    """Walk the estimate through gate 1's opening, reporting fixes as asked."""
    gate = tracker.sequence[0]
    quat = G.quat_from_ahrs(0.0, 0.0, float(np.arctan2(gate.through[1], gate.through[0])))
    for k in range(240):
        d = 6.0 - speed * k * dt
        if d < -1.5:
            break
        pos = gate.centre - gate.through * d
        det = _detection_of(gate, pos, quat, camera)
        ev = tracker.update(dt, [det] if det is not None else [], pos, quat)
        if report_range and det is not None:
            # Report the true range, but never closer than ``closest`` -- which
            # is how a gate we only ever saw from a distance would look.
            tracker.note_fix(True, range_m=max(abs(d), closest))
        if ev is not None:
            return ev
    return None


def main() -> int:
    camera = G.Camera.real()
    seq = G.race_sequence(laps=1)
    print()
    print("Crossing evidence: a pass needs geometry AND apparent size to agree")
    print("-" * 70)

    # 1. A normal approach, ranged right up to the gate, is corroborated.
    tr = T.GateTracker(sequence=list(seq), camera=camera)
    ev = _fly_through(tr, camera, report_range=True, closest=1.5)
    assert ev is not None, "a straight run through the opening was not counted at all"
    assert ev.clean, "a centred crossing should be clean"
    assert ev.corroborated, (
        "a gate ranged to 1.5 m must corroborate; if this fails the threshold is "
        "too tight and real crossings will go unscored")
    print(f"  normal approach          counted, clean={ev.clean}, "
          f"corroborated={ev.corroborated}   <- must be True")

    # 2. The same geometry, but the gate was never ranged closer than 9 m. The
    #    estimate says we went through something we were never near.
    tr = T.GateTracker(sequence=list(seq), camera=camera)
    ev = _fly_through(tr, camera, report_range=True, closest=9.0)
    assert ev is not None, "the crossing should still be counted, just not trusted"
    assert not ev.corroborated, (
        "a crossing never ranged closer than 9 m must NOT be corroborated -- this "
        "is the check that catches a drifted estimate declaring phantom passes")
    print(f"  never ranged under 9 m   counted, clean={ev.clean}, "
          f"corroborated={ev.corroborated}  <- must be False")

    # 3. A tracker nobody reports ranges to must not invent a verdict. The
    #    stress suite drives the tracker directly, with no PnP anywhere.
    tr = T.GateTracker(sequence=list(seq), camera=camera)
    ev = _fly_through(tr, camera, report_range=False, closest=0.0)
    assert ev is not None
    assert ev.corroborated, (
        "with no range information at all the tracker must abstain, not accuse")
    print(f"  no ranges reported       counted, clean={ev.clean}, "
          f"corroborated={ev.corroborated}   <- abstains, must be True")

    # 4. The evidence resets between gates: a well-corroborated gate 1 must not
    #    vouch for gate 2. This is the bookkeeping error that would make the
    #    whole check decorative.
    tr = T.GateTracker(sequence=list(seq), camera=camera)
    _fly_through(tr, camera, report_range=True, closest=1.5)
    assert tr.index == 1, f"expected to advance to gate 2, at {tr.index}"
    assert tr._min_fix_range == float("inf"), (
        "the closest-range evidence must reset at each crossing, or gate 1's good "
        "sighting silently corroborates gate 2")
    print("  evidence resets per gate  yes")

    print("-" * 70)
    print("  PASS - crossings are only scored when both witnesses agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
