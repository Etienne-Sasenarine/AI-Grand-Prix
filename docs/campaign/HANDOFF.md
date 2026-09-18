# Handoff — flight software, 17 Sep 2026

Written to survive a context reset. If you are picking this up cold, read this
file, then `flight/README.md`, then `SPEC.md`.

---

## Where things stand in one paragraph

The onboard stack is built and runs closed-loop: a simulated aircraft flies the
real course using only what the real drone would have — flight-controller
attitude, gyro, camera detections, and the surveyed map. It takes off, localises
itself, works out which gate is next without any markers, and flies gates.
**It now completes all 22 crossings in most configurations** (see the table
below). Median position error is 0.10–0.20 m, down from 0.71–2.39 m this
morning. Four bugs were found and fixed today; the first was responsible for
almost everything else.

---

## What exists

All in `flight/`. Pure NumPy plus OpenCV; no deep-learning framework, because the
Jetson image has none.

| Module | What it does | State |
|---|---|---|
| `geometry.py` | Course map, gate corners, camera projection, frame conventions | Verified against training code |
| `observation.py` | The 51-number frame and 32-frame history | **Exact** match to training code |
| `policy_runtime.py` | Runs an skrl checkpoint in NumPy | 2.5e-6 vs PyTorch, 126 µs |
| `pnp.py` | Gate-relative pose from 8 corners | Verified; see accuracy table |
| `pose_filter.py` | Position, velocity, thrust scale, yaw bias | Verified; see drift study |
| `gate_tracker.py` | Which gate is next, and when we crossed | 22/22 across 17 conditions |
| `betaflight_curves.py` | Physical units ↔ stick values | Self-tested; forward model UNVERIFIED |
| `control_adapter.py` | Action → envelope → curves → channels | Self-tested; **axis signs UNVERIFIED** |
| `dynamics.py` | Quadrotor plant, for closed-loop testing | Self-tested with sign assertions |
| `controller.py` | Guidance: flies the course from the estimate. Doubles as Plan B | Self-tested |
| `flight_loop.py` | State machine: takeoff → race → hold → land/abort, watchdog | Runs closed-loop |
| `sim.py` | Synthetic flight + detector model | Used by the tests |

`python run_all.py` runs everything.

---

## The bugs worth remembering

**Five of the six are the same class: a quantity computed in one frame, or under
one convention, used as if it were in another.** They do not announce
themselves — every one of them looked fine until something downstream was
measured against ground truth.

### 1. The yaw bearing residual was not tilt-compensated — *the big one*

`flight_loop` handed the pose filter `atan2(y, x)` of the gate's position in the
**body** frame and the filter compared it against a bearing from the **map**,
which is a world-frame quantity. The body is rolled and pitched, so those are not
the same angle.

Measured with a perfect pose, a perfect map and zero true heading error: the
residual reports **4.3° of phantom yaw error at 25° of bank, 11.8° at 35° of bank
with the nose 15° down.** It is not noise — it is *signed with the turn*. This
course turns predominantly one way, so the filter integrated it. By gate 7 the
estimated heading bias had walked to **+21°** against a true error of 3°.

At fx = 425, 21° puts the predicted gate **163 px** from where it actually is.
The tracker then stopped recognising the gate directly in front of it, started
matching detections to the wrong gate, and the position estimate followed the
wrong fixes to 8 m of error — at which point it declared crossings that never
happened. Every downstream symptom traced back here.

**Fixed**: `geometry.level_bearing_from_ahrs()` is now the only way to turn a
body-frame direction into a bearing. `_bearing_consistency_check()` asserts the
recovered world bearing is independent of bank, and also asserts that the naive
version *is* badly wrong, so nobody simplifies it back.

### 2. Roll/pitch used the wrong convention (caught by parity test)

The training code derives roll and pitch from **gravity in the body frame**, AHRS
style — not from a ZYX Euler decomposition. Roll agreed to 5e-16; **pitch was
sign-flipped**, disagreeing by up to 1.786 rad. The policy reads roll and pitch
in observation slots 24 and 25, so this told it the aircraft was leaning the
opposite way.

### 3. The same mismatch, again, in the filter's prediction

`pose_filter.predict` built the thrust direction inline with an FLU-Euler formula
from AHRS angles. That **mirrors the horizontal components** — vertical looks
fine, horizontal acceleration points the wrong way across the ground. Position
diverged 10 m in 1.4 s, with the controller amplifying it.

**Fixed centrally**: `geometry.body_up_from_ahrs()` is now the only way to get a
thrust direction from attitude. Never rebuild this inline.

### 4. The pitch *sign* in the control adapter

`AxisMap.sign_pitch` was `-1` based on incorrect reasoning. Body NED has x
forward, y right, z down; a pitch rate is a rotation about +y, which by the
right-hand rule carries +z toward +x, so the nose goes **up**. All three signs
now default to `+1`.

**Still UNVERIFIED against Betaflight** and the most dangerous unchecked
assumption in the package. `control_adapter.verify_axis_map()` prints a props-off
bench script.

### 5. Association could not tell two gates apart

The tracker accepted the detection nearest its prediction, inside a window that
**widens the longer the gate goes unseen** — so the more it struggled, the more
willing it became to accept the wrong gate. Two measured failures:

- **The double gate.** Its two openings are one structure 2.70 m apart
  vertically, which at 3.5 m range is 328 px — inside the widened 420 px window.
  Hunting for the lower opening, the tracker locked onto the upper one and the
  estimate stepped 2.70 m. The aircraft then flew over the lower opening
  believing it was lined up on it.
- **Gates 7 and 8**, 7.54 m apart and both in shot after the turn.

**Fixed**: `TrackerConfig.claim_margin`. A detection may only be claimed by the
expected gate if no *other* nearby gate explains it substantially better. Being
nearest to a prediction is not enough.

And `FilterConfig.gate_reject_m` was **3.0 m, justified by "the nearest gates are
7.54 m apart"** — a spacing that excluded the double gate's 2.70 m. The rejection
threshold was wider than the ambiguity it existed to resolve. Now 1.5 m.

### 6. Takeoff assumed it knew the hover throttle

The takeoff law could only ever command `hover_stick × 1.24`, and the filter's
thrust-scale estimate is gated on being airborne (above 0.40 m). An aircraft 15 %
down on thrust therefore **sat on the pad for 47 seconds and scored zero**: it
could not climb without a better thrust model, and could not learn one without
climbing.

Worse, the integrator was driven by the filter's *climb rate*, which is dead
reckoned through the very thrust model that is wrong. The filter believed it was
climbing at 1.7 m/s² while the aircraft sat still, so the error stayed near zero.

**Fixed**: the takeoff trim integrator is driven by **height**, which is pinned
by gate fixes and owes the thrust model nothing. And `Guidance.command` now
divides by the learned `thrust_scale` — without that, the filter learned the
aircraft was down on thrust, wrote the number down, and the controller carried on
commanding too little anyway.

Measured after the fix — takeoff succeeds across a **2.2× span of hover
throttle**, and the filter recovers the true scale to within 0.005:

| True thrust scale | Stick actually needed | Liftoff | Height at 8 s | Scale learned |
|---|---|---|---|---|
| 1.20 | 0.213 | 1.16 s | 1.37 m | 1.202 |
| 1.00 | 0.255 | 1.39 s | 1.41 m | 1.007 |
| 0.85 | 0.300 | 2.77 s | 1.74 m | 0.867 |
| 0.75 | 0.340 | 3.82 s | 2.69 m | 0.764 |
| 0.55 | 0.464 | 5.42 s | 1.11 m | 0.549 |

This matters because **the hover throttle is the single least-known number in the
model** and we do not get to measure it before the first flight.

### 7. A crossing had only one witness (added, not a bug fix)

The plane-crossing test asks "did my position estimate pass through the
opening" — which is self-assessed, and therefore silent about exactly the
failure it needs to catch. When the estimate had drifted 8.5 m it declared
crossings that never happened, each reported clean and under 0.15 m off centre.

Apparent size is an independent witness: to pass through a gate you must have
been right up against it, and how big it looks owes nothing to where you think
you are. `note_fix()` now records PnP's closest range to each target, and a
crossing is `corroborated` only if the gate was ranged within 2.5 m.
`scoring_crossings` requires both. A crossing with one witness is logged, not
scored.

`test_crossing_evidence.py` checks that it fires when it should, *does not* fire
on a normal approach, abstains when nobody reports ranges, and resets between
gates so a well-seen gate cannot vouch for the next one.

### 8. Two older ones, already fixed

- The estimator integrated **downwards through the floor** while the aircraft sat
  on the pad during the throttle ramp, arriving at liftoff tens of metres out. A
  drone cannot be underground; saying so is free.
- On the pad, the filter explained "thrust commanded, no motion" by deciding the
  **motors were weak** — it learned a thrust scale of 0.59 against a true 1.0.
  Ground reaction is not in the model, so thrust-scale learning is gated on being
  airborne. (This gate is what bug 6 then collided with; the takeoff integrator
  is the resolution, because it is bounded and explicit.)

---

## Results worth keeping

**Parity with the training code** — what the network sees in flight matches what
it saw in training:

| Check | Result |
|---|---|
| 51-number observation frame | **exactly 0** difference, 200 cases |
| 1632-number history stack | **exactly 0** difference |
| Projection to pixels | 3.9e-6 px, 0 visibility mismatches over 3200 corners |
| Roll and pitch | 4.4e-16 rad over 500 random attitudes |
| World bearing vs bank angle | **< 1e-9 rad** over 400 attitudes (was 11.8°) |

**PnP** — discard the rotation, keep the translation, use the FC's attitude:

| Noise | Range | Full 6-DOF PnP | PnP + FC attitude |
|---|---|---|---|
| 2 px | 8 m | 2.12 m | **0.09 m** |
| 5 px | 8 m | 3.15 m | **0.29 m** |
| 10 px | 12 m | 6.81 m | **1.38 m** |

**Gate tracking**: 22/22 crossings, zero bookkeeping errors, across 17 adverse
conditions including 50 % dropout, 20 false positives/s, 100 ms latency and 1 m
pose noise.

**The gate label matters** (Isaac, Gene's model, his own track, only the label
source changed): correct label 26.9 gates per crash and 515 laps; no label 1.1
and **zero laps**; off-by-one 1.0; frozen 1.8.

**Known limit**: heading drift. The filter corrects it from gate bearings, which
now works properly — but with no gate in view for a long time there is no
absolute heading reference at all.

---

## Where PnP stops, and why (measured today)

The old note here guessed that close-range PnP was returning a bad solution and
that the gravity tie-break was mispicking. **Both guesses were wrong.** Measured
over three closed-loop runs, by range to the expected gate:

| Range | Frames | Corners in frame | Solved | Reason for failure |
|---|---|---|---|---|
| 1.0–1.4 m | 296 | 2.4 | 0 | fewer than 4 corners |
| 1.4–1.8 m | 292 | 3.2 | 89 | fewer than 4 corners |
| 2.2–2.6 m | 346 | 4.6 | 287 | — |
| 7–10 m | 1342 | 7.8 | 1064 | — |

PnP is not failing. **The corners leave the frame**, and `solve_gate_pose`
correctly returns None. The gravity tie-break was picking correctly essentially
always (`gravity_score < 0` on 9 frames out of 4325).

The binding constraint is the **360-pixel frame height**, not the 640-pixel
width. With a 45.9° vertical field of view, a 2.7 m gate frame simply does not
fit inside about 3 m, whatever the tilt. This costs nothing: those gaps are
0.7–0.9 s long and the estimate goes into them at 0.15 m and comes out at 0.10 m.

**On camera tilt** — the old note suggested lowering it toward the other team's
0°. That is not supported. Measured corner counts at racing attitude (0.35 m
below the gate centre, 12° nose-down), our 20° tilt keeps all 8 corners in frame
from 3 m out, while 0° tilt holds only 4 corners from 3 m to 8 m. The nose-down
pitch of racing flight cancels the up-tilt. **Keep 20°.**

---

## Closed-loop status

`python test_closed_loop.py` — the whole stack flies the real course, given only
attitude, gyro, camera detections and the map. 22 crossings is a complete run.

| Case | This morning | Now | Position error |
|---|---|---|---|
| nominal | 9/22 | **22/22** | 0.15 m |
| 15 % thrust error | 0/22 | **22/22** | 0.22 m |
| slow rate loop (90 ms) | 9/22 | **22/22** | 0.23 m |
| 50 ms command latency | 9/22 | **22/22** | 0.12 m |
| 3° attitude noise | 9/22 | **22/22** | 0.16 m |
| faster: 7 / 4 m/s | 20/22 | **22/22** | 0.15 m |
| heavy drag | 9/22 | 21/22 | 0.11 m |
| poor detector | 8/22 | 6/22 | 0.45 m |
| everything at once | 0/22 | 10/22 | 0.43 m |

**6 of 9 configurations now complete all 22 crossings, against 0 of 9 this
morning.** Median position error fell from 0.71–2.39 m to 0.11–0.23 m.

Note "poor detector" went *down*, 8 → 6. That is not a regression in the sense
that matters: this morning's 8 included crossings declared from an estimate that
was metres wrong, and those are now correctly refused. The number got smaller
because it got honest. It is still a real failure and it is the top item below.

---

## What I would look at next, in order

1. **The remaining failures are the degraded-detector cases.** "Poor detector"
   (8 m range, 30 % dropout, 15 % corner dropout, 8 px noise) and "everything at
   once" still lose the count. Diagnose them the same way: instrument first.
   The diagnostic scripts used today are in the session scratchpad and are worth
   keeping — `diag_pnp_range.py`, `diag_gaps.py`, `diag_index.py`,
   `diag_assoc.py`. The last one is the most valuable: it tags each simulated
   detection with the gate that produced it, so a mis-association is visible
   directly instead of being inferred from its consequences.
2. **Give the double gate a dedicated manoeuvre.** It works now, but by a
   general mechanism rather than a designed one. Backing off along −through
   while descending, holding yaw on the gate, would be more deliberate.
3. **Verify the axis signs on real hardware.** Still the most dangerous unknown.

---

## Things to be careful of

- **`PYTHONPATH` matters for Isaac runs.** Without `PYTHONPATH=$HOME/aigp/idr_exp`
  the evaluation imports the repo copy, which still has the gate-counting bug,
  and results collapse by 43× while looking like a model failure. Run from
  `~/aigp/idr_exp`.
- **`pkill -f <pattern>` matches this shell's own command line.** It has killed
  my own process twice. Use PID loops.
- **Betaflight curve parameters and axis signs come from another team's drone.**
  Replace from our own `diff all`. Use `RateCurve.fit_from_samples()` once there
  is a blackbox log.
- **Measure before changing anything.** Every one of today's fixes came from an
  instrumented run contradicting a plausible-sounding theory. The theory in the
  previous version of this file — that close-range PnP was returning bad
  solutions — was wrong, and acting on it would have wasted the day.
- The `flight/` code has **never touched hardware**. Nothing here is validated
  against a real drone, a real camera or a real flight controller.

---

## Immediate on-site actions this unblocks

1. `scp -r dcl@192.168.55.1:~/target ./` — the organizers' MSP library and their
   `fake_fc.py`. Everything in `control_adapter` can then be tested end-to-end.
2. Props-off bench sweep with `control_adapter.verify_axis_map()` — settles the
   sign question.
3. `diff all` per airframe — replaces the borrowed Betaflight parameters.
4. Hover and read the throttle. **Less urgent than it was**: takeoff now works
   across a 2.2× span of hover throttle and learns the true value in flight. Still
   worth measuring, but it is no longer a single point of failure.
5. `companion_listener_msp.py --listen-only` — the achievable loop rate, which
   sets the simulator's policy rate.
6. `v4l2-ctl -d /dev/video0 --list-formats-ext` — which camera modes exist.
7. **Do not change the camera tilt from 20°** without re-running the visibility
   sweep. See the section above.
