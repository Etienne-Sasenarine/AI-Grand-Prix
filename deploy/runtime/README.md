# `flight/` — the onboard code

Pure NumPy. **No PyTorch, no framework**, because the Jetson image the
organizers ship has none and installing one there is a multi-hour job that
breaks on venue Wi-Fi. Run everything with:

```
python run_all.py
```

| Module | What it does | Runs on the drone |
|---|---|---|
| `policy_runtime.py` | Exports an skrl checkpoint to `.npz` and runs it as four matrix multiplies | ✅ (export side is laptop-only) |
| `geometry.py` | Course map, gate corners, camera projection, pass tests | ✅ |
| `observation.py` | The 51-number frame and the 32-frame history | ✅ |
| `gate_tracker.py` | Which gate is next, and when we went through it | ✅ |
| `betaflight_curves.py` | Physical units ↔ Betaflight stick values | ✅ |
| `control_adapter.py` | Action → safety envelope → curves → channel values | ✅ |
| `sim.py` | Synthetic flight and detector, for testing the above | laptop |
| `test_*.py`, `run_all.py` | The test suites | laptop |

---

## What is verified, and how

**The policy runs identically without PyTorch.** The exporter rebuilds the
network the way skrl instantiates it and compares, so an activation silently
missing from the last hidden layer cannot slip through. On the real
`pq_speed_best` checkpoint the NumPy path agrees with PyTorch to
**2.5 × 10⁻⁶**, at 126 µs per inference.

It also refuses to export a checkpoint with no `state_preprocessor`. The network
was trained on **normalised** observations; skipping that normalisation doesn't
crash anything, it just produces a policy that flies like it has never seen the
course.

**What the network sees in flight matches training.** `test_parity.py` checks our
NumPy geometry against the actual torch code the policy was trained with:

| Check | Result |
|---|---|
| Gate corners in world | float32 epsilon (the training table is float32) |
| Projection to pixels | **3.9 × 10⁻⁶ px**, 0 visibility mismatches over 3 200 corners |
| 51-number frame | **exactly 0** difference, over 200 random cases |
| 1 632-number history stack | **exactly 0** difference |

**The gate tracker counts correctly under abuse.** 17 conditions × 5 seeds, on
the real course, two laps, 22 crossings each (gate 9 counts twice per lap):

| Condition | Crossings | Errors | Lag | Confident |
|---|---|---|---|---|
| Reference | 22 | 0 | 0 | 97 % |
| 50 % detection dropout | 22 | 0 | 0 | 97 % |
| 20 false positives/s | 22 | 0 | 0 | 97 % |
| 8 px corner noise | 22 | 0 | 0 | 97 % |
| 6 m detector range | 22 | 0 | 0 | 78 % |
| 100 ms detection latency | 22 | 0 | 0 | 96 % |
| 1.0 m pose noise | 22 | 0 | 1 | 97 % |
| 9 m/s flight | 22 | 0 | 0 | 99 % |
| Everything at once | 22 | 0 | 1 | 96 % |

**The whole pipeline holds together.** A full two-lap run through camera →
detector → tracker → observation → history → policy → envelope → curves →
channels: 3 142 frames, 0 NaNs, 0 out-of-range channel values, 0 envelope
breaches, 22/22 crossings, using **9 % of the 60 Hz budget** on this laptop.

---

## What is **not** verified — read before flying

**A pose is not optional.** Measured on this course: **34 % of frames have two
or more gates in view at once.** Without a pose estimate there is no principled
way to know which is the target, and the vision-only fallback was measured
drifting **13 crossings ahead of reality while still declaring a plausible 22
passes** — confident and wrong, the worst failure mode there is.

So that path is a *degraded* mode, not an alternative. It keeps a best-effort
count for the log, but `tracker.confident` is False throughout and the flight
loop must hold rather than race on a guessed index. The normal path has a pose:
the detector gives eight corners, the gate geometry is known, so PnP gives
gate-relative pose, and the surveyed map turns that into a world pose.

**The Betaflight forward curves are unverified.** The *inverse* is a bisection
over a monotonic function, so it is correct for whatever forward model you give
it. The forward model itself is written from the 4.x source as understood, with
parameters read from **another team's** drone. Two ways to settle it: compare
`RateCurve.max_rate_deg_s()` against Betaflight Configurator, or log
stick/setpoint pairs and call `RateCurve.fit_from_samples()`, which replaces the
analytic model with a measured one.

**The axis and sign conventions are unverified and are the most dangerous thing
here.** Isaac is forward-left-up, Betaflight is forward-right-down, and channel
order depends on the receiver map. A reversed sign flips the drone on takeoff.
`control_adapter.verify_axis_map()` prints an eight-command props-off bench
script. Run it, with two people, before anything spins with props on.

**The transmitter seam is a guess.** The organizers' `RCTransmitter.set_control()`
signature is not in their written guide, so `ControlAdapter` takes a `send`
callable instead of calling a method. When `~/target/` is in hand, adapt it in
exactly one place.

**The detector is not written.** `sim.py` models one — dropouts, corner noise
growing with range, false positives, latency, limited range — but a model of a
detector is not a detector.

**The end-to-end test measures plumbing, not flying.** The simulated path is
kinematic and ignores what the policy asks for, so the policy sees a trajectory
it did not command. It saturates its outputs **86 %** of the time in that test.
That is expected open-loop and is *not* evidence the controller is sound — but it
is worth re-checking once the loop is closed, because it is also what an
out-of-distribution policy looks like.

---

## Still to build

1. **Altitude without a barometer.** Another team measured the barometer
   dropping ~6 m within 6 s of spin-up and staying there; an accelerometer
   cannot substitute (vibration rectification, 0.985 g at throttle 0.18). Plan B's
   altitude estimator was built on a barometer and needs replacing with the
   vision-plus-thrust-bias path.
2. **Takeoff.** Gyro-spread liftoff detection does not work on this airframe —
   measured firing on the pad at one threshold and never firing at another,
   with the drone lifting and running away. Takeoff must not depend on detecting
   liftoff.
3. **The flight loop**: state machine, watchdog wiring, logging, abort.
4. **The gate detector itself**, and the PnP that turns its corners into a pose.
