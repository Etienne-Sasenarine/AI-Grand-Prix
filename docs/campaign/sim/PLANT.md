# The plant: what the simulated aircraft is

Everything here is read from the code that builds the aircraft:
`contract/plant.py` (the constants), `dynamics/rate_control.py` (thrust curve,
rate loop, derived gains), `tasks/drone_racer/mdp/actions.py` (delay, filter,
per-episode randomisation) and `tasks/drone_racer/mdp/events.py` (mass and
inertia written into PhysX at startup). Where a number is estimated rather
than measured, it says so. Nothing here is the 0.6 kg / 5-inch / fx=320
aircraft the upstream project modelled; those documents were removed.

**The aircraft is not a 5-inch quadcopter.** It is the organizer-supplied
Neros Archer B2: 8-inch props at 4.1 pitch, Betaflight 4.4.3 on an H743 flight
controller, a Jetson Orin NX 16 GB, a 6S pack, **1745 g all-up race weight**
measured on site on 17 Sep 2026. The asset under `assets/5_in_drone/` and its
0.6076 kg URDF are inherited from the upstream `isaac_drone_racer` project and
are used for geometry and joints only; `events.set_body_mass` and
`events.set_body_inertia` overwrite mass and inertia in PhysX at startup with
the values below, and the thrust curve is sized for this airframe. What makes
the simulated aircraft *this* aircraft is those overrides, not the mesh.

## Two halves, and why they are kept apart

**Contract**: what the flight client assumes and what the wire carries. One
hover constant, one rate limit, one straight-line thrust map. The commanded
velocity channels of the observation are derived from these, so they are part
of the observation contract and are hashed (`docs/OBSERVATION.md`).

**Plant**: what the aircraft actually is. Fixed where it is known, randomised
per episode where it genuinely varies. Not hashed: the policy is supposed to
see a spread. The client cannot know which airframe, pack or set of props it
is flying, so the simulator holds the client's constants still and moves the
aircraft underneath them.

## Mass

| | value | status |
|---|---|---|
| all-up mass | **1.745 kg**, props included | scale reading, confirmed with the person who weighed it |
| one propeller | 0.005 kg | the code's figure, a 5-inch tri-blade weighed as a class; the real 8-inch props are heavier. Body mass is derived from the total, so the error moves mass between links without changing what the aircraft weighs |
| airframe link | 1.745 − 4 × 0.005 = 1.725 kg | derived |

Mass is written at startup by `events.set_body_mass`, per link. The spawn
config deliberately sets **no** `mass_props`: `MassPropertiesCfg(mass=1.745)`
applies to every rigid body in the articulation, and the USD has five (body
plus four spinning props), so the first live run flew an 8.725 kg aircraft
against thrust sized for 1.745 kg, a thrust-to-weight of 0.2. A test parses
the asset config to make sure nobody puts it back.

Mass is the one plant parameter that is not randomised. SimpleFlight measured a
30% mass error as unrecoverable on hardware even with domain randomisation, and
randomising a well-calibrated mass mostly adds learning difficulty.

## Inertia

| axis | kg·m² | status |
|---|---|---|
| roll | 0.0040 | **estimated**: four arm assemblies at 45° plus a central box (battery, Orin, plates); plausible mass distributions span 0.0034–0.0050 |
| pitch | 0.0055 | estimated; a racing stack is longer front-to-back than wide |
| yaw | 0.0075 | estimated |

The URDF inherited (0.003, 0.003, 0.006) from a 0.5 kg aircraft; those were
replaced. Props get (3.4e-6, 3.4e-6, 6.7e-6) as thin planar bodies of 0.127 m
span. A `scale_range` hook exists for randomising inertia and is off.

**The COM-quaternion finding.** PhysX does not store a 3×3 inertia tensor. It
stores three principal moments plus the rotation from body axes to principal
axes, and that rotation lives in the body's COM pose. `set_inertias` writes the
moments and leaves the rotation alone. This asset's body link carried a 7.16°
tilt about y from the USD conversion, so the diagonal we wrote read back as
(0.004054, 0.0055, 0.007446) with 4.32e-4 in the products, coupling roll into
yaw on every input. `set_body_inertia` now also resets the COM quaternion to
identity, in PhysX's **xyzw** convention (Isaac Lab's utilities are wxyz;
writing wxyz identity here stores a 180° rotation about x, which maps a
diagonal tensor to itself and passes every obvious check while the body is
upside down in its own mass frame). A test asserts the real part is in the
last slot.

## Thrust

The action's first channel is a dimensionless throttle stick. Zero policy
action is `THRUST_HOVER`.

| constant | value | note |
|---|---|---|
| `THRUST_MIN` | 0.05 | |
| `THRUST_HOVER` | 0.225 | was 0.255; radio stick-hover runs on the aircraft put it at 0.21–0.24. Must equal the client's `HOVER_THRUST` |
| `THRUST_MAX` | 0.90 | |

**What the client assumes**: a straight line, 1 g at hover stick, pinned there.

**What the plant does**, per episode:

* `plant_hover ~ U(0.21, 0.29)`: where this aircraft actually hovers. The top
  of the range is a drained pack, not a different aircraft: thrust goes as V²
  at fixed duty, so a 6S pack falling 4.2 → 3.5 V/cell loses ~31% of thrust and
  the hover stick climbs to ~0.28.
* `plant_twr ~ U(4.29, 8.0)`: thrust-to-weight at full stick. The real chain is
  throttle → duty → RPM → thrust; thrust goes as RPM² but duty→RPM is markedly
  sublinear, and four 6S thrust-stand datasets agree the composite is about
  stick^1.6 (quarter stick ≈ 12% of max, half ≈ 35%, three-quarters ≈ 68%).
  Pooled, the fraction of full-stick thrust carried by the s² term is
  `q ≈ 0.68`. The floor 4.29 = 0.90 / 0.21 is derived: any curve weaker than
  the straight line through its own hover point is concave, which no motor
  produces. The ceiling is what the thrust-stand shape (measured on 5-inch
  hardware, the only data available) implies. This
  airframe has never been on a thrust stand, so training spans the range.
* `stick_to_newtons(s) = (q·s² + (1−q)·s) / (q·h² + (1−q)·h) · m·g`, pinned to
  1 g at the episode's hover stick `h`, with `q` derived from the sampled TWR.

Two thrust values exist per step on purpose: the commanded one, which the
observation's velocity integrator uses with the client's straight line, and the
applied one, which the plant uses with its own curve.

## Rates

The other three channels are NED body rates, ±`RATE_LIMIT` = **3.2 rad/s** at
action ±1 (matches the flight client's `ACTION_RANGES`; a test compares them).

A PD rate loop at the 120 Hz physics rate turns the setpoint into a body
moment: `m = kp·(ω_des − ω) − kd·ω`, clamped to a per-axis moment limit.

The gains are **derived from the inertia**, not written beside it. From
`tau = I / (kp + kd)` and `kd = ratio · kp`:

| | roll | pitch | yaw |
|---|---|---|---|
| target time constant `tau` | 0.036 s | 0.036 s | 0.072 s |
| `kd / kp` | 0.0375 | 0.0375 | 0.0375 |
| peak angular acceleration `alpha_max` | 100 rad/s² | 100 rad/s² | 33.3 rad/s² |
| moment limit = I · alpha_max | 0.40 N·m | 0.55 N·m | 0.25 N·m |

The targets restore the response the inherited gains (kp 0.08, kd 0.003,
moments 0.30/0.30/0.20) had on the 0.5 kg aircraft. Left as pinned numbers on
the corrected inertia they gave pitch a 66 ms time constant (+84%), and the
first corrected-plant run lost 73% of its episodes to gate strikes.

Two honest limits. Physics at 120 Hz (dt 8.3 ms) cannot represent a time
constant much below ~25 ms, and a real Betaflight rate loop on this class of
aircraft settles in single-digit milliseconds. The simulated aircraft is
therefore slower to respond than the real one, in the direction that makes the
policy more conservative. Closing that gap needs a props-off bench measurement:
step a rate command, log the gyro, fit the time constant.

## Latency

Per episode, drawn uniform:

| | range | models |
|---|---|---|
| action delay | 0–2 policy steps (0–33 ms at 60 Hz) | Jetson → UART → Betaflight command path. Ring buffer per env, cleared on reset |
| rate setpoint lag `tau` | 0.008–0.045 s, first order | `rc_smoothing_setpoint_cutoff` = 15 Hz measured on our FC (~10.6 ms) plus motor lag |
| vision delay | 1–4 control steps (17–67 ms) | sensor readout, copy to the Orin, YOLO-pose inference; see `docs/OBSERVATION.md` |

The lag is discretised exactly as `1 − exp(−dt/tau)`; forward Euler was 44%
off at dt/tau ≈ 0.8. Why it is here at all: 50 ms of delay took our own stress
runs from 2.6 to 55 crashes per 100 gates; Sun et al. report 0 / 2.7 / 68%
failure at 10 / 30 / 50 ms; Agilicious measure ~40 ms Betaflight
command-to-actuation.

## Disturbances

`push_robot`: every 0–0.2 s, a random force in ±0.1 N and torque in ±0.05 N·m.
Off in play.

## Camera

On-site ChArUco calibration at 1920×1080 (fx 1302.94, fy 1303.11, cx 952.29,
cy 529.04, five Brown-Conrady coefficients), scaled to the policy's 640×360
frame: **fx ≈ 434.3**, 73.6° horizontal field of view, tilted **20° up**. The
rendered Isaac camera (aperture 17.684 × 9.946, focal 12) agrees with the
analytic projector to 0.05 px. Earlier checkpoints trained at fx = 320 (the
virtual qualifier's 90° camera) scored 42 crashes per 100 gates on the real
focal length against 2.6 on their own.

## What is still not modelled

* **No motor model.** `dynamics/motor.py` and `allocation.py` were removed
  unused; thrust and moments are applied directly. `use_motor_model` remains
  in the cfg only so old Hydra overrides parse.
* **Inertia is estimated and not randomised.**
* **Detector error is only partly modelled.** Corners drop out with a
  calibrated Markov chain; a corner that is present but *misplaced* is
  `AIGP_KP_JITTER_PX`, whose magnitude is uncalibrated and defaults to 0.
* **Ground effect, battery sag within an episode, and aerodynamic drag on the
  airframe** are absent (drag appears only inside the commanded-velocity
  integrator, which is an observation, not a force).
* **The rate loop is slower than Betaflight's**, see above.
