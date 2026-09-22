# The observation contract

The policy is trained in Isaac against `utils/aigp_obs.py` (torch, 4096 envs
at once) and flown on the Orin against the flight repo's `race_obs.py` (numpy,
one aircraft). Those are two implementations of one vector. If they disagree
by a channel order, a clip or a focal length, nothing raises: the policy
receives numbers that mean something slightly different from what it trained
on and flies confidently into a gate. So `contract/` defines the vector once,
with no dependencies, and `contract/verify.py` hashes every value both ends
must agree on.

## One frame

Indices within a frame. Every value is a float.

| index | channels | meaning |
|---|---|---|
| 0–15 | `u0 v0 … u7 v7` | the eight gate corners in the 640×360 frame, normalised to [0, 1]. Outer ring ids 0–3 then inner ring 4–7, each clockwise from top-left *as seen*: viewed against the through-axis the left/right order is swapped, so the ids still name the corner the detector would label. `NOT_SEEN = −1.0` when off-frame (a corner may sit up to `OFF_FRAME_MARGIN` = 0.15 of the frame outside it and still count as seen) |
| 16–23 | `vis0 … vis7` | 1.0 seen, 0.0 not |
| 24–25 | `roll pitch` | gravity-referenced, NED |
| 26–28 | `gx gy gz` | NED body rates, clipped to ±`GYRO_CLIP` = 8 rad/s |
| 29–31 | `vx vy vz` | **commanded** body velocity in FRD, clipped ±20 m/s: integrated from commanded thrust, attitude and a linear drag model `k = (−0.50, −0.50, −0.15)` s⁻¹, never IMU acceleration and never the simulator's true twist |
| 32–49 | `gate0 … gate17` | one-hot of which gate is next (`N_GATES` = 18 slots) |
| 50 | `gate_frac` | next-gate index / 17 |
| 51–54 | `a_thrust a_roll_rate a_pitch_rate a_yaw_rate` | **v2 only**: the action the policy *issued* on the previous step, clamped, before any delay |

* **v1** = 51 channels × 32 frames = **1632** (1639 with `GATE_LOOKAHEAD=1`,
  which appends the next gate's pose in the current gate's frame).
* **v2** = 55 × 32 = **1760**. A strict append: a v1 reader sees exactly v1 in
  the first 51 channels.

Why v2 exists: with no latency a policy can infer what it commanded from what
happened next. With ~40 ms of real delay it cannot; it commands roll, sees
nothing for two or three steps, commands more, and the accumulated commands
land together. Eschmann's ablation: trajectory tracking went from 10/10 to
0/10 when action history was removed with delay still simulated. The action
fed back is the *issued* one, not the delayed one, because the aircraft knows
what it sent the instant it sends it; feeding back the delayed command would
hand the policy the delay as free information.

`HISTORY` = 32 frames. At 60 Hz that is 0.53 s; at the runner's 40 Hz it is
0.8 s, which is one of the reasons a 60 Hz policy degrades at 40
(`docs/TRAINING_RESULTS.md`).

## Timing model

* **Camera latch.** Keypoints update at 30 Hz and are held between frames
  (`CameraFrameLatch`, remainder-carrying so 100/30 gives a 3,3,4 cadence and
  60/30 an exact stride of 2). Attitude, rates and commanded velocity update
  every control step.
* **Vision delay.** The latched keypoints are then delayed 1–4 control steps
  per env (`KeypointDelayLine`, backfilled with the current frame on reset).
  Set `OBS_VERSION=v1` to switch this off together with the action channels.

## Detector augmentation (opt-in)

Corners in simulation are projected analytically and always present. On the
aircraft they are not: measured at a real gate over 1345 frames, a corner's
visibility flips between consecutive frames **4.2%** of the time and about
**13%** of the corners the geometry offers are missed.

`_keypoint_dropout` gives each in-frame corner a two-state Markov chain per
env with `p(visible → dropped) = (1−ρ)·q` and `p(dropped → visible) =
(1−ρ)(1−q)`, steady-state dropped fraction `q`, flip rate `2q(1−q)(1−ρ)`.
`AIGP_KP_DROP=0.13`, `AIGP_KP_STICKY=0.81` reproduce the measurements
(verified 0.131 and 0.0430 against 0.13 and 0.042). The clustering is the
point: independent draws at the same rate would flip 32% of the time and leave
the runner (which needs four corners and clears its history on a bad frame)
ready 0.5% of the time against a measured 58%. Dropped corners carry
`NOT_SEEN`, not 0: 0 is a valid position (top-left) and writing it collapsed
racing from 12.80 to 0.13 gates per episode.

`_keypoint_jitter` (`AIGP_KP_JITTER_PX`) adds Gaussian error per corner per
axis after dropout. Its magnitude is **not calibrated** and defaults to 0; the
measurement that would set it is corner displacement on a static gate from the
same 1345-frame set.

## The hash

`verify.contract_fields(version)` collects: the channel list, `HISTORY`, the
frame and flat dims, sentinels and clips, `N_GATES`, frame size, fx/fy/cx/cy
(rounded to 9 dp), distortion (12 dp), camera tilt, gate sizes, thrust
min/hover/max, rate limit, action names. JSON with sorted keys, SHA-256;
`short_hash()` is the first 12 hex digits. Current values:

| version | short hash | flat |
|---|---|---|
| v1 | `efe3550e9012` | 1632 |
| v2 | `a20c14d6a335` | 1760 |

Randomised plant parameters (hover range, TWR range, inertia, mass) are
deliberately **not** in the hash; a test asserts it. The policy is meant to see
a spread of those, and hashing them would make every run incompatible with
every other.

`train.py` writes `verify.stamp()` to `<run>/contract.json`; `play.py` calls
`verify.check_run_dir()` and refuses a checkpoint whose stamp disagrees with
the running code. The same `contract/` directory is vendored wholesale into
the flight repo (never edited in one copy) so the Orin can check the hash
without this repo checked out.

## Environment variables

| variable | read by | effect |
|---|---|---|
| `OBS_VERSION` | env cfg, `verify.version_from_env` | `v2` (default) or `v1`; v1 also switches the vision delay off |
| `GATE_LOOKAHEAD` | env cfg | `1` appends the 7-channel next-gate pose; changes width, so a checkpoint only loads under the setting it trained with |
| `SEPARATE_NETS` | `train.py`, `play.py` | `1` uses separate policy/value trunks (measured worse; see `TRAINING_RESULTS.md`) |
| `AIGP_POLICY_HZ` | racing cfg | policy rate; physics stays 120 Hz, decimation must come out whole |
| `AIGP_HOVER_HZ` | hover cfg | same, hover only, applied after the racing one |
| `AIGP_KP_DROP` | observations | corner dropout steady-state fraction (measured 0.13) |
| `AIGP_KP_STICKY` | observations | dropout stickiness ρ (measured 0.81) |
| `AIGP_KP_JITTER_PX` | observations | corner position noise in 640×360 px; uncalibrated, default 0 |
| `VISUAL_GATE_COUNTER` | `play.py`, `utils/gate_counter.py` | `1` drives the one-hot from the onboard visual lock instead of the plane-crossing check; needs the flight repo (`AIGP_FLIGHT_REPO`) |
| `AIGP_FLIGHT_REPO` | `utils/flight_repo.py` | where `ai-grand-prix` is checked out, for the counter and the cross-repo tests |
| `PLAY_START_OFFICIAL` | play cfg | official gate number to start in front of (default 1) |
| `PLAY_SPEED_CAP_MPS` | `play.py` | post-policy brake in play; weights unchanged |
| `ENABLE_CAMERAS`, `OVERHEAD_LIGHTS`, `OVERHEAD_HEIGHT`, `OVERHEAD_INTENSITY`, `DOME_INTENSITY`, `GATE_ROOT` | scene cfg | rendering only |
