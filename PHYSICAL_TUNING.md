# Physical-drone constants → Isaac Sim

Offline calculator for all phases: [tools/physical_constants.py](tools/physical_constants.py).
See [input schemas and usage](tools/PHYSICAL_CONSTANTS.md) to generate a JSON
worksheet and a report of measured candidate constants with destination files.
It does not arm the aircraft or edit configuration. Unmeasured values stay pending.

**Branch correction:** the camera code currently uses FX=423.6 and FY=422.3
in both `utils/aigp_obs.py` and `AI_GP/camera_model.py`. The 320 px / 296 px
camera examples below are historical; at FY=422.3, a same-height gate center
at 20° tilt is approximately 333.7 px. Also, the implemented `rate_kd` is
rate damping: for an unsaturated isolated axis, increasing it lowers the time
constant AND steady gain. Use the calculator's implemented-model fit rather
than interpreting it as derivative-of-error damping.

The Isaac plant is **not** motors, ESCs, KV, or battery sag. It is a rigid
body plus collective thrust plus a 120 Hz rate PD. The policy outputs the same
four numbers the real stack sends on `SET_ATTITUDE_TARGET`: thrust stick and
NED body rates.

Measure the bird, write the numbers into **both** places that share a
constant, then decide whether the keeper policy can fly as-is or needs one
warm-start. Do not train a zoo of guessed variants.

Current keeper: `models/pq_speed_best.pt`.

---

## What you are identifying

| Bucket | Goes into | Retrain? |
|---|---|---|
| **A. Scale** — mass, hover stick, thrust rails | observation `T/m` and stick→Newtons | No. Patch runtime + Isaac, fly the keeper. |
| **B. Rate envelope** — what ±1 actually means | action decode + FC limit | Only if the real rail is far from 3.2 rad/s. |
| **C. Rate plant** — how fast rates track | PD + inertia + moment limits | Only if the bird is mushy or twitchy vs Isaac. |
| **D. Drag** — `k ⊙ v` in the velocity integrator | commanded `vx,vy,vz` | Optional. Fit from a straight flight, then one continue. |
| **E. Camera** — tilt, intrinsics | keypoint projection | Only if the real image is not 640×360 / 20° / fx=fy=320. |
| **F. Signs** — roll / pitch / yaw command polarity | AI_GP wire only | Never an Isaac number. Wrong sign = instant crash. |

Leave KV, ESC, props, and `use_motor_model` alone. That model is off.

---

## Current Isaac / AI_GP values

Fill the **measured** column on the pad. Do not edit Isaac until the
checklist for that row is done.

| Constant | Isaac now | AI_GP now | File(s) | Measured |
|---|---|---|---|---|
| `mass_kg` | **0.6076** | (same via `T/m`) | `dynamics/rate_control.py` `DEFAULT_MASS_KG`; `mdp/actions.py` `ControlActionCfg.mass_kg` | |
| URDF mass | **0.5** (wrong) | — | `assets/5_in_drone/urdf/5_in_drone.urdf` | must match `mass_kg` |
| `THRUST_HOVER` | **0.225** | `HOVER_THRUST=0.255` **(stale)** | `rate_control.py`; `AI_GP/config.py` | **0.21-0.24**, John on the sticks |
| `PLANT_HOVER_RANGE` | **(0.21, 0.24)** | n/a (one baked constant) | `rate_control.py`; `mdp/actions.py` `plant_hover_range` | same runs |
| `THRUST_MIN` | **0.05** | `MIN_THRUST=0.05` | same | |
| `THRUST_MAX` | **0.90** | `MAX_THRUST=0.90` | same | |
| `RATE_LIMIT` | **3.2 rad/s** | `race_obs.ACTION_RANGES` ±3.2 | `rate_control.py`; `AI_GP/race_obs.py` | |
| `rate_kp` | **(0.08, 0.08, 0.08)** | (FC does this) | `mdp/actions.py` | |
| `rate_kd` | **(0.003, 0.003, 0.003)** | (FC does this) | same | |
| `moment_limit` | **(0.30, 0.30, 0.20) N·m** | — | same | |
| `Ixx, Iyy, Izz` | **0.003, 0.003, 0.006** | — | URDF `<inertia>` | |
| drag `k` FRD | **(-0.50, -0.50, -0.15)** | `DRAG_KX/Y/Z` | `utils/aigp_obs.py` `DEFAULT_DRAG_BODY`; `AI_GP/config.py` | |
| camera | 640×360, fx=fy=320, tilt **20°** | `camera_model.py` / spec | `utils/aigp_obs.py` | |
| policy / phys Hz | **60 / 120** | `CONTROL_HZ=99` | env cfg `decimation=2`, `sim.dt=1/120` | log the real loop |
| `RATE_SIGN_*` | Isaac uses `flu_to_ned` | pitch **−1**, roll **+1**, yaw **+1** | `AI_GP/config.py` | |

The URDF mass of 0.5 kg and the thrust map of 0.6076 kg **already disagree**.
The scale reading wins. Set both to the same number.

---

## Order of operations

Work the phases in order. Later phases assume earlier numbers are frozen.
Log every flight (`AI_GP` CSV + `--panel` if you can). Write the number on
the table *before* you change a file.

### Pad kit

- Scale (0.01 kg).
- Tape measure or laser (camera height, lever arm).
- Fully charged pack you will race.
- Props you will race. Do not swap mid-ID.
- `HIGHRES_IMU` + command log at ≥100 Hz if the FC will give it.
- A clear 10–15 m straight (hall or outdoor).
- One person on a kill / disarm.

Safety: first flights are hover and rate steps, not a lap. Stay below 2 m
until Phase 1–3 are signed off.

---

## Phase 0 — signs (2 minutes, before anything else)

Wrong polarity makes every later number garbage.

1. Props off. Arm. Command a **small +roll rate**. Gyro `x` (body) must
   increase the same way Isaac’s NED roll does after `flu_to_ned`.
2. Repeat **+pitch** and **+yaw**.
3. If an axis is inverted, fix `RATE_SIGN_ROLL` / `PITCH` / `YAW` in
   `AI_GP/config.py`. Do **not** flip Isaac inertia or PD to paper over a
   sign error.

Pass: three short hops, each axis rotates the way the stick says.

---

## Phase 1 — mass (2 minutes)

1. Weigh the **race-ready** airframe: frame, FC, camera, antennas, props,
   and the pack you will fly.
2. Write `mass_kg = <scale>`.

**Isaac**

- `dynamics/rate_control.py` → `DEFAULT_MASS_KG`
- `ControlActionCfg.mass_kg` (same number)
- URDF `<mass value="..."/>` — **same number**, not 0.5

**AI_GP**

- No separate mass knob. Hover stick below sets `T/m = (stick/hover)·g`.
  The integrator uses that, not a mass field. Mass only matters so hover
  Newtons in Isaac equal `m·g`.

Do not retrain.

---

## Phase 2 — hover stick (10 minutes)

This is the 1 g calibration. The observation and the plant both use

```
T/m = (thrust_stick / hover_stick) · g
Newtons = (thrust_stick / hover_stick) · m · g
```

1. Props on. Level pad. Arm.
2. Raise collective slowly until altitude is **constant** (baro / visual /
   MoCap — not “feels hover”).
3. Average stick for 3–5 s while rates < 20 °/s and tilt < 8°.
   AI_GP already has this window in `ekf/commanded_accel.py`
   (`hover_is_observable`). Prefer the logged hover-trim observer
   (`EKF_HOVER_TRIM_TAU_S`) over a eyeball.
4. Repeat three times. Use the median.

That number is `THRUST_HOVER` / `HOVER_THRUST`.

**Also record the rails**

- Lowest stick that still produces noticeable thrust → `THRUST_MIN`
  (keep 0.05 unless the FC deadzone is clearly higher).
- Highest stick the FC will actually send / that does not brown out on a
  punch → `THRUST_MAX` (keep 0.90 unless the FC clips earlier).

**Isaac:** `THRUST_HOVER`, `THRUST_MIN`, `THRUST_MAX` in
`dynamics/rate_control.py` and `ControlActionCfg`.

**AI_GP:** `HOVER_THRUST`, `MIN_THRUST`, `MAX_THRUST` in `config.py`.

Do not retrain. Fly the keeper with the new hover. If it systematically
climbs or sinks in a level hover, you measured wrong — do not “fix” it
with PD.

---

## Phase 3 — rate limit (15 minutes)

Isaac maps policy action ±1 to `±RATE_LIMIT` (3.2 rad/s ≈ 183 °/s).
The real FC may clip lower.

For each axis (roll, then pitch, then yaw):

1. Hover. Command a **known** rate: 1.0, 2.0, then 3.2 rad/s, for ~0.4 s
   each, then back to 0. Leave 2 s between steps.
2. From the gyro, record peak rate and 10–90 % rise time.

**Read**

| What you see | What to write |
|---|---|
| Peak ≈ command up to 3.2 | keep `RATE_LIMIT = 3.2` |
| Peak clips at `ω_max` < 3.0 | set `RATE_LIMIT = ω_max` in Isaac **and** `ACTION_RANGES` in `race_obs.py` |
| Peak overshoots then settles | keep the limit; Phase 4 owns the overshoot |

If you change `RATE_LIMIT` by more than ~0.3 rad/s, **warm-start**
`pq_speed_best` overnight (see “When to retrain”). The policy’s flicks
were learned against 3.2.

---

## Phase 4 — rate plant: rise time, inertia, PD (30–40 minutes)

Isaac’s inner loop is

```
ω_des_flu = ned_to_flu(ω_des_ned)
m = rate_kp ⊙ (ω_des − ω) − rate_kd ⊙ ω
m = clamp(m, ±moment_limit)
```

You cannot get a clean inertia tensor from a hover. You **can** match the
closed-loop rate step.

1. Use the same 2.0 rad/s steps from Phase 3 (below saturation).
2. Measure 10–90 % rise time `t_r` and overshoot on each axis.
3. In Isaac play (or a 1-env diagnose), command the same step and compare.

**Fit, in this order**

1. **`rate_kp` / `rate_kd`** — if the bird is slower than Isaac, lower `kp`
   or raise `kd` until rise time matches. If it rings, raise `kd`.
2. **`moment_limit`** — if the real step saturates (rate slope goes
   linear, then stops rising), lower the Isaac clamps until the Isaac
   slope matches. Yaw is already 0.20 vs 0.30 on roll/pitch.
3. **Inertia** — only if PD + clamps still cannot match and you have a
   physical estimate:
   - Bifilar pendulum, or
   - `I ≈ τ / α` from a known moment and measured angular accel on a
     short unsaturated pulse.

   Write `ixx`, `iyy`, `izz` in the URDF. Keep `ixx ≈ iyy` unless the
   airframe is obviously asymmetric. `izz` is usually ~2× `ixx` on a
   5" (current 0.006 vs 0.003).

**Do not** spend a session on this if Phase 3 said the envelope is 3.2
and the steps look “crisp enough.” A 20 % inertia error is smaller than
a 10 % hover error.

Retrain only if you changed `kp`/`kd`/`I` enough that a play of the
keeper in Isaac with the **new** plant looks drunk. Then one warm-start.

---

## Phase 5 — drag (one straight flight)

Commanded body velocity is

```
a = g_body − (T/m) ẑ + k ⊙ v − ω × v
```

with `k = (-0.50, -0.50, -0.15)` in FRD (`k < 0` opposes motion).

1. Level, constant-thrust crawl down a 10–15 m hall. No gates required if
   you have a speed truth (optical flow, PnP range rate, or a tape +
   stopwatch as a last resort).
2. Prefer vision range-rate over IMU accel. IMU specific force is **not**
   a velocity sensor on a quad.
3. Fit `k_x` (and `k_y` if you have a sideslip) with
   `AI_GP/tools/identify_drag.py --telem <csv>` or the same regression:
   `k = (v · a) / (v · v)` on `|v| > 0.2 m/s`.
4. `k_z` is weaker. Fit only if you have a clean climb/sink at constant
   attitude. Otherwise keep −0.15.

**Isaac:** `utils/aigp_obs.py` `DEFAULT_DRAG_BODY`

**AI_GP:** `DRAG_KX`, `DRAG_KY`, `DRAG_KZ`

If `|k|` moves by more than ~30 %, warm-start once so the policy sees
the new `vx,vy,vz` scale. If it is within 30 %, just patch and fly.

---

## Phase 6 — camera (5 minutes if spec-true, 30 if not)

Isaac projects privileged corners with

| Param | Value |
|---|---|
| Resolution | 640 × 360 |
| `fx`, `fy` | 320 |
| `cx`, `cy` | 320, 180 |
| Tilt | 20° up about body +Y |
| Stream | 30 Hz hold, control 60 Hz |

1. Confirm the race camera is the spec pinhole (no fisheye crop).
2. Photograph a known rectangle (gate inner 1.5 m or a checkerboard).
   Check that a gate at ~5 m fills the width the pinhole predicts.
3. Level the airframe. A same-height gate must sit **below** image
   centre by `fy · tan(20°) ≈ 116 px`. If it sits on centre, tilt is 0
   and every trained approach is wrong.
4. If tilt or `fx` differ, change `CAMERA_TILT_UP_DEG` / `FX` / `FY` in
   `utils/aigp_obs.py` **and** the matching AI_GP camera model, then
   retrain. This is a new observation, not a plant tweak.

Do not “fix” a tilt error with rewards.

---

## Phase 7 — timing (log only)

| Clock | Isaac | Real |
|---|---|---|
| Physics / rate PD | 120 Hz | FC rate loop (read the FC) |
| Policy | 60 Hz (`decimation=2`) | AI_GP `CONTROL_HZ=99` |
| Camera | 30 Hz hold | spec 30 Hz |

You will not retune Isaac to 99 Hz at the venue. Note the real control
period and any extra MAVLink delay (command → gyro). If delay is
\>30 ms and the keeper overshoots hard, a warm-start with a 1–2 step
action delay is the fix — not a new mass.

---

## Phase 8 — IMU (do not put in the observation)

The accelerometer is **not** a policy channel. Measure it so you know
the chip, not so you can swap it for commanded velocity.

On the pad, quiet:

- `|a|` should sit near 9.81, `zacc ≈ −g` (same convention as `ahrs.py`).
- Gyro bias: 5 s average at rest. If an axis is >0.05 rad/s, the FC
  should zero it; Isaac’s gyro is clean.

If you later get an IMU noise model (Allan variance, bias random walk),
that is for domain randomization — **add** noisy accel, do not replace
`vx,vy,vz`. Until then, leave the observation alone.

---

## Where to write the numbers

After the table is filled, edit in this order so train and play cannot
drift:

1. `isaac_drone_racer/dynamics/rate_control.py` — mass, hover, min/max, rate limit
2. `isaac_drone_racer/tasks/drone_racer/mdp/actions.py` `ControlActionCfg` — same + PD + moments
3. `isaac_drone_racer/assets/5_in_drone/urdf/5_in_drone.urdf` — mass + `ixx,iyy,izz`
4. `isaac_drone_racer/utils/aigp_obs.py` — drag, camera
5. `AI_GP/config.py` — `HOVER_THRUST`, thrust rails, `DRAG_K*`, `RATE_SIGN_*`
6. `AI_GP/race_obs.py` `ACTION_RANGES` — if `RATE_LIMIT` changed

Then play `pq_speed_best` in Isaac with the **new** plant before you
retrain. If it still flies the PQ hall, you do not need a new
checkpoint.

---

## When to retrain

Keep `pq_speed_best` until one of these is true:

- `RATE_LIMIT` moved by more than ~0.3 rad/s
- Hover/mass were wrong in Isaac by more than ~10 % **and** play of the
  keeper on the corrected plant looks drunk (not just a bit floaty)
- Drag `|k|` moved by more than ~30 %
- Camera tilt or `fx` changed
- Rate PD / inertia changed enough that Isaac play of the keeper cannot
  track a 2 rad/s step

Then **one** continue, same task, same obs flags
(`GATE_LOOKAHEAD=0`, `SEPARATE_NETS=0`):

```bash
OMNI_KIT_ACCEPT_EULA=YES GATE_LOOKAHEAD=0 SEPARATE_NETS=0 \
python -u scripts/rl/train.py --task Isaac-Drone-Racer-v0 \
  --headless --num_envs 4096 \
  --checkpoint models/pq_speed_best.pt
```

Do not spawn parallel models with guessed `Izz`. If you want robustness
before the venue, one domain-randomized continue is enough:
mass ±10 %, hover ±0.02, rate limit 2.8–3.6, drag ±30 %.

---

## Venue day (short form)

The phases above are the full ID. On site you only run this. Scripts live
in `AI_GP/tools/pad_constants.py` and write `AI_GP/logs/pad_constants.json`.

```bash
cd AI_GP
python tools/pad_constants.py            # what's done / what's next
```

### Battery 1 — get the numbers

| Step | What you do | Script |
|---|---|---|
| 0. Signs (props **off**) | Tip roll / pitch / yaw by hand. Gyro must go **positive** for +NED. | `python tools/pad_constants.py signs` or `signs --listen` |
| 1. Mass | Weigh race-ready bird. | `python tools/pad_constants.py mass 0.62` |
| 2. Hover | Arm, hold, see if it climbs. | `python tools/pad_constants.py fly-hover` (or ingest a `tune_flight.py hover` CSV) |
| 3. Rates | Pulse 2 rad/s on each axis. | `python tools/pad_constants.py fly-rates`  (`--full` adds 1.0 and 3.2) |
| 4. Camera | Same-height gate cy. Expect ~296 px, not 180. | `python tools/pad_constants.py camera --gate-cy 290` |

```bash
python tools/pad_constants.py report
```

`report` is the table to copy into Isaac / `config.py`. Hover goes into
`HOVER_THRUST` **before** battery 2.

`fly-hover` / `fly-rates` **arm the bird**. Ctrl+C disarms. They do not
send sim-reset 31000. Same MAVLink as `main.py` (`127.0.0.1:14550`, or
`MAV_IP` / `MAV_PORT`).

### Battery 2 — fly the keeper

Mass and hover are in the sheet. Hover is in AI_GP. Signs are good.

Fly the keeper gently. If it climbs or sinks while level, re-run
`fly-hover --thrust <new>` — do not touch PD.

### Stop

You should have `mass_kg`, `HOVER_THRUST`, `RATE_LIMIT` (or “keep 3.2”),
and signs. That is everything you must get on site.

Do **not** start drag or inertia on battery 1. Optional later:
`python tools/identify_drag.py --telem <csv>`.

---

## What this file is not

`AI_GP/TUNING.md` is the classical / assist / Kalman runbook
(`KALMAN_KP_ATT`, lean boost, yaw image gain). Those do not go into
Isaac. This file is only the numbers the PPO plant and the commanded-
velocity observation share with the bird.
