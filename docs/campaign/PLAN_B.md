# Plan B: slow and sure

A fallback for the AI Grand Prix Physical Qualifier if the learned stack (Plan A: HG-DAgger TCN / Isaac PPO policy) is not flying reliably by the end of Sunday, Sept 20.

Drafted 2026-09-16 from the PQ spec (VADR-TS-005), the organizer gate-coordinate table, the Betaflight 4.4.3 source, and a competitor's public PQ repo (`JCorbin406/AIGP-Physical-Qualifier`). Their notes describe **their** drone. Treat every hardware number here as "check on ours" until someone reads it off our FC.

This document lives outside `AI_GP/` on purpose. The repo is read-only for us (see `CLAUDE.md`).

---


## 0. Tested in simulation (2026-09-16)

I built a Plan B simulator on CPU (`~/aigp/tools/planb_sim.py`) and ran 500 runs per setting.

**What it models:**
- Betaflight ANGLE-mode airframe: lags, drag, and thrust calibration error.
- The official course, with every gate shifted a little in each run.
- The real 74 x 46 deg camera. Gate fixes (front or back view) are available only in view at 1.2-14 m, with noise, dropouts and latency.
- Between fixes: drag-model dead reckoning plus the FC barometer.
- Scoring on the true state, with 0.3 m half-span and 0.12 m half-height clearance.

**Results:**

| Setting | Gate / cruise speed | Band A (expected) | Band B (plausible) |
|---|---|---|---|
| safe | 2.5 / 2.5 m/s | **100 %**, 2:21 | 93 %, 2:34 |
| **balanced** | 2.5 / 4.0 m/s | **100 %, 2:05** | 88 %, 2:18 |
| fast | 3.0 / 5.0 m/s | 99.4 %, 1:52 | 79 %, 2:10 |

A gate speed of 3.5 m/s drops to 75-91 % even in band A.

**Recommendation:** bank the first scored run with **balanced**. Use **safe** if conditions look rough.

**What sets the run time:** the two forced turnarounds (G6 split-S, G9 double gate) and the alignment pauses, not the cruise speed.

**Design changes the simulation forced:**
- Takeoff uses a staged throttle ramp, with no integrator wind-up on the pad.
- Commit to a gate only when lined up (height < 0.2 m, lateral < 0.25 m), with escalating time-boxed fallbacks (1.5 s / 4 s / 8 s).
- On approaches, point the camera at the gate until about 2.5 m out, then square up to the gate axis.
- On the G6 bypass, point the camera at G6. Stop about 5.5 m west and 2.2 m north of it, so it is in view after the turn.
- Give G10 a 4.5 m approach, because it is the longest leg and arrives about 50 deg off-axis.
- Barometer: use it only for changes between vision fixes, and learn its offset from vision height. Vision height also estimates the thrust bias, which makes the barometer optional.
- Velocity: use a drag-model velocity observer, not accelerometer integration. Calibrate drag on day 2.

**Assumption checks** (balanced setting, 300 runs each):

| Assumption changed | Band A | Band B |
|---|---|---|
| as designed | 100 % | 87 % |
| no barometer (vision height only) | 100 % | 83 % |
| detector range 8 m / 5 m | 100 % / 97 % | 86 % / 47 % |
| no back-of-gate detection | 98 % | 64 % |
| gate fixes at 10 Hz, +100 ms latency | 99.7 % | 69 % |
| camera tilt unknown by 2 deg / 5 deg | 100 % / 92 % | 83 % / 47 % |
| all pessimistic at once | 98 % | 45 % |

What this means:
- **The barometer is optional.** Vision height also teaches the thrust bias.
- **Measure the camera tilt:** it is the biggest single risk.
- **Under rough conditions,** the detector's range, back-view detection and fix rate matter most.

**Assumptions to verify on site:**
- A usable FC barometer.
- A detector that gives gate-relative pose (front or back view).
- ANGLE mode is allowed.
- Airframe size.

**Not modelled:** real aerodynamics near frames, ground effect, detector false positives.

## 1. Why a slow plan can win

Scoring (spec §3.3):
1. **Primary:** the most gates passed.
2. **Tie-break:** time to the last valid gate.

So:
- A 6-minute run that clears every gate beats a 40-second run that crashes at gate 7.
- Scoring uses the best run from the **final two days (Mon 21 – Tue 22)**. The first scored slot should **bank a safe run**, and only then should we attempt anything faster.
- A human touching the sticks makes the run invalid. Plan B is built so the pilot never *needs* to touch them, but can at any moment.
- **Slow, but not too slow.** A crawl carries its own risks:
  - an unknown maximum run time (VQ capped runs at 8 min);
  - 15-min practice slots that must also fit setup and reset;
  - battery drain, since hovering costs power by the second, not by the metre;
  - drift that grows with time spent between gate sightings.
- So Plan B **approaches gates at about 2.5 m/s** (cruise up to 4 m/s) and only stops where the course forces it. The simulation gives about 2:05 for two laps (§0).
- For comparison, Gene's Isaac policy does about 7.9 s per lap in sim, an average of 13+ m/s. That comes from the training logs of the Sep 5 keeper: 33.2 gates in 23.7 s, over a lap of at least 100 m.

## 2. Design principles

| Principle | Concretely |
|---|---|
| Let the FC stabilize | Betaflight **ANGLE mode** (self-level, tilt capped by `level_limit`, 30° in the other team's dump). The Jetson sends *angles* and throttle, not rates. If ANGLE is not allowed, fall back to ACRO with a gentle Jetson-side attitude loop on `MSP_ATTITUDE`. |
| Flow, and slow only where needed | Cruise at about 2.5 m/s and keep correcting on the gate while moving. Slow to about 1.5 m/s in the last 3 m only if the alignment error is large. Full stops only at takeoff, the G6 split-S turnaround and the G9 double-gate turnaround. Finish every turn *before* the final 3 m. Never turn and pass at the same time. |
| The gate is the position sensor | No global position exists (no mocap, no GPS). Dead-reckon at most one short leg (≤ 15 m), then re-anchor on the next gate. |
| One process, few moving parts | One Python 3.10 process on the Jetson: MSP link, camera, detector, state machine, logger. No ROS, no GPU-dependent stages in the critical path. |
| Everything logged | Every armed second gets IMU, MSP status, commands, detector output, state-machine state and H.264 video. |
| Reuse, don't rewrite | Take pieces from `AI_GP` **by copying into a separate Plan B folder**: `vision/gate_detector.py` (orange opening detector), `control/pid.py`, `ahrs.py`, `camera_model.py` math. The only part that has to be new is the MSP/camera bring-up layer, and Plan A needs that too. |

## 2a. Speed budget: why 2.5 m/s

Assumptions:
- Lap path about 135 m. That is the racing line (about 129 m by another team's estimate) plus approach-point offsets.
- Fixed overheads per lap: the G6 split-S turnaround (brake, yaw 180°, re-align, about 5 s), the G9 stacked-gate climb/turn/descend (about 8 s), and a few short slow-downs.

| Tier | Cruise | Lap | 2 laps + takeoff/finish | Use |
|---|---|---|---|---|
| B-slow | 1.5 m/s | ~120 s | ~4.3 min | first autonomous gates in the cage and on the track |
| **B-cruise** | **2.5 m/s** | **~75 s** | **~2.9 min** | **default scored run** (bank this) |
| B-quick | 3.5 m/s | ~57 s | ~2.2 min | only after B-cruise has completed two laps cleanly |

Physics limits at each speed:
- Bank capped at 20°.
- 100 ms from photon to command.
- 30 fps camera.
- Clearance inside the 1.5 m opening is about ±0.45 m for a roughly 0.6 m drone. **Measure our drone.**

| Cruise | Tightest turn | Stop distance (15° pitch-back) | Travel during latency | Blind time after the opening leaves the frame (~1 m, 75° HFOV) |
|---|---|---|---|---|
| 1.5 m/s | 0.6 m | 0.4 m | 15 cm | 0.67 s |
| **2.5 m/s** | **1.8 m** | **1.2 m** | **25 cm** | **0.40 s** |
| 3.5 m/s | 3.4 m | 2.3 m | 35 cm | 0.29 s |
| 5.0 m/s | 7.0 m, too wide for the 50° arrivals | 4.8 m | 50 cm, about the whole clearance | 0.20 s |

At 2.5 m/s every turn fits inside the 6–15 m legs, braking for the two turnarounds takes about 1 m, and latency error stays well inside the opening. Above about 4 m/s a simple look-and-servo controller runs out of margin; that regime is Plan A's job.

### How speed is controlled with no velocity sensor

In ANGLE mode, forward speed settles where tilt balances drag: v ≈ g·tan(pitch)/k. The drag constant k of the Archer B2 is unknown.

| k | 3° | 5° | 8° | 12° |
|---|---|---|---|---|
| 0.3 | 1.7 m/s | 2.9 m/s | 4.6 m/s | 7.0 m/s |
| **0.5** (the value in the team's Isaac/AI_GP code) | 1.0 m/s | 1.7 m/s | 2.8 m/s | 4.2 m/s |
| 0.8 | 0.6 m/s | 1.1 m/s | 1.7 m/s | 2.6 m/s |

So:
1. **Day 2:** calibrate pitch → speed from one logged straight flight (the team's `tools/identify_drag.py` regression). Use range-rate to a gate (from the gate's apparent size) or a tape and stopwatch as ground truth.
2. **During each approach:** close the loop on **range rate** (the gate's apparent size growing) so the target speed holds even if k was wrong.
3. **Cap pitch** (for example 12°) so a bad estimate can never become a sprint.

### "Won't finish" traps, and the rule for each

| Trap | Rule |
|---|---|
| ALIGN waits forever for a perfect picture | ALIGN is **time-boxed** to 1.5 s. After that, go if errors are inside the *safe* envelope (lateral < 0.35 m, heading < 10°), otherwise slow to 1.5 m/s and keep correcting. |
| Gate not found | Sweep ±30° once, then fall back to the map bearing and creep. Budget per leg is 2× the expected time. Degrade speed, don't abort. Land only on safety triggers. |
| Battery sag near the end of a long run | A roughly 3 min run keeps sag small. Scale hover throttle by pack voltage. |
| Unknown run time limit | Ask (§8.4). If it is under 3 min, start with B-quick on the straight legs only. |

## 3. Hardware facts to verify on day 1

From the spec, the other team's FC dump, and the Betaflight 4.4.3 source:

| Item | Expected | How to check |
|---|---|---|
| FC firmware | Betaflight **4.4.3**, H743, MSP API 1.45 | `version` in the CLI |
| FC link | `/dev/ttyTHS1`, 115200, MSP v1 | organizer demo app |
| MSP throughput | about 100 replies/s total, **one command per 100 Hz tick** | bench |
| IMU via MSP | `MSP_RAW_IMU`: acc 512 counts = 1 g, gyro 16.4 counts = 1 °/s (raw counts, **not** °/s) | at rest: \|a\| ≈ 512 |
| Attitude via MSP | `MSP_ATTITUDE`: roll/pitch in 0.1°, yaw in whole degrees | tilt by hand |
| Camera | IMX477 via `nvarguscamerasrc`. Spec says 1080p60, the other team logs 1080p30. Tilt is adjustable (`feature SERVO_TILT` is on). | organizer demo |
| Arming | Pilot's CRSF transmitter, **ARM on AUX1**. MSP cannot arm. | `aux` in the CLI |
| Override | **MSP OVERRIDE (mode 50) on AUX5**. `msp_override_channels_mask = 11` (roll, pitch, yaw) **excludes throttle**; it needs to be **15** for Plan B. | `get msp_override` |
| Modes | **No ANGLE/HORIZON mode assigned**, so the drone flies ACRO by default | `aux` |
| Rates | Betaflight rates, rc_rate 0.55, super rate 0.75, expo 0, so **nonlinear** stick-to-rate (about 440 °/s at full stick) | `rateprofile` |
| Throttle curve | `thr_mid 54`, `thr_expo 68`, so **nonlinear** stick-to-thrust | `rateprofile` |
| Failsafe | `failsafe_procedure = DROP`, `failsafe_delay = 50` | `get failsafe` |
| Baro | `baro_hardware = AUTO`; presence unknown | `status`, `MSP_ALTITUDE` |
| Board clock | resets to 2023-11-21 on every boot | name logs by counter, not date |

### Safety fact from the firmware source (not a guess)

In Betaflight 4.4.3 `src/main/rx/msp_override.c`, while the override switch is on and a channel is in the mask, the FC uses the **last `MSP_SET_RAW_RC` values it received, with no staleness timeout**.

If the Jetson process hangs mid-flight, the drone keeps flying on its last stick command until the pilot flips AUX5 off or disarms. Before AUX5 goes on, that buffer is all zeros, which are invalid pulses, so `rxfail` handling takes over. Consequences:
- The Jetson must stream a **safe neutral frame** (level, hover-or-idle throttle) *before* the pilot enables override.
- A **Jetson-side watchdog** (separate thread) must send a "level + slow descend" frame if the main loop stalls for more than 150 ms.
- The pilot's thumb stays on AUX5 and ARM. Brief the pilot: **"override off" is the recovery, "disarm" is the kill.**
- The mask must **never** include the ARM or override channels. Use exactly 15, no wider.

## 4. What Plan B needs changed on the FC

All of this goes through the CLI (the FAQ allows rates, PIDs, filters and telemetry). **Ask Neros/organizers before touching modes.**

1. `set msp_override_channels_mask = 15`, so the Jetson owns AETR while AUX5 is high.
2. ANGLE mode on a channel. Two options:
   - Pilot switch: `aux <n> 1 <AUXk> 1300 2100`.
   - Always on while override is on: put ANGLE on the same AUX5 range.
3. `set level_limit` to about 25. Cruise needs ≤ 12° pitch and turns at 2.5 m/s need ≤ 20° bank, so 25 leaves margin without allowing a dive.
4. Optionally flatten the throttle curve (`thr_mid 50`, `thr_expo 0`) so thrust is closer to linear.
   - If we are not allowed to, measure it: hover stick plus two climb points.
5. `diff all` backup **before** and **after**, saved off-drone.

If ANGLE is refused, use ACRO:
- Set `roll/pitch_srate 0`, `expo 0` so rate ≈ 200·rc_rate·stick (linear).
- Close a slow P loop on `MSP_ATTITUDE` at 25–50 Hz.

## 5. The state machine

```
PREFLIGHT ─▶ WAIT_ARM ─▶ WAIT_OVERRIDE ─▶ TAKEOFF ─▶ HOVER_CHECK
   │                                                  │
   └ fail: never arms                                 ▼
             ┌────────────────────────────────── GOTO_APPROACH(i) ◀───────────┐
             │   (dead-reckon ≤ 15 m to a point 3 m in front of gate i)       │
             ▼                                                                │
       FACE_GATE(i) ─▶ ACQUIRE(i) ─▶ ALIGN(i) ─▶ CREEP(i) ─▶ PUNCH(i) ─▶ BRAKE ┘
             (yaw to       (slow yaw     (center     (≈2.5 m/s,   (hold heading,
             gate axis)     sweep if     x, y and    keep          fixed distance
                            not seen)    skew)       servoing)     after the gate
                                                                   leaves frame)
  after the last crossing: BRAKE ─▶ LAND ─▶ IDLE (pilot disarms)
  any watchdog: HOLD (level, hover) ─▶ LAND
```

- **PREFLIGHT.** Checks:
  - MSP link at ≥ 85 Hz with no CRC errors.
  - IMU at rest reads \|a\| ≈ 1 g.
  - Camera at ≥ 25 fps.
  - Detector sees gate 1, with bearing within 10° of the map prediction from the start box (7.4 m away, 7° right).
  - Battery OK, log file open, neutral RC frame streaming, FC mode flags as expected.
- **TAKEOFF.** Ramp throttle to the measured hover + 8 % until attitude shows lift-off, then hold hover. Height comes from vision: the opening of gate 1 (center 1.35 m) should sit at the image row the camera model predicts for a 1.35 m eye height. Baro can help if present.
- **GOTO_APPROACH(i).** Integrate *commanded* velocity (the team's `ekf/commanded_accel.py` idea) and gyro yaw from the last gate crossing. Aim at the approach point `P_i − 3 m · axis_i`. Time out at 2× the expected leg time.
- **ALIGN(i).** Runs *while moving*: at cruise on well-aligned legs, slowing to about 1.5 m/s on the 35–50° arrivals. There are three loops, all on the detected inner-opening quad:
  - **Yaw** centers the horizontal pixel offset.
  - **Roll** zeros the lateral offset. Skew comes from the left/right edge height ratio (perspective), and range comes from the 1.5 m opening size with calibrated `fx`.
  - **Throttle** puts the opening center at the predicted row.
  - ALIGN is **time-boxed** (§2a): leave when errors are small, or after 1.5 s if they are merely safe.
- **CREEP(i)** (the approach). Pitch is set from the calibrated table for the tier's speed (about 2.5 m/s for B-cruise), with closed-loop control on range rate and a hard cap of 12°. Servos stay active. The gate's apparent size growing is the progress signal.
- **PUNCH(i).** With a 75° HFOV, the 2.7 m outer frame leaves the image inside ~1.8 m and the 1.5 m opening inside ~1.0 m. When the opening fills more than ~70 % of the width, freeze yaw and roll, hold pitch for `(range + 1.5 m) / speed`, then BRAKE. Mark gate i passed and reset dead-reckoning to `P_i + 1.5 m · axis_i`.

### Course-specific moves

These come from the official table; the double-gate heights come from the other team's on-site check.

| Leg | Distance | Arrival off-axis | Plan B move |
|---|---|---|---|
| start → G1 | 7.4 m | 7° | take off facing north, align, pass |
| G1 → G2 | 8.3 m | 8° | straight |
| G2 → G3 | 9.0 m | 35° | approach point, then face east |
| G3 → G4 | 10.8 m | 50° | approach point, then face south |
| G4 → G5 | 7.9 m | 22° | straight-ish (G5 faces 215°) |
| G5 → G6 | 7.7 m | **152°, split-S** | pass **north** of G6 with ≥ 2.5 m clearance, continue ~3.5 m west, stop, turn to face east (90°), align, pass |
| G6 → G7 | 9.8 m | 50° | approach point, then face south |
| G7 → G8 | 6.2 m | 4° | straight (G8 faces 215°) |
| G8 → G9 top | 7.0 m | 50° | approach point, **climb to 4.05 m**, face south, align on the **upper** opening, pass |
| G9 top → G9 bottom | turnaround | 180° | brake, descend to 1.35 m, yaw 180° (face north), align on the **lower** opening, pass |
| G9 bottom → G10 | ~13.6 m | 32° | longest dead-reckoned leg; approach point, then face north |
| G10 → G1 (lap 2) | 10.4 m | 2° | straight |

- **Double gate:** the stacked frames put the top opening near 4 m. Confirm the ceiling and the real heights on site.
- **If G9 is judged unsafe, ask the organizers first.** We need to know whether a skipped gate stops the count. If it does, the run ends at G8 (8 gates) and nothing is lost by stopping there.
- **Run length:** one lap is about 11 crossings, and a complete run is 2 laps (the other team counts 23 crossings, ending on gate 1). At B-cruise that is about 75 s per lap, so **about 3 min** in total (§2a). Confirm the battery holds for 2× that in the cage, and check any slot or run time limit.

## 6. Build order: shared with Plan A

Layers 1–2 are prerequisites for **both** plans, so none of this work is wasted.

1. **Bring-up and logging (day 1).** Ship nothing else until this is solid.
   - Organizer demo apps running.
   - A `diff all` backup of the FC.
   - An MSP reader (attitude, raw IMU, status, analog).
   - Camera capture at a known fps with CLOCK_MONOTONIC stamps.
   - A logger that records every armed flight unattended.
   - Pilot flies manual laps while logging. This gives real gate images, hover throttle and noise levels.
2. **MSP override bench (props off).**
   - The RC echo follows the Jetson only while AUX5 is high.
   - Stick signs are correct on every axis.
   - Kill the Jetson process mid-override and watch the channels hold (the stall watchdog must catch this).
   - Neutral-frame-before-override procedure works.
3. **Detector on real frames.**
   - Tune `vision/gate_detector.py` HSV on logged frames, with rejection of the red signage the other team reports.
   - Measure hit rate and false positives per gate.
   - Calibrate camera intrinsics with a checkerboard (15 min; not provided by organizers).
4. **Cage.** Autonomous hover (throttle loop), yaw hold, ±1 m nudges, watchdog → HOLD → LAND.
5. **Track slots.** G1 only, then G1–G3, then G1–G6 (split-S), then full lap without G9 (if allowed), then full lap, then 2 laps.
   - Each new stretch is flown at **B-slow** first, then repeated at **B-cruise**.
   - Only move to B-quick after two clean B-cruise runs.
6. **Freeze.** From Sunday night, change only parameters and keep a written log of every change.

## 7. Day plan (team arrives Thu 9/17 AM)

| Day | Plan B goal | Exit criterion |
|---|---|---|
| Wed 9/16 (tonight, off-site) | Offline prep | MSP codec plus a fake FC for tests; logger skeleton; replay harness for detector tuning; kinematic sim of the state machine on the official map; organizer question list |
| Thu 9/17 | Layer 1 and the layer 2 bench | Logged manual flights and FC backups; override bench passes on one drone |
| Fri 9/18 | Layers 3–4 | Detector ≥ 95 % hit rate on logged approach frames; stable autonomous hover in the cage |
| Sat 9/19 | Layer 5, first half | Autonomous G1–G3 at B-slow, then at B-cruise |
| Sun 9/20 | Layer 5, full; freeze | One full autonomous lap at B-cruise; full 2-lap duration checked against the battery; **Plan A go/no-go decision** |
| Mon 9/21 | **Scored** | First slot: **B-cruise** (bank it). Later: B-quick, or Plan A if it passed its go/no-go |
| Tue 9/22 | **Scored** | Improve on the banked score. Never risk the drone before a banked valid run exists |

**Plan A go/no-go (Sunday):**
- The policy has flown at least 3 gates autonomously on the real drone, more than once.
- The observation has been checked against the real camera (intrinsics, tilt, fps) and real rates/throttle.
- Its failure mode is a safe HOLD, not a flyaway.

If any of these is false, Monday starts with Plan B.

## 8. Questions for the organizers

Contacts: Florian Mott for race rules, Brandon Porter (Neros) for the drone.

1. Can we assign modes (ANGLE) and change `msp_override_channels_mask`, rates and the throttle curve through the CLI?
2. Please send the official **gate coordinate PDF** (with double gate) and the **as-built gate heights**. Is the double gate two stacked 2.7 m frames?
3. Scoring:
   - Are gates counted only in order?
   - Does a missed or skipped gate end the count?
   - Is a run that crashes after N gates scored as N?
   - Exactly how are 2 laps and the finish defined?
4. Run procedure:
   - Who arms, and is a pilot arming plus flipping the override switch before the start line OK?
   - Is landing or disarming by the pilot *after* the finish allowed?
   - Is there a maximum run duration or slot length?
5. Camera: 1080p60 or 30? Which sensor mode do the demo apps use? How is tilt set?
6. Are all 4 of our drones on identical firmware and config? Is a baro or rangefinder fitted?
7. Where are the demo apps on the Jetson image, and which Python/packages are preinstalled?

## 9. Things that would sink Plan B, and the mitigation

| Risk | Mitigation |
|---|---|
| Detector false positives (red signage, orange clutter) | Tune on real logs; demand a quad shape with an inner opening (the team's detector already does); keep a map-consistency gate (bearing and range must be plausible) |
| Height drift without a baro | Re-anchor height on every gate sighting; keep hover-throttle battery compensation; keep legs short |
| Yaw drift | `MSP_ATTITUDE` yaw plus gyro integration; re-anchor at each gate (its axis is known) |
| Jetson hang with override on | Separate stall watchdog thread; pilot briefed; bench-tested |
| Battery sag over a 6–8 min run | Log voltage; scale hover throttle; test a full 2-lap duration in the cage |
| Split-S / double gate too risky | Decide per organizer answer (§8.3); stop cleanly after the last safe gate |
| Too slow: run time limit, slot time, battery, drift | B-cruise (about 3 min run); time-boxed ALIGN; degrade speed instead of waiting; confirm limits (§8.4) |
| Too fast for the controller | Pitch cap 12°, bank cap 20° (`level_limit` 25), speed tiers raised only after clean runs |
