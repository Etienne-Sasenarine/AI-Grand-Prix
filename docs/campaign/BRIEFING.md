> **Superseded (2026-09-16, 15:00):** the full, up-to-date write-up is `briefing/main.pdf`. It includes the stress tests, the retrain feasibility test and the Plan B simulation. This markdown is the earlier morning version.

# AI Grand Prix: team briefing (as of Wed 2026-09-16)

Written for someone new to the codebase. It covers what the competition is, what the team has built, where the code stands, what is missing, and what to do next.

Companion documents:
- [`PLAN_B.md`](PLAN_B.md): the slow, safe fallback.
- [`CLAUDE.md`](CLAUDE.md): workspace rules. The `AI_GP` repo is read-only for us.

Tags:
- **[V]** = verified by reading code, running it, or checking git.
- **[I]** = inference.
- **[ext]** = from a public source outside our repo.

---

## 0. TL;DR

1. **Repo.** `Code-Red-Cables/AI_GP`, **public** on GitHub.
   - `main` is stale (June 6).
   - **`isaacsim`** (pushed Sept 16, 04:49 UTC) is the truth. It contains every earlier line of work plus the Isaac Lab reinforcement-learning project in `isaac_drone_racer/`. [V]
2. **The team has used five approaches,** always on the organizers' *simulator* (VQ1/VQ2):
   - waypoint and spline replay;
   - classical vision with a Kalman filter;
   - DreamerV3 (abandoned);
   - imitation learning (HG-DAgger);
   - Isaac Lab PPO trained on a digitized copy of the **physical** course.
3. **The PQ "keeper" model** is `isaac_drone_racer/models/pq_speed_best.pt`.
   - Training run of Sept 5; about 33 gates per 40 s episode, about 15 m/s through the gates, in simulation.
   - No deterministic evaluation or real-drone test exists in the repo. [V]
4. **Nothing in the repo talks to the real drone yet.**
   - The client speaks MAVLink over UDP to the simulator.
   - The real drone is Betaflight, speaking MSP over UART to a Jetson Orin NX.
   - There is no Jetson camera code and no MSP code. [V]
   - This may exist on Geneustace's machine but was never pushed; ask.
5. **Gene says everything is pushed.** The remote is unchanged since 2026-09-16 04:59 UTC [V].
   - The repo's `.gitignore` keeps these out of git **by design**, so if they're needed they have to be shared by drive or USB:
     - root `models/*.pt` (`gate_pose_v5.pt`, `policy_seed_17.pt`);
     - datasets;
     - seed/coach logs.
   - Two Python modules that the default `assist` mode imports were never committed [V].
   - There are no Isaac runs after Sept 6.
   - Checklist in §8.
6. **Four issues to raise with the team before anyone flies or retrains** (§7):
   - (a) a gate-counting bug in the current Isaac code, **confirmed by an A/B training run**, that explains why the Sept 6 run collapsed; any retrain collapses until it is fixed;
   - (b) the camera model is the simulator's 90° lens, but the PQ camera is 75°;
   - (c) the Isaac track is about 7% too small, with assumed gate heights of 2.0 / 4.7 m vs about 1.35 / 4.05 m on site [ext];
   - (d) Betaflight's rate and throttle curves on the drones are nonlinear [ext].
7. **Strategy.** Scoring is *most gates first, time second*, and uses the best run of **Mon 9/21 – Tue 9/22**.
   - Bank a slow, valid run first ([`PLAN_B.md`](PLAN_B.md)), then try the fast policy.
8. **This laptop is set up.**
   - The flight-client environment works.
   - Isaac Sim 4.5 / Isaac Lab 2.1 train headless on the RTX 4060 (§10).

---

## 1. The competition

**AI Grand Prix**
- Run by Anduril with the Drone Champions League (DCL); drones supplied by Neros.
- More than 3,000 teams entered. [ext: theaigrandprix.com]
- Finals are in Columbus, Ohio in November 2026: a head-to-head live race with a $500K prize pool. [ext]
- The number of teams advancing from the PQ and the prize split are **not published**. [ext]

| Stage | When | What |
|---|---|---|
| VQ1 (virtual qualifier 1) | Opened May 31 | DCL "FlightSim" simulator (Windows); short course, just finish. Spec VADR-TS-001 (`AI_GP/reference/AI Grand Prix Tech Specs.pdf`). |
| VQ2 | Opened Jun 29, closed Aug 3 | Harder course, and state data **blocked**: only IMU + 640×360 camera. Spec VADR-TS-003. Top 10 teams advanced to the PQ. [ext] |
| **Physical Qualifier (PQ)** | **Sept 15–22**, Anduril LC3 (OC-23, Santa Ana) | Real drones in a hall. Spec VADR-TS-005 (`260902_PQ_Technical_Spec_0002.pdf`). |
| Finals | Nov 2026, Ohio | Live head-to-head |

### PQ rules and hardware (VADR-TS-005, plus [ext] where marked)

**Scoring and run rules**
- **Scoring:**
  1. Most gates passed.
  2. Tie-break: time to the last valid gate.
  3. Uses the best result from the **final two days**.
  - A run is 2 laps. Any human intervention makes it invalid.
- **Practice:** several daily slots per team (15 min each per TS-004 [ext]), plus a training cage. Each team gets 4 drones. Neros repairs them and handles all charging.

**Course**
- Footprint 60×21 m. The drawn box is about 26×50 m.
- 10 gates, flown in order. Gate 9 is a **stacked double gate**: top opening southbound, then a half-loop, then the bottom opening northbound. Gate 6 is a **split-S**.
- Gates are 2.7 m outer / 1.5 m inner, 80 mm deep, orange, and stand on the floor, so the opening center is about 1.35 m up. [ext]
- An official gate-coordinate PDF exists ("Drone_Race_Track_Gate_Coordinates_with_doublegate_5800.pdf"). **Get it from the organizers.** [ext]

**Drone**
- Neros Archer B2, 8" props.
- Betaflight flight controller (FC): you may set rates, PIDs, filters and telemetry, but not reflash.
- FC talks to the Jetson over UART using **MSP**.
- The pilot's RC override always wins.
- Other teams report [ext]:
  - Betaflight 4.4.3, `/dev/ttyTHS1` at 115200 baud;
  - roughly 100 MSP replies per second in total;
  - the default override mask covers roll, pitch and yaw but **not throttle**;
  - the pilot's transmitter arms the drone.

**Compute and sensors**
- Jetson Orin NX 16 GB (Seeed A603 carrier, JetPack 6.2, 25 W), full root access.
- Organizers provide demo apps for camera bring-up and FC serial communication.
- Camera: IMX477 rolling shutter, 1080p **60** per spec (another team measures **30 fps** [ext]), **75° HFOV**, adjustable tilt, no calibration provided.
- IMU: 50 Hz per spec. About 92 Hz is achievable over MSP [ext].
- No motion capture in the hall.

**Logistics** (welcome packet)
- Hours 7:00–22:00, off-site by 23:00. Day badges, closed-toe shoes.
- Mandatory safety briefing on day 1.
- Contacts:
  - Florian Mott (race / tournament)
  - Brandon Porter (Neros, drone tech)
  - Patrick Bark (general)
  - Brandon Ruffin (safety)
- The racing Wi-Fi password is in the packet. **Never commit that PDF to the public repo.**

**[ext] Another PQ team works in public:** `github.com/JCorbin406/AIGP-Physical-Qualifier` ("AI of Sauron").
- Its docs are the best available description of the real drone: FC dump, MSP notes, bring-up logs, course map.
- It has **no license**. Use the *facts*; don't copy the code.

---

## 2. People and repo layout

- **Geneustace Wicaksono** (GitHub SentientPlatypus): main developer, owns the sim/training machine (Windows plus WSL, RTX 4070 Laptop, `D:\Code\Competitions\`).
- **Rocky Shao:** June starter and vision work; Qualifier2 testing.
- **Grant Lin** (lingrant74): July OpenCV navigation, keyboard teleop.
- **Etienne Sasenarine:** his own VQ2 repo (`Etienne-Sasenarine/AI-Grand-Prix`), including **GateNet CNN and TensorRT deployment tooling**. That is relevant for the Jetson.
- **bojro** (you).
- The team leans heavily on AI coding agents (Claude Code, Cursor, Codex). Docs are written as hand-off notes, so some decisions live only in chat history. [V]

Branches: every useful one is already merged into `isaacsim` [V].

| Branch | Last commit | Why you'd look |
|---|---|---|
| `isaacsim` | Sep 16 | **Current.** Everything below plus `isaac_drone_racer/` |
| `classical-vision-racing` | Aug 30 | Same as isaacsim minus the Isaac vendoring commit |
| `Q2_kalman`, `Q2_spline`, `Q2_CV`, `Q2_new`, `Q2_pnp`, `keyboard_teleop` | Jul 26–31 | VQ2 experiments (history) |
| `Qualifier2`, `Qualifier2_testing` | Jul 14–19 | VQ2 "vision reboot" plans and gap analysis (good reading) |
| `spline-path`, `manual-control`, `preplanning`, `modified-starter`, `main` | Jun | VQ1 era |

---

## 3. How we got here

| Era | Dates | Approach | Outcome |
|---|---|---|---|
| Starter | Jun 3 | Organizer example client plus `PLAN.md`; HSV (colour-threshold) gate detection; attitude + thrust commands | Controller working, untuned |
| VQ1 reactive, then waypoints | Jun 6–16 | Chase gates visually, then fixed waypoints | Gate 1 passed |
| VQ1 mapping + spline | Jun 17–20 | Fly by keyboard to record the course, then replay as a Catmull-Rom spline with pure pursuit (using sim position data) | "sub 14 seconds" (commit message only) |
| VQ2 vision reboot | Jun 28–Jul 18 | VQ2 blocks position and attitude, so IMU plus vision only | `GAP_ANALYSIS.md`: vision not connected to flight; abandoned |
| DreamerV3 | Jul 23–25 | Model-based RL against the live sim (real-time only, about 30 steps/s) | 168k steps, **0 gates**; deleted |
| Grant's OpenCV / PnP / VIO | Jul 26–27 | Hough/contour detection, then YOLO 4-corner pose, then solvePnP with IMU | "passed gates 1 and 2" |
| Dual-gate Kalman | Jul 28–Aug 1 | EKF using PnP from two gates; `tools/tune_flight.py` harness | Human PB 35.96 s |
| **classical-vision-racing** | Aug 10–30 | Li & de Croon classical stack; then **HG-DAgger**: YOLO 8-keypoint pose into a TCN policy, seeded with human laps and improved with human corrections; on-sim PPO fine-tune | Human PB **14.036 s** (18/18 gates). `policy_seed_17.pt` "leaves the pad". **No autonomous lap time recorded.** |
| **isaacsim (PQ)** | Aug 30–Sep 16 | Isaac Lab PPO on the digitized PQ track, same observation vector as the simulator client | Keeper `pq_speed_best.pt` (Sep 5). Last run Sep 6 collapsed (see §7a). Vendored into AI_GP on Sep 16. |

---

## 4. What the code is: the `isaacsim` branch

The repo has two halves that share one observation vector.

### 4a. Flight client (repo root). Written for the VQ simulator.

**Threads** (`main.py`, `setup.py`), all writing to one shared dict:
- `mavlink_rx.py`: IMU, race status and gate counter from the sim.
- `vision_rx.py`: UDP JPEG frames from the sim, then YOLO.
- `ekf_estimator.py`: state estimation.
- `logger.py`: logging.
- `timesync.py`: clock sync.

**Control loop:** `planner.compute_target()` → `controller.py` → MAVLink `SET_ATTITUDE_TARGET` (thrust + body rates) at about 50–99 Hz.

**Flight modes** (`FLIGHT_MODE` in `config.py`):

| Mode | What it is |
|---|---|
| `policy` | **The timed path.** HG-DAgger TCN: `policy_net.py`, `policy_planner.py`, `race_obs.py`. Launched by `tools/run_policy.py`. |
| `isaac` | Isaac PPO actor: `isaac_planner.py`, launched by `tools/run_isaac.py` |
| `assist` (default), `kalman`, `spline`, `race` | Classical experiments |

**Perception:**
- `vision/yolo_pose_gate_detector.py`: 8 gate corners (outer 0–3, inner opening 4–7).
- `vision/gate_detector.py`: deterministic HSV orange-opening detector.
- Plus PnP, snake and GateNet variants.

**The observation** (`race_obs.py`, one frame = 51 numbers):

| Count | Content |
|---|---|
| 16 | 8 corners as normalized pixels (−1 if unseen) |
| 8 | Visibility flags |
| 2 | Roll, pitch |
| 3 | Gyro |
| 3 | "Commanded" body velocity: integrated from thrust, attitude and drag, never from the accelerometer |
| 18 + 1 | Gate one-hot plus lap fraction |

The Isaac policy stacks 32 frames, giving **1632 inputs**.

**Tools:** `tools/tune_flight.py` is the big harness (modes: acro, pilot, coach, policy, hover…), `train_policy.py`, `train_gate_pose.py`, `train_ppo.py`, `pad_constants.py` (on-site measurement of the real drone), and ~35 others.

**Tests:** `pytest --ignore=isaac_drone_racer` on this laptop gives **441 pass, 79 fail, 1 skip** [V].
- 68 of the failures are `test_assist_planner.py`, which imports `vision/cam_bank_bias_learner.py` and `vision/lateral_yaw_learner.py`. **Neither was ever committed on any branch** [V].
- The rest are small test/code drift. One is in the timed-path detector's persistent lock and is worth a look.

### 4b. `isaac_drone_racer/`: Isaac Lab PPO for the PQ

It is a fork of `kousheekc/isaac_drone_racer` (BSD-3), copied in without its git history.

- **Task ids:**
  - `Isaac-Drone-Racer-v0`: training; random start gate, no camera.
  - `Isaac-Drone-Racer-Play-v0`: viewing; start at gate 1, FPV camera.
- **Rates:** 120 Hz physics, 60 Hz policy, keypoints held at 30 Hz, 40 s episodes.
- **Observation:** the same 51-D frame as above, but the gate corners are *projected* from known gate poses (privileged) instead of detected. There is no observation noise.
- **Plant:** not a motor model. A rigid body, with collective thrust mapped linearly from the stick (hover stick 0.255, mass 0.6076 kg, max thrust-to-weight about 3.5) and a PD loop on body rates (±3.2 rad/s). No drag or latency. Minimal randomization (±0.1 N pushes).
- **Rewards:** progress toward the gate (+20), gate passed (+600), speed through the gate (+3.4), camera and center alignment at the crossing (+4 each), plus small shaping terms; crash −100. A missed gate, a crash or flying 20 m away ends the episode.
- **PPO (skrl 1.4.2):** shared 256-256-256 MLP, 4096 envs, 24-step rollouts, lr 1e-4, 50k timesteps per run (about 1.4 h on a laptop GPU).
- **Track:** 11 entries (10 gates, with the double gate as two), digitized from the spec picture. See §7c.

### 4c. Training lineage of the keeper [V]

Warm-start chain recovered from normalizer sample counts:

1. `pq_scratch_35k` (Sep 4, from scratch on the PQ track)
2. `pq_vel_50k` (+ run-in and velocity shaping)
3. `pq_hook_best` (the gate 11 → 1 wrap-around)
4. **`pq_speed_best`** (Sep 5, +speed through gates): 33.2 gates per episode, 463 reward, episodes 42% time-out / 52% collision / 5% missed / 2% flyaway.
5. The Sep 6 continuation used **identical settings** and collapsed to 0 gates, because of the §7a bug (confirmed). `pq_hook2_best.pt` is that broken model.

Earlier Aug 30–Sep 4 runs were on 7- and 17-gate VQ-style tracks.

### 4d. The bridge and the plan for the real drone (as written in the repo)

- `isaac_planner.py` runs an Isaac checkpoint inside the simulator client. It uses YOLO keypoints, attitude, gyro and the velocity integrator, and expects `race_status` for the gate index.
- `isaac_drone_racer/PHYSICAL_TUNING.md` and `tools/pad_constants.py` are the on-site procedure:
  1. check signs;
  2. weigh the drone;
  3. find hover stick;
  4. measure rate limit;
  5. check camera tilt;
  6. copy the constants into Isaac and `config.py`;
  7. fly the keeper;
  8. retrain once, only if a constant moved a lot.
- The procedure assumes a MAVLink link to the bird (`127.0.0.1:14550`) [V]. The real FC speaks MSP.

---

## 5. Two ways it could fly at the PQ

| | Plan A (existing stack) | Plan B ([`PLAN_B.md`](PLAN_B.md)) |
|---|---|---|
| Brain | `pq_speed_best` PPO (behavior cloning retired) | State machine: stop, align, creep, punch through |
| FC mode | ACRO (Jetson sends body rates) | ANGLE (FC self-levels; Jetson sends angles + throttle) |
| Speed | Trained for ~15 m/s through gates; about 7.9 s per sim lap | About 2.5 m/s cruise, about 75 s per lap, about 3 min per run (tiers 1.5 / 2.5 / 3.5 m/s; see `PLAN_B.md` §2a) |
| Needs | MSP bridge; real-camera keypoints matching the training camera; linear rates/throttle; onboard gate counter; probably a retrain with real intrinsics and track | MSP bridge; HSV/YOLO gate detector tuned on real frames; hover throttle; FC mode change (ask first) |
| Risk | High (sim-to-real gap at speed) | Low |

Both need the same foundation: MSP link, camera capture, logging, bench checks. Day 1 goes to that regardless.

---

## 6. What is *not* in the repo (possibly done on Geneustace's machine)

Grep confirms **zero** occurrences of MSP, serial, Betaflight, Jetson, TensorRT, GStreamer/nvargus or CLOCK_MONOTONIC in the client [V]. Before assuming these are gaps, **ask Geneustace whether he has them locally**. If not, this is the Day 1–2 work:

1. **MSP-over-UART bridge.**
   - Out: `MSP_SET_RAW_RC`.
   - In: `MSP_ATTITUDE`, `MSP_RAW_IMU`, `MSP_STATUS`, battery.
   - Unit conversion: gyro raw counts ÷ 16.4 = °/s [ext].
2. **Camera capture on the Jetson** (Argus/GStreamer), resized to 640×360, carrying capture timestamps.
3. **YOLO on the Orin NX:** JetPack torch or a TensorRT engine; Etienne has TensorRT tooling.
4. **Gate detector retrained on real gate images.** Current weights were trained on simulator frames.
5. **Onboard gate/lap counter.** The simulator's `race_status` does not exist on the real drone, and both learned planners **raise an error without it** [V].
6. **Real-drone safety:**
   - The crash monitor, auto re-arm and finish handling are all simulator-based [V].
   - IMU staleness timeout is 1.25 s [V].
   - With MSP override on, Betaflight 4.4.3 **keeps using the last command with no timeout** if the Jetson stops sending [V: `src/main/rx/msp_override.c`].
7. **Linux/headless cleanup:** Windows-only quit key, OpenCV display on by default [V].

---

## 7. Findings to raise with the team

### 7a. Isaac gate-crossing bug

**Evidence** [V: code, and Isaac Lab 2.1 step order]:
- `tasks/drone_racer/mdp/commands.py:234`: on *any* environment reset, `prev_robot_pos_w` is overwritten for **all** environments.
- Isaac Lab then runs the crossing check in the same step (reset first, then `command_manager.compute`).
- So any step with a reset anywhere records **no** gate crossings.
- With 4096 environments, resets happen almost every step.

**Symptom:** the Sep 6 run, same settings as the keeper, logged about 0 gates and exactly 0 misses from its first window.

**Impact:**
- Any warm-start retraining (which `PHYSICAL_TUNING.md` plans for on site) will likely collapse.
- Single-environment play is mostly unaffected.

**Candidate fix** (not applied to the repo): update only the reset environments' entries.
```python
prev = self.prev_robot_pos_w.clone()
prev[env_ids] = self.robot.data.root_pos_w[env_ids]
self.prev_robot_pos_w = prev
```
Clone first: line 327 stores `root_pos_w` without copying, so writing into it in place could corrupt Isaac's own buffer.

**A/B confirmation (run on this laptop, 2026-09-16)** [V]:
- Two scratch copies, both warm-started from `pq_speed_best.pt`: seed 7, 2048 envs, 30 iterations.
- Logs: `~/aigp/idr_A/logs` and `~/aigp/idr_B/logs`.

| | Current code | One-line fix |
|---|---|---|
| `Episode_Reward/gate_passed` (mean / last) | 0.014 / 0.000 | 1.11 / 1.33 |
| `Episode_Termination/gate_missed` (mean) | **0.000** | 0.16 |
| Total reward (mean / last) | −1.6 / −2.2 | +66.7 / +84.5 |

The current code reproduces the Sep 6 signature (zero passes, zero misses) from the keeper itself. The fix restores gate crossings immediately.

- **Implication:** the keeper was trained *before* this line was introduced *(inference)*.
- **Action:** don't retrain until the fix is in, and Geneustace should decide how to apply it.

### 7b. Camera model does not match the PQ camera

| | Code (`aigp_obs.py`, `camera_model.py`, `config.py`, `race_obs.py`) | PQ spec |
|---|---|---|
| Resolution | 640×360 | 1920×1080, downscaled to 640×360 |
| Focal length | fx = fy = 320 | fx ≈ 417 |
| Horizontal FOV | 90° | 75° |
| Tilt | Fixed 20° | Adjustable, unknown |
| Distortion | None | M12 lens distortion |
| Shutter / rate | Global, 30 Hz | Rolling shutter, 30 or 60 fps |

- Gates therefore look about 30% bigger and farther off-center than the policy and YOLO were trained on.
- A 75° image cannot be remapped to 90° (the missing edges don't exist).
- The team's own `PHYSICAL_TUNING.md` says "camera fx changed ⇒ retrain".
- Calibrate on day 1 with a checkerboard.

### 7c. Isaac track versus the real course

Mapping of Isaac track entries to official gates:

| Isaac entry | 1 | 2 | 3 | 4–5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|
| Official gate | G6 | G7 | G8 | G9 top / bottom | G10 | **G1 (start)** | G2 | G3 | G4 | G5 |

- **Matches:** gate order, direction (clockwise, left side northbound) and the stacked double gate.
- **Scale:** about 7% small. North–south span 37.3 m vs 39.8 m; east–west 16.7 m vs 18.0 m.
- **Heights:** 2.0 m and 4.7 m assumed; about 1.35 m and 4.05 m on site [ext].
- **Headings:** G5 is 35° off and G8 13° off.
- **Start:** Play starts at G6. The real start is G1, from a start box 7.4 m south of it, **taking off from the ground** (never trained).
- **Not modelled:** pylons and walls.
- The official coordinate table would fix positions and headings exactly.

### 7d. The FC's input curves are nonlinear [ext, another team's drone; verify on ours]

- **Rates:** Betaflight rates with super rate 0.75 give about 440°/s at full stick, nonlinear. The Isaac decode assumes linear ±3.2 rad/s.
- **Throttle:** `thr_mid 54`, `thr_expo 68`. The plant assumes stick maps linearly to thrust.
- **Modes:** no ANGLE mode is assigned; override covers roll, pitch and yaw only; ARM is on AUX1 and override on AUX5.

### 7e. Smaller items [V]

- The bridge's default weights `models/isaac_thrust_rates.pt` are missing.
- `SNAP_VISUAL` defaults on in the bridge but was never used in Isaac training. Consider `OBS_SNAP_VISUAL=0`.
- The two AI_GP-vs-Isaac parity tests silently skip after vendoring (path moved). With the path fixed, all 29 pass, so the two observation implementations agree.
- `isaac_drone_racer/.gitattributes` routes `*.pt` to Git LFS, but they were committed as plain files. That is harmless without git-lfs.

---

## 8. Files outside git (priority order)

Gene reports everything is pushed. The items below are either gitignored (so they only move by drive or USB) or referenced but absent.

**Scope update (team, Sep 16):** the VQ-simulator models and the behavior-cloning / HG-DAgger work are retired. Only the Isaac PQ stack and Plan B matter. Items 2–6 and 9 below are therefore **not needed**; they stay listed for the record. The Isaac keeper and all its runs are already in the repo.

1. **`D:\Code\Competitions\isaac_drone_racer`** [V: every run's `agent.yaml` points there]:
   - `git log` and `git remote -v`;
   - any `logs/skrl` runs **after 2026-09-06**;
   - `logs/runlogs/*.log`;
   - a diff against the vendored copy.
2. Root `models/`:
   - `policy_seed_17.pt` (default timed policy), `policy_seed_18.pt`;
   - **`gate_pose_v5.pt`** (default detector; without it `vision_rx` refuses to start);
   - `isaac_thrust_rates.pt`, `gate_pose_isaac_hsv.pt`, `ROBOFLOW_*.pt`, `gate_detector.pt`.
3. `datasets/AIGP_8keypoints.v5i.yolov8/` (or Roboflow access), plus `isaac_drone_racer/datasets/isaac_gate_pose_*`. The nested `.gitignore` blocks these even with `git add`.
4. `logs/seed/` (18 human laps), `logs/coach/`, `logs/best/telem_best.csv`, `pilot_best.csv`.
5. `vision/cam_bank_bias_learner.py` and `vision/lateral_yaw_learner.py`. The default `assist` mode crashes without them.
6. Never-committed tools the docs reference: `tools/preflight.py`, `replay_attitude.py`, `probe_race_clock.py`, `score_runs.py`, `approach_metrics.py`, `build_tracking_tape.py`.
7. Any real-drone work (MSP, Jetson, camera) and `logs/pad_constants.json`, if pad checks happened.
8. Environment recipes: `pip freeze` of `winvenv` and `D:\isaacsim_venv`, `conda env export` of `env_isaaclab`, and the `sitecustomize.py`.
9. Also ask Etienne for GateNet `best.pt`; ask Rocky about the unpushed `manual-control-vq2` branch (historical only).

**Transfer tip:** the repo is public, so weights and data are better shared by drive or USB than committed.

---

## 9. Next steps

### Tonight (off-site)
- Send the §8 list to Geneustace. Ask directly: *"Do you have MSP/Jetson code, and did you train after Sep 6?"*
- Send the §7a–7d findings to the team for discussion.
- Email the organizer questions (`PLAN_B.md` §8), especially:
  - the gate-coordinate PDF and heights;
  - whether FC mode and override-mask changes are allowed;
  - how gates are counted in a partial run.
- Optional prep here: an MSP codec with a fake-FC test, a logger skeleton, a detector replay harness.

### Thu 9/17 (arrival)
- Safety briefing and badges.
- **Bring-up before any autonomy:**
  - organizer demo apps;
  - FC `version` and `diff all` backup on every drone;
  - camera capture and fps;
  - MSP read of IMU and attitude.
- Props-off override bench: RC echo, signs, what happens when the Jetson stops.
- Pilot flies manual laps **with logging**, for real gate images, hover throttle and IMU noise.
- Checkerboard camera calibration; measure the tilt.

### Fri 9/18 – Sun 9/20
- Run Plan A and Plan B in parallel on the shared foundation.
- **Sunday: go/no-go for Plan A** (criteria in `PLAN_B.md` §7).

### Mon 9/21 – Tue 9/22 (scored)
- Bank a valid run first, then improve. Change only parameters, and log everything.

---

## 10. This laptop

- **Hardware:** RTX 4060 Laptop (8 GB), 15 GB RAM (WSL sees about 7.6 GB), WSL Ubuntu 26.04, no sudo.
- **Repo:** `ai-grand-prix/AI_GP`, branch `isaacsim`, **read-only for us**.
- **Flight client:** `~/aigp/venv-client`.
  - Python 3.12, torch 2.14 + CUDA, ultralytics, OpenCV, pymavlink, pygame, pytest, tensorboard.
  - Run tests: `cd AI_GP && ~/aigp/venv-client/bin/python -m pytest -q --ignore=isaac_drone_racer`
- **Isaac:** `source ~/aigp/isaac_env.sh`.
  - Python 3.10, Isaac Sim 4.5, Isaac Lab 2.1 in `~/aigp/IsaacLab`, skrl 1.4.2, torch 2.5.1+cu121.
  - Version pins: numpy<2, transformers<4.50, warp-lang<1.8, gymnasium<1.2.
  - Missing system libraries were unpacked from `.deb` files into `~/aigp/syslibs`.
  - Train from a scratch copy, **not** from the repo (training writes `logs/` into the current directory). Example:
    ```bash
    source ~/aigp/isaac_env.sh
    cd ~/aigp/idr_smoke
    python -u scripts/rl/train.py --task Isaac-Drone-Racer-v0 --headless --num_envs 2048 --max_iterations 30
    ```
  - Isaac often hangs on exit after writing checkpoints; `pkill -9 -f scripts/rl/train.py`.
  - The Isaac GUI and FPV camera **do not render under WSL** (no Vulkan device). Headless training only. Watching needs the Windows install (see `AI_GP/ISAACSIM.md`).
  - With only about 7.6 GB of WSL RAM, 4096 environments may not fit. A `%UserProfile%\.wslconfig` with `memory=12GB` followed by `wsl --shutdown` would help, but it restarts WSL.

---

## Glossary

| Term | Meaning |
|---|---|
| **VQ1 / VQ2 / PQ** | Virtual qualifiers 1 and 2, physical qualifier |
| **MAVLink / MSP** | Drone telemetry and command protocols: MAVLink for PX4/ArduPilot and the VQ simulator; MSP for Betaflight |
| **ACRO / ANGLE** | Betaflight flight modes. ACRO: sticks command rotation *rates*. ANGLE: sticks command *tilt*, and the FC self-levels. |
| **MSP override** | Betaflight feature: while an AUX switch is on, chosen RC channels come from a companion computer over MSP |
| **PPO** | Proximal Policy Optimization, the RL algorithm used in Isaac |
| **skrl** | The RL library used to run PPO |
| **HG-DAgger** | Human-gated DAgger: the policy flies, a human takes over on mistakes, and the corrections become training data |
| **TCN** | Temporal convolutional network (the imitation policy) |
| **Keypoints** | The 8 gate corners the YOLO-pose model outputs |
| **PnP** | Perspective-n-Point: solving camera pose from known 3-D corners |
| **EKF / AHRS** | Extended Kalman filter / attitude-heading estimator |
| **Keeper** | The team's word for the best checkpoint worth keeping |
