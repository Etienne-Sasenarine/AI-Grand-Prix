# 02 — The simplest path to a stable autonomous hover

Research report, 19 Sep 2026. Scope: get the 1745 g, 8-inch Betaflight 4.4.3 drone to hold a hover
for 10 seconds under Jetson control, with the sensors it actually has.

**Tags.** `[CITED: url]` = published source. `[TEAM-MEASURED: file]` = our own data or code.
`[SAURON]` = the other team's notes, as already summarised in `SPEC.md` §11 (their repo was not
present at `/tmp/sauron` when this was written, so every [SAURON] item here is second-hand via
`SPEC.md`). `[MY ESTIMATE]` = my arithmetic or judgement, not a measurement — check before relying on it.

**Terms.** *Feedforward* = the throttle you send before any correction, i.e. your best guess of
hover throttle. *P / I / D* = correction proportional to the height error / to its running sum / to
the vertical speed. *ANGLE mode* = the flight controller (FC) self-levels; stick = tilt angle.
*Override mask* = which radio channels the Jetson is allowed to replace (4 = throttle only,
15 = roll, pitch, throttle, yaw). *PnP* = computing camera position from the pixel positions of
points whose real-world layout is known. *Ground effect* = extra lift close to the floor.

---

## 0. The answer in six sentences

1. **The hover problem is a height-sensor problem.** The drone has no working height sensor once the
   props turn, and the current script flies on the one sensor that is known to be broken.
2. **Do not start from the ground.** Let the pilot take off and hover in ANGLE mode, then hand over
   *throttle only*. This skips ground effect, liftoff detection and the unknown hover throttle in one move,
   because the pilot's own stick position at the moment of handover *is* the hover throttle.
3. **Get height from the forward camera looking at one known-size target** 3–6 m ahead: an ArUco
   marker on a stand first (an afternoon's work with OpenCV only), a race gate second (same maths, needs a detector).
4. **The same target also gives left/right and forward/back position**, which is the pose file
   `xy_hold.py` is waiting for and nobody produces.
5. **Use gentle gains.** With ~25 Hz feedback and 0.1–0.25 s of total delay the height loop can only
   be slow; my simulation says the teammate's gains are 3–4× too hot even with a perfect sensor (§4).
6. **Ask the organizers today whether a 5 g downward laser rangefinder wired to the Jetson is
   allowed.** The 2019 AIRR winners flew with one [CITED: https://arxiv.org/html/2109.14985v1]. It is the single best fix if permitted.

---

## 1. Recommended ladder to a first 10-second hover

Each rung is small, has a pass/fail check, and says what to do on a fail. Do not skip rungs. The
pilot owns **arm** and the **override switch** throughout [SAURON]. Betaflight's MSP override
replaces only the masked channels while the override mode switch is on
[CITED: https://github.com/betaflight/betaflight/issues/14004].

| # | Rung (risk) | What you do | PASS when | If it FAILS |
|---|---|---|---|---|
| **0** | **Props OFF, bench** (none) | Run the hover script with a *fake* height fed in (a number you type or a slow sine). Watch `MSP_MOTOR` and `rc_channels()` respond. Then test the three takeovers: (a) override switch OFF, (b) kill the script with override ON, (c) pull the Jetson UART. | Motors follow the commanded throttle with the right sign; switch OFF returns throttle to the radio in < 0.2 s every time, 10 of 10; you have **written down** what (b) and (c) do. | If (b)/(c) freeze the last throttle (the teammate's doc suspects this [TEAM-MEASURED: orin-setup/HOVER_10S.md]; [SAURON] saw the same), the rule is: *pilot flips override OFF before anything else, always.* Brief the pilot. Do not fly until (a) is 10/10. |
| **1** | **Pilot hover, Jetson only logs** (normal flying) | ANGLE mode, pilot hovers at ~1 m for 3 × 20 s, away from walls. Log stick, `motors()`, RPM, voltage, attitude, baro (for the record). `tools/flight_recorder.py` / `hover_logger.py` exist. | Three medians of throttle stick agree within **±0.01** (10 µs). You now have `u_hover` (stick), mean motor output, mean RPM² and volts. | Spread bigger than that → pilot was climbing/descending or battery sagging; repeat with a fresh pack and a height reference (pilot holds the drone level with a gate's lower bar). |
| **2** | **Open-loop throttle hold** (low) | Mask = 4. Pilot hovers at ~1 m, flips override ON. Jetson sends **a constant throttle = the pilot's mean stick over the previous 1 s** and nothing else, for 5 s, then beeps; pilot flips OFF. Pilot flies roll/pitch/yaw the whole time. | No jump at handover; height drifts < 0.5 m in 5 s; handback is smooth. | A jump at handover = channel scaling or throttle-curve mismatch between radio µs and what the script sends; fix on the bench (rung 0). Fast climb/sink = the averaging window caught a stick movement; lengthen it to 2 s. |
| **3** | **Vision height, read-only** (none → normal flying) | First props off: hold the drone by hand at 0.5 / 1.0 / 1.5 m (tape measure) facing the target from 4 m; log the vision height. Then the pilot flies gentle ups and downs while it logs. | Static error ≤ 5 cm, noise ≤ 2 cm RMS, ≥ 20 valid fixes per second, dropouts < 0.3 s, latency measured (blink an LED or wave the drone and compare with gyro). | Constant offset → camera tilt is not exactly 20°; calibrate it out (it does not hurt hovering anyway). Noisy → process at full 1080p in a cropped window, fix the exposure short. Dropouts → bigger marker / closer. |
| **4** | **Closed-loop height, pilot on XY** (medium) | Mask = 4. As rung 2, but after handover the Jetson runs the height controller of §4 with target = *height at handover*. Start with gains halved. 10 s, then beep. | Height stays within ±0.15 m for 10 s, no growing oscillation. **This is the "first autonomous hover" milestone for the vertical axis.** | Slow growing oscillation (period 2–4 s) → halve `Kp` and `Kd`. Steady offset → integrator too slow, raise `Ki` by 50 %. Twitchy throttle → more velocity filtering, or lower `Kd`. |
| **5** | **Add XY hold** (medium-high) | Mask = 15. Feed the same target's sideways/forward position into `xy_hold.py` (it already limits tilt to 0.3°–2° [TEAM-MEASURED: orin-setup/xy_hold.py]). **Verify stick direction props-off first**: push the drone left by hand, see the roll command go right. | Stays in a 0.5 m circle for 10 s. | Runs away in one direction = sign error → override OFF, fix, rung 0 again. |
| **6** | **Takeoff from the floor** (high) | Only now. Fast ramp to `u_hover − 0.04`, then slow ramp at 0.02 /s until vision height > 0.3 m, then hand to the height loop with its integrator pre-loaded to the current throttle. | Leaves the floor without bouncing or ballooning above 1.2 m. | Balloons → the ramp overshot through ground effect (§2.2); lower the slow-ramp rate and start the loop at 0.2 m. |

Optional rig: a **slack ceiling line** (light cord from the drone's underside to a floor weight,
length 1.5 m) stops a fly-away into the net. It must stay slack in normal hover — a taut tether
flips a quadrotor. [MY ESTIMATE] Hand-holding a 1.7 kg 8-inch drone with props on is not worth the risk.

---

## 2. Height-sensing options for THIS drone, ranked

| Rank | Option | Works here? | Accuracy / rate | Effort | Verdict |
|---|---|---|---|---|---|
| **1** | **Downward ToF rangefinder wired to the Jetson** (VL53L1X ≈ 4 m range, I²C; or TFmini, UART). Not through Betaflight. | Only if organizers allow added hardware. | ~1–3 cm, 30–50 Hz, no drift, works from the floor up [MY ESTIMATE from datasheet-class figures]. | 2–3 h incl. mounting; `i2c-tools` is already on the image [TEAM-MEASURED: SPEC.md §"Installed"]. | **Best if legal. Ask first (§5).** The AIRR-2019 winners fused a downward laser rangefinder with filtered vertical acceleration for height [CITED: https://arxiv.org/html/2109.14985v1]. |
| **2** | **ArUco / ChArUco marker on a stand, seen by the forward camera** | Yes. OpenCV ≥ 4.7 has `cv2.aruco` in the main package [MY ESTIMATE — run `python3 -c "import cv2;print(cv2.__version__, hasattr(cv2,'aruco'))"` on the Jetson]. | ~1–2 cm range error and < 1 cm height noise at 4 m with a 25 cm marker [MY ESTIMATE, §2.1]; plus ~7 cm per degree of pitch error. 30–60 Hz. | Half a day. No training, no detector. | **Recommended first.** Gives height **and** XY **and** yaw from one target. Practice-only: there are no markers on the scored course. |
| **3** | **Race gate (known 1.5 m opening / 2.7 m outer) seen by the forward camera**, PnP — what `flight/pnp.py` + `pose_filter.py` already do | Yes, *once a gate-corner detector exists on the Jetson*. None does today [TEAM-MEASURED: EXECUTION.md C7/C8]. | Similar to the marker; geometry limits in §2.1. | The detector is the cost (classical colour detector: ~2 h tuning per `EXECUTION.md` C7). | **The race solution** — every serious team does this: Swift fuses gate-corner PnP with VIO in a Kalman filter [CITED: https://www.nature.com/articles/s41586-023-06419-4]; MonoRace uses camera + IMU only, *no baro, no rangefinder*, PnP on gate corners into an EKF [CITED: https://arxiv.org/html/2601.15222v1]. Step to this after the marker works; the code downstream is identical. |
| **4** | **Open-loop thrust model** (constant hover throttle, optionally scaled by RPM² or voltage) | Partly. | Holds height for seconds, not tens of seconds: a 0.005 throttle error ≈ 0.25–0.5 m/s² ≈ 0.5–1 m of drift in 2 s [MY ESTIMATE, §4]. | Trivial. | **Use as the feedforward and as the 1-second fallback when vision drops out — never alone.** |
| **5** | **Accelerometer integration** | No. | [SAURON] measured 0.985 g vs 1.000 g at throttle 0.18 (vibration rectification: vibration turns into a false constant offset); they built an accel hold and removed it. Our own log agrees: mean `az` falls from 505 counts (props off) to 502 (idle) to 500.5 near hover, i.e. ≈ −0.9 % g ≈ 0.09 m/s², and its scatter grows from 0.5 to 15–125 counts [TEAM-MEASURED: logs/orin/hover_20260918_004725.csv]. 0.09 m/s² integrates to 4.5 m in 10 s. Heavy vibration also clips/aliases; ArduPilot treats > 30 m/s² vibration as estimator-breaking [CITED: https://ardupilot.org/copter/docs/vibration-failsafe.html]. At 23 Hz we also alias everything. | — | **Not as a height source.** Possibly later as short-term damping blended with vision; not for the first hover. |
| **6** | **Barometer** | **No.** | Reads 0.2 m low at idle, 0.6 m low at 1046 µs, 1.6 m at 1118 µs, 3.9 m at 1173 µs, 4.5 m at 1190 µs; snaps back to 0 on disarm; `vario` is 0.000 on all 2828 samples [TEAM-MEASURED: logs/orin/hover_20260918_004725.csv]. Known mechanism: prop-wash pressure on an unshielded baro [CITED: https://betaflight.com/docs/wiki/guides/current/Barometer]; Betaflight's own tracker describes this causing "excessive thrust" in altitude control [CITED: https://github.com/betaflight/betaflight/issues/15583]. | — | **Dead. The error is a function of throttle, which makes it worse than noise — see §3.** Open-cell foam over the sensor is the textbook cure, but it means opening the stack; ask (§5). |

Betaflight-side rangefinder: needs a custom build with `USE_RANGEFINDER` + a sensor define, and even
then the reading "needs further integration in flight code" before 4.6
[CITED: https://github.com/betaflight/betaflight/discussions/14314]; altitude hold arrives only in
4.6 / 2025.12 [CITED: https://oscarliang.com/betaflight-4-6/]. Our dump does list `feature -RANGEFINDER`
and `SONAR_*` resources [TEAM-MEASURED: logs/orin/bf_dump_before_beginner_20260918.txt], so it may be
compiled in, but with no reflashing and nothing in 4.4.3 using it, **wire any rangefinder to the Jetson, not the FC.**

### 2.1 Geometry: what does a 20°-up camera see from a low hover?

Numbers: 1920×1080, `fx ≈ 1275 px` (= 425 at 640×360 × 3) [TEAM-MEASURED: SPEC.md §Camera]. All below is [MY ESTIMATE] trigonometry.

- Vertical field of view = 2·atan(540/1275) = **45.9°**. Tilted 20° up and with the body level
  (ANGLE-mode hover), the image covers elevations from **−2.9° (bottom edge) to +42.9° (top edge)**. The camera barely looks below the horizon.
- A gate opening spans 0.6 m to 2.1 m above the floor (centre 1.35 m [TEAM-MEASURED: SPEC.md]). The **lower bar of the opening is visible only if the camera height `h ≤ 0.6 + 0.05·d`** (d = distance to gate):

| Distance d | Max hover height to see all 4 inner corners | What you see from 1.35–1.5 m |
|---|---|---|
| 3 m | 0.75 m | top half of the gate only |
| 4 m | **0.80 m** | top bar + two top corners (top bar at +8° to +11°) |
| 6 m | 0.90 m | same |
| 8 m | 1.00 m | same; lower bar appears only beyond ~15–18 m |

- The floor at the gate's feet is never visible inside 20 m from a 1 m hover. The top outer edge (2.7 m) is in view from d > 2.3 m.
- **Conclusion: hover at 0.7–0.8 m, 4–6 m in front of the gate (or marker), facing it.** All four inner corners are then in frame with ~60 px to spare at the bottom. The teammate's 0.6 m target is compatible. At the race height of 1.35 m only the top two corners are visible when close — still enough for height (below), but not for a 4-corner PnP.
- **Height observability** (how far the target moves in the image per cm of height change ≈ `fx / d`):

| | at 1080p | at 640×360 |
|---|---|---|
| 4 m | **3.2 px per cm** | 1.06 px per cm |
| 8 m | **1.6 px per cm** | 0.53 px per cm |

  With ~1 px corner noise, height noise is ≈ 0.3 cm at 4 m and ≈ 0.6 cm at 8 m (1080p), or 1–2 cm at 640×360. That is far better than needed. Process a cropped 1080p window around the last detection rather than down-scaling.
- **The real error is pitch, not pixels.** 1° of pitch error moves the image 22 px = **7 cm of false height at 4 m, 14 cm at 8 m**. FC attitude resolution is 0.1° [TEAM-MEASURED: SPEC.md], and in a steady ANGLE hover it is good; during acceleration the FC's accelerometer-based levelling can be off by a degree or more. Two consequences: stay close (4 m, not 8 m), and a *constant* tilt-mount error only gives a constant height offset, which a hover does not care about.
- **Range** comes from apparent size: the 1.5 m opening is 478 px wide at 4 m, 239 px at 8 m; a 1 px error is 0.8 cm / 3.3 cm of range. Two top corners + FC roll/pitch are enough: range from their spacing, height from their row. That is the fallback when the lower bar is out of view.
- **Marker sizing:** a 25 cm ArUco at 4 m is 80 px across at 1080p — comfortable. Mount its centre ~1.8–2.0 m high (the image centre from a 0.8 m hover at 3–4 m looks at 1.9–2.3 m), so it is also visible from the floor for rung 6.
- The camera sits 12 cm ahead of the body origin; for a hover this is a constant and can be ignored. Rolling shutter is irrelevant at hover speeds.

### 2.2 Ground effect for 8-inch props (rotor radius R = 0.10 m)

- Classical single-rotor formula (Cheeseman–Bennett): thrust ratio = 1 / (1 − (R/4z)²) → +33 % at z = 0.5 R (5 cm), +6.7 % at z = R (10 cm), +1.6 % at z = 2R (20 cm) [MY ESTIMATE from the formula; formula described in CITED: https://www.sciencedirect.com/science/article/pii/S1000936124000645].
- Quadrotors feel more than a single rotor: ~15 % (isolated rotor) vs ~**20 % (quadrotor layout) below z/R ≈ 1**, negligible above z/R ≈ 2 for a single rotor [CITED: https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/aerodynamic-effects-of-rotorrotor-interaction-on-a-twinrotor-system-in-ground-effect/B57E828030CE5C29E4315DF0F05D96D8]; the classical model under-predicts for quadrotors, where the effect reaches higher [CITED: https://onlinelibrary.wiley.com/doi/10.1155/2017/1823056].
- For us: props sit roughly 5–10 cm off the floor on the pad, so expect **~10–30 % free lift on the ground, fading to nothing by about 0.3–0.5 m** [MY ESTIMATE]. This is why [SAURON] saw the drone "lift onto the cushion near 0.18" while hover is ~0.21–0.25, and why a throttle that just unsticks the drone is *not* hover throttle. **Hover at ≥ 0.6 m and never calibrate hover throttle below 0.5 m.**

---

## 3. Critique of the current controller (`orin-setup/hover_10s.py`)

The code is careful in most respects: pilot owns arming, bounded throttle (≤ 0.28), slew limit, a
checks file that currently blocks launch, honest caveats. `HoverSequence.update()` takes a plain
`Sample(t, height, vz, tilt)` — so the control maths is cleanly separable from the sensor. The
problem is the sensor.

**What will happen if it is flown as written** [MY ESTIMATE, from TEAM-MEASURED data]:

1. Ground zero is captured disarmed, where the baro is fine (the 10 s noise check passes — that check only proves the baro works with props *off*).
2. Pilot arms: baro already reads ≈ −0.2 m at idle. The script ramps its setpoint up; the baro says "below target", so throttle rises.
3. **Positive feedback.** Our log shows the baro error grows with throttle: from 1118 µs to 1190 µs the reading falls 2.9 m, i.e. roughly **−40 m per unit of throttle**. The controller's effective P gain is 0.10 + 0.08×0.8 = 0.164 throttle per metre. Loop gain of this false path ≈ 40 × 0.164 ≈ **6.6 — far above 1**. Every bit of extra throttle makes the baro read lower, which demands more throttle.
4. Throttle therefore runs to the 0.28 cap at the slew limit (0.08 /s, about 3.5 s) and stays there. With hover near 0.21, 0.28 is 30–60 % excess thrust: the drone accelerates upward at several m/s² into the net.
5. None of the guards fire: the 0.91 m ceiling and the hover-trim integrator both need a *positive* height and the baro is reading −1 to −6 m; the 0.40 m-per-sample discontinuity check is not tripped by a fall of ~5 cm per sample; `vz` from the baro is just the derivative of the throttle ramp. Only the pilot's switch or the 40 s timeout ends it — and on a fault the stream stops, which may **freeze throttle at 0.28** (their own doc's warning).
6. This is exactly what [SAURON] reported: their baro altitude-hold "latched a fake liftoff 0.14–0.36 s into every run", and another attempt "lifted and ran away".

Even with a perfect height sensor the gains are too hot: P+D on a velocity that has passed through two
0.15 s filters (≈ 0.3 s of lag on the damping term), on top of link and motor delay. In my simulation
it oscillates with growing amplitude in every case tried, including the most favourable (§4). This is the
general reason "P on a noisy derivative oscillates": to make the derivative usable you must filter it,
the filter adds delay, and delayed damping is not damping.

**Minimal change to make it safe and useful:**

1. **Delete the baro as a control input** (keep logging it). Feed `Sample.height/vz` from vision (§2 rank 2/3) or a ToF. If neither is ready, run rung 2 (constant throttle) — it is still progress.
2. **Start in the air from a pilot handover**, not from the ground: target height = height at handover; feedforward = pilot's mean stick over the last 1–2 s instead of the fixed 0.208.
3. **Cut the gains** to §4's values and shorten the velocity filter to ~0.06–0.10 s (vision height is clean enough; the baro was not).
4. **On any fault, do not stop streaming**: keep sending the feedforward throttle minus 0.01 (a gentle sink) while shouting for the pilot. A frozen hover-ish throttle is survivable; a frozen 0.28 is not. Also cap at `u_hover + 0.05`, not at a fixed 0.28.
5. The read-only rehearsal mode proves nothing about flight (disarmed baro). Replace it with rung 3.

---

## 4. Control design for the vertical axis

**The plant.** Near hover, vertical acceleration ≈ `k · (u − u_hover)`, with `k` between `g/u_hover`
(thrust linear in stick) and `2g/u_hover` (thrust ∝ stick²). For `u_hover ≈ 0.21` that is
**k ≈ 47–94 m/s² per unit throttle**; `thrust_linear = 40` in our dump [TEAM-MEASURED: bf_dump]
pushes it toward the lower end. So **0.01 of throttle (10 µs) ≈ 0.5–0.9 m/s²** [MY ESTIMATE]. This one number explains the struggle:
a hover-throttle guess that is off by 0.03 is a 1.4–2.8 m/s² shove — the drone is on the floor or a
metre up within a second, before any gentle controller can react. **Feedforward accuracy matters more than gains.**

**The delay budget** [MY ESTIMATE unless tagged]: camera + detection 40–80 ms; control tick ≤ 40 ms;
MSP send at 50 Hz ≤ 20 ms; FC throttle smoothing at 10 Hz ≈ 16 ms [TEAM-MEASURED: bf_dump `rc_smoothing_throttle_cutoff = 10`];
8-inch motor/prop response 50–80 ms. Total ≈ **0.15–0.25 s**. A loop with this much delay cannot have a bandwidth much above ~1–1.5 rad/s: expect settling in 3–5 s, not 0.5 s. That is fine for a hover.

**Recommended law** (run it at the vision rate, 25–30 Hz; send RC at 50 Hz):

```
e   = clamp(z_target − z, ±0.30 m)
u   = u_ff + I + Kp·e − Kd·vz          # vz = low-passed d(z)/dt, time constant 0.06–0.10 s
u   = clamp(u, u_ff − 0.05, u_ff + 0.05), then slew-limit to 0.30 per second
I  += Ki·e·dt   only while u is not clamped (anti-windup), |I| ≤ 0.04
```

| Parameter | Start value | Range to explore | Note |
|---|---|---|---|
| `u_ff` | pilot's mean stick over the 1–2 s before handover | — | never a constant from a file for flight #1 |
| `Kp` | **0.04** throttle per m | 0.02–0.06 | teammate's effective value: 0.164 |
| `Kd` | **0.04** throttle per (m/s) | 0.03–0.06 | teammate's: 0.08 on a 0.3 s-lagged signal |
| `Ki` | **0.015** per m·s | 0.01–0.03 | this *is* the hover-throttle learner and the battery-sag compensator |
| Authority | ±0.05 around `u_ff` | ±0.03 first flight | ±0.05 ≈ ±2.5–4.5 m/s² |
| Vision dropout | hold `u_ff + I` (no P, no D) up to 1 s, then beep for takeover | | never cut throttle in the air |
| Ceiling | z > target + 0.6 m → `u_ff − 0.02` until below | | |

Simulation evidence [MY ESTIMATE — a 40-line double-integrator model with 70 ms motor lag, 25 Hz
sampling, 1 cm height noise; not the real drone]: with these gains the hover holds ±1 cm and a 0.2 m
step settles without sustained oscillation for k = 47–94 and delays of 0.10–0.18 s; it becomes marginal only at
k = 94 *and* 0.25 s delay — so if you see a slow growing oscillation, halve `Kp` and `Kd`. With the
teammate's gains and filters the same model diverges in all nine cases. With a ±0.03 feedforward error even the good gains hit the floor
or overshoot by > 1 m, which is the argument for the handover trick.

**Battery sag.** Pack voltage fell 24.0 → 23.2 V within one short hover [TEAM-MEASURED: hover_20260918_004725.csv];
thrust at a fixed command falls roughly with voltage squared, so expect hover throttle to creep up
5–15 % across a pack [MY ESTIMATE]. The slow integrator absorbs this. Later refinement: since
bidirectional DShot RPM is live [TEAM-MEASURED: SPEC.md], log mean RPM² at hover — thrust ∝ RPM² regardless of voltage — and use it to predict `u_ff` for the next flight. Not needed for the first hover.

**Takeoff later (rung 6):** "punch through" ground effect is the racing-pilot habit, but it needs a
height sensor that works from 0 m and a known thrust; with neither, use the two-stage ramp in rung 6
and let the vision loop catch the drone at 0.3 m. `flight/flight_loop.py` already takes the right
stance — no liftoff detection, integrator learns hover [TEAM-MEASURED: flight/flight_loop.py] — but its gains (kp 0.30, ki 0.15) were tuned in a simulator without this delay; re-check them against §4 before flight.

**What to log, every tick** (one CSV row): monotonic time; frame capture timestamp; vision `z, x, y`,
range, number of corners, valid flag; filtered `z`, `vz`; `u_ff`, `I`, P term, D term, final `u`,
clamp/slew flags; FC roll/pitch/yaw; raw gyro and accel; four motor outputs; four RPMs; voltage;
baro altitude (as evidence only); armed / override / ANGLE flags; pilot's own throttle stick (so you
can see what the pilot would have done); loop period. Label the file with the airframe `mcu_id` [TEAM-MEASURED: SPEC.md].

**How others do it, for orientation.** PX4 users without GPS feed an external/vision position into
the estimator at 30–50 Hz, set height reference to "Vision", tune a delay parameter, and verify axis
signs by hand before the first flight [CITED: https://docs.px4.io/main/en/ros/external_position_estimation.html] — the same recipe as rungs 3–5. MAVLab's AIRR winner deliberately avoided feature-based visual odometry, used gate-corner PnP combined
with the onboard attitude (found "more robust" than full PnP attitude), and predicted motion with a drag model weighted 85 % against
15 % accelerometer [CITED: https://arxiv.org/html/2109.14985v1] — i.e. trust the model and the gates, not the accelerometer. That matches `flight/pose_filter.py`'s design.

---

## 5. Questions for the organizers (ask today)

1. **May we add a small sensor wired only to the Jetson** — a downward time-of-flight rangefinder (VL53L1X, ~5 g, I²C) or an optical-flow board (PMW3901)? If yes: any mounting or weight rules, and is it allowed in scored runs or practice only?
2. **May we place our own visual marker (printed ArUco board on a stand) inside the netted area during our practice slots?**
3. Is there a supported way to hover-test away from the course (a smaller netted cage), and may we use a slack safety line?
4. The barometer reads metres low as soon as the props spin on our airframes. Is that known? May we open the stack and put open-cell foam over the DPS310, or will you?
5. **What does this firmware do with throttle when `MSP_SET_RAW_RC` frames stop while the override switch is ON** — hold the last value, fall back to the radio, or failsafe? After how long? (`failsafe_procedure = DROP` in our dump [TEAM-MEASURED: bf_dump].)
6. Is `msp_override_channels_mask` ours to set (we need 4 for throttle-only and 15 for full control; one dump shows 11 [TEAM-MEASURED: bf_dump])? Entering the CLI over the MSP UART wedged our board twice [TEAM-MEASURED: SPEC.md] — what is the safe way to change it?
7. Was this firmware built with rangefinder support (the dump lists `feature -RANGEFINDER` and `SONAR` resources)? Is any height source planned by you?
8. Does a scored run have to begin from the floor with an autonomous takeoff, and may the pilot arm and then flip a switch to start it?
9. As-built gate dimensions: is the opening exactly 1500 mm and its lower edge exactly 600 mm above the floor on every gate? (Our vision height is only as good as these numbers.)
10. What hover throttle and thrust-to-weight do *you* measure for this airframe at race weight?

---

## 6. Things I could not verify

- `/tmp/sauron` was absent; [SAURON] items are as quoted in `SPEC.md` §11.
- Whether `cv2.aruco` exists in the Jetson's OpenCV build — a one-line check (§2, rank 2).
- The exact ground-effect curve of Sanchez-Cuevas et al. (publisher page blocked); the 15 % / 20 % / z/R figures come from the JFM paper's summary of Conyers et al.
- All gains come from a toy model. They are starting points to be halved at the first sign of oscillation, not tuned values.
- The "MSP frames below 5 Hz fall back to radio" behaviour seen in web results is from an iNav pull request, not Betaflight 4.4.3 — hence organizer question 5 and rung 0(b).
