# 05 — Code review: why the first computer-controlled hover does not work

19 Sep 2026. Reviewed: John's `orin-setup/` on branch `codex/hover-10s` (HEAD `006575b`), the
organizers' `target/msp/`, our `flight/` and `tools/`, and the real FC dump. Nothing was changed,
no serial port was opened. Every number below comes from a throwaway script in
`<scratchpad>/review05/` (`demo1_xy.py`, `demo2_baro.py`, `demo2b_ideal.py`) that imports John's
real classes, or from the CSVs in `logs/orin/`.

**One-paragraph answer.** `hover_10s.py` cannot hover for three independent reasons, any one of which
is enough. (1) With our current `--beginner` switch setup it aborts the instant the pilot flips
override on, so it never spins up. (2) If that is fixed, its only height sensor is the barometer,
which on this airframe reads about **20 m low per unit of throttle** once the props load up; the
controller sees "−3 m", pins throttle at its 0.28 cap, and the real aircraft is through the 0.91 m
"ceiling" 0.9 s after liftoff and at ~9 m (i.e. in the net) 3 s later. (3) Even with a *perfect*
height sensor, the filtering added in `fly()` makes the loop overshoot to 1.8 m; the unit tests pass
only because they test a different, unfiltered loop. Separately, the XY hold has its pitch sign
backwards (runs away), and every abnormal exit freezes the last throttle. The safe part: the
checks file correctly blocks launch today. Do not "tick the boxes" to get past it.

Terms: *override* = Betaflight "MSP override" mode, where chosen stick channels come from the Jetson
instead of the radio. *Mask* = `msp_override_channels_mask`, which channels (bit 0 roll, 1 pitch,
2 throttle, 3 yaw; 4 = throttle only, 15 = all four). *ANGLE* = self-levelling mode (stick = tilt
angle). *ACRO* = rate mode (stick = rotation speed; no self-levelling). *us* = microseconds of RC
pulse, 1000–2000, 1500 = centre.

---

## 1. Prioritised defects

### D1 — CRITICAL — barometer as the only height sensor: positive feedback into the throttle cap
`hover_10s.py:355-369` (reads `fc.altitude()`), controller at `:94-98`.

Evidence from `logs/orin/hover_20260918_003956.csv` and `_004725.csv` (24 Hz, 2×120 s):
- Baro error tracks throttle: 1078 us → −0.9 m, 1111 → −1.4, 1130 → −1.8, 1165 → −3.0,
  1190 → −5.3 m. Least-squares fit: **error ≈ −20 m × normalised throttle, lag 0.5–2 s**
  (slope 20.5 and 20.5 in the two files). At John's hover 0.208 that is **−4.2 m**.
- Disarming recovers +6 to +7 m within 1 s. `vario` is exactly 0.000 in all 5,657 rows.
- Noise with props loaded: per-sample step std 0.14–0.18 m; **2.3–3.0 % of samples step more than
  0.40 m**. At rest the reading is quantised in ~0.09 m steps.
- Caveat: those logs were taken on/near the ground, where trapped prop-wash is worst. In free air
  the slope may be smaller. Halving it (K = 10) does not change the outcome below.

Simulation (`demo2_baro.py`: John's `HoverSequence` + the exact filter from `fly()`, 25 Hz loop,
his own plant model with true hover = 0.208, baro = true − 20·lag(throttle)):

| t | true height | baro says | throttle | state |
|---|---|---|---|---|
| 2 s | 0.00 m | −1.8 m | 0.160 (still ramping) | TAKEOFF |
| 3.8 s | **0.91 m (ceiling) — not noticed** | ≈ −3 m | **0.280 (cap)** | TAKEOFF |
| 5 s | 5.2 m | −0.1 m | 0.248 | TAKEOFF |
| 6 s | **8.9 m** | +4.4 m | 0.168 | LAND (ceiling finally seen) |

- The throttle **does go to the 0.28 cap**. At the cap, with hover 0.208, acceleration is
  (0.28/0.208 − 1)·g = **3.4 m/s²**: 0.91 m in 0.73 s, 3 m in 1.33 s. Ten seconds at the cap would
  be ~170 m on paper; in the cage it is the net in about two seconds.
- The 0.9144 m software ceiling fires ~2 s late and ~8 m too high, because it reads the same baro.
- The hover-trim learner (`:89-93`) never runs: it needs 0.20 < height < 0.82 and the baro says −3.
- Same result with K = 10 (peak 6.2 m) and with true hover 0.25 (peak 6.2 m).
- **Does the 0.40 m discontinuity check (`:360`) fire?** With a smooth collapse, no: the baro falls
  ~0.06 m per sample. With realistic noise, yes, but at a random moment: with 0.15 m noise it
  aborted at t = 2.1 s on the ground (harmless); with 0.08 m noise it aborted at t = 7.5 s with the
  aircraft at 8 m. From the real logs' 2–3 % big-step rate, expect it within ~1.5 s of loading the
  props. It is a coin toss whether it saves the flight or freezes the throttle mid-air (see D4).

Fix: not a code tweak. Height must come from something other than the barometer (section 2).
Until then the script must refuse `--fly`; its checks file does that today — keep it that way.

### D2 — CRITICAL — ANGLE/override conflict: script aborts at the moment override is switched on
`hover_10s.py:335-336` and `:370`; `tools/setup_angle_mode.py:248-258` (`angle_range_beside_override`).

Confirmed by reading both. `--beginner` assigns ANGLE to ch9 **900–1700 only**, i.e. exactly where
override (1700–2100) is off. John's preflight (`:308`) needs ANGLE on with override off — passes.
Then the pilot flips override on, ANGLE drops in the same instant, and the wait loop's ANGLE check
at `:335` runs *before* the override check at `:337` → `RuntimeError('Angle mode disabled')`.
Even if reordered, `:370` feeds `active(st, angle_bit)` into `healthy` → fault "unhealthy telemetry"
on the first sample. Result: override on, armed, stream stopped, throttle frozen at 0.0 (idle on
the ground — harmless this time). Worse, if someone deletes the check to "make it work", John's
hover then flies in **ACRO**: with mask 4 the pilot holds a 1.7 kg 8-inch level by hand in rate
mode; with mask 15 roll/pitch 1500 means "zero rotation rate", not "level", and it drifts off.
Fix: section 3 (configuration per flight type).

### D3 — CRITICAL (if `--xy-pose` is ever used) — XY hold pitch sign inverted
`xy_hold.py:76-81`. Comment says "negative pitch tilts forward"; Betaflight and the organizers'
`msp_rc.py:151` say pitch > 1500 = forward / nose down. Code: target ahead → `forward > 0` →
`pitch = −atan(...)` → `pitch_us < 1500` → nose **up** → flies **away**.

`demo1_xy.py` (true Betaflight plant, start 0.30 m south of the hold point):

| heading | as written | pitch sign fixed |
|---|---|---|
| yaw 0 (facing the target) | first cmd 1497; error 0.30 → 1.16 m @5 s → 4.4 m @10 s → **14.9 m @20 s, RUNS AWAY** | 0.02 m @5 s, converges |
| yaw 90 (error is pure roll) | converges (0.02 m) | converges |
| yaw 180 | first cmd 1503; **RUNS AWAY**, same numbers | converges |

- Roll sign is correct. The N/E → body rotation (`:74-75`) is correct for a clockwise compass yaw.
- `test_xy_hold.py:63` "passes" because its toy plant uses the same wrong convention
  (`an = −g·tan(pitch)`), so the test is circular.
- Second, independent problem: there is no magnetometer, so the FC's yaw zero is wherever it was
  at boot, while the pose feed's "north" is whatever the camera code decides. Nothing aligns them.
  With the sign fixed, a 60° mismatch still converges, **90° diverges (5.8 m @30 s), 180° runs away.**
- With mask 15 the pilot has no roll/pitch authority while this happens; only the switch.

```diff
-        pitch = max(-self.budget_deg, min(self.budget_deg,
-                     -math.degrees(math.atan2(forward, 9.80665))))
+        pitch = max(-self.budget_deg, min(self.budget_deg,          # BF: >1500 = nose down = forward
+                     math.degrees(math.atan2(forward, 9.80665))))
```
…and flip the plant line in the test (`an = +g·tan(pitch)`). Also pass `--angle-limit-deg 25`
after `--beginner` (it rewrites `level_limit` 55 → 25; John's default is 55, so he sends 9 us per
degree where 20 is needed and every correction comes out 2.2× *weaker* than intended: his 0.3°
starting budget (3 us) really commands 0.14°).

### D4 — CRITICAL — every abnormal exit freezes the last throttle
`hover_10s.py:396-402` (`except → raise`, `finally: stream.stop()`), `ThrottleStream.run:237-245`.

Paths that end with "frames simply stop": any exception in the main loop (baro discontinuity,
`MSPTimeout`, stale pose file, tilt > 15°, the 40 s timeout, controller fault), producer stale
> 250 ms, Ctrl-C, SSH drop (SIGHUP), process kill, Orin crash. The team's working hypothesis
(SPEC §override, tagged unverified; John's own docstring agrees) is that Betaflight 4.4.3 **holds the
last MSP values forever** while override is on. The dump adds `rxfail 3 h` (throttle *hold*) and
`failsafe_delay = 50` (5 s), both pointing the same way. So "stop sending" is **not** a safe state:
it is "continue at whatever throttle I last sent" — 0.21 (hover forever, drifting) up to 0.28
(3.4 m/s² climb). The only recovery is the pilot's switch, and with the radio throttle stick
parked at minimum (see D7) that recovery is a motor cut.

The organizers' `RCTransmitter` (`msp_rc.py:196-232`) on a stale caller **keeps streaming** an idle
frame (sticks centred, throttle 1000, ARM low); `close()` disarms and holds idle 0.3 s; `msp_bench
demo` wraps everything in `try/finally: tx.close()`. Theirs fails *downward*; John's fails *frozen*.

Right behaviour for a drone at 0.6 m: **keep the stream alive and command a bounded descent.**
On any software fault: sticks 1500, throttle = 0.85 × current hover trim for 2 s (about −1.5 m/s²,
reaches the ground from 0.6 m in under 1 s, touchdown ≈ 1.3 m/s), then ramp to 1000 and keep
streaming 1000 until disarmed or override goes off. A straight cut to 1000 from 0.6 m is a 3.4 m/s
impact — ugly but bounded, and still far better than a frozen 0.28. Nothing in software covers
Orin death or a pulled UART; that is what the pilot's switch is for, and it must be bench-proven
(section 5). Install SIGTERM/SIGHUP handlers and run under `tmux` so a dropped SSH does not kill it.

```diff
         except BaseException:
-            print('PILOT: SWITCH OVERRIDE OFF AND LAND. ...')
-            raise
+            print('FAULT: descending. PILOT: OVERRIDE OFF WHEN READY.', flush=True)
+            descend = 0.85 * (ctrl.hover_trim if ctrl else 0.0)
+            for value, secs in ((descend, 2.0), (0.0, 10.0)):   # stream stays alive throughout
+                end = time.monotonic() + secs
+                while time.monotonic() < end:
+                    stream.set(value); time.sleep(0.02)
+            raise
```
(and `ThrottleStream.run` must send throttle 1000 on a stale producer instead of raising.)

### D5 — HIGH — the flown loop is not the tested loop; it overshoots even with a perfect sensor
`hover_10s.py:363-367` vs `simulate():125`. `fly()` low-passes height (τ = 0.15 s), differentiates,
and low-passes again; `simulate()` and the tests hand the controller true height and true velocity.
Add the ±0.08/s output slew limiter (`:98`), which delays *reductions* as much as increases.
`demo2b_ideal.py`, ideal sensor, John's plant:

| loop | result |
|---|---|
| `simulate()` as tested (50 Hz, no filter) | max 0.61 m, TAKEOFF → HOVER → LAND → DONE |
| `fly()` filter, 50/30/25 Hz | **max 1.8 m**, ceiling trip at 6.6 s, never reaches HOVER |
| same, true hover 0.19 instead of 0.208 (−9 %) | **max 2.9 m**, ends in 40 s timeout fault |
| same, true hover 0.23 | OK (0.59 m) |

14/14 tests pass and prove nothing about the path that flies. Fix: make `simulate()` call the same
filter code as `fly()`; remove the second low-pass; make the slew limit asymmetric (fast down);
take velocity from a height+accelerometer complementary filter rather than a double-filtered
derivative.

### D6 — HIGH — telemetry timing margins
`fly()` opens the link with `timeout=.08` and default `retries=2`, so one request can block
**3 × 0.08 = 0.24 s**; the loop makes three per pass (`status`, `altitude`, `attitude`) → worst case
0.72 s. The producer-stale limit (`:237`) and the controller's `dt > 0.25` fault (`:56`) are both
0.25 s. One request losing all three tries, plus the other two, crosses both limits → stream dies →
D4. A single lost reply (+0.08 s) is survivable.
Rate: SPEC measured 23 Hz × 4 requests ≈ 92 commands/s, consistent with Betaflight serving about
one MSP command per 10 ms task tick (from memory of the source — unverified). John's 50 Hz RC
frames spend half of that budget, leaving ~50 requests/s ÷ 3 ≈ **16 Hz** for the control loop, not
50. Measure it. At 16 Hz one 0.09 m baro quantisation step is a 1.5 m/s raw velocity spike
(≈ 0.16 m/s after the two filters) — larger than the `|vz| < 0.10` "stable" gate and the `< 0.06`
hover-sample gate, so HOVER entry and hover learning would be flaky even on a good day.
Also: if `tools/flight_recorder.py` is installed as the `aigp-recorder` service it holds
`/dev/ttyTHS1` with `exclusive=True` and the hover script cannot open the port; stop it first.
Fix: `request(..., retries=0)` in the flight loop and treat one miss as "reuse last, count it";
poll `status()` every 5th pass; stream RC at 25–33 Hz.

### D7 — MEDIUM — procedure gap: pilot's throttle stick at takeover (mask 4 and 15)
The pilot must arm with throttle low, and `fly()` insists on it (`:304`). When the pilot later flips
override **off** in the air, throttle authority jumps back to the radio stick — still at minimum →
motors to idle from wherever the aircraft is. `HOVER_10S.md` never says "once the script has
lifted off, raise your throttle stick to about hover (≈ 1200 us) and keep it there". It must.

### D8 — LOW — three smaller items
- **One `MSPLink`, two threads: OK as written, fragile.** `send()` takes `_write_lock`
  (`msp.py:265-270`), so the stream thread's `set_raw_rc` cannot interleave bytes with the main
  thread's `request()`. All `request()` calls are on the main thread, so the "same command from two
  threads" hazard (`_waiters` is keyed by command) is not triggered — until someone adds a
  `fc.status()` to another thread. Keep every request on one thread and say so in a comment.
- **`monitor()` truncation: already fixed.** In `c7bcbfa` it ended after the baseline loop and its
  body sat after `fly()`'s `finally:` — unreachable, using an undefined `baseline`. Repaired in
  `7391b80`; HEAD is fine. Check which version is on the Orin.
- `fly()` clears the MSP arming block (`:320`) and never restores it on exit; the organizers' demo
  does (`link.set_arming_disabled(True)` in `finally`).

### D9 — CRITICAL for our own stack — pitch sign in `flight/control_adapter.py:115`
`AxisMap.sign_pitch = +1.0`. The docstring correctly derives the policy convention (positive pitch
rate = nose **up**) and then states Betaflight is "believed to be the same (stick up = nose up)".
It is the opposite: pitch > 1500 = stick forward = nose **down** (organizers' docstring,
`msp_rc.py:151`). So `sign_pitch` must be **−1**, and the self-test at `:258-259` asserts the wrong
thing. Roll and yaw (+1) agree with Betaflight. Still bench-verify all three (section 5) —
but go to the bench expecting −1.

### Throttle mapping (question 5) — no defect, but know what the numbers mean
John sends `1000 + 1000·value`. Betaflight first discards everything below `min_check` 1050, rescales
1050–2000 to 0–1, then applies `thr_mid 54 / thr_expo 68`. Using `flight/betaflight_curves.py`
(forward curve itself unverified against firmware):

| value | us | post-curve throttle | vs hover |
|---|---|---|---|
| 0.05 | 1050 | 0.000 (dead zone: first 5 % does nothing) | |
| 0.168 (trim floor) | 1168 | 0.234 | 0.80× |
| **0.208** | 1208 | **0.293** | 1.00× |
| 0.248 (trim ceiling) | 1248 | 0.347 | 1.19× |
| 0.28 (cap) | 1280 | 0.378 | 1.29× |

- Local slope is 1.46: a 1 % stick change is a 1.5 % throttle change. Thrust rises faster than
  throttle (between linear and square; `thrust_linear = 40` sits in between), so the cap is
  1.3–1.7× hover thrust = **2.9–6.5 m/s²** upward. The cap is meaningful as a bound on *how fast* it
  goes wrong, not as protection: it is well above hover, as it must be to climb at all.
- The learned range ±0.04 (±20 % throttle, roughly ±20–40 % thrust) is a sensible width **if 0.208
  is right**. 0.208 is the mean of two medians from manual-flight logs that the automatic detector
  rejected; SPEC's only other figure is "≈ 0.25, eyeball". If the truth is 0.25 the trim ceiling
  0.248 cannot reach it and the loop leans on the P term permanently. Battery sag over a pack moves
  hover by several percent too. Widen to 0.16–0.30 once a real height sensor exists.
- "Hover 0.208" is a **stick** number; post-curve it is 0.29. Say which whenever quoting it.

---

## 2. Our own `flight/` stack as a hover candidate (question 6)

It cannot hover today, for structural reasons:
1. **No hardware entry point.** Nothing in `flight/` imports `MSPLink` or `RCTransmitter`;
   `ControlAdapter.send` is a bare callable nobody supplies. `flight_loop.step()` is only ever
   driven by `sim.py`.
2. **No gate detector exists anywhere** in `flight/` or `tools/` (no contour/ArUco/chessboard code,
   no trained model). `pnp.py` needs 8 gate-corner pixels from something. `_takeoff()` trims
   throttle from `filt.pos[2]`, which is "pinned by gate fixes" — with no fixes, height is
   dead-reckoned through the very thrust model we do not know.
3. It commands **body rates (ACRO)**, so it needs fast, sign-correct attitude feedback at the
   ~16–23 Hz MSP allows, a verified rate-curve inverse, and D9 fixed. A sign error in ACRO flips the
   aircraft; in ANGLE it only leans the wrong way by a capped angle.
4. `models/pq_speed_best.npz` is for the wrong camera (known).

**Shortest path: ANGLE mode, throttle-only (mask 4) first, height from vision.** Betaflight does
levelling; the pilot does XY; we close one loop. Reusable as-is:
- organizers' `MSPLink` + **`RCTransmitter`** (`rate_hz=33, max_throttle=0.30, command_timeout_s=0.25`)
  — gives the keep-streaming watchdog for free. Its ARM handling is inert because ch5 is not in
  the mask; the pilot arms. Its watchdog drops to throttle 1000 (a cut, not a descent): acceptable
  at ≤ 0.6 m over a mat; wrap our loop in `try/finally` that commands the D4 descent first.
- `flight/pnp.py` — `solve_gate_pose` works as-is **only if** something supplies the 8 gate-corner
  pixels, and nothing does. Fastest zero-ML substitute: the **calibration chessboard (or a large
  ArUco marker) on a stand 2–3 m in front of the pad**, at 1–1.5 m because the camera looks 20° up;
  `cv2.findChessboardCorners` + `cv2.solvePnP` directly (stock OpenCV, already on the Jetson), then
  reuse `pnp.py`'s camera-tilt/attitude handling (`camera_position_world_from_attitude`) to turn the
  pose into height. Check `python3 -c "import cv2; print(hasattr(cv2,'aruco'))"` on the Orin.
- `flight/betaflight_curves.py` `ThrottleCurve` — only for reporting post-curve numbers.
- John's `HoverSequence` state machine (TAKEOFF/HOVER/LAND, ceiling, timeouts) — keep, after D5.
- `tools/flight_recorder.py`-style logging of `motors()` — but in-process (port is exclusive).
Not reusable for this: `pose_filter.py` (assumes gate map + thrust model; a 15-line height/velocity
complementary filter using `raw_imu()` az, ~512 counts per g, is simpler), `controller.py` Guidance
and `flight_loop.py` (rate-mode, gate-driven).

Could not verify: that chessboard detection survives prop vibration and rolling shutter at hover;
that vision height at 10–15 Hz with ~50–100 ms latency is enough. D5's numbers say the loop is
already marginal, so gains must be re-tuned in a sim **that includes the measured latency**.

---

## 3. Switch/mode configuration per flight type (question 7)

| Flight type | override sw | mask | ANGLE while override ON? | today's `--beginner` config |
|---|---|---|---|---|
| Human flying | OFF | any | (n/a) ANGLE on when OFF | correct |
| John hover, pilot XY | ON | **4** | **must be ON** (pilot levels via radio in ANGLE) | **wrong: ANGLE drops → abort D2** |
| John hover, auto XY | ON | **15** | **must be ON** (script sends tilt angles) | **wrong**, same |
| RL policy | ON | **15** | **must be OFF** (policy sends rates = ACRO) | correct |

The same switch position needs ANGLE on for hover work and off for the policy, so one fixed
assignment cannot serve both. Cleanest tool behaviour:
- `setup_angle_mode.py --hover-test`: ANGLE on ch9 **900–2100** (always on), `level_limit 25`.
- `setup_angle_mode.py --beginner` (rename `--race`): ANGLE 900–1700 as now.
- Both print, in capitals, which flight types the resulting config is valid for, and `--show`
  prints the same line. Mask is CLI-only on 4.4 (`tools/bf_cli.py`): have the tool read and print
  it, and set it only with an explicit `--mask 4|15 --yes`.
- If ch9 is a 3-position switch (the 18 Sep log shows it at 1094 and 1520, so probably yes), a
  no-reconfiguration alternative: override 1300–2100, ANGLE 900–1700 → low = human+ANGLE,
  middle = Jetson+ANGLE (hover), high = Jetson+ACRO (policy). Elegant, but it edits the kill
  switch's range; only do it with props off and re-verify takeover in all three positions.
- `hover_10s.py` should check ANGLE **after** override is on and say exactly which setup command to
  run if it is off, instead of "Angle mode disabled".
- The airframe was swapped on 18 Sep: mask (was 11 → set to 15 on the old one), mode ranges, active
  PID profile (was 3, `level_limit 55`) are **all unconfirmed on the current drone.** Re-read first.

---

## 4. Minimum viable hover — 10 steps, reusing what exists

1. Re-read the current FC: `setup_angle_mode.py --show`, `bf_cli.py` for `msp_override_channels_mask`,
   `aux`, `level_limit`, `rxfail`, `failsafe_*`, `mcu_id`. Save the dump with the `mcu_id` in its name.
2. Set **mask 4** and ANGLE always-on (section 3). Stop `aigp-recorder` if installed.
3. Props off: run the section 5 bench checks. Do not skip B3/B4 (what happens when frames stop).
4. Human hover in ANGLE with `tools/test_hover.py` (operator marks the hover by key press, reads
   `motors()`): get the real hover **stick** value for this airframe and pack. Three hovers.
5. Height sensor: board/marker on a stand, `findChessboardCorners`/ArUco + `solvePnP` via the
   GStreamer capture path from `tools/test_camera_capture.py`. Validate by hand-carrying the drone
   (props off) 0 → 1 m against a tape: error < 5 cm, rate ≥ 10 Hz, note the latency.
6. New ~150-line script: `MSPLink` + `RCTransmitter` (33 Hz, `max_throttle` = hover + 0.06),
   John's `HoverSequence` for the state machine, vision height + accelerometer complementary
   filter for `vz`, D4 descent in `finally`, SIGTERM/SIGHUP handlers, run under `tmux`.
7. Put the *same* filter, measured loop rate and measured vision latency into `simulate()`;
   retune until it holds 0.6 ± 0.1 m for true hover anywhere in ±15 % of the step 4 value.
8. Tethered / over-mat hops: target 0.3 m, 3 s hover. Pilot raises the radio throttle to hover as
   soon as it lifts (D7) and flies XY. Abort criterion agreed beforehand: anything above 1 m → switch.
9. Extend to 0.6 m / 10 s. Log `motors()`; feed the measured hover back into SPEC.
10. Only then consider mask 15 + XY (after D3, a yaw-frame alignment step, and its own bench check).

## 5. Bench checks that must pass before props go on

All with propellers **off**, battery in, radio on, a second person watching.
- **B1 Channel order.** Script sends throttle 1100 on index 2 with mask 4; `MSP_RC` readback (order
  roll, pitch, **yaw, throttle**) shows index 3 = 1100 only while override is on.
- **B2 Mask really is what we think.** Override on, script sends roll 1600: with mask 4 the readback
  roll must stay on the radio's value; with mask 15 it must follow the script.
- **B3 Takeover.** Script streaming throttle 1150, motors spinning: flip override off → motors
  follow the radio stick within one blink. Repeat 5×. Repeat with the radio stick at 1200 (D7).
- **B4 Stream death.** Script streaming 1150 → `kill -9` it. **Record what the motors do and for how
  long** (this turns SPEC's "holds forever" hypothesis into a measurement). Then flip override off:
  must recover. Repeat by pulling the UART plug and by `sudo shutdown -h now` on the Orin.
- **B5 Radio loss with override on.** Turn the transmitter off: note the behaviour and timing
  (`rxfail 3 h`, `failsafe_delay 50`, `failsafe_procedure DROP` predict ~5 s hold, then motor stop).
- **B6 Watchdog.** Add a debug flag that makes our loop stop calling `set_control` while the process
  stays alive: `RCTransmitter` must drop throttle to 1000 within 0.25 s and keep streaming.
- **B7 Fault descent.** Inject each fault (bad height, MSP timeout, Ctrl-C, SIGHUP): motors must
  step down to the descent value, then to idle, never freeze.
- **B8 Signs (for mask 15 and for the RL adapter).** Send pitch 1600 in ACRO: the **rear** motors
  speed up (nose-down). Roll 1600: **left** motors speed up. Yaw 1600: tilt by hand and confirm
  clockwise. Then `MSP_ATTITUDE`: nose down by hand → note the sign of pitch; rotate clockwise →
  yaw increases. Write the four results into SPEC; set `AxisMap.sign_pitch` from them (expect −1).
- **B9 ANGLE present with override on** (for hover): `status()` shows the ANGLE box active in the
  override-on position. For the RL config: shows it *inactive*.
- **B10 Loop rate.** Run the real loop for 60 s with the RC stream on: record achieved Hz, worst
  gap, MSP timeouts. Must be ≥ 10 Hz with worst gap < 0.15 s, or lower the RC stream rate.
- **B11 Height sensor** per step 5, including with motors running props-off (vibration).

## 6. What I could not verify

- Betaflight 4.4.3 holding the last MSP values forever, and its ~one-MSP-command-per-10 ms budget,
  are from memory of the source — not read, not measured. B4 and B10 settle both.
- The baro model comes from ground runs; the free-air magnitude is unknown.
- All sims use John's plant (thrust ∝ stick, linear drag, no motor lag, no ground effect); the real
  aircraft makes D5 worse, not better. The forward throttle curve is itself unverified.
- Which commit of `hover_10s.py` is on the Orin; whether `cv2.aruco` exists there; whether ch9 is a
  3-position switch; every FC setting on the swapped airframe.
