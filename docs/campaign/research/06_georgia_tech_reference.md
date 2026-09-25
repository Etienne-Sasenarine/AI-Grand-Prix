# The Georgia Tech ("AI of Sauron") approach — our reference design

19 Sep 2026. Decision by the team: **treat their measured setup as the thing to
reproduce**, not just as hints. They fly the same organizer-supplied drone, the
same firmware build, the same transmitter, and they are through five gates; we
cannot hover.

**Where this comes from.** Their repository was public on 17 Sep, when we read
it; it is private now, and our clone in `/tmp` did not survive. What follows was
recovered from our own saved session notes of that reading — about 30 KB: their
hover and ramp test write-ups, their 16 Sep status log, their Betaflight
settings, the file listing, and commit subjects. **We do not have their flight
code, and none of it is reproduced here**: the repository had no licence. These
are their facts, settings and method, in our words. Anything below tagged
*inferred* is my reading, not something they wrote.

---

## 1. Their flight-controller setup — copy this exactly

Read from their saved Betaflight dump of 16 Sep (`5_failsafe_autoland`), same
firmware string as ours (`BF_BLOCK2 (SH74) 4.4.3 Aug 31 2026`):

| Setting | Theirs | Ours (last board read) |
|---|---|---|
| ANGLE mode | `aux 5 1 4 900 2100` — **ANGLE always on**, whole travel of the override channel | not assigned at all |
| Override mask | `msp_override_channels_mask = 4` — **throttle only** | 11, later 15 |
| MSP override switch | `aux 4 50 4 1700 2100` | identical |
| ARM | `aux 0 0 0 1600 2100` | identical |
| Failsafe | `failsafe_procedure = AUTO-LAND` | `DROP` |
| Rates, throttle curve, `min_check`, idle | 55/75, 57/70, `thr_mid` 54, `thr_expo` 68, 1050, 550 | identical |

Their settings folders are named in the order they tried things: mask (11) →
"angle always" → mask 8 "throttle only" → mask 4 "throttle only". So they hit
the same mask confusion we did and settled on **4**, which our source reading
(report 01) confirms is throttle under `map AETR`.

What this setup means in flight: **the pilot always flies roll, pitch and yaw,
in self-levelling mode; the Jetson only ever touches throttle, and only while
the override switch is on.** The pilot owns arming and the override switch.
Their failsafe on a lost link or a dead script: sticks freeze 5 s, then the
flight controller levels, holds throttle 1200 for 5 s and disarms — it sinks.

This settles our open design question: ANGLE stays **on** during override.
`setup_angle_mode.py --beginner` should assign ANGLE over the whole travel
(900–2100), as they did, not only where override is off.

## 2. What they tried for height, in order, and what happened

1. **Barometer hover.** Failed. The barometer fell about 6 m within 6 s of
   spin-up and stayed there; vario read 0.00 on every sample. It faked a liftoff
   0.14–0.36 s into every attempt, latched a hover throttle of 0.10, and the
   drone sat on the ground-effect cushion (their runs 0032–0040). *This is the
   design our `hover_10s.py` is on today.*
2. **Accelerometer instead of barometer.** Built, then removed. Vibration biases
   the vertical reading (0.985 g at throttle 0.18 against 1.000 g props off, and
   growing with throttle); in their toy model a 126 m climb read as −106 m.
3. **Open-loop ramp with automatic liftoff detection from gyro noise.** Failed
   both ways: at one threshold it fired on the pad, at another it never fired and
   the drone "lifted and ran away" until the pilot took over (runs 0041–0043).
4. **Open-loop ramp with the pilot's three-position switch setting the phase.**
   Up = throttle creeps up 0.008/s; middle = hold (estimates hover from the last
   0.5 s of throttle and vertical force, then damps vertical speed within ±0.05);
   down = walk down 0.02/s. They say plainly it is *not* altitude hold: nothing
   measures height, holds must be short.
5. **"Gate hold": throttle flown on a state estimator corrected by gate
   position fixes from the camera — no barometer anywhere.** Merged 16 Sep, a
   runbook for its first flight written 17 Sep. *Inferred:* this is what carries
   them through gates now. It is the same idea as report 02's recommendation and
   as our own `flight/pnp.py` + `flight/pose_filter.py`.

Numbers they measured on the same airframe: lifts onto the ground cushion near
throttle **0.18**; hover about **0.25** by the pilot's read (they set
`hover_throttle 0.25`, `liftoff_throttle 0.18`). Our teammate's two logs gave
0.208 — consistent with a reading taken low, in ground effect.

## 3. How they run a test — the procedure worth copying

- Cage or net, props on, fresh battery, **USB cable unplugged** before props on.
- Logger service confirmed *not* running (it would hold the serial port).
- ANGLE confirmed active on the display before arming.
- Start the script and wait for READY **before** the pilot does anything.
- Pilot: stick down, arm, **then** flip override; hand stays on that switch.
- The script checks the flight controller is echoing its throttle, and stops
  with a clear message if the mask does not include throttle.
- One throttle cap per run, raised in 0.02 steps (0.20, 0.22, 0.24…), changed
  only between runs.
- To end: **override off first, then disarm, then stop the script.** "Never stop
  the script with override on" — that freezes the last throttle.
- They did an RC range check with the Jetson's Wi-Fi active.

## 4. Link budget they measured (same board, same library)

The flight controller answers about one message per 100 Hz tick (96–97 per
second at 115200 baud). IMU alone: ~92 Hz. With a 50 Hz command stream plus
attitude/altitude/status polling: IMU falls to about 45–50 Hz, and once fell to
~11 Hz. Their hover script sends commands at **25 Hz**, polls status and
attitude at 5 Hz, and gives every remaining slot to the IMU. Expected
command-to-response delay about 130 ms, of which ~30 ms is the fixed 15 Hz stick
smoothing. Camera logging ran at a steady 30 fps alongside.

## 5. What it cost them

On 16 Sep their first drone crashed during a *manual* flight and the Jetson was
destroyed; cause never established. The storage drive survived and every run was
recovered because logging was on the drive, not over the network.

## 7. How much did *they* trust each number?

Judged from how they wrote about it and what they did with it — a number they
bet a flight on counts for more than one they jotted down.

| Number | Their value | How they got it | How they treated it | Our confidence |
|---|---|---|---|---|
| **Hover throttle** | **0.25** | "by the pilot's read" — an eyeball, during runs where the barometer faked liftoff and the liftoff detector failed. Their results tables were left empty | They **did** write it into the flight config for the gate-hold flight (`hover_throttle 0.25`), so they trusted it enough to fly on — but their hold mode re-estimates hover in flight and only allows ±0.05 around it, which says they expected it to be off by a few hundredths | **Medium-low as a value, good as a starting point.** Three rough, independent signals now agree on 0.23–0.27: their 0.25; our log analysis (report 04: liftoff near 1200–1220 µs, hover estimated 1230–1270 µs); and our teammate's 0.208, which was taken low in ground effect and so reads under. **Start at 0.25, let the controller trim ±0.05, cap at 0.32** |
| **Liftoff onto the ground cushion** | **0.18** | Repeated across runs 0035–0039: the throttle was clamped at 0.18 and the drone sat on the cushion every time | Stated as fact, used to size the cap ladder (start 0.20) | **Medium-high.** Repeated, and it is a floor not a guess: below ~0.18 nothing happens. Pack voltage will move it |
| **Barometer collapse** | ~6 m in 6 s, vario 0.00 | Nine runs, raw readings | Abandoned the barometer entirely | **High** — we measured the same on our own airframe |
| **Accelerometer bias** | 0.985 g at throttle 0.18; fitted as −0.46·u² g | One run (0038), one point, a fitted curve | Used it inside their hold mode, but called the hold "not altitude hold" and told the pilot to keep holds short | **Low as a formula** (one data point), **high as a warning** |
| **Gyro liftoff detection** | fails at every threshold | Three runs, failed both ways, one runaway | Told future readers not to fly it | **High** — a negative result they paid for |
| **Link budget** | ~96 messages/s; IMU 92 Hz alone, ~45–50 Hz beside 50 Hz commands; once 11 Hz | Logged counts over whole runs, zero timeouts | Redesigned their polling around it; chose 25 Hz commands | **High.** Ours measured 23 Hz for a heavier polling pattern, consistent |
| **Command-to-response delay** | ~130 ms | "Expected path", added up from parts | Written as an expectation, not a measurement | **Low-medium.** Plan for 100–200 ms |
| **Camera intrinsics** | fx 1270.7 / 1276.9 / 1280.1 at 1080p | ChArUco, reprojection 0.24–0.49 px, three units | Stored per drone as files | **High** — three units within 0.7 % |
| **Camera tilt** | 0° | Config field on all three | — | It is *their* choice, not a measurement of ours. Ours is 20° |
| **Failsafe timing** | freeze 5 s, then level, throttle 1200 for 5 s, disarm | Read from their settings; described as known behaviour | Written into the pre-flight briefing | **Medium-high**; matches the source reading in report 01 |

The pattern: **their negative results and their settings are solid; their
positive flight numbers are rough.** That suits Plan B, which needs a starting
throttle and a controller that corrects it — not a precise one.

## 6. What this changes for us

1. **Flight-controller setup = theirs** (section 1). One bench session with
   Configurator over the flight controller's own USB port: ANGLE always on,
   mask 4, and consider AUTO-LAND. Ask the organizers if mode changes are allowed.
2. **Stop work on barometer height.** They proved it dead on this airframe three
   days before we tried it.
3. **First hover = their step 4 shape**: pilot in ANGLE, Jetson on throttle only,
   a cap per run, pilot's hand on override. It needs no height sensor and gets a
   real hover throttle number. Expect ~0.25, cushion at ~0.18.
4. **Then their step 5**: height from gate or marker fixes feeding our
   `pose_filter.py`. This is the path to gates.
5. **Command at 25 Hz, not 50**, and budget telemetry slots deliberately.
6. Procedure in section 3 goes into `FLIGHT_SESSION.md`.

What we still do not know about them: how they detect gates, their estimator's
details, their gains, and how they steer between gates. The best source for that
is a conversation — they worked openly for most of the week.
