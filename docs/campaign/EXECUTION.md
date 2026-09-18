# Execution plan — what we do, in order

**17 Sep 2026.** Working days 17–20. Scored runs 21–22. A valid run is 2 laps, fully autonomous.
Most gates wins; time only breaks ties; best run of the final two days counts.

### Where things live

| Document | Purpose |
|---|---|
| **`SPEC.md`** | **Every number, with its source and how much to trust it.** Check here first |
| **`EXECUTION.md`** (this) | What we do, in what order, and what "done" means |
| `FIELD_PLAN.md` | Reference detail: safety architecture, bring-up ladder, step-by-step procedures, contingencies |
| `PLAN_B.md` | The slow fallback controller |
| `flight/` | Code we have written and tested |
| `briefing/main.pdf` | Onboarding and history. Read the "Status at 17 September" section first |
| `organizer_docs/`, `reference_text/` | The official documents and their plain-text extracts |

---

## 1. What changed today, and what it does to priorities

We got the organizers' Orin NX guides, the official gate table, and pulled another team's public
repo (AI of Sauron — same drone model, further ahead, working openly). Several risks got smaller
and several new ones appeared.

### Risks that shrank

| Was | Now |
|---|---|
| "Write the MSP protocol layer" — 1 day | **The organizers ship it.** `~/target/msp/` has the library, an RC transmitter with a watchdog, and a fake flight controller for offline development. We write an adapter — hours |
| "Hover throttle is probably 35–45 %, the simulator is optimistic by 56 %" | **Unfounded.** Another team reports ≈ 0.25 on the same airframe, but only as a pilot's estimate taken in ground effect. Honest position: **we do not know**, and 0.25 is as likely as 0.40. Measuring it stays a top action |
| "Camera field of view is wrong, fix to 417" | **Three independent calibrations say fx ≈ 423.6–426.7 at 640×360.** The 423.6 already in our code is a measured value. Use ≈ 425. The spec's nominal 75° is ~1° optimistic |
| "ANGLE mode may not be available — Plan B at risk" | **Confirmed available and assignable.** Their config has it permanently on |
| "The override mask is 11 and throttle is not overridden" | **It is a setting, not a constraint.** Set it to 15 |
| "Betaflight curve parameters unknown" | **Known** (their dump): roll/pitch 0.55/0.75, yaw 0.57/0.70, `thr_mid` 54, `thr_expo` 68, `rates_type = BETAFLIGHT` |

### Risks that grew, or are new

| Risk | Why it matters |
|---|---|
| **The barometer is unusable with props spinning** | Altitude dropped ~6 m within 6 s of spin-up and stayed there. **Plan B's altitude estimator was built on a barometer.** That design is dead as written |
| **No reliable liftoff detection** | Gyro spread does not separate on-pad from airborne at any threshold. Their high threshold never fired and the drone "lifted and ran away". Our takeoff routine cannot use it |
| **The link may be slower than our policy** | Organizers state 30–50 Hz attitude. Our policy runs at 60 Hz and eats gyro. **We may be training against a sensor that does not exist** |
| **Camera is 12 cm forward of the body origin** | The simulator puts it at the origin. At 1–2 m from a gate that is a real projection error |
| **Lens distortion is significant and unmodelled** | k1 ≈ +0.04…+0.08, k2 ≈ −0.20…−0.31 across three lenses. The simulator assumes a perfect pinhole |
| **The principal point is not at centre** | ~27–43 px above centre at 1080p. The simulator assumes dead centre |
| **The board's Wi-Fi shares the RC band** | 2.4 GHz, centimetres from the receiver. They added an RC range check with Wi-Fi active |
| **Crashes destroy Jetsons** | One of theirs did, on 17 Sep. This is the argument for the ladder |

### The re-rank

**Down:** camera field of view — three real calibrations make ≈ 425 a solid starting value.
**Unchanged:** thrust calibration. The one outside number is an eyeball estimate, so this is still
an open measurement, not a confirmation.
**Up:** policy rate matching, altitude estimation without a barometer, camera extrinsics and
distortion in the simulator.
**Unchanged at the top:** total command latency, the gate detector, the course survey.

---

## 2. The critical path

```
copy ~/target/ off a board  →  adapter + curves tested against fake_fc
                            →  props-off bench test
                            →  hover under Jetson control
                            →  one gate  →  one lap  →  score
```

Everything else runs beside it. **Five workstreams can start this hour without touching that path.**

---

## 3. Actionables

Each has an owner slot, a time estimate, and a definition of done. Numbered for reference.

### A. Board and link — the critical path

| # | Action | Time | Done when |
|---|---|---|---|
| **A1** | `ssh dcl@192.168.55.1` (USB-C to **micro-USB**, the small port by the barrel jack), then `scp -r dcl@192.168.55.1:~/target ./` and share it | 10 min | The team has the toolbox locally |
| **A2** | `sudo ~/target/bringup-check.sh` | 5 min | Passes, or its last line names the broken layer |
| **A3** | `sudo ~/target/msp/setup_jetson_uart.sh --apply` — once per board | 5 min | The serial console has released `/dev/ttyTHS1` |
| **A4** | `msp_bench.py --port /dev/ttyTHS1 info` | 5 min | Firmware, sensors, battery and arming blockers all print |
| **A5** | Save `diff all` from the Betaflight CLI, per airframe | 10 min | A file per drone. **Moves ~8 spec rows from third-party to measured** |
| **A6** | Set `msp_override_channels_mask = 15` and confirm an ANGLE switch exists | 10 min | All four channels overridable |
| **A7** | `companion_listener_msp.py --listen-only` — **record the poll rate** | 10 min | A number. It decides our policy rate |
| **A8** | `imu_check.py --motion`, and settle the gyro-units conflict empirically | 20 min | All checks pass; units known |
| **A9** | Wire our adapter to their `RCTransmitter`, test against `fake_fc.py` including the `frozen` fault | 3 h | Adapter commands the fake FC; watchdog proven |
| **A10** | Props-off bench sweep using `verify_axis_map()` output | 30 min | All four axes correct in sign and magnitude, checked by two people |

### B. Vehicle measurement — needs a pilot, no autonomy software

| # | Action | Time | Done when |
|---|---|---|---|
| **B1** | Weigh, measure diagonal and prop-tip span, find the centre of gravity | 20 min | Numbers in `SPEC.md` |
| **B2** | Enable blackbox (`SPIFLASH`, setpoint logging on) | 10 min | Logging confirmed |
| **B3** | The 30-minute measurement flight (`FIELD_PLAN.md` P3): hover, rate steps, full-throttle climb, fixed-lean run, hover on a tired pack | 30 min | One log containing all six manoeuvres |
| **B4** | **Confirm hover throttle** against the expected ≈ 0.25 | 5 min | A number. If it is far from 0.25, stop and tell everyone |
| **B5** | Extract the rate step response → `rate_kp`, `rate_kd`, moment limits for the simulator | 1 h | Simulator rate loop matches the measured response |
| **B6** | Fit the rate curve from logged stick/setpoint pairs, replacing the analytic model | 1 h | `RateCurve.fit_from_samples()` populated; the UNVERIFIED caveat retired |

### C. Camera and perception

| # | Action | Time | Done when |
|---|---|---|---|
| **C1** | ~~List the camera modes~~ **DONE 18 Sep.** Two modes exist: **3840×2160@30 and 1920×1080@60**. The two-lane wiring did not cost us 1080p60, so the spec's 75° figure applies to a mode we have. No lower-resolution sensor mode exists, so 640×360 has to come from scaling | 0 h | ✅ Done |
| **C2** | ~~Solve the exposure problem~~ **CLOSED 18 Sep — there is no exposure problem.** Measured over plain SSH: correctly exposed 1920×1080 colour frames, auto-exposure converging in ~1 frame. The 3 h budgeted here is freed | 0 h | ✅ Done. `tools/test_camera_capture.py` measures exposure per run and will say so if this ever stops being true |
| **C3** | Print a ChArUco board — 7×5, 36 mm square, 27 mm marker, `DICT_5X5_100` (their recipe, known to work) on something rigid and flat | 30 min | Board in hand |
| **C4** | Calibrate our own cameras. Expect fx ≈ 425 at 640×360 | 1 h | Intrinsics and distortion per airframe, reprojection error < 0.5 px |
| **C5** | **Set** the camera tilt to **20° up** on every airframe and verify with an inclinometer, then measure its position offset from the body origin | 30 min | Tilt reads 20 ± 1° on each airframe; offset in `SPEC.md` |
| **C6** | Start capturing real gate footage — every gate, 1–15 m, all angles, cut off, overlapping, backlit, plus negatives and confusers, plus every manual flight for motion blur | ongoing | A growing dataset |
| **C7** | Tune the classical colour detector on real frames as the immediate fallback | 2 h | It finds gates at a usable range |
| **C8** | Label, train, measure the detector's error against distance (`FIELD_PLAN.md` P6) | 2 days | An error model to feed the simulator |

### D. Course

| # | Action | Time | Done when |
|---|---|---|---|
| **D1** | Survey all 10 gates from a fixed datum — both uprights per gate | 90 min | A CSV, diffed against the official table |
| **D2** | Measure gate heights, including the double gate's upper opening | 20 min | Numbers confirmed rather than inherited |
| **D3** | Re-measure each morning and before every scored run | 15 min | Drift tracked |

### E. Simulator and training

| # | Action | Time | Done when |
|---|---|---|---|
| **E1** | **Set the policy rate to the measured link rate** (A7). 120 Hz physics: decimation 2 = 60 Hz, 3 = 40 Hz, 4 = 30 Hz | 1 h | Simulator matches reality before the long run |
| **E2** | Set fx = fy ≈ 425, principal point offset, **camera 12 cm forward of the body origin** | 1 h | Projection matches the real camera |
| **E3** | Add lens distortion to the projection, or verify it is negligible at gate scale | 2 h | Either modelled or dismissed with numbers |
| **E4** | Set mass 1745 g **and** re-derive `rate_kp` from B5 — never mass alone | 1 h | The simulated drone turns like the real one |
| **E5** | Add latency and detector-noise randomisation | 2 h | Bands from `SPEC.md` §8 |
| **E6** | Launch the cloud training run, 3–4 seeds | overnight | Checkpoints by morning |
| **E7** | Evaluate every candidate with the stress harness, never with training reward | 1 h | A ranked table |
| **E8** | **Replay a real flight's commands through the simulator and overlay the trajectories** | 2 h | We know where the simulator still diverges |

### E9–E11. Newly measured, feeds the simulator

| # | Finding | Action |
|---|---|---|
| **E9** | **34 % of frames have 2+ gates in view at once** on this course | Gate identity *requires* a pose. Budget time for PnP, and treat "no pose" as a hold condition, not a degraded-but-flying mode |
| **E10** | With a 12 m detector and 10 % dropout, the target gate is **unseen 28 % of frames** | The policy must tolerate long unseen stretches — check this is in the training randomisation |
| **E11** | Policy saturates its outputs **86 %** of the time when fed a trajectory it did not command | Expected open-loop, but re-check once the loop is closed; it is also what an out-of-distribution policy looks like |

### F. Plan B — needs rework

| # | Action | Time | Done when |
|---|---|---|---|
| **F1** | **Remove the barometer from the altitude estimator.** It is unusable with props spinning | 3 h | Altitude comes from vision plus a thrust-bias estimate, as our own simulation already found was viable |
| **F2** | Replace liftoff detection — gyro spread does not work | 2 h | Takeoff does not depend on detecting liftoff |
| **F3** | Re-run the Plan B Monte-Carlo without the barometer path | 1 h | Updated success numbers |
| **F4** | Confirm ANGLE mode is permitted in a scored run | — | Answer from the organizers |

### G. Ask the organizers

| # | Question |
|---|---|
| **G1** | May we place tape or markers on the gates to identify them? **A yes largely removes the gate-counting problem** |
| **G2** | Is ANGLE mode permitted during a scored run? |
| **G3** | When are our practice slots, and how many scored attempts? |
| **G4** | Who arms for a scored run — does arming before the start line count as intervention? |
| **G5** | Are gates re-placed between runs, and to what tolerance? |
| **G6** | Is 1920×1080@60 actually available given the two-lane camera wiring? If not, what is the field of view in the modes that are? |
| **G7** | What is the intended way to get properly exposed frames in flight, headless? |
| **G8** | Is there a maximum run duration? Are gates counted only in order? |

### H. Already done

| # | Item |
|---|---|
| **H1** | `flight/policy_runtime.py` — policy runs in NumPy, verified against PyTorch to 2.5 × 10⁻⁶. No framework needed on the drone |
| **H2** | `flight/betaflight_curves.py` — rate and throttle curves with a bisection inverse, now carrying the real per-axis parameters |
| **H3** | `flight/control_adapter.py` — action decode, safety envelope, curve conversion, channel mapping, props-off bench script generator |
| **H4** | `SPEC.md` — consolidated source of truth, every row audited for provenance |
| **H5** | Official gate table obtained; our corrected track matches it exactly |
| **H6** | `flight/geometry.py` + `observation.py` — **verified against the training code**: the 51-number frame and the 1632-number history match *exactly*, projection to 4e-6 px with zero visibility mismatches |
| **H7** | `flight/gate_tracker.py` — counts 22/22 crossings with zero errors across 17 adverse conditions (50 % dropout, 20 false positives/s, 100 ms latency, 1 m pose noise, 9 m/s). Correctly reports low confidence when it has no pose |
| **H8** | `flight/sim.py` — synthetic flight and detector model used to find and fix three real bugs before any hardware existed |
| **H9** | End-to-end pipeline test: 3142 frames, 0 NaNs, 0 out-of-range channels, 0 envelope breaches, 9 % of the 60 Hz budget |
| **H10** | `flight/run_all.py` — one command, six suites, all passing |

---

## 4. Sequencing

**Today (17th).** A1–A8 in one push — that is the whole board bring-up and it unblocks everyone.
In parallel: B1–B3 with a pilot, D1 with two people, C1/C3 while waiting. Evening: E1, E2, E4 with
whatever measurements landed, then E6 overnight.

**Thursday (18th).** A9–A10 and the first Jetson-commanded hover. C2 and C4 to get usable images.
C6 capture running all day. F1–F2 rework. Evening: recalibrate and launch training again.

**Friday (19th).** **A complete lap by any means** — that is the floor under our score. C7/C8
detector work. E8 replay comparison.

**Saturday (20th).** Rehearse both plans under run conditions. Re-measure gates. Freeze.

**Sunday (21st).** First slot: the most reliable configuration, flown slowly, to bank gates.

**Monday (22nd).** Improve on the banked score. Never risk the airframe before a valid run exists.

---

## 5. Decision gates

**Before any powered flight:** props-off bench sweep passed on all four axes (A10), watchdog proven
against `fake_fc.py` (A9), pilot briefed on override and arm.

**Before the long training run:** policy rate matched to the measured link rate (E1), camera
geometry corrected (E2), mass *and* rate loop corrected together (E4).

**Plan A go/no-go, Sunday:** three gates flown autonomously more than once; observations checked
against the real camera and the real curves; failure mode is a hold, not a flyaway.

**Scoring:** a slow run that finishes beats a fast run that stops at gate four. Bank first, push
after.

---

## 6. Standing corrections

Two things I asserted earlier today turned out to be wrong. Both are corrected in `SPEC.md`, and
both are worth remembering as a calibration on confidence:

1. **"The drone probably hovers at 35–45 %, so the simulator overstates thrust by ~56 %."** Measured
   ≈ 0.25 on the same airframe. The mass argument ignored how much air 8-inch props at 4.1 pitch
   move on 6S.
2. **"Use fx = 417 from the spec's 75°."** Three independent calibrations say 73.7–74.1°, i.e.
   fx ≈ 425. A nominal lens figure is not a calibration, and the value already in our code was
   measured.

The pattern in both: a specification number and a derivation beat nothing, but they lose to a
measurement. Prefer measurements, and say plainly which one a number is.

---

## 7. Three parallel workstreams — logged 17 Sep, evening

These run **at the same time as the A100 training run** and block nothing. Each has a different
owner-shaped dependency: one needs the drone, one needs only a laptop, one needs only the laptop.

### W1 — Bench measurement session (needs the drone, no flying)

Two thirds of the total measurement value, zero airframe risk. Full procedures in
`SIM_MEASUREMENTS.md` §7. Running order:

| Step | What | Time | Done when |
|---|---|---|---|
| W1.1 | Betaflight CLI bridge over the MSP UART (see W2 — same tool) | 30 min | `diff all` returns over SSH |
| W1.2 | `diff all` per airframe, dated, one file each | 15 min | 4 files saved. Replaces another team's curve constants |
| W1.3 | MSP poll rate: idle, with vision running, **and with RC transmitting at 50 Hz** | 30 min | Three numbers. The third is the only one that matters |
| W1.4 | `v4l2-ctl -d /dev/video0 --list-formats-ext`, then `frame-timestamps.py --verify` | 20 min | Mode list + true fps. Decides whether the 74 deg figure applies at all |
| W1.5 | Gyro units — settles the 16x [CONFLICT] in `SPEC.md` | 20 min | `imu_check.py --motion` plus a hand-rotation cross-check |
| W1.6 | `rc_smoothing` read and re-test at a higher cutoff | 30 min | Possibly ~30 ms of latency recovered for free |
| W1.7 | **MSP override hold behaviour, props off** | 20 min | We currently design around an unverified hypothesis whose failure mode is a flyaway. **Never test this in flight** |
| W1.8 | Axis signs, props off | 10 min | Three signs confirmed. Most dangerous unverified assumption in `flight/` |
| W1.9 | Camera tilt **set to 20 deg** and verified; camera body offset measured | 40 min | Inclinometer median of three, per airframe |
| W1.10 | ChArUco calibration | 1.5 h | Our own intrinsics + distortion, per airframe. **Needs a printer and rigid backing** |

### W2 — Jetson MSP logger and CLI bridge (laptop only, testable with no hardware)

~60 lines on `MSPLink.request_batch()`. Logs `attitude()`, `raw_imu()`, `rc_channels()`,
`motors()`, `analog()` and command **139 `MSP_MOTOR_TELEMETRY`** to CSV with `time.monotonic()`
stamps. Plus a ~30-line Betaflight CLI bridge (the FC enters CLI when it sees `#` on an MSP port
while disarmed).

**Build this before anyone flies.** It makes the Jetson its own flight recorder: hover throttle,
hover attitude bias, battery sag, achieved rates and link rate all fall out of a single pilot
hover, with no Betaflight Configurator, no USB cable to the FC, and no dependency on blackbox
working. Test it against the organizers' `fake_fc.py` on the laptop.

Done when: a CSV comes back from `fake_fc.py` with all fields populated and monotonic timestamps,
and `diff all` returns over the bridge.

### W3 — The two remaining closed-loop failures (laptop only)

`flight/test_closed_loop.py` completes 22/22 in 5 of 9 configurations. The two that do not:

| Case | Crossings | Symptom |
|---|---|---|
| everything at once | 18/22 | `lost the gate count for too long` |
| slow rate loop / latency / heavy drag | 21/22 | one crossing short, run finishes |

Method, which worked all day: **instrument first, do not tune first.** The diagnostic scripts are
in the session scratchpad and are worth keeping — `diag_pnp_range.py`, `diag_gaps.py`,
`diag_index.py`, `diag_assoc.py`. The last is the most useful: it tags every simulated detection
with the gate that produced it, so a mis-association is visible directly instead of inferred from
its wreckage.

Done when: 9/9, or a written reason why a case is not worth fixing.

**Note:** every failure chased today turned out to be a logic bug rather than a tuning problem —
a vacuous `inf` comparison, a frame-convention mismatch, a threshold wider than the ambiguity it
guarded. Expect the same here before reaching for a constant.
