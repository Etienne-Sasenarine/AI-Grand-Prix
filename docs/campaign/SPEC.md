# Source of truth — every number in one place

Last updated 17 Sep 2026. **Consult this before answering any question about the hardware, the
course, the rules or the simulator.** If a number is not here, it has not been checked.

**Source audit, 17 Sep.** Every row was re-checked against the document it claims to come from.
Result: 43 rows are backed by an official organizer document, 14 by our own direct observation or
measurement, 8 are derived with the formula shown, **17 have no source at all** and are now marked
as such, 2 are active conflicts, and 14 are still unknown. The 17 unsourced rows were previously
tagged as mere "assumptions", which understated the problem — most of them describe **another
team's drone**, not ours.

**Correction, 17 Sep (later).** That audit removed two claims — "4 drones per team" and "personal
chargers not allowed" — as fabrications, on the grounds that the welcome packet did not support them.
**That removal was itself the error.** Both are stated verbatim in the **PQ-SPEC FAQ**, which the
audit failed to search; the packet was simply the wrong document to check. They are restored in §1 as
[SPEC], along with three more FAQ facts the same oversight had left out: no financial liability for
damage, a training cage with manual piloting permitted, and **no external tracking of any kind inside
the building**. The last one matters most — it rules out every measurement procedure that assumes a
ground-truth position reference.

The lesson worth keeping: an audit that only checks the document a claim was *attributed* to will
delete true statements. Search every official source before calling something unsourced.

### Confidence legend

| Tag | Meaning |
|---|---|
| **[SPEC]** | Stated in an **official organizer document**. The section is cited |
| **[MEAS]** | We measured or read it ourselves — a physical measurement, or a direct read of our own code and checkpoints |
| **[DERIV]** | Computed from a [SPEC] or [MEAS] value. The formula is given so it can be rechecked |
| **[NO SOURCE]** | **No official document says this.** It comes from a third party, from general knowledge, or from a guess. Treat as a placeholder and verify before relying on it |
| **[CONFLICT]** | Two sources disagree. Do not act on either until resolved |
| **[OPEN]** | Nobody knows yet. On the question list |

### A. Official organizer documents — the only citable sources

All six read in full. Plain-text extracts of the first five are in `reference_text/`.

| Key | Document | Read |
|---|---|---|
| **PQ-SPEC** | PQ Technical Specification **VADR-TS-005**, issue 00.02, 2026-09-03 (`organizer_docs/`, also `AI_GP/reference/260902_PQ_Technical_Spec_0002.pdf`) | ✅ |
| **QUICK** | **Orin NX Quick Start**, Manual I, rev. 2026-09 (`organizer_docs/orin-nx-quickstart.pdf`) | ✅ |
| **SW** | **Orin NX Software Guide**, Manual II, rev. 2026-09 (`organizer_docs/orin-nx-software-guide.pdf`) | ✅ |
| **GATES** | **Drone Racing Track — Gate Layout & Coordinate Reference for Contestants** (`organizer_docs/`) | ✅ |
| **PACKET** | AI Grand Prix Welcome Packet, Electric Fire. Logistics only. **Contains the team Wi-Fi password — never copy its text into any file or commit it** | ✅ |
| **VQ-SPEC** | **VADR-TS-001** — the *Virtual* Qualifier spec (`AI_GP/reference/AI Grand Prix Tech Specs.pdf`). **A different race with a different simulated camera (fx = fy = 320, i.e. 90°) and a MAVLink interface.** Listed so it is recognised and *not* used for the physical drone; it is the origin of the 90° camera error | ✅ |

### B. Our own direct observation

Not an outside source, but verifiable by re-running it.

| Key | What |
|---|---|
| **CODE** | Direct reads of `AI_GP/` — the observation contract, action decode, rate loop constants, the drone URDF, the checkpoint tensors |
| **OURMEAS** | Our own physical measurements on site (currently: the 1745 g race weight) |
| **OUREXP** | Our own simulation experiments, 16 Sep — the durability sweep, the A/B test of the gate-counting bug, the Plan B Monte-Carlo runs |

### C. Third-party measured data — [SAURON]

**AI of Sauron** (Georgia Tech), `github.com/JCorbin406/AIGP-Physical-Qualifier`, pulled 17 Sep.
Another physical-qualifier team running the **same organizer-supplied drone**, working openly.
They are ahead of us on hardware: three airframes calibrated, flight tests flown, one Jetson
destroyed in a crash.

**Status: a good working basis, not an authority.** Their numbers are real measurements on real
hardware of the same model, which makes them far better than a guess — so we adopt them as
starting values and label them **[SAURON]**. But they measured *their* airframes, so every one
must still be confirmed on ours. Where a [SAURON] value contradicts an organizer document, §11
says which we follow and why.

**Licence: none.** Facts and measurements only — **never their code.** Do not copy files,
functions or structure from that repository.

**Not all [SAURON] values are equally good.** Their repo is a working log, not a results document —
some entries are superseded, some tables were never filled in, and their newest commit is 17 Sep, so
anything could have moved since. Grade before use:

| Grade | What | Why |
|---|---|---|
| **S-HIGH** | Betaflight config dumps, camera calibrations | Literal files. The calibrations carry reprojection errors (0.24–0.49 px) and three units agree to 0.7 %. But the config dumps are **date-stamped and they were actively changing settings** — the override mask went 11 → 8 → 4 across one day |
| **S-MED** | Hardware identity: gyro ICM42688P, baro DPS310, 6S, board NONEH743, firmware 4.4.3 | Read from `status.txt` / `version.txt`, unlikely to drift, but it is *their* airframe |
| **S-LOW** | **Hover throttle 0.25 / liftoff 0.18** | **A pilot's eyeball estimate, not a measurement.** Their own notes: the drone "lifted onto the cushion near 0.18" — in ground effect — during a sequence where the barometer faked liftoff on every run and the gyro liftoff gate failed both ways. The results tables were never filled in, and that airframe then crashed. Treat as "probably nearer 0.25 than 0.45", nothing more |
| **S-OPS** | Operational findings (§11): baro unusable with props, accelerometer bias, gyro liftoff unreliable | Negative results from real flights. These are the most trustworthy thing in the repo, because they were learned the hard way and there is no incentive to overstate them |

### D. Not a source at all — flagged [NO SOURCE]

| Key | What it is | Why it is not a source |
|---|---|---|
| **BF-LORE** | Betaflight firmware behaviour taken from general knowledge of the 4.x source — the rate and throttle curve formulas, the override hold behaviour | Not read from the source tree and not verified against running firmware. The curve formulas in particular have variants in circulation. Note that [SAURON]'s dump confirms `rates_type = BETAFLIGHT`, so at least the correct *family* of formula is in use |
| **GUESS** | Our own inference where nothing else was available | Marked so it can be replaced |

**Nothing in section C should ever be described as "the spec says".**

---

## 1. Airframe

| Quantity | Value | Tag | Note |
|---|---|---|---|
| Frame | **Archer B2** | [SPEC] | PQ-SPEC §3.6 says "Platform: Archer B2 Frame". ("Neros" is our inference from the Neros drone-support contact in PACKET) |
| Propellers | **8 inch, 4.1 pitch** | [SPEC] | PQ-SPEC §3.6 |
| All-up race weight | **1745 g** | [MEAS] | 17 Sep, race configuration |
| Motor-to-motor diagonal | — | [OPEN] | Measure. Simulator assumes 250 mm (a 5-inch frame) |
| Prop-tip span | — | [OPEN] | Sets how much of the 1.5 m opening is free |
| Centre of gravity offset | — | [OPEN] | Balance on a ruler edge, both axes |
| Rotational inertia | — | [OPEN] | Infer from the rate step response, not a pendulum |
| Hover throttle | ≈ 0.25 stick, liftoff ≈ 0.18 — **S-LOW, a pilot's eyeball estimate** | [SAURON] | Not a measurement: taken in ground effect during a test sequence their own notes call unsuccessful. Useful only as "probably nearer 0.25 than 0.45". **Still [OPEN]. Measuring it ourselves remains a top action** |
| Max thrust / thrust-to-weight | ≈ 3.5 *if* hover really is 0.25 | [DERIV] from an S-LOW input | 8-inch props at 4.1 pitch on 6S are genuinely powerful, so the simulator's 3.53 is plausible — but this is now "not obviously wrong", not "confirmed" |
| Battery | **6S** (24.86 V observed) | [SAURON] |  |
| Gyro / accelerometer | ICM42688P | [SAURON] |  |
| Barometer | **DPS310 fitted** — but see §8, it is unusable with props spinning | [SAURON] |  |
| Provided on site | "Chargers, batteries, and drones", plus drone storage and a Neros support desk | [SPEC] | PACKET |
| **Fleet size** | **4 dedicated drones per team** | **[SPEC]** | PQ-SPEC FAQ: *"Each team is assigned a fleet of 4 dedicated drones."* Data is not shared between teams |
| **Personal chargers** | **Not permitted** | **[SPEC]** | PQ-SPEC FAQ: *"Teams may not use personal battery chargers."* Charging is run by the Neros repair team |
| **Liability for damage** | **None** — "Teams are not financially liable for drone damage" | **[SPEC]** | PQ-SPEC FAQ. Repairs and maintenance are on-site. The real cost of a crash is a slot, an airframe for the day, and possibly the SD card, not money |
| **Training cage** | Dedicated cage time per day **in addition to** track slots; **manual piloting permitted** | **[SPEC]** | PQ-SPEC FAQ. **Every measurement flight belongs in the cage, not in a scored slot** |
| **External tracking** | **None exists in the building** — no motion capture, no true-state system | **[SPEC]** | PQ-SPEC FAQ. Rules out every measurement method that needs ground-truth position |

---

> ### ⚠ Correction, 17 Sep — the hover-throttle prediction was wrong
>
> Earlier today this document, the briefing and several messages argued that the simulator's
> hover figure of 0.255 was "a 600 g racing quad's number" and that a 1745 g machine carrying a
> Jetson would "more likely hover near 35–45 %", making the simulator optimistic about available
> acceleration by roughly 56 %.
>
> **[SAURON] report ≈ 0.25 on the same airframe model** — though only as a pilot's estimate taken
> in ground effect (S-LOW; see the grading table). The confident prediction of 35–45 % was
> unfounded: 8-inch props at 4.1 pitch on a 6S pack move far more air than the mass argument
> assumed. The honest position is now **"we do not know, and 0.25 is as likely as 0.40"** — which
> is weaker than either of my two previous claims.
>
> **What survives:** measuring hover throttle is still the cheapest high-value check available,
> and the durability finding — that a 10 % thrust error costs ~58 crashes per 100 gates — is
> unchanged. What changes is the *expected* answer, and therefore the priority: this is now a
> confirmation, not an expected surprise.

## 2. Flight controller and the link

| Quantity | Value | Tag | Note |
|---|---|---|---|
| Firmware | Betaflight | [SPEC] | PQ-SPEC §3.6 |
| Version | **Betaflight 4.4.3**, board `NONEH743`, MSP API **1.45** | [SAURON] | From their `version.txt`. `msp_bench.py info` prints ours in seconds |
| Link | **MSP over UART**, `/dev/ttyTHS1`, 115200 baud | [SPEC] | QUICK §6. Not MAVLink |
| Achievable telemetry rate | **"attitude at 30–50 Hz is realistic; a hard real-time control loop is not"** | [SPEC] | QUICK §6, SW §4. **Measure it** — this sets our policy rate |
| Access permitted | Rates, PIDs, filters, telemetry via CLI. **No reflashing** | [SPEC] | PQ-SPEC FAQ |
| Serial console conflict | Linux holds a console on `/dev/ttyTHS1` by default | [SPEC] | SW §4. `sudo setup_jetson_uart.sh --apply`, once per board |
| Override switch | Manual RC override "immediately supersedes autonomous functions at any time" | [SPEC] | PQ-SPEC FAQ. This is the kill path |
| Override timeout | **None** — the FC holds the last MSP stick values indefinitely | **[NO SOURCE]** | 3RD-PARTY + BF-LORE, attributed to `src/main/rx/msp_override.c`, not read by us. **We design for it being true because the failure is a flyaway**, but it is a hypothesis. The organizers' `RCTransmitter` having its own watchdog is consistent with it |
| Override channel mask | **It is a setting, not a constraint.** [SAURON] ran 11, then 8, then settled on **4 = throttle only** (their Jetson flies throttle; the pilot flies attitude in ANGLE). Bit *n* = channel *n*+1, so **we need 15** for full autonomy | [SAURON] | Their folder names track the experiment. **Check and set ours** |
| Arming | ARM on AUX1 from the pilot's radio. An "MSP arming lock" exists; `msp_bench.py demo` clears it, arms, then re-asserts it. Betaflight also blocks arming while an MSP client is connected | [SPEC] | SW §4 |
| Flight mode | ANGLE **is available and assignable**. [SAURON] have it mapped `aux 5 1 4 900 2100`, i.e. permanently on | [SAURON] | Removes the main doubt over Plan B. Still confirm it is permitted in a scored run |
| Failsafe | [SAURON] moved from `DROP` to **`AUTO-LAND`**: on RC loss or a dead script with override on, channels **freeze for 5 s**, then Betaflight levels, holds throttle 1200 (0.20) for 5 s and disarms. Open loop — it sinks | [SAURON] | The 5 s freeze is consistent with the no-timeout hold. **Never stop the onboard script with override on** |
| Gyro units over MSP | **Raw counts at 16.4 per °/s** | **[MEAS]** | **Resolved 18 Sep on our own airframe.** A two-minute log peaks at **4894** — impossible as °/s (13 rev/s), normal as counts (298 °/s). The organizers' guide saying "already °/s" is wrong for this firmware. Evidence: `logs/orin/hover_20260918_004725.csv`. **Divide raw gyro by 16.4 before use** |
| Accelerometer scale | 512 counts = 1 g (Betaflight) | [SPEC] | SW §4, `imu_check.py --acc-1g` |
| IMU rate | 50 Hz, calibrated at power-on | [SPEC] | PQ-SPEC §3.8 |

### Betaflight curves — `flight/betaflight_curves.py`

| Quantity | Value | Tag |
|---|---|---|
| ✅ **Verified on the real airframe** | `get` over the CLI on dcl-orin, 18 Sep: `thr_mid 54`, `thr_expo 68`, `roll/pitch_rc_rate 55`, `yaw_rc_rate 57`, `roll/pitch_srate 75`, `yaw_srate 70`, `rates_type BETAFLIGHT`. **The rows below are correct** | **[MEAS]** 18 Sep, `tools/bf_beginner.py --probe --dev /dev/ttyTHS1` | An earlier note here doubted these because our `diff all` (`logs/orin/diff_all.txt`) shows every `profile` and `rateprofile` block empty. **That was a misreading of `diff`.** The board reports `# config: YES`: Betaflight 4.4 applies a board/vendor configuration on top of defaults, and `diff` prints only what differs *from that*, so vendor-set values are invisible to it. **Use `get` or `dump all`, never `diff`, to read what the aircraft is actually running.** The agreement with [SAURON] is real: both teams fly the same vendor config |
| `rates_type` | **BETAFLIGHT** | **[MEAS]** 18 Sep, our airframe via CLI |
| Roll / pitch `rc_rate` · `srate` | **0.55 · 0.75** (`expo` 0) | **[MEAS]** 18 Sep — matches [SAURON] exactly |
| Yaw `rc_rate` · `srate` | **0.57 · 0.70** | [SAURON] |
| `quickrates_rc_expo` | OFF | [SAURON] |
| `thr_mid` / `thr_expo` | **54 / 68** | **[MEAS]** 18 Sep — matches [SAURON] exactly |
| `msp_override_channels_mask` | **Was 11; we set it to 15 on 18 Sep** | **[MEAS]** | 11 left one of the four channels un-overridable, so the Jetson could not command it. Now all four. Verified persisted across an FC reboot |
| `dshot_bidir` | **ON** — per-motor RPM telemetry is live | **[MEAS]** | 18 Sep. Already enabled; no change needed. This is what makes the thrust coefficient measurable rather than estimated |
| `motor_poles` | **14** | **[MEAS]** | 18 Sep. Needed to convert eRPM to RPM |
| `rc_smoothing_setpoint_cutoff` | **15 Hz, fixed** (not auto; `auto_factor` 35) | **[MEAS]** | 18 Sep. A deliberate 15 Hz filter on the setpoint. **A latency source worth testing** — our sensitivity sweep puts 50 ms at 55 crashes per 100 gates |
| MSP override switch | **AUX5, above 1700** (`aux 4 50 4 1700 2100`) | **[MEAS]** | 18 Sep |
| Receiver channel map | **AETR1234** | **[MEAS]** | 18 Sep. Note MSP_RC reports in Betaflight's internal order, which is not the receiver order |
| FC build | BTFL **4.4.3**, API **1.45**, board **SH74**, built 2026-08-31 | **[MEAS]** | 18 Sep, our airframe |
| Barometer with props turning | **Unusable — confirmed on our airframe** | **[MEAS]** | 18 Sep: `vario` held exactly 0.000 for all 2828 samples of a 2-minute flight while altitude fell to −6.5 m. **Any stability gate built on vario filters nothing** — it accepted a log of the drone on the ground as a hover |
| Max rate at full stick | **440 °/s roll & pitch, 380 °/s yaw** | [DERIV] from [SAURON] settings via `200 × rc_rate / (1 − srate)`. Formula is BF-LORE |
| Stick for the policy's ±3.2 rad/s (183 °/s) | **0.741 → PWM 1870** (roll/pitch) | [DERIV] from [SAURON] settings |
| Stick 0.50 gives | **88 °/s**, not 220 | [DERIV]. The curve is deliberately flat in the middle |

> **Two consequences worth internalising.**
> **(a)** The curve is deliberately flat in the middle. The policy's full ±3.2 rad/s envelope costs
> **74 % of the stick**, not the 42 % a linear guess gives — a 44 % error, 162 PWM counts. Any code
> that maps rate to stick linearly is badly wrong.
> **(b)** The simulator's hover figure of 0.255 is a **stick position**. Put through the throttle
> curve it becomes **0.390 effective throttle**. So when somebody reports "it hovers at 40 %", find
> out whether that is the stick or the post-curve value before comparing it to 0.255.

---

## 3. Compute

| Quantity | Value | Tag |
|---|---|---|
| Module | Jetson Orin NX 16 GB on a Seeed reComputer A603 carrier | [SPEC] PQ-SPEC §3.7 |
| OS | Ubuntu 22.04, JetPack 6.2 / L4T r36.4.3, on an NVMe SSD | [SPEC] QUICK §1 |
| Login | `dcl` / `dcl`, over USB at **192.168.55.1** (micro-USB port, next to the barrel jack) | [SPEC] QUICK §2 |
| Serial console fallback | `/dev/ttyACM0` at 115200 on the same cable | [SPEC] QUICK §2 |
| Power | 25 W profile; sustained load thermally throttles. `tegrastats`, `nvpmodel -q` | [SPEC] QUICK §1, SW §8 |
| Installed | OpenCV (**GStreamer-enabled**), NumPy, pyserial, python3-gi, rich, pymavlink, MAVProxy, v4l-utils, i2c-tools, gpiod | [SPEC] SW §1 |
| **Not** installed | **No PyTorch, no ultralytics, no deep-learning framework** | [SPEC] SW §1 |
| Never do | `pip install opencv-python` (breaks GStreamer silently); change the device tree (writes verify clean and do nothing) | [SPEC] QUICK §7, SW §6 |
| Toolchain | `~/target/` — see §9 | [SPEC] QUICK §4 |

---

## 4. Camera

| Quantity | Value | Tag | Note |
|---|---|---|---|
| Sensor | Arducam 12.3 MP HQ (Sony IMX477), **rolling shutter**, M12 lens mount | [SPEC] | PQ-SPEC §3.8 |
| Quoted mode | 1920×1080 @ 60 fps | [SPEC] | PQ-SPEC §3.8 |
| **Airframes are distinguishable by `mcu_id`** | We hold configuration from **two different flight controllers**: `003a00313433510a32363337` (full `dump all`, 1903 lines, 18 Sep) and `0031003e3433510a32363337` (the earlier `diff_all.txt`). Both report `board_name NONEH743` | **[MEAS]** 18 Sep | **Label every dump, log and measurement with its `mcu_id`.** `SIM_MEASUREMENTS.md` §9 already says hover throttle must be measured per airframe because four supposedly identical drones rarely are — the same goes for configuration, and until now we had no way to tell which drone a file came from. Two of four airframes captured; two to go |
| **ANGLE mode is available but on no switch** | `box_ids()` includes boxId 1 (ANGLE). The assigned modes are boxIds **0, 40, 41, 43, 50** on aux channels 0, 6, 2, 5, 4. **ANGLE is not among them** | **[MEAS]** 18 Sep, MSP `box_ids()` + `MSP_MODE_RANGES` on dcl-orin | **Blocks beginner flying entirely** — self-levelling is the single biggest aid and it cannot be switched on. Assigning it needs an `aux` line, so Configurator or a CLI session, and we must know which transmitter switch is free first. `MSP_BOXNAMES` (116) times out on this firmware, so the assigned IDs are not yet resolved to names |
| **Human flying defaults to ANGLE; the scored run is ACRO** | Decision, 18 Sep: a person on the transmitter gets **ANGLE (self-levelling) unless they deliberately flip to ACRO**, with a beginner tilt limit. The scored run is the trained policy, which commands rotation *rates* and therefore needs ACRO. `tools/setup_angle_mode.py --assign --channel N --acro-at <us>` assigns ANGLE everywhere except the ACRO end of the switch; `--tilt-limit 25` lowers `level_limit` from 55 | **[DECISION]** user, 18 Sep | **The ANGLE switch stays with the pilot's radio even under MSP override** (the override mask covers the four sticks only), so it must be **on ACRO for an autonomous run** — nothing on the Jetson can set it. Taking over mid-run = override off **and** ANGLE on. `level_limit` exists only in self-levelling modes, so it cannot affect the policy. **Neither write path has run on the real flight controller yet** |
| **PID profiles are per-airframe tunes, not spare slots** | `profile_name`: 0 = `10"`, 1 = `5"`, **2 = `8"` (active, ours)**, 3 = `8" Fiber`, 4 = `-`. Roll P/D 38/57, 49/56, **40/48**, 40/48. `level_limit` 30 / 23 / **55** / 55; `angle_level_strength` 50 / 45 / **50** / 50 | **[MEAS]** 18 Sep, full `dump` lines 1174–1646 | **Never fly this airframe on profile 0 or 1.** `tools/bf_beginner.py` defaulted to writing its "beginner" tilt limit into profile 1 and telling the pilot to select it — i.e. the 5-inch gains on an 8-inch aircraft. Corrected to profile 2 on 18 Sep. The three distinct `level_limit` values are also what `setup_angle_mode.py` uses to locate that byte in `MSP_PID_ADVANCED` without trusting a remembered offset |
| **`msp_override_channels_mask`** | **11**, i.e. roll, pitch and yaw overridden but **NOT throttle** | **[MEAS]** 18 Sep, full `dump all` | Confirms the concern in EXECUTION §1 on *our* airframe. **A6 is still outstanding**: it needs to be 15 before the Jetson can fly the aircraft |
| **Bidirectional DShot** | **ON**, `motor_poles = 14` | **[MEAS]** 18 Sep | **Already done.** `SIM_MEASUREMENTS.md` §1 lists this as the five-minute job that turns measurements 2 and 3 from estimates into measurements by giving per-motor RPM. It is on, and the pole count is right |
| **Accelerometer calibration** | `acc_calibration = -72,-37,2,1` — calibrated, not zeros | **[MEAS]** 18 Sep | A common beginner failure is flying an uncalibrated accelerometer and blaming the tune. This one is done |
| **`crash_recovery`** | **OFF** (`crash_recovery_angle 10`, `crash_recovery_rate 100` are its unused parameters) | **[MEAS]** 18 Sep | Corrects an earlier note here that read the angle parameter as the mode being set to "10". It is off |
| **MSP profile selection works** | `MSP_SELECT_SETTING` (210) switches PID profiles live: selecting 0 / 1 / 2 returned roll-P 38 / 49 / 40, matching the dump exactly | **[MEAS]** 18 Sep | A **safe** way to change profiles that does not touch the CLI and does not wedge the board. Rate-profile selection via `128 + n` did not visibly work, but all four rate profiles are identical so the test is inconclusive |
| ⚠ **Betaflight CLI wedges this board** | **Entering the CLI over the MSP UART and leaving it leaves the flight controller silent on every baud rate until the flight battery is pulled.** Reproduced twice out of two on 18 Sep, once with `exit` and once with `save`. Recovery took a physical power cycle both times; waiting 80 s did nothing. The `save` did take effect | **[MEAS]** 18 Sep, dcl-orin | **Do not run `bf_cli.py` or `bf_beginner.py --apply` anywhere near a scored run or a packed flight line.** Budget a battery pull after every CLI session. `bf_cli.leave()` now drains the port and waits rather than closing mid-reboot, and `bf_beginner --apply` verifies MSP came back and says plainly if it did not — but neither prevents the wedge, they only stop it being discovered later by something that matters |
| **Telemetry rate, camera running** | **23.0 Hz** for a full sample (attitude + gyro + motors + sticks every cycle, battery/altitude/armed every tenth), 0 read errors over 422 samples, while capturing 1920×1080 at the same time | **[MEAS]** 18 Sep, `test_camera_capture.py --flight` | Closes part of **A7**. This is the composite-sample rate, not the single-request rate the organizers quote at 30–50 Hz. **Our policy runs at 60 Hz and this is 23 Hz**, so the simulator's decimation has to match measurement, not hope — see EXECUTION E1. Frame-to-telemetry pairing came out at a median of 21.9 ms, worst 68.4 ms |
| **Available modes** | **3840×2160 @ 30 fps and 1920×1080 @ 60 fps.** Both confirmed present and both start; 1080p60 captured successfully. Sensor reports 10-bit Bayer `RG10`, analog gain 1.0–22.25, exposure 13 µs–683 ms | **[MEAS]** 18 Sep, `v4l2-ctl --list-formats-ext` + Argus on dcl-orin | **Closes the two-lane worry: 1920×1080@60 does exist**, so the spec's 75° field of view is quoted for a mode we actually have. Only two modes are offered, so there is no lower-resolution option to fall back to — 640×360 must be produced by scaling |
| Horizontal field of view | Spec says **75°**; three independent calibrations say **73.7–74.1°** | **[CONFLICT]** | PQ-SPEC §3.8 vs [SAURON]. **Follow the measurements** — three units agree to within 0.4°, and a nominal lens figure is not a calibration |
| Focal length at 1920×1080 | fx = **1270.7 / 1276.9 / 1280.1**, fy within 4 px of fx | [SAURON] | bilbo / frodo / gollum. Reprojection error 0.28 / 0.24 / 0.49 px |
| **Focal length at 640×360** | **fx ≈ 423.6 / 425.6 / 426.7; use ≈ 425** | [DERIV] from [SAURON] | Divide by 3. **Unit-to-unit spread is only 0.7 %**, so the lens is consistent. Note the code's existing 423.6 came from bilbo and is therefore already a measured value, not a guess |
| **Our own ChArUco calibrations, 18 Sep** | **Two runs disagree by 13 % and neither matches the spec.** 18:10 run: fx 1579.4, 62.6° HFOV, principal point offset (−43, −15) px, RMS 0.371 px. 18:15 run: fx 1396.1, 69.0° HFOV, offset (+32, −77) px, RMS 0.690 px. Both 25 views, both 1920×1080 | **[CONFLICT]** — `target/camera-calibration/overlay_*/calibration.json` | **Do not adopt either.** Same camera, five minutes apart, yet fx differs by 183 px and the principal-point offset flips sign in x. Both read *narrower* than the 74.2° three other units gave and the 75° the spec claims. A low RMS over views that are all alike is exactly what an under-constrained calibration looks like — focal length and distance trade off and the solver picks arbitrarily. `SIM_MEASUREMENTS.md` §0 names the likely cause: **"a curled printout gives a confident, wrong calibration."** Redo on rigid flat backing with the board tilted well off fronto-parallel and filling the frame corners, and require two consecutive runs to agree within ~2 % before believing either. Also note the capture came through an MJPEG stream (`127.0.0.1:8080`), not the camera directly — rule out rescaling |
| Vertical field of view | ≈ 46.1° | [DERIV] | `2·atan(180 / 425)` |
| Principal point | **Not at centre.** Measured (976, 497), (968, 508), (969, 513) at 1920×1080 against a centre of (960, 540) — i.e. ~8–16 px right and **27–43 px above** centre. At 640×360 that is about (322, 168) rather than (320, 180) | [SAURON] | Three airframes. Ours needs its own calibration |
| Lens distortion | **Significant and consistent**: k1 ≈ +0.04…+0.08, k2 ≈ −0.20…−0.31, k3 ≈ +0.23…+0.29 across three lenses. **The simulator models zero** | [SAURON] | A real gap. Measure ours |
| Calibration provided? | **No** — "formal camera calibration matrices will not be provided" | [SPEC] | PQ-SPEC FAQ. Calibrate ourselves |
| Tilt | **LOCKED at 20° up** (decided 17 Sep 2026). Adjustable in hardware; [SAURON] run `camera_tilt_deg: 0.0` on all three of their airframes, and we are deliberately not following them | [DECIDED] from [MEAS] | See below. Set it on the airframe and verify with an inclinometer; do not change it without re-running the visibility sweep |
| Camera position in the body frame | **[0.12, 0.02, 0.013] m** — 12 cm forward, 2 cm right, 1.3 cm up of the body origin | [SAURON] | **The simulator places the camera at the body origin.** A 12 cm forward offset matters when a gate is 1–2 m away |
| Calibration board that worked | ChArUco, 7×5 squares, 36 mm square, 27 mm marker, `DICT_5X5_100` | [SAURON] | A directly reusable recipe — print this |
| Exposure / gain | Controllable | [SPEC] | PQ-SPEC FAQ |
| ISP (colour, AE, AWB) | **Works over plain SSH.** Argus exposed correctly headless with no `DISPLAY`: median pixel mean 104, spread 62, full 0–255 range, auto-exposure settling within ~1 frame at 1920×1080@60 | **[MEAS]** 18 Sep, `tools/test_camera_capture.py` on dcl-orin | **Supersedes [SPEC]** QUICK §5 / SW §1, which say it needs an EGL context and falls back to a flat debayer. That did not happen on this board. Re-check if the board, JetPack image or capture mode changes |
| Frame timestamps | Hardware-latched at the frame boundary. **`cv2.VideoCapture` discards them**; use the GStreamer path | [SPEC] | SW §2 |
| Camera↔IMU sync | **No hardware sync line exists** on this carrier. Solve in software; pin exposure first | [SPEC] | SW §3 |
| Clock drift | Frame timestamps use the SoC clock; a nominal 30.000 fps measures ~29.9997 and the error accumulates | [SPEC] | SW §3 |

### The double gate is a hairpin, not two passes the same way

Settled 17 Sep 2026 from the organizers' own figure.

The gate coordinate reference lists **gate 9 once** — `39.7, 147.7, 180, Down` —
and gives no heights anywhere. The technical spec's entire contribution is one
sentence: *"The track consists of multiple gate gates including a double gate."*
So the text says nothing about how the structure is flown.

The **figure** does. The document embeds a bird's-eye map with the racing line
drawn on it, and at gate 9 that line comes down from gate 8, passes through the
structure heading down the map, makes a tight hairpin immediately beyond it, and
comes back **up** through the structure before running on to gate 10. Two
crossings of the same plane, in opposite directions.

The text extract in `reference_text/` does not contain this — the racing line is
a raster image. To see it, pull the embedded PNG out of the PDF with `pypdf`
(`page.images`) and look at it.

Corroboration: the training environment has had it right all along, with the two
openings at yaw −90° and +90°. `flight/geometry.py` had both at −90°, which would
have meant three direction reversals and an approach to gate 10 from the wrong
side. Fixed 17 Sep.

This matters because it sets **crossings per lap (11, not 10)**, which sets the
one-hot index for every crossing after it — and an off-by-one gate label was
measured at **1.0 gates per crash against 29.4 for a correct one**, worse than no
label at all.

### The training environment over-counts passes on angled gates

Found 17 Sep 2026 while matching our onboard crossing rule to the simulator's.

`isaac_drone_racer` declares a gate passed when the drone crosses the gate plane
**and** all three components of `drone_pos - gate_centre` are under 0.75 m
**in world axes** (`tasks/drone_racer/mdp/commands.py`, `gate_size = 1.5`). The
physically correct test is the offset in the *gate's own frame* — sideways and
vertical within the 1.5 m opening. For a gate square to the world the two agree.
For a rotated gate they do not, and the world-axis box is the looser of the two:

| Gate yaw off-axis | Correct limit | What the sim allows | Over-permissive |
|---|---|---|---|
| 0° | 0.75 m | 0.75 m | 0 % |
| 15° | 0.75 m | 0.78 m | 4 % |
| 30° | 0.75 m | 0.87 m | 15 % |
| **35°** | 0.75 m | **0.92 m** | **22 %** |
| 45° | 0.75 m | 1.06 m | 41 % |

**Two of our ten gates are affected**: gates 5 and 8 both sit at yaw −125°, which
is 35° off axis. In training, a policy is rewarded for passing those gates up to
0.92 m off centre — a trajectory that on the real course clips the frame.

This is a **reward-shaping gap, not a scoring gap**: it does not change what the
organizers count, it changes what the policy was taught to aim for. It teaches
sloppier centring on exactly the two gates that need the most, and it is
invisible in simulated results because the simulator is also the scorer.

**Action before the A100 retrain**: make the training environment's pass test and
`flight/gate_tracker.py`'s crossing rule the same function, using the gate-frame
offsets. Our onboard rule is already the correct one, so the change belongs in
the (scratch copy of the) training environment, not in the flight code. Note the
`AI_GP/` repo is read-only — do this in `~/aigp/idr_exp`.

### Why the tilt is 20°, not 0°

Decided 17 Sep 2026, from measurement, against the other team's practice.

A racing quadrotor flies **nose-down** — tilting the thrust vector forward is the
only way it accelerates — and that pitch subtracts directly from the camera's
up-tilt. So the tilt must be chosen for the attitude the aircraft actually races
at, not for level hover.

Corners still inside the frame at racing attitude (camera 0.35 m below the gate
centre, 12° nose-down). Four are the minimum for a pose:

| Range | Tilt 0° | Tilt 10° | Tilt 20° |
|---|---|---|---|
| 3 m | 4 | 6 | **8** |
| 4 m | 4 | 8 | **8** |
| 6 m | 6 | 8 | **8** |
| 8 m | 8 | 8 | **8** |

At 0° the gate climbs out of the *top* of the frame under exactly the attitude we
spend the race in. Two further reasons to keep 20°: it is what the policy was
trained with, so holding it removes a sim-to-real gap rather than adding one; and
the binding constraint on close-range corner visibility is the **360-pixel frame
height** (45.9° vertical field of view), which no tilt fixes — a 2.7 m gate frame
does not fit inside about 3 m at any tilt.

The cost is at the other end: a gate *below* the aircraft falls outside the cone,
which is why the double gate's lower opening needs a stand-off approach. That is
handled in `controller.waypoint`, and it is the cheaper problem.

Constant: `flight/geometry.py:CAMERA_TILT_DEG`.

---

## 5. The course

| Quantity | Value | Tag |
|---|---|---|
| Boundary | **85 ft × 165 ft = 25.91 m × 50.29 m** | [SPEC] GATES |
| Grid squares on the map | 16.4 ft = 5 m | [SPEC] GATES |
| ~~Footprint "60 m × 21 m"~~ | **Wrong.** PQ-SPEC §3.4 contradicts GATES; GATES is the layout authority | [CONFLICT→resolved] |
| Gate outer | 2700 × 2700 mm, **80 mm deep** | [SPEC] PQ-SPEC §3.5 |
| Gate inner opening | 1500 × 1500 mm, 80 mm deep | [SPEC] PQ-SPEC §3.5 |
| Opening centre height | ~1.35 m (frames stand on the floor) | [DERIV] |
| Double gate upper opening | ~4.05 m | [SAURON] | Reported confirmed on site 15 Sep. Measure it ourselves |
| Gate count | **10**, flown in numerical order from gate 1 | [SPEC] GATES |
| Double gate | Gate 9 | [SPEC] GATES filename + PQ-SPEC §3.4 |
| **Double gate is flown twice, in OPPOSITE directions** | Down through the structure, tight hairpin immediately beyond it, back up through it, then on to gate 10 | **[SPEC]** | **Read off the racing line drawn in the GATES bird's-eye figure** (extract the embedded PNG; the text layer does not carry it). The gate table lists gate 9 once, giving the direction of the *first* crossing only. See below |
| Which opening is taken first | Upper, then lower — **inferred, not sourced** | [DERIV] | Gates 8 and 10 are both at 1.35 m, so taking the high opening first leaves the aircraft low and already pointing at gate 10. The map is a plan view and cannot show height. **Confirm on site** |
| Double gate opening heights | Our code 4.05 / 1.35 m; training sim 4.7 / 2.0 m | **[CONFLICT]** | Neither is sourced. Randomize both for training; measure on site |
| As-built placement | — | [OPEN] Survey it |
| Re-placement tolerance between runs | — | [OPEN] Ask |

### Coordinate convention [SPEC]

Origin at the **top-left** of the boundary. X to the right (0–85 ft), Y **downward** (0–165 ft).
Rotation is the direction of flight, clockwise from 0° = toward the top edge (decreasing Y).
Simulator frame: `x = X_ft × 0.3048`, `y = (165 − Y_ft) × 0.3048`.

| Gate | X (ft) | Y (ft) | Rot (°) | Direction | x (m) | y (m) |
|---|---|---|---|---|---|---|
| 1 (start) | 12.0 | 71.0 | 0 | up | 3.66 | 28.65 |
| 2 | 16.0 | 39.0 | 0 | up | 4.88 | 38.40 |
| 3 | 40.0 | 17.0 | 90 | right | 12.19 | 45.11 |
| 4 | 72.0 | 40.0 | 180 | down | 21.95 | 38.10 |
| 5 | 66.0 | 70.0 | 215 | down-left | 20.12 | 28.96 |
| 6 | 41.0 | 86.0 | 90 | right | 12.50 | 24.08 |
| 7 | 70.5 | 106.6 | 180 | down | 21.49 | 17.80 |
| 8 | 60.0 | 129.0 | 215 | down-left | 18.29 | 10.97 |
| 9 (double) | 39.7 | 147.7 | 180 | down | 12.10 | 5.27 |
| 10 | 13.0 | 110.0 | 0 | up | 3.96 | 16.76 |

Our corrected simulator track already uses this convention and matches the table.

---

## 6. Rules and scoring [SPEC]

| Item | Value | Source |
|---|---|---|
| A run | **2 laps** | PQ-SPEC §3.3 |
| Validity | "the drone must fly **without any intervention of human pilots**" | PQ-SPEC §3.3 |
| Primary ranking | **Maximum number of gates successfully passed** | PQ-SPEC §3.3 |
| Tie-break | Fastest elapsed time to the last validly crossed gate | PQ-SPEC §3.3 |
| Qualification | Best overall result during the **final two days** (21–22 Sep) | PQ-SPEC §3.3 |
| Timing start | First timing-line cross / gate trigger | PQ-SPEC §3.2 |
| Practice | Multiple scheduled slots per team per day | PQ-SPEC §3.2 |
| Spectating | No standing near the flight path, PPE or not | PQ-SPEC FAQ |
| Video | Onboard VTX fitted; footage may be transmitted | PQ-SPEC FAQ |
| Equipment | Ground station and RC controller provided per team | PQ-SPEC FAQ |
| Run timeout | — | [OPEN] |
| Attempts per day | — | [OPEN] |
| Gates counted in order only? | — | [OPEN] |

### Logistics [SPEC] — PACKET

| Item | Value |
|---|---|
| Event dates | 15–22 September 2026 |
| Daily hours | 07:00–22:00, official close 23:00. Staying past hours is not permitted |
| Badging | Day badge (LC3), worn visibly, returned daily |
| Safety briefing | **Mandatory on day 1**, with written confirmation that you have read the guidelines |
| Provided | Chargers, batteries, drones, drone storage, water and snacks |
| Bring | Team equipment and personal tools, photo ID, closed-toe shoes, laptop and chargers |
| Support | On-site Neros support desk for equipment; any staff member for safety |
| Contacts | Drone technical support: Brandon Porter (Neros). Safety and security: Brandon Ruffin (Anduril) |
| Photography | No Anduril badges visible in photos; nothing marked internal, proprietary or confidential |

---

## 7. Our policy's contract

| Item | Value | Tag |
|---|---|---|
| Observation | 51 numbers per frame × 32 frames = **1632** | [MEAS] checkpoint |
| Per frame | 8 gate corners as normalised pixels (16) + per-corner visibility (8) + roll, pitch (2) + body rates gx, gy, gz (3) + commanded body velocity (3) + 18-gate one-hot + lap fraction (19) | [MEAS] code |
| Unseen corner marker | −1.0 | [MEAS] code |
| Network | 3 × 256 ELU, then linear to 4 | [MEAS] checkpoint |
| Preprocessor | skrl `RunningStandardScaler`: `clip((x−mean)/(√var+1e−8), −5, 5)`. **Mandatory** | [MEAS] checkpoint |
| Action | 4 values in [−1, 1]: thrust, roll rate, pitch rate, yaw rate (NED) | [MEAS] code |
| Thrust decode | hover 0.255, min 0.05, max 0.90; piecewise about hover | [MEAS] code |
| Rate decode | ±3.2 rad/s = ±183 °/s at full deflection | [MEAS] code |
| Policy rate | 60 Hz in the simulator (120 Hz physics, decimation 2) | [MEAS] code |
| **Real achievable rate** | — | [OPEN] **Match the simulator to it before the long training run** |
| Simulated drone mass | 0.6076 kg, 250 mm diagonal (a 5-inch frame) | [MEAS] URDF |
| Simulated inertia | Ixx = Iyy = 0.00384, Izz = 0.00768 kg·m² | [DERIV] URDF body + 4 prop links |
| Simulated rate loop | PD, kp 0.08, kd 0.003, moment limit (0.30, 0.30, 0.20) N·m | [MEAS] code |

> **The mass trap.** Thrust is computed as `(stick / hover_stick) × mass × g`, so acceleration is
> `(stick/hover_stick) × g` and **the mass cancels**. Setting the real mass alone changes nothing.
> What matters is thrust-to-weight, rotational inertia (the real airframe is **11–16×** the
> simulated one) and drag. **If you set the real inertia and leave `rate_kp` at 0.08, the simulated
> drone becomes ~13× sluggish and every policy trained on it is worthless.** That gain stands in for
> Betaflight's own controller and must be re-derived from a measured step response:
> `kp ≈ inertia / response time`.

---

## 8. The measurement queue

Ordered by what it costs us to be wrong, from the durability study.

| Rank | Unknown | Cost of a wrong guess | How |
|---|---|---|---|
| 1 | Total command latency | 50 ms alone: 2 → **55** crashes/100 gates | LED + timestamps, both ends |
| 2 | Hover throttle / thrust-to-weight | 10 % error: 2 → **58** crashes/100 gates. Still genuinely unknown — the only outside figure is an eyeball estimate | Hover 20 s, read the log. **5 minutes** |
| 3 | Camera field of view | wrong mode: every gate corner in the wrong pixel. **[SAURON] gives us ≈ 425 at 640×360 as a solid starting value** | Mode list, then ChArUco |
| 4 | Camera tilt | 5 % error: → **32** crashes/100 gates | Level surface + inclinometer |
| 5 | Policy rate vs link rate | trains against a sensor that does not exist | `companion_listener_msp.py` |
| 6 | Course as-built | 7.5 % scale: 2 → **15.5** crashes/100 gates | Survey from a datum |
| 7 | Rate response / inertia | sets whether the simulated drone can turn like the real one | Rate steps, blackbox |
| 8 | Drag | significant above 10 m/s | Fixed lean, terminal speed |
| 9 | Detector error vs distance | the largest single sim-to-real gap | Tape-measured positions |
| 10 | Battery sag | thrust drifts during the run | Hover at full and empty |

---

## 9. The organizers' onboard toolchain [SPEC]

`~/target/` on the Jetson. Pre-installed; needs no pip, network or root (except where noted).
**"Do not re-implement MSP framing."** (SW §5)

| Path | What it is |
|---|---|
| `msp/msp.py` | `MSPLink`: framing + `attitude()`, `raw_imu()`, `altitude()`, `analog()`, `rc_channels()`, `motors()`, `status()`, `uid()`, `decode_arming_flags()`, `decode_sensor_flags()`. Thread-safe, background RX, counts CRC errors rather than raising |
| `msp/msp_rc.py` | `RCTransmitter`: streams `MSP_SET_RAW_RC` at a fixed rate from a background thread, **with a watchdog** (`command_timeout_s`). `arm()` refuses unless throttle is at minimum; `close()` disarms, holds the idle frame, then stops |
| `msp/fake_fc.py` | **Simulates a Betaflight FC on a pty.** Fault modes: `healthy`, `frozen`, `saturated`, `badscale`, `noaccel`, `i2cerr`, `nouid` |
| `msp/setup_jetson_uart.sh` | Frees the UART from the serial console. `--apply` to act; bare to inspect |
| `msp_bench.py` | `info` / `telemetry` / `rc` / `demo`. **`demo --props-off` is the only program that can arm** |
| `imu_check.py` | Functional IMU test: identity, sensor presence, sample rate, CRC/I²C errors, 1 g at rest, gyro near zero, data actually changing, attitude↔accel cross-check. `--motion` adds a rotation test |
| `companion_listener_msp.py` | Live telemetry table. `--listen-only` is safe; bare **transmits a throttle ramp** |
| `live-view.py` | Camera → browser MJPEG on :8080. Works over SSH (flat colour — expected) |
| `live-view-pts.py` | Same, with **hardware capture timestamps**; `--csv` logs them |
| `live-view-imu.py` | Camera + live attitude burned into the frame, `/imu.json` endpoint |
| `show-camera.py` | Local display, full ISP control (`--exposure-time`, `--gain`, `--ae-lock`, …) |
| `raw-view.py` | Raw Bayer, ISP bypassed — "is the sensor alive?" |
| `frame-timestamps.py` | Per-frame timing and jitter, CSV out. `--verify` proves the clock domain |
| `bringup-check.sh` | Walks the whole stack in dependency order, stops at the first real failure |
| `camera-bind-check.sh` | Why the camera driver is not bound — DT node → I²C → driver → module → bus → probe |
| `usb-device-mode.sh` | Why 192.168.55.1 is not answering |
| `signoff.sh` | The factory acceptance test |

**Safety rules from SW §7:** props off before anything that can arm; start
`companion_listener_msp.py` with `--listen-only`; never treat a stale reading as current (their
tools distinguish LIVE / STALE / DOWN — copy that); shut down with `sudo shutdown -h now`; the
Jetson is not a flight controller.

---

## 10. What we have built — `flight/`

| Module | Status |
|---|---|
| `policy_runtime.py` | **Working, verified.** Exports an skrl checkpoint to `.npz` and runs it in NumPy. Agreement with PyTorch on the real checkpoint: **2.5 × 10⁻⁶** max absolute error. Removes PyTorch from the drone entirely |
| `betaflight_curves.py` | **Working, self-tested.** Rate and throttle curves with a bisection inverse that is correct regardless of the forward model. Forward curves marked UNVERIFIED until checked against real firmware |
| `control_adapter.py` | **Working, self-tested.** Action decode → safety envelope → curves → channel values, plus `verify_axis_map()` which prints the props-off bench script |
| `models/pq_speed_best.npz` | Exported policy, 2.2 MB |

Run any of them directly to execute its self-test.


---

## 11. Operational findings from [SAURON]

Things they learned the expensive way, on the same hardware. None of this is in an organizer
document. Treat each as a strong prior to confirm, not as proof.

### The barometer is unusable with the propellers spinning

Raw `MSP_ALTITUDE` fell **about 6 m within 6 s of spin-up**, stayed there, and snapped back when
the props stopped. The flight controller's vertical-speed reading was 0.00 on every sample. Their
altitude-hold latched a fake liftoff 0.14–0.36 s into every run and sat on the ground-effect
cushion.

**This matters to us directly: Plan B's altitude estimate was built around a barometer with an
alpha-beta filter.** That design is dead as written. Their answer was to fly throttle on a state
estimate corrected by gate detections and no barometer at all — which is the same conclusion our
Plan B simulation reached when it found a barometer optional because vision teaches the thrust
bias. Drop the barometer path; keep the vision path.

### An accelerometer cannot replace it

At throttle 0.18 the vertical accelerometer read 0.985 g against 1.000 g with props off, tilt under
1°. That offset is vibration rectification and it grows with throttle. Integrated, the bias swamps
the signal within seconds — in their toy model a 126 m climb read as −106 m. They built an
accelerometer-based hold and then removed it.

### Gyro-based liftoff detection did not work

On the pad at throttle 0.10–0.11, gyro spread ran 30–66 with a 145 spike; airborne was not
separable at any single threshold. A low threshold fired 0.5 s into the creep with the drone still
on the ground; a high one never fired at all and the drone "lifted and ran away" until the pilot
took over. **Do not build a liftoff detector on gyro spread.**

### The board's Wi-Fi sits in the RC band

The Jetson's Wi-Fi runs at 2.4 GHz, the same band as the ELRS link, centimetres from the receiver.
They added an RC range check *with Wi-Fi active* to their pre-flight. Worth copying.

### Operational discipline worth stealing

- The pilot always owns **arm** and the **override switch**; the Jetson only ever touches the
  channels in the mask.
- **Never stop the onboard script with override on** — that is the same channel freeze as a link
  loss.
- An override-on *while already airborne* is **refused** by their code, deliberately.
- Their dump shows `blackbox_disable_setpoint = OFF`, i.e. setpoint is logged — which is exactly
  what we need to verify the rate curve empirically.

### The risk is real

One of their airframes crashed on 2026-09-17 and **destroyed its Jetson**. The cause was not
established. The SSD survived and the logs were recovered. This is the argument for the bring-up
ladder, for props-off bench tests, and for not flying new code on the last day.
