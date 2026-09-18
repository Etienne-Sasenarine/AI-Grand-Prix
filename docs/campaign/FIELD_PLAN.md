# Physical Qualifier — field plan (rev. 2, 17 Sep 2026)

Working days 17–20 Sep. Scored runs 21–22 Sep. A valid run is **2 laps, fully autonomous, with no
human pilot intervention at any point**. Ranking is by **most gates passed**; time is only the
tie-break; best single run across the final two days counts.

Companion documents: `SIM_MEASUREMENTS.md` (what to measure and why), `INSTALL_PLAN.md` (what to
install, ordered by how badly it blocks), `PLAN_B.md` (the slow fallback), `briefing/main.pdf`
(onboarding and history — read its "Status at 17 September" section first).

**Organizer documents** (`organizer_docs/`, plain text in `reference_text/`): the PQ technical spec,
the **Orin NX quick start**, the **Orin NX software guide**, and the **official gate coordinates**.
The two Orin guides are the authority on the onboard toolchain and have already changed this plan
substantially — read them before writing anything that touches the flight controller or camera.

---

## 1. The one-page version

**The critical path is the link to the flight controller — but it is much smaller than we thought.**
Our flight client speaks MAVLink to the virtual qualifier's simulator; the real flight controller
runs Betaflight, which speaks MSP, and a search of our whole codebase for "MSP" returns nothing.
**However, the organizers ship a complete working MSP layer on the drone** (`~/target/msp/`,
documented in `organizer_docs/orin-nx-software-guide.pdf`), including a transmitter with a built-in
watchdog and a *fake flight controller* we can develop against on a laptop with no hardware. So this
is an adapter job measured in hours, not a protocol job measured in days.

**But only one workstream waits on it.** Three others need no autonomy software whatsoever and
should start in the same hour:

| Starts now, needs nothing | Why it doesn't wait |
|---|---|
| **Vehicle measurement flight** | A human pilot and the flight controller's own recorder. Zero code |
| **Course survey** | Two people and a tape measure |
| **Camera calibration + data capture** | A camera and a printed checkerboard |
| **Simulator training** | Randomize what we can't measure yet; narrow the ranges as numbers arrive |
| **The control adapter** | Testable on the laptop against the organizers' `fake_fc.py` |

**The highest-value five minutes of the week:** hover the drone and read the throttle percentage.
The simulator assumes it hovers at 25.5 %, which is a 600 g racing quad's figure. The real drone is
1745 g with a computer on it. If it actually hovers at 40 %, the simulator has been promising the
policy roughly 56 % more acceleration than exists — and a 10 % thrust error alone took crashes from
~2 to **58 per 100 gates** in testing.

**The finding that may force a retrain:** the organizers state, twice, that the flight-controller
link gives "attitude at 30–50 Hz; a hard real-time control loop is not" realistic. Our policy runs at
60 Hz and takes gyro readings as input. Sending sticks at 60 Hz is fine — that is just an RC link,
and Betaflight keeps the fast stabilization loop internally. But if the *feedback* only arrives at
30–50 Hz, the simulator is training against a faster sensor than exists. **Measure the achievable
rate on day one, then set the simulator's policy rate to match before the long training run.**

**Strategy:** get a complete slow lap by Saturday. That puts a floor under the score. Sunday's first
run is the most reliable configuration we have, flown slowly, to bank gates. Push only afterwards.

---

## 2. Standing rules

1. **Two people minimum for any powered flight.** Pilot on the sticks, engineer on the laptop. The
   pilot's only job is the kill path; they do not debug.
2. **The pilot owns the override switch.** It is the one safety layer that lives in firmware rather
   than in our code, which is why it is the one we trust.
3. **Nothing goes on the drone that has not run on the bench with propellers off.** No exceptions,
   including one-line fixes, including on the last day.
4. **Every powered event is logged on both ends** — blackbox on the flight controller, and a Jetson
   CSV of inputs, outputs, detections and message timings. A flight with no log is a wasted battery.
5. **Config is a file, not a memory.** Every flight records the git commit, the checkpoint filename
   and the full parameter set in its log header.
6. **Bank a score before chasing a time.**

### Safety architecture

Betaflight's MSP override **holds the last commanded stick values indefinitely — there is no
timeout** (confirmed in the 4.4.3 source, `src/main/rx/msp_override.c`). If the Jetson process dies
at 12 m/s, the drone keeps flying that command into a wall.

| Layer | Mechanism | Trust |
|---|---|---|
| 1 | Pilot flips the override switch off → sticks take over | Firmware. Highest |
| 2 | Pilot disarms | Firmware |
| 3 | Radio link lost → failsafe cuts motors (`failsafe_procedure = DROP`) | Firmware |
| 4 | Watchdog: control loop misses its deadline → fall back to an idle frame | **Provided** by `RCTransmitter`'s `command_timeout_s`, in its own background thread |

Layer 4 is already built for us, and it is built the right way — its own thread with its own timer,
because the failure it guards against is the control loop stopping. Our job is to **configure the
timeout and verify the behaviour against `fake_fc.py`'s fault modes**, not to write it.

Note the rule interaction: using the override **voids that run**. The pilot can abort, not rescue.
That raises the stakes on layer 4 considerably.

---

## 3. Technical gap register

Everything between what we have and what we need. Nothing here is a criticism of the existing code —
almost all of it was written for the virtual qualifier, which was a different race with a different
interface.

### First, what Betaflight is (skip if you know)

The **flight controller** is a small board with a gyroscope, running firmware called **Betaflight**.
Its job is narrow and fast: take four numbers — throttle, roll, pitch, yaw — thousands of times a
second and spin the motors so the drone does what they ask. It handles the millisecond-scale
balancing. We don't write that and couldn't do it better.

Those four numbers normally come from a **radio receiver** listening to the pilot's remote.

**MSP** is just a message format for talking to the flight controller over a serial wire. **MSP
override** is the Betaflight feature we need: while a switch on the pilot's remote is on, Betaflight
takes its four numbers from MSP messages — from our Jetson — instead of the radio.

**ACRO vs ANGLE** are flight modes. In ACRO, stick position means *rate of rotation*: centre the
stick and it stops rotating but stays tilted. In ANGLE, stick position means *lean angle*: centre it
and it levels itself. Our policy outputs rotation rates, so it needs ACRO. Plan B wants ANGLE
because "hold still" becomes nearly free.

**Blackbox** is Betaflight's built-in flight recorder — throttle, gyro, motors, voltage, downloaded
over USB afterwards. Almost every measurement we need comes out of it.

### What the organizers already built (do not rebuild it)

`~/target/` on the Jetson, pre-installed, no network or pip needed. Log in over a USB cable:
`ssh dcl@192.168.55.1`, password `dcl`.

| File | What it gives us |
|---|---|
| `msp/msp.py` — `MSPLink` | All MSP framing, plus `attitude()`, `raw_imu()`, `analog()`, `rc_channels()`, `motors()`, `status()`, and arming-blocker decode. Thread-safe, background receive loop, counts corrupt frames instead of crashing |
| `msp/msp_rc.py` — `RCTransmitter` | Streams stick commands at a fixed rate from a background thread **with a watchdog already built in** — stop calling `set_control()` and it falls back to an idle frame by itself. `arm()` refuses unless throttle is at minimum; `close()` disarms cleanly |
| `msp/fake_fc.py` | **Simulates a Betaflight flight controller on a laptop**, with fault modes: `frozen`, `saturated`, `badscale`, `noaccel`, `i2cerr`, `nouid`. The full test suite runs with nothing plugged in |
| `msp/setup_jetson_uart.sh` | Frees the serial port from Linux's login console. **Run this once per board before anything else** — skipping it looks exactly like a dead flight controller |
| `msp_bench.py` | `info` / `telemetry` / `rc` / `demo`. `demo --props-off` is the *only* program that can arm |
| `imu_check.py` | Functional IMU test that catches frozen buffers, dead axes, wrong scaling — not just "did it reply" |
| `companion_listener_msp.py` | Live telemetry table; reports completed poll cycles per second, which is how we measure the achievable rate |
| Camera tools | `live-view.py`, `live-view-pts.py` (hardware capture timestamps), `live-view-imu.py`, `show-camera.py`, `raw-view.py`, `frame-timestamps.py` |
| Diagnostics | `bringup-check.sh` walks the whole stack and stops at the first real failure; `camera-bind-check.sh`; `usb-device-mode.sh` |

The software guide states it plainly: **"do not re-implement MSP framing."** Our job is the adapter
from the policy to `RCTransmitter`.

**First action for whoever reaches a board:** `scp -r dcl@192.168.55.1:~/target ./` and share it, so
the rest of the team can develop against `fake_fc.py` without the drone.

### A — Flight controller and the link to it

*Substantially reduced by the organizer toolchain. Struck items are no longer ours to build.*

| ID | Gap | What it means | Effort | Blocks |
|---|---|---|---|---|
| **A0** | **Get `~/target/` off a board** | `scp -r dcl@192.168.55.1:~/target ./`. Until this happens nobody can develop against the fake flight controller | 10 min | All adapter work |
| **A1** | ~~No MSP implementation~~ → **adapter only** | The protocol is provided. We write the layer that turns our policy's output into `RCTransmitter.set_control()` calls | 4 h | Everything autonomous |
| **A2** | Free the serial port | `sudo ~/target/msp/setup_jetson_uart.sh --apply`, once per board. Linux runs a login console on that port by default. **Skipping this looks exactly like a dead flight controller** | 5 min | Any link at all |
| **A3** | Rate curve inversion | **Still ours.** Betaflight maps stick to rotation rate along a curve (`rc_rate`, `super_rate`, `expo`), gentle in the middle, aggressive at the ends. Our policy speaks rad/s. Invert it exactly — a linear approximation is wrong by tens of percent mid-range | 4 h | Accurate control |
| **A4** | Throttle curve inversion | **Still ours.** Same for throttle (`thr_mid`, `thr_expo`) | 1 h | Height control |
| **A5** | Sign and axis mapping | **Still ours, and the most dangerous item on this list.** Our simulator uses forward-left-up; Betaflight uses forward-right-down. Plus channel order. A reversed sign flips the drone on takeoff | 2 h | Any flight |
| **A6** | ~~No watchdog~~ → **configure and verify** | `RCTransmitter` has one built in. Set the timeout, then prove it against `fake_fc.py`'s `frozen` mode | 1 h | Safety sign-off |
| **A7** | **Loop rate mismatch** ⚠ | Organizers: "attitude at 30–50 Hz is realistic; a hard real-time control loop is not." Our policy runs at 60 Hz and eats gyro as input. Measure the real rate with `companion_listener_msp.py`, then **match the simulator to it before the long training run** | 2 h | **Training validity** |
| **A8** | Gyro units unverified | Sources conflict: one says raw counts at 16.4 per °/s, the organizers' `imu_check.py` says Betaflight gyro is already in °/s. **Verify empirically**, don't assume — the wrong choice is a silent 16× error | 30 min | Sensor fusion |
| **A9** | Arming flow | `msp_bench.py demo` clears an "MSP arming lock", arms, then re-asserts it. Betaflight also blocks arming while an MSP client is connected. Read their code and copy the sequence | 2 h | Any real run |
| **A10** | Override mask — **verify, don't assume** | The mask=11 finding (throttle not overridden) came from *another team's* dump. Since `msp_bench.py demo` drives throttle successfully, ours may already be right. Check `diff all` on our drone | 10 min | Autonomous hover |
| **A11** | ANGLE mode not assigned | Flies ACRO; Plan B needs ANGLE. CLI-assignable — confirm it's permitted | 15 min | Plan B |
| **A12** | Blackbox not configured | Must be on and recording gyro debug data, or the measurement flight yields nothing | 15 min | All diagnostics |
| **A13** | Camera/IMU time sync | **No hardware sync line exists on this carrier** — the guide is explicit. Must be solved in software: a shared physical edge, timesync messages, or estimated online. **Pin the exposure first** or auto-exposure moves the capture instant every frame | 4 h | Latency measurement |

### B — Onboard software

| ID | Gap | What it means | Effort | Blocks |
|---|---|---|---|---|
| **B1** | Camera capture with real timestamps | Tools exist (`live-view-pts.py`, `frame-timestamps.py`). **But `cv2.VideoCapture` throws the hardware timestamp away** and gives arrival time with tens of ms of jitter. Use the GStreamer/`python3-gi` path for anything timing-sensitive | 4 h | Perception |
| **B2** | **Image processor needs a display** ⚠ | Colour, auto-exposure and white balance run in NVIDIA's ISP, which needs a graphics context a plain SSH session doesn't have. Headless gives flat, dim, un-exposed frames **by design**. Either arrange a display context on the board or design the detector for minimally-processed frames | 4 h | Usable images in flight |
| **B3** | **No deep-learning framework on the board** | The image ships OpenCV (GStreamer-enabled), NumPy, pyserial, python3-gi, pymavlink. No PyTorch. **The NumPy policy runtime is now mandatory, not a preference** — three matrix multiplies, ~50 lines. The detector needs the same treatment: export to ONNX and run via OpenCV's DNN module or TensorRT | 3 h | Plan A |
| **B4** | **Never `pip install opencv-python`** | It silently replaces the GStreamer-enabled build; `cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)` then returns False with no explanation | — | Camera pipeline |
| **B5** | No gate counter | The "which gate is next" input has no source on the real drone | 1 day | Plan A |
| **B6** | No takeoff routine | The policy was trained starting *in the air*. The real race starts on the floor | 4 h | Any real run |
| **B7** | Crash logic is simulator-era | It auto-re-arms after a crash — a simulator behaviour, dangerous on a real drone | 3 h | Safety sign-off |
| **B8** | No logging standard | Synchronized record of inputs, outputs, detections | 2 h | All learning |
| **B9** | Compute budget unverified | Detector + policy + capture at 25 W. Sustained load thermally throttles — watch `tegrastats` | 2 h | Loop rate |

### C — Perception

| ID | Gap | What it means | Effort | Blocks |
|---|---|---|---|---|
| **C1** | Detector has never seen a real gate | Existing detectors were tuned on simulator images. See §5, P6 | 2 days | Any autonomy |
| **C2** | **Sensor mode may not be what the spec says** ⚠ | The carrier wires only **2 of the camera's 4 data lanes**, capping resolution × framerate. The guide says do not assume the headline modes exist — run `v4l2-ctl -d /dev/video0 --list-formats-ext` and believe that. **The 75° field of view is quoted for 1920×1080 @ 60 fps; if that mode is unavailable, our field of view is wrong before we start** | 10 min | Everything perception |
| **C3** | Camera not calibrated | Spec states none will be provided; it's a screw-mount lens | 1 h | Perception accuracy |
| **C4** | Camera tilt unknown | Adjustable even in flight, so it's *our choice* — then measure it. A 5° error took crashes to 32 per 100 gates | 30 min | Perception accuracy |

### D — Simulator fidelity

| ID | Gap | Cost of being wrong | Effort |
|---|---|---|---|
| **D1** | Thrust-to-weight is a guess (hover assumed 25.5 %) | 58 crashes/100 gates at 10 % error | 5 min to measure |
| **D2** | Inertia and rate response are a 5-inch airframe's | See the warning below | 4 h |
| **D3** | Zero latency modelled | 50 ms alone: 2 → 55 crashes/100 gates | 3 h |
| **D4** | No drag model | Significant above 10 m/s | 3 h |
| **D5** | Perfect vision assumed | The largest single sim-to-real difference | 1 day |
| **D6** | Course map — **official table now in hand** | The official gate coordinates confirm an 85 × 165 ft (25.91 × 50.29 m) boundary, and our corrected track already matches it exactly. The spec's "60 × 21 m" is wrong. **Still survey for as-built placement** — a 7.5 % scale error was worth 2 → 15.5 crashes/100 gates | 2 h |
| **D7** | **Policy rate may be too fast** ⚠ | Simulator runs a 60 Hz policy; real gyro feedback may only be 30–50 Hz. Set the simulator's decimation to the measured rate **before** the long training run, or we train against a sensor that doesn't exist | 1 h |

> **The weight trap — read before touching the simulator.** The real drone is 1745 g, the simulated
> one 607.6 g. Setting the mass alone changes **nothing**: the simulator computes thrust as
> `(stick / hover_stick) × mass × g`, and acceleration is force over mass, so mass cancels. What
> does not cancel is (a) thrust-to-weight, (b) rotational inertia — the real airframe resists
> rotation 11–16× more — and (c) drag. **And if you set the real inertia but leave `rate_kp` at
> 0.08, the simulated drone becomes ~13× sluggish and every policy trained on it is worthless.**
> That gain isn't physics; it stands in for Betaflight's own controller, which is retuned per
> airframe. Re-derive it from the measured response: `gain ≈ inertia / response time`.

### E — Operations

| ID | Gap | Note |
|---|---|---|
| **E1** | ~~No compiler on the laptop~~ | **Downgraded.** The Betaflight software-in-the-loop build is no longer needed — the organizers' `fake_fc.py` is pure Python and does the same job better. `sudo apt install -y build-essential cmake ffmpeg poppler-utils` is still nice to have, no longer blocking |
| **E4** | Laptop↔Jetson link | USB-C to **micro-USB** (the small port next to the barrel jack, not a USB-A port). The board always answers on `192.168.55.1`. If the network half fails, the same cable gives a serial console on `/dev/ttyACM0` at 115200 |
| **E2** | Practice slot schedule unknown | Flight time is rationed. Get the list first thing |
| **E3** | Battery and propeller count unknown | The real limit on attempts |

---

## 4. Workstreams

The mistake to avoid is queueing everything behind the link. Five of eight streams start immediately.

| Stream | Name | Content | Waits for |
|---|---|---|---|
| **WS-1** | Link and control | A0–A13: copy the organizer toolchain, adapter, curve inversion, sign table, loop-rate measurement | **Nothing — critical path**, and testable on a laptop |
| **WS-2** | Vehicle characterization | D1–D4: weigh, balance, measurement flight, diagnostics | **Nothing** — pilot + blackbox |
| **WS-3** | Perception | B1–B5, C1–C4, D5: sensor mode, display context, calibration, data, detector, gate counter | Camera only |
| **WS-4** | Course survey | D6: survey, reconcile, build the map | **Nothing** — tape measure |
| **WS-5** | Simulator and training | Fold measurements in; cloud training; evaluate | **Nothing** — start randomized |
| **WS-6** | Integration and flight test | The bring-up ladder, in practice slots | WS-1; WS-3 + WS-4 for gate work |
| **WS-7** | Plan B | The slow stop-and-line-up controller | WS-1, WS-3 |
| **WS-0** | Operations | Slots, batteries, organizer questions, installs, kit | **Nothing** |

**Staffing.** Strongest systems person on **WS-1 alone**, with a second person reviewing the sign and
axis conventions — Isaac uses forward-left-up, Betaflight uses forward-right-down, and a reversed sign
there flips the drone on takeoff. It is the most common first-flight failure in this field.

Because the organizers ship `fake_fc.py`, **WS-1 no longer needs the drone to make progress.** Once
someone copies `~/target/` off a board, the adapter, the curve inversion, the sign table and the
watchdog verification can all be written and tested on a laptop — including against deliberate
fault modes (frozen IMU, saturated gyro, wrong scaling, bus errors). Use that. Arriving at the first
practice slot with tested code is worth more than any other hour this week.

**WS-2** is the cheapest high-value work available: one battery, one log, and the simulator stops
guessing. **WS-4** is two people and ninety minutes and retires a risk worth 15 crashes per 100
gates. **WS-3** is the longest pole after WS-1 and is mostly patient work — start the capture the
hour the camera produces a frame.

### The bring-up ladder (WS-6)

One rung at a time, one limit at a time. If a rung fails, fix it and re-fly that rung.

| Rung | What | Pass criterion |
|---|---|---|
| 0a | `setup_jetson_uart.sh --apply`, then `msp_bench.py info` | Firmware identity, sensor list, battery, arming blockers all readable |
| 0b | `imu_check.py`, then `imu_check.py --motion` | Every check passes, including the frozen-buffer and axis-mapping tests |
| 0c | `companion_listener_msp.py --listen-only` | Record the achievable poll rate. **This number sets the simulator's policy rate** |
| 1 | Bench, props off, command sweep on all four axes | Correct sign, correct magnitude, mask covers all four |
| 2 | Props on, strapped down, 10 % throttle | Correct motor order and direction, no oscillation |
| 3 | Pilot hover, manual | Trims out, blackbox clean, hover throttle noted |
| 4 | Jetson sends fixed hover, pilot holds override | Holds attitude; handoff in and out is bump-free |
| 5 | Jetson commands rate steps on each axis | Gyro follows within measured lag. **This is also the plant measurement** |
| 6 | Vision station-keeping 3 m in front of one gate | ±0.5 m for 20 s |
| 7 | Plan B through one gate from 5 m | 10 consecutive passes, no contact |
| 8 | Plan B, three gates | Counter advances correctly 10/10 |
| 9 | Plan B, full lap | One clean lap |
| 10 | Plan A, speed limiter at minimum, one gate | Passes without divergence |
| 11 | Plan A, full lap, limiter stepped up | Clean lap |
| 12 | Two laps at target pace | Repeatable |

### Deploy-time limits, ratcheted open

Applied in our code between the policy and the link. Each opens only after a clean flight at the
current setting. Log the active envelope in every flight header.

| Limit | Start | Target | Opens after |
|---|---|---|---|
| Max commanded rotation rate | 1.0 rad/s | 3.2 rad/s | Rung 5, then one step per clean rung |
| Max throttle stick | hover + 0.10 | 0.90 | Rung 4 |
| Min throttle stick | hover − 0.10 | 0.05 | Rung 5 |
| **Speed limiter** (scales the commanded-velocity input) | 30 % | 100 % | One step per clean lap |
| Altitude ceiling | 2.5 m | 4.5 m | Before the double gate |
| Watchdog deadline | 100 ms | 100 ms | **Never relaxed** |

The speed limiter is the most useful knob we have — it lets the *same* policy fly the course slowly
first, by scaling the commanded-velocity channels in the observation.

---

## 5. Procedures

### P0 — Board bring-up (30 min, do this first on every board)
1. Plug USB-C to **micro-USB** (small port next to the barrel jack). `ssh dcl@192.168.55.1`, password `dcl`.
2. `sudo ~/target/bringup-check.sh` — walks the whole stack in dependency order and stops at the first real failure, so its last line names the layer that actually broke.
3. `ls /dev/video0` and `nvpmodel -q` (expect 25 W). If the camera node is missing, `sudo ~/target/camera-bind-check.sh`.
4. `sudo ~/target/msp/setup_jetson_uart.sh --apply` — frees the serial port from Linux's login console.
5. `python3 ~/target/msp/msp_bench.py --port /dev/ttyTHS1 info` — firmware, sensors, battery, and the current reasons it won't arm.
6. `scp -r dcl@192.168.55.1:~/target ./` — **copy the toolchain off the board** so the team can work without it.
7. Save a `diff all` from the Betaflight CLI for this specific airframe.

**Shut down with `sudo shutdown -h now`.** The filesystem is on a real SSD; pulling power mid-write
eventually corrupts it.

### P1 — Camera calibration (1 h)
1. **First, find out which sensor modes actually exist:** `v4l2-ctl -d /dev/video0 --list-formats-ext`. The carrier wires only two of the camera's four data lanes, so the headline modes may not be available. The spec's 75° field of view is quoted for 1920×1080 @ 60 fps — **if that mode isn't there, our field of view number is wrong before we start.**
2. Print a checkerboard or ChArUco target and **glue it to something rigid and flat**. A curled printout gives a confident, wrong calibration.
3. Capture 30–50 views at race exposure and gain: near and far, tilted both axes, in all four corners of frame.
4. Solve with OpenCV. Expect reprojection error under ~0.5 px; worse means the target moved or isn't flat.
5. **Cross-check independently:** tape measure flat on a wall, camera exactly 2.00 m away, confirm the width spanned matches.
6. Record the result with the date and *which drone*. The four airframes may not match.

### P2 — Camera tilt (30 min)
1. Set the drone on a surface confirmed level with a spirit level.
2. Phone inclinometer against the lens face or bracket. Three readings, take the median.
3. Cross-check optically: known target, known height, known distance; confirm it lands where the calibrated model predicts.
4. Tilt is adjustable, so **decide it**: ~10° suits Plan B's slow near-level flight, ~20° suits fast forward-leaning flight. Choose, set, measure, write it down, put that number in the simulator.

### P3 — Vehicle diagnostics: one battery, one log (30 min)

The highest-value half hour of the week. Human pilot, blackbox on, in this order. **No autonomy
software required**, so this can happen on day one regardless of WS-1.

| # | Manoeuvre | Yields |
|---|---|---|
| 1 | Armed on the ground, 10 s idle | Gyro noise floor, vibration spectrum, motor idle |
| 2 | Stable hover at 1.5 m, 20 s | **Hover throttle** (D1) — the most valuable number available |
| 3 | 3 sharp rolls, 3 pitches, 3 yaws, settling between | **Rate response** (D2): rise time, overshoot, peak angular acceleration per axis. Replaces guessing inertia |
| 4 | Full-throttle climb 1.5 s, recover | **Max thrust**, true thrust-to-weight |
| 5 | Hold ~20° pitch down a straight until speed plateaus | **Drag** (D4): `k = m·g·tan θ / v²` |
| 6 | Hover 20 s near the end of the pack | **Battery sag**: how much hover throttle rises as it empties |
| 7 | Land, disarm, download the log, save `diff all` from the CLI | The settings dump. **Do not skip** — it's what will be missing at 2 a.m. |

Repeat manoeuvre 2 on **every airframe you might fly**. Hover throttle is what the whole simulator
hangs from, and four supposedly identical drones rarely are.

### P4 — Latency and clock alignment (4 h)
The most damaging factor in testing, and the one people estimate instead of measuring. The organizers
call their timing section "the section most likely to save you a week"; read it before starting.

Three facts about this hardware, from their guide:
- The frame timestamp is the Jetson's clock, not the camera's. The crystals differ by tens of parts per million, so a nominal 30.000 fps stream measures about 29.9997 and the error **accumulates**.
- The timestamp marks a frame boundary *after* the first row's exposure ended. The exposure midpoint you actually want to pair with a gyro sample is **behind** the timestamp, not ahead of it.
- **There is no hardware sync wire** between camera and flight controller on this carrier. The offset must be solved in software — and **pin the exposure first** (`show-camera.py --exposure-time` with equal min and max, plus `--ae-lock`), or auto-exposure moves the effective capture instant every frame and no calibration holds.

Then:
1. Establish the clock relationship, or the answer is meaningless. `frame-timestamps.py --verify` proves the clock domain.
2. Put a bright LED in the camera's view, switchable by the Jetson.
3. Flash it and record: command time → first frame showing it lit → detector output → MSP send → first gyro response.
4. That gives each stage separately. Knowing the total is 90 ms is useful; knowing 60 ms of it is the detector tells you what to fix.
5. Feed the total **and its variation** into the simulator as a randomized band.

### P5 — Course survey (90 min, two people)
1. Pick a datum: a permanent corner as origin plus a second fixed feature for the x-axis direction. Tape both, photograph them.
2. For each gate, measure **the base of both uprights** from the datum. Two points give centre *and* heading in one operation — far more reliable than measuring an angle.
3. Record bottom-bar height for one gate of each type; confirm the double gate's upper opening.
4. Note what the drawing omits: pylons, netting, lighting rigs, anything above 4 m.
5. Convert to the simulator frame and diff against **both** the organizer table and the drawing. Reconcile anything off by more than 0.25 m before training on it.
6. **Re-measure every morning and before every scored run.** Ask what tolerance gates are re-placed to — that number becomes the gate-jitter range in training.

### P6 — Gate detection and data labelling (the long pole)

**Why this is hard when the simulator makes it look easy.** In simulation the policy receives the
eight gate corners as exact pixel coordinates, computed by projecting known geometry through a
perfect pinhole camera. No noise. Corners never missed. Never the wrong gate. Visibility known
exactly. On the real drone every one of those is false: a compressed image of an orange-and-white
frame in a cluttered hall, motion blur, a rolling shutter that skews fast-moving objects, changing
light, other gates in frame, orange pylons that look like gate material, and a detector that
sometimes returns nothing and occasionally returns something confidently wrong.

**Closing this has two halves, and the second is the one teams skip:** make the detector as good as
is reasonable, *and* make the policy tolerant of a realistic detector by reproducing its **measured**
error statistics inside the simulator. Perfect detection isn't achievable. Graceful degradation is.

**Stage 0 — classical detector, first hour.** Tune the existing colour-and-geometry detector
(`vision/snake_gate_detector.py`, thresholds via `tools/hsv_tuner.py`) on real frames. No data, no
training, no installation. Noisier and shorter-ranged than a learned detector, but available
immediately, and it's the fallback that cannot fail to be ready. Plan B tolerates a noisy detector
far better than Plan A does.

**Stage 1 — capture, starting the hour the camera works.** Capture is the bottleneck, so begin
before you need it. Must cover:
- 1 m to 15 m, all approach angles including oblique;
- **every gate on the course** — backgrounds differ, and that's what a detector learns;
- gates cut off by the frame edge, and two gates overlapping;
- the double gate, from below and from the upper approach;
- backlit and shadowed, and whatever the lighting does across a day;
- **negatives**: frames with no gate, and frames with confusers — pylons, red signage, other drones, people;
- **motion blur** — which means recording every manual pilot flight. Hand-carried footage is easy to collect and missing exactly the degradation that matters.

**Stage 2 — labelling, done cheaply.** This is where teams lose a day. Three techniques cut the work
5–10×:
1. **Auto-label then correct.** Run the classical detector over all footage, keep its output as draft labels. Correcting a box is far faster than drawing one.
2. **Geometric auto-labelling — the big win.** We know the gate geometry exactly (2700 mm outer, 1500 mm inner) and, after P5, exactly where each gate is. If the camera pose for a frame is known, the eight corners project into the image exactly — the label is free and pixel-perfect. Bootstrap: hand-label ~40 frames, solve camera pose by PnP, then track along the video and re-project labels for every frame between. This turns an afternoon into thousands of labelled frames instead of a few hundred.
3. **Label what the policy needs.** The observation wants eight keypoints plus visibility flags, not bounding boxes — and keypoint labelling is slow. Cheaper hybrid: learned model finds the gate as a box, classical line-fitting finds the quadrilateral inside it. `vision/yolo_hybrid_gate_detector.py` is already shaped this way.

For a single visually distinctive class, fine-tuning a pretrained detector typically needs **300–800
well-chosen labelled frames**, not tens of thousands. Spend effort on coverage and hard cases, not
volume.

**Stage 3 — train and deploy.** Fine-tune on the laptop GPU (small model, minutes). **Note the
runtime constraint:** the Jetson image has no PyTorch and no ultralytics, so export to ONNX and run
through OpenCV's DNN module, or through TensorRT if it's present on the image (verify — it ships
with JetPack but isn't in the guide's package list). TensorRT engines must be built **on the Jetson
itself**; they're device- and version-specific and cannot be prepared in advance.

**Stage 4 — measure the detector, because this is what feeds the simulator.** Not the metrics
detector papers report. Measure: detection rate vs distance (what *is* our reliable range?); corner
pixel error vs distance and angle, median and 95th percentile; false positives per frame and what
triggers them; identity error rate; throughput and latency at 25 W. Get ground truth by placing the
drone at tape-measured positions relative to a gate — those frames have exactly known corner
positions. `tools/eval_detectors.py`, `tools/eval_gate_pose.py`, `tools/vision_rate.py` exist for
this.

**Stage 5 — close the loop.** Feed those measurements back as the simulator's vision model: pixel
noise as a function of distance, dropout rate, max range, frame rate, latency. Retrain. This is the
step that converts a detector measurement into a policy that survives a real detector, and it is why
Stage 4 is not optional.

### P7 — Gate counter
Maintain the expected gate number. Predict where that gate should appear using the surveyed map and
current pose estimate; accept a detection only if it lands near the prediction. Advance on a
confirmed pass, with a short lockout so one crossing can't count twice. **Encode explicitly that the
double gate counts twice per lap** — one physical gate carrying two numbers is where a counter
desynchronizes. If the expected gate isn't seen for a while: widen the search, then fall back to a
hold. Never silently re-label; log every re-association. **Test it offline by replaying a
hand-carried lap before it ever flies.**

### P8 — Simulator calibration loop
```
field measurement → update sim → train (cloud) → evaluate in stress harness
       ↑                                                    ↓
  compare logs  ←  fly and log  ←  deploy
```
**The closing step is what makes it honest:** after each flight, replay the *recorded command
sequence* through the simulator from the same initial state and overlay simulated against actual
trajectory. A simulator that cannot reproduce a flight it has the commands for cannot be trusted to
train a policy. Where they diverge tells you which parameter is still wrong, and in which direction.

Randomization bands — wide now, narrowed as measurements land:

| Parameter | Before measurement | After |
|---|---|---|
| Hover stick (thrust-to-weight) | 0.25 – 0.50 | measured ±0.03 |
| Thrust scale error | ±15 % | ±7 % |
| Rate response time | 30 – 90 ms | measured ±30 % |
| Total latency | 0 – 150 ms | measured ±30 % |
| Camera tilt | nominal ±8° | measured ±2° |
| Camera field of view | 70 – 80° | measured ±1° |
| Corner pixel noise / dropout | 0–5 px, 0–15 % | from Stage 4 |
| Vision rate | 15 – 60 Hz | measured |
| Gate position / heading jitter | ±0.3 m, ±5° | organizer re-place tolerance |
| Drag | ±30 % | ±15 % |
| Thrust-axis misalignment | ±2° | from CG measurement |
| Battery sag | thrust decaying over the episode | measured curve |

Ramp randomization in over the first ~20 % of training rather than starting at full width. Launch a
run every evening — one that finishes at 07:00 is one you fly at 09:00. For scale: Gene's runs were
~205 M samples; the 1-hour laptop retrain was 73 M, which is why it was still a toddler; a 10-hour
cloud run at 8–16 k environments reaches 2–4 B. Run 3–4 seeds in parallel. **Evaluate with the
stress harness, never with training reward** — training reward hid the 6 Sep collapse.

---

## 6. Schedule and decision gates

| Day | Goal | Exit criterion |
|---|---|---|
| **Thu 17** | We can command the drone, and we know the course | `~/target/` copied off a board; board bring-up (P0) passes; **poll rate measured**; **sensor mode list captured**; bench test props-off passes all four axes; measurement flight logged; hover throttle known; course surveyed; overnight training launched at the measured policy rate |
| **Fri 18** | We can see and count gates | Detector working on real gates; counter verified offline on recorded footage; autonomous hover; station-keeping at one gate |
| **Sat 19** | **A complete lap, by any means** | One clean Plan B lap — this is the floor under our score. Plan A begins with the limiter at minimum |
| **Sun 20** | Rehearsal and freeze | Both plans rehearsed under run conditions; gates re-measured; config frozen; **Plan A go/no-go** |
| **Mon 21** | **Scored** | First slot: most reliable configuration, flown slowly, to bank gates. Then step up |
| **Tue 22** | **Scored** | Improve on the banked score. Never risk the drone before a valid run exists |

**Plan A go/no-go (Sunday) — all three must hold:**
1. It has flown ≥3 gates autonomously on the real drone, more than once.
2. Its observations have been checked against the real camera and the real flight-controller curves.
3. Its failure mode is a safe hold, not a flyaway.

---

## 7. Contingencies

- **Link not working by end of Day 1** → this is now unlikely, since the protocol layer is the organizers' and is known-good. If it happens, the cause is almost certainly one of: the serial console not released (`setup_jetson_uart.sh --apply`), wrong port or baud, or the FC unpowered. Run `sudo ~/target/bringup-check.sh` and believe its last line. Escalate to the organizers — they wrote this stack and have diagnostics for it.
- **Camera gives flat, dim, unusable images** → expected over a plain SSH session; the image processor needs a graphics context. Either arrange a display on the board or switch the detector to minimally-processed frames. Do not spend hours assuming the camera is broken.
- **Detector unreliable on real gates** → fall back to the classical detector, tuned on site.
- **Gate counter desynchronizes** → Plan B doesn't strictly need the gate number, only "is there a gate ahead and where". Prefer Plan B until the counter is proven.
- **Plan A can't complete a lap by end of Day 4** → score with Plan B. A completed slow run beats a fast run that ends at gate 4.
- **Drone damaged** → repair takes priority over everything; go to simulator and offline work.

---

## 8. Open questions for the organizers

Already answered by the spec: MSP over UART with demo software provided; Betaflight CLI access
covers rates, PIDs, filters, but no reflashing; camera tilt adjustable in flight; exposure and gain
controllable; no calibration matrices; valid run is 2 laps with no human intervention; manual
override always supersedes autonomy; practice is in scheduled slots.

Still open:
1. May we assign an ANGLE mode switch, and change `msp_override_channels_mask`, the rate curve and the throttle curve?
1b. What poll rate should we expect on the flight-controller link in practice, and is raising the baud rate above 115200 supported?
2. **May we place tape or markers on the gates to identify them?** A yes largely removes the gate-counting problem — ask early.
3. Who arms for a scored run? Does arming and enabling override before the start line count as intervention?
4. Is there a maximum run duration? How many scored attempts per team per day?
5. Are gates re-placed between runs, and to what tolerance?
6. Are gates counted only in order — does skipping one stop the count? Is a run that crashes after N gates scored as N?
7. The 75° field of view is quoted for 1920×1080 @ 60 fps, but the carrier wires only two CSI lanes — is that mode actually available, and if not, what is the field of view in the modes that are?
7b. What is the intended way to get ISP-processed (properly exposed) frames in flight, headless?
8. Are all four drones on identical firmware and configuration? Is a barometer fitted?
9. Where do we get the camera and serial demo applications?
10. Exact as-built gate heights, and the double gate's upper opening height.
