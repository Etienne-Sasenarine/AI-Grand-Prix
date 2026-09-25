# Experiments — what to run, in order, and what each one yields

Written 17 Sep 2026. Companion to `SIM_MEASUREMENTS.md`, which explains *why*
these and not others. This one is the running order and the commands.

**Tooling is built and tested.** Everything below runs against `tools/`, and
every tool in there passes a self-test against a simulated flight controller
with known parameters — so the analysis code is checked against a ground truth
it was never told, not merely checked for not crashing.

```
cd tools && python3 run_all.py        # 5/5, no hardware needed
```

| Tool | What it does |
|---|---|
| `tools/msp_logger.py` | Polls the FC and writes a CSV. Read-only: never arms, never transmits, never writes a setting |
| `tools/analyze_hover.py` | Hover log → hover throttle, attitude bias, battery sag, telemetry rate |
| `tools/analyze_rates.py` | Rate-step log → `tau` and the acceleration ceiling per axis, and the simulator gains to set |
| `tools/bf_cli.py` | Betaflight CLI over the MSP port, from the Jetson. `diff all` with no Configurator and no second cable |
| `tools/mock_link.py` | A flight model standing in for the FC, so all of the above are testable on a laptop |

Validation numbers from the self-tests: hover throttle recovered to **0.0003**,
`tau` to within **8 %**, and a saturated axis correctly distinguished from an
unsaturated one — which matters because only a saturated step measures the
ceiling.

---

## E0 — Prerequisite, one command

```bash
scp -r dcl@192.168.55.1:~/target ./
```

Nothing below runs without the organizers' MSP library. One minute. Until it
lands, the tools run against the mock and prove only their own arithmetic.

---

## Bench experiments — no flying, no airframe risk

Do all of these before anyone arms a drone. Two thirds of the total value.

### E1 — Free the UART, confirm the link *(5 min)*

```bash
sudo ~/target/msp/setup_jetson_uart.sh --apply     # once per board
python3 ~/target/msp_bench.py --port /dev/ttyTHS1 info
```

**Yields:** firmware version, sensor list, arming blockers. **Done when:** the
board answers. Linux runs a serial console on `/dev/ttyTHS1` by default and will
fight for every byte until this is run.

### E2 — Full settings dump, per airframe *(15 min)*

```bash
python3 tools/bf_cli.py --dev /dev/ttyTHS1 --dump logs/diff_bilbo.txt
```

**Yields:** the rate curve (`*_rc_rate`, `*_srate`, `rates_type`), the throttle
curve (`thr_mid`, `thr_expo`), `msp_override_channels_mask`, `rc_smoothing*`,
`dshot_bidir`, `motor_poles`, `blackbox_device`, `failsafe_procedure`.

The tool prints each of these with a note on why it matters, and flags two
things automatically: an override mask that is not 15, and bidirectional DShot
being off.

**Replaces:** every Betaflight constant in `flight/betaflight_curves.py`, which
currently comes from **another team's airframe**. **Done when:** one dated file
per drone.

### E3 — Turn on bidirectional DShot *(5 min)* — do this early

```bash
python3 tools/bf_cli.py --dev /dev/ttyTHS1 --allow-write --save \
    --cmd "set dshot_bidir = ON" --cmd "set motor_poles = 14"
```

**Yields:** per-motor RPM over the existing signal wire, logged in blackbox and
readable over MSP. This is what turns the thrust coefficient from an estimate
into a measurement, and it costs five minutes and no hardware.

**Check `motor_poles` against the actual motors** — a wrong value scales every
RPM reading by a constant and the error is invisible.

**Ask the Neros desk first.** The rules grant us "rates, PIDs, filters,
telemetry"; bidirectional DShot is arguably telemetry, but it is worth the
question rather than the assumption.

### E4 — Achievable telemetry rate, in the configuration we fly *(30 min)*

```bash
python3 ~/target/companion_listener_msp.py --port /dev/ttyTHS1 --listen-only
python3 tools/msp_logger.py --dev /dev/ttyTHS1 --hz 200 --seconds 20 --out logs/rate_idle.csv
```

Three runs: **(a)** idle, **(b)** with the vision pipeline running, **(c)** with
an RC stream transmitting at 50 Hz. Watch `tegrastats` in a second session.

**(c) is the only one that matters** and is the one everyone forgets — polling
telemetry while also commanding is a different budget from polling alone.

**Yields:** the real control rate, which sets the simulator's `decimation`.
The logger prints achieved Hz and the worst gap; if it falls well short of the
request, that shortfall *is* the result.

**Settles:** whether the guides' "30–50 Hz is realistic" is pessimistic. Another
team measured ~92 Hz IMU-only and ~50 Hz alongside a 50 Hz RC stream.

### E5 — Gyro units *(20 min)* — settles a 16× conflict

```bash
python3 ~/target/msp/imu_check.py --dev /dev/ttyTHS1 --motion --json logs/imu.json
python3 tools/msp_logger.py --dev /dev/ttyTHS1 --seconds 10 --out logs/gyro_units.csv
```

Then rotate the board about 90° in about a second by hand, and compare the
integral of the raw gyro against the change in `attitude()`.

**Yields:** whether raw gyro is counts at 16.4 per °/s, or already °/s.
`SPEC.md` records this as an open **[CONFLICT]**, and the library's own
docstring says only "raw FC units", so it will not save you. A 16× scale error
on the policy's gyro input is silent and fatal.

**Feeds:** `--gyro-scale` on both analysis tools.

### E6 — `rc_smoothing` — possibly 30 ms for free *(30 min)*

```bash
python3 tools/bf_cli.py --dev /dev/ttyTHS1 --cmd "get rc_smoothing_auto_factor"
```

Betaflight auto-selects a cutoff from the *observed* RC rate. MSP-injected RC
arrives slowly, so it can settle near 15 Hz and add ~30 ms of lag.

**Why it is worth 30 minutes:** latency is our top sensitivity — 50 ms alone
took crashes from 2 to 55 per 100 gates. Filters are explicitly inside the CLI
scope the rules grant.

**Procedure:** read the current values, then props-off, stream RC at 50 Hz and
compare the commanded step against the `rc_channels()` readback and the motor
outputs. Try a fixed higher `rc_smoothing_setpoint_cutoff` and re-measure.

### E7 — MSP override hold behaviour *(20 min)* — **props off, always**

Arm props-off via `msp_bench.py demo --props-off --max-throttle 0.05`, stream RC
from our adapter, then **kill the process** and watch `rc_channels()` and
`motors()`.

**Yields:** whether Betaflight freezes at the last commanded value, and for how
long, before failsafe acts.

**Why it matters:** we currently design the watchdog around an *unverified
hypothesis* whose failure mode is a flyaway. **Never test this in flight.** The
props-off version answers it completely.

### E8 — Axis signs *(10 min)* — props off

```bash
python3 -c "import sys; sys.path.insert(0,'flight'); \
            import control_adapter; control_adapter.verify_axis_map()"
```

Then tip the airframe by hand on each axis and confirm the gyro moves the way
the NED convention says it should.

**Yields:** the three sign conventions in `flight/control_adapter.py`, which are
currently **UNVERIFIED and the most dangerous assumption in the package**. A
reversed sign flips the drone on takeoff. Two people.

### E9 — Camera modes and true frame rate *(20 min)*

```bash
v4l2-ctl -d /dev/video0 --list-formats-ext
python3 ~/target/frame-timestamps.py --verify
python3 ~/target/frame-timestamps.py --limit 600 --csv logs/frames.csv
```

**Yields:** which modes actually exist (only 2 of 4 CSI lanes are wired, so
1920×1080@60 may not), the true frame rate, and the capture→userspace latency
median and p95 from the `latency` column — one of the four latency stages,
measured for free.

**Settles:** whether the spec's 75° HFOV even applies. It is quoted for a
specific mode; a different mode means a different field of view.

### E10 — Camera tilt: set it, then verify *(30 min per airframe)*

**Set to 20°.** This is a decision, not a discovery — see `SPEC.md`, "Why the
tilt is 20°, not 0°". Level surface confirmed with a bubble level, phone
inclinometer against the lens face, median of three.

**Yields:** `CAMERA_TILT_DEG` confirmed. A 5° error took crashes to 32 per 100
gates.

**Add the hover attitude bias from E12** — if the aircraft hovers at −2° pitch,
the effective tilt is 22°, and `analyze_hover.py` prints that arithmetic for you.

### E11 — Camera intrinsics *(1.5 h per airframe)*

ChArUco 7×5, 36 mm square, 27 mm marker, `DICT_5X5_100`. **Glue it to something
rigid** — a curled printout gives a confident, wrong calibration. 30–50 views at
race exposure and gain, in the mode E9 found. Cross-check with a tape measure
against a wall at exactly 2.00 m.

**Yields:** our own `fx`, `fy`, `cx`, `cy` and distortion, per airframe,
replacing another team's numbers. The organizers state calibration will not be
provided.

**Needs a printer and rigid backing — confirm both exist on day one.** This is
the only bench item with an equipment dependency we cannot work around.

---

## Flight experiments — training cage, one battery

The rules grant **dedicated cage time in addition to track slots**, and manual
piloting is permitted. **Every measurement flight belongs in the cage.**

Start the logger before the pilot arms, and leave it running for the whole pack:

```bash
python3 tools/msp_logger.py --dev /dev/ttyTHS1 --hz 200 --seconds 300 --out logs/flight1.csv
```

One file covers every experiment below.

### E12 — Hover *(20 s)* — the highest-value measurement we have

Pilot hovers at ~1.5 m, ANGLE mode, as still as they can. Repeat three times.

```bash
python3 tools/analyze_hover.py logs/flight1.csv
```

**Yields, all from the same 20 seconds:**

| Output | Feeds |
|---|---|
| **Hover throttle** | `THRUST_HOVER` — the top-ranked sensitivity in every source we have |
| **Hover attitude bias** | Effective camera tilt, and it supersedes the ruler-balance CG method |
| **Battery sag** | A thrust-scale *drift* to randomise over, which is what a 2-lap run experiences |
| **Telemetry rate** | Cross-check on E4 |

The tool reads **motor outputs, not the throttle stick** — post-mix, post-curve,
post-TPA, which is the unambiguous number. `SPEC.md` records a live ambiguity
here that this settles.

It also averages only over frames where the aircraft is genuinely holding
station (tilt ≤ 8°, rates ≤ 20 °/s). A pilot is constantly correcting, and
averaging the corrections biases the answer by however hard they were working.

### E13 — Rate steps *(10 min)* — this replaces measuring inertia

Three sharp steps per axis, left and right, settling between. Step sizes
**1.0, then 2.0, then 3.2 rad/s**, ~0.4 s each, 2 s apart.

```bash
python3 tools/analyze_rates.py logs/flight1.csv
```

**Yields:** `tau` and the angular-acceleration ceiling per axis, and the tool
prints the `rate_kp` and `moment_limit` to set for any chosen inertia.

**Why three step sizes and not one:** only a step big enough to *saturate* the
axis measures the ceiling. A gentle step yields a number that is really just
`kp` times the step size. The tool detects saturation from the shape of the rise
and says which case each axis is in — verified in its self-test, where only the
axis with the lowest ceiling is flagged.

**Do not measure inertia.** Only `kp/I`, `kd/I` and `M/I` appear in the
dynamics; the absolute value is not identifiable and not needed. It is also not
settable — PhysX auto-computes it from the collision hulls, so the URDF edit
`PHYSICAL_TUNING.md` Phase 4 instructs **does nothing at all**.

**`tau` from 50 Hz telemetry is indicative only** — a 40 ms time constant is two
samples. The ceiling and the plateau survive the low rate; `tau` wants blackbox.
The tool warns when the rise is thinly sampled.

### E14 — Full-stick continuous roll *(3 min)*

Full deflection, hold 2 s, read the plateau.

**Yields:** the real maximum rate, validating the whole curve chain end to end.
A steady-state number, so 50 Hz sampling is entirely adequate. Expect ~440 °/s
if the dumped `rc_rate`/`srate` and our curve model are both right — this is the
measurement that retires the UNVERIFIED caveat on `betaflight_curves.py`.

### E15 — Brief full-throttle climb *(≤0.5 s)*

From a hover, full throttle for under half a second, then recover.

**Yields:** `T/W = 1 + a_peak/g`, which **settles the linear-vs-quadratic thrust
question** — currently a 3.5× ambiguity (3.53 or 12.5) on the most sensitive
parameter in the model.

**Read peak RPM if E3 was done, not the accelerometer.** The onboard
accelerometer is biased by **−0.46·u² g** from vibration rectification — that is
**−0.37 g at full throttle**, a third of the signal. If DShot telemetry is
unavailable, film it: the gates are 2.7 m tall and make a perfectly good ruler.

**Risk:** a ceiling strike. Keep it under 0.5 s and brief the pilot on the abort.

### E16 — Hover on a tired pack *(20 s)*

Repeat E12 near the end of the battery. **Yields:** the sag slope, converting a
fixed thrust scale into a drifting one.

### E17 — Drag *(10 min air + 1 h analysis)* — only if time allows

Hold a fixed ~20° pitch down a straight until speed plateaus, at 3–4 lean
angles. `D = m·g·tan(θ)` at the plateau.

**There is no onboard speed source** — no GPS, no optical flow, the barometer is
unusable, and the rules confirm **no external tracking exists in the building**.
Speed comes from a phone at 60 fps filming the drone crossing two floor marks a
known distance apart, or from gate-to-gate timing on the surveyed course.

**Deprioritised, with a caveat.** The simulator has no drag term at all, so this
is worthless until one is added. But the literature puts drag at ~0.7 g of force
at 14 m/s, which is first-order rather than a correction, and our own sensitivity
sweep never tested it because there was nothing to vary. **It is the weakest-
grounded entry in the randomisation table.** If a second flight slot appears,
spend it here.

---

## If the flight budget collapses to one battery

Fly **E12 and E13**. Hover throttle and the rate step response are the two
numbers the simulator genuinely hangs from, and between them they cover the top
of every ranking we have — our own durability sweep, Gene's `PHYSICAL_TUNING.md`,
and the published sim-to-real literature, which agree on this despite
disagreeing on much else.

---

## Risk

The rules say teams are **not financially liable** for damage and each team has
**4 dedicated drones** with on-site repair. That is not permission to be casual:
another team lost a Jetson to a crash on 17 Sep, cause never established. A
crash costs **a slot, an airframe for the day, and possibly the logs on that
board**.

- Nominate one airframe as the measurement drone; put E15 and E17 on it.
- **Hover throttle must still be measured per airframe** (E12) — four supposedly
  identical drones rarely are.
- Mirror `~/target/` and every log to the laptop at the end of each session.
- E7 props-off, always. E8 with two people.
