# Field measurement card

Rewritten 17 Sep 2026 after a four-way review: Gene's `PHYSICAL_TUNING.md`, a full audit of the
live simulator parameters (read out of PhysX at runtime, not out of the docs), the published
sim-to-real quadrotor literature, and a feasibility pass against what we actually have on site.

**The short version.** Measure four things. Skip inertia entirely — it is not identifiable, not
needed, and not even settable in this simulator. Randomise the rest at 10–20 %.

---

## 0. Physical items to obtain

Everything else on this page is free. These are not.

| Item | Needed for | If we cannot get it |
|---|---|---|
| **Printer + rigid flat backing** (foam board, clipboard) | ChArUco camera calibration | **Blocks the largest perception item.** A curled printout gives a confident, wrong calibration. Ask the Neros desk on day one |
| **Small spirit level** (£5) | Confirming the camera tilt surface is level | Phone levelling app on a separately-confirmed surface |
| **Kitchen scale** (0.01 kg) | Race-ready mass per airframe | Ask the Neros desk — they weigh drones |
| **Long tape measure, 30–50 m** | Course survey, drag-test floor marks | A laser rangefinder makes the survey ~3× faster with two people |
| **Phone tripod** | Video methods for climb / drag | A gate leg, a chair, tape |

**Do not go looking for:** a thrust stand, a fish scale, a ratchet strap, an LED and jumper wires.
Every test that needed them has been replaced by something better (see §4).

---

## 1. Measure these four. Nothing else is close.

Ranked by measured cost of getting them wrong. Our own 16 Sep durability sweep and the published
literature agree on the ordering.

| # | Constant | Procedure | Time | Flight? |
|---|---|---|---|---|
| **1** | **All-up mass, per airframe** | Kitchen scale, race-ready, with the pack we fly | 30 s | no |
| **2** | **Hover throttle → thrust coefficient** | 20 s hover at ~1.5 m, ANGLE mode. Take the **mean of `motors()` across all four**, not the stick — that is the post-mix, post-curve value the ESCs saw. Log `analog()['voltage_v']` alongside. Median of three. **Repeat per airframe** | 10 min | hover |
| **3** | **Peak acceleration, full throttle ≤1.5 s** | From a hover, brief full-throttle climb, recover. `T/W = 1 + a_peak/g` | 5 min | yes |
| **4** | **End-to-end latency** | Bench, in stages: `frame-timestamps.py` for capture→userspace; tape on a motor bell filmed with `live-view-pts.py` for command→actuation; `companion_listener_msp.py --listen-only` for poll rate | 1 h | no |

**Enable bidirectional DShot first** — five minutes, and it turns 2 and 3 from estimates into
measurements by giving per-motor RPM over the existing signal wire, logged in blackbox. Set
`motor_poles` correctly (usually 14) or every RPM is wrong by a constant factor.

### Why mass is #1 despite cancelling

Mass cancels in `a = (stick/hover)·g`, so changing it alone does nothing — that is still true and
still the most misunderstood thing here. It ranks first because **mass and the thrust coefficient
are the denominator and numerator of thrust-to-weight**, and T/W is what actually transfers. In the
one controlled study that ranks these directly, a +30 % mass error was *total failure* — the policy
would not fly — while a +30 % inertia error cost only 1.5× tracking error.

---

## 2. Resolve before the retrain: the thrust law is ambiguous by 3.5×

Both the simulator (`dynamics/rate_control.py:stick_to_newtons`) and our plant
(`flight/dynamics.py:specific_thrust`) use

```
thrust = (stick / hover_stick) · m · g          ← LINEAR in stick
```

Real propeller thrust goes as RPM², and motor command → RPM is roughly linear. So:

| Thrust law | Implied T/W ceiling at `THRUST_MAX = 0.90`, `hover = 0.255` |
|---|---|
| Linear (what we simulate) | **3.53** |
| Quadratic (what propellers do) | **12.5** |

This is the single most sensitive parameter in the model and it is currently uncertain by 3.5×.
**Measurements 2 and 3 settle it empirically** — no theory needed. Do them before the GPU spend,
not after.

---

## 3. Do not measure inertia. Match the step response instead.

The rotational dynamics are `ω̇ = (kp/I)(ω_des − ω) − (kd/I)ω`, clamped at `±M/I`. **Only ratios
appear.** The absolute inertia is neither needed nor identifiable.

It is also **not settable**. The USD sets `physics:diagonalInertia = (0,0,0)` and `density = 0.0`,
so PhysX auto-computes inertia from the collision hulls. `PHYSICAL_TUNING.md` Phase 4 step 3 says
to write `ixx/iyy/izz` into the URDF — **the URDF is not loaded by the simulator. That instruction
changes nothing.** Measured from the live sim: `Ixx 2.75e-3, Iyy 3.16e-3, Izz 5.2e-3` kg·m².

**Do this instead.** Fly rate steps, log blackbox, deconvolve `setpoint → gyro` to recover the real
closed-loop step response per axis, then fit `rate_kp`, `rate_kd`, `moment_limit` and an action
delay until the simulated response overlays the logged one. This subsumes inertia, motor lag,
Betaflight's PIDs and its filter phase lag into one identified curve — which is what the policy
actually experiences. It also makes the inertia/gain coupling failure *structurally impossible*
rather than something to remember to avoid.

Rate steps: three sharp steps per axis, left and right, settling between. ~10 min of airtime.

---

## 4. Not worth measuring — closed by decision

| Item | Why not |
|---|---|
| Rotational inertia (bifilar pendulum) | Not identifiable, not needed, not settable. See §3 |
| Motor KV, ESC specs, thrust coefficients from databases | `Allocation` and `Motor` are **dead code** in this environment — never instantiated, `use_motor_model=False` |
| Motor-to-motor diagonal, arm length | Enter nothing. Record in 2 minutes for the log; do not plan around them |
| Centre-of-gravity offset by ruler balance | The hover attitude bias (free, from measurement 2) gives the same information in the units that matter — a camera pitch bias in degrees |
| Static thrust via tie-down and luggage scale | We have no stand, no strap, no anchor. An 8-inch 6S quad at full throttle on a tether is the most dangerous thing anyone proposed. Measurement 3 is free, safe and better |
| **Full-throttle climb read from the IMU** | **Biased −0.46·u² g by vibration rectification — that is −0.37 g at full throttle.** Not a marginal error, a third of the signal. Use RPM telemetry or video |
| Rolling-shutter readout time | Bound it from the mode's line time and fold it into the latency band |
| Ground-effect height | Takeoff is untimed; the clock starts at the first gate |
| Barometer anything | Settled: drops ~6 m within 6 s of spin-up. Do not re-confirm |
| HVAC air movement | Indoor hall. A ribbon test in 60 s, then leave wind out of the randomisation |

---

## 5. Randomisation ranges

The training environment currently has **no domain randomisation at all** —
`enable_corruption = False`, no parameter sampling, only a ±0.1 N disturbance push. That has to
change: on a racing quad through 1.5 m gates, **zero randomisation fails outright** (5–13 gates
real, against 46–60 in sim) *even with a system-identified model*.

| Blanket randomisation | Real-flight gates |
|---|---|
| 0 % | 5, 5, 7 — fails |
| **10 %** | 39, 39, 38 — best reward |
| **20 %** | 36, 35, 35 — most gates |
| 30 % | 37, 37, 37 — robust but slower |

**10–20 % blanket is the sweet spot.** The refinement that matters: *randomise what you could not
measure; do not randomise what you did.* Randomising mass ±30 % after measuring it more than
doubled the error in one study versus simply using the measured value.

| Parameter | Range | Note |
|---|---|---|
| Thrust-to-weight | **±15 % around measured** | Plus battery-sag drift within an episode |
| **Total latency** | **`U(30, 130)` ms** | Wide and asymmetric on purpose. Most under-modelled quantity in the field |
| Closed-loop rate response | rise ±20 %, delay ±30 % | Around the *matched* response, not around raw inertia |
| Motor time constant | `U(0.02, 0.10)` s | Weak for us — we command body rates, so Betaflight's loop hides it |
| Drag `d_x, d_y` | `U(0.2, 0.9)` s⁻¹ mass-normalised, `d_z = 0` | Weakest-grounded number here; 5-inch values widened by judgement |
| Camera `fx` | `U(415, 435)` | Three calibrations gave 423.6–426.7 |
| Double-gate heights | upper 3.5–4.8 m, lower 1.2–2.2 m | Neither is sourced |
| Everything else | ±10–20 % | |

---

## 6. Possible free win — 10 minutes, props off

Betaflight's `rc_smoothing` auto-selects a **~15 Hz cutoff** when RC arrives slowly over MSP, worth
roughly **30 ms**. Filters are explicitly inside the CLI scope the rules grant us. Latency is our
top sensitivity — 50 ms alone took crashes from 2 to 55 per 100 gates — so reading `rc_smoothing_*`
out of `diff all` and testing a higher cutoff may be the best ten minutes of the week.

---

## 7. The two sessions

**Bench, day one, ~4 h, nothing flies.** Two thirds of the total value is here.

1. Betaflight CLI over the MSP UART, then `diff all` per airframe (4×)
2. MSP poll rate — idle, with vision running, **and with RC transmitting at 50 Hz** (the only
   configuration that matters, and the one everyone forgets)
3. `v4l2-ctl -d /dev/video0 --list-formats-ext`, then `frame-timestamps.py --verify`
4. Gyro units check — settles a 16× scale [CONFLICT] in `SPEC.md`
5. `rc_smoothing` (§6)
6. **MSP override hold behaviour, props off** — we design around an unverified hypothesis whose
   failure mode is a flyaway. Never test this in flight
7. Camera tilt: **set to 20°**, verify with an inclinometer, median of three
8. Camera position in the body frame — tape measure, 10 minutes
9. ChArUco calibration, in parallel on a second person

**Flight, one battery, training cage, in this order.** The spec grants cage time separately from
scored slots, and manual piloting is permitted — **every measurement flight belongs in the cage.**

1. 10 s armed idle — gyro noise floor
2. 20 s hover at 1.5 m — **measurement 2**, plus hover attitude bias, free
3. Rate steps, roll / pitch / yaw — **§3**
4. Full-stick continuous roll, 2 s — validates the whole rate-curve chain
5. Brief full-throttle climb, ≤0.5 s — **measurement 3**
6. 20 s hover on a tired pack — battery sag
7. Land, save the log, save `diff all`

≈12 minutes of airtime. **If it collapses to one battery, fly only the hover and the rate steps.**

---

## 8. Build first: a Jetson MSP logger

~60 lines on `MSPLink.request_batch()`, logging `attitude()`, `raw_imu()`, `rc_channels()`,
`motors()`, `analog()` and command **139 `MSP_MOTOR_TELEMETRY`** (the constant is already in the
organizers' `msp.py`; `request(cmd, payload)` is generic, so RPM needs only a `struct.unpack`), to
CSV with `time.monotonic()` stamps.

This makes **the Jetson the flight recorder**: hover throttle, hover attitude bias, battery sag,
achieved rates and link rate all fall out of one pilot hover, with no Betaflight Configurator, no
USB cable to the FC, and no dependency on blackbox working. Testable on the laptop against the
organizers' `fake_fc.py`. Build it before anyone flies.

---

## 9. Risk

The rules say teams are **not financially liable** for drone damage and each team has **4 dedicated
drones** with on-site repair. That is not permission to be casual: another team lost a Jetson to a
crash on 17 Sep with the cause never established. A crash costs **a slot, an airframe for the day,
and possibly the logs on that board**.

- Nominate one airframe as the measurement drone; put every risky manoeuvre on it.
- Hover throttle must still be measured **per airframe** — four supposedly identical drones rarely are.
- Mirror `~/target/` and every log to the laptop at the end of each session.
