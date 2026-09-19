# The next flight session — one page

Written 18 Sep after a 25-minute slot produced no data. Take this to the flight
line. **Nothing here needs a laptop, Wi-Fi, SSH, or me.**

---

## Why the last session produced nothing

The recorder ran over SSH from the laptop, so the Jetson had to be on Wi-Fi and
reachable for the whole flight. It wasn't. The aircraft flew fine — the radio
talks straight to the flight controller and never needed us — but the thing that
writes the data was unreachable, so nothing was written.

That dependency is now gone. **The drone records itself.**

---

## Step 0 — make ANGLE the default (once per drone, ~5 minutes, propellers off)

**Do this first. Nothing else matters as much.**

ANGLE is Betaflight's self-levelling mode: let go of the right stick and the
aircraft levels itself. ACRO is full manual: it stays at whatever lean you left
it. Out of the box Betaflight has **no ANGLE assignment at all**, so the aircraft
is always in ACRO — the hardest possible way to fly a 1745 g 8-inch quad, and why
the last attempts were hard.

**What the one command below sets up:**

| Who is flying | Switch | What the drone does |
|---|---|---|
| **A person** (the normal, resting position) | autonomy switch **off** | **ANGLE** — self-levelling, leans 25° at most |
| **The Jetson** | autonomy switch **on** | **ACRO** — what the trained policy expects |

"The autonomy switch" is whichever switch engages **MSP override**, Betaflight's
name for handing the sticks to the Jetson. The command reads that off the flight
controller and puts ANGLE on every *other* position of the same channel, so there
is nothing to look up and nothing that can be left in the wrong position. Taking
the aircraft back from the Jetson puts it straight back into ANGLE.

**⚠ The label on the transmitter is not to be trusted.** The switch marked
ACRO / ANGLE does *not* select ANGLE, and flipping it to "ANGLE" has made the
drone die (reported 18 Sep) — most likely because that position is the autonomy
switch, and it handed roll, pitch and yaw to a Jetson that was not flying.
**Never flip to the autonomy position unless the autonomy program is running.**

If the recorder service below is already installed, stop it first
(`sudo systemctl stop aigp-recorder`, and `start` it again afterwards): the
flight controller's serial port can only be open in one program at a time.

**1. Look, without changing anything:**

```
python3 ~/aigp/setup_angle_mode.py --dev /dev/ttyTHS1 --beginner --dry-run
```

**2. Do it:**

```
python3 ~/aigp/setup_angle_mode.py --dev /dev/ttyTHS1 --beginner --yes
```

It finishes by reading the flight controller's own state back and printing
`ANGLE -- self-levelling is ON`. It goes over MSP, **not** the Betaflight CLI, so
it will not wedge the board; it refuses to write while armed, touches no other
mode, reads everything back before saving, and changes nothing about ACRO, the
rates or the throttle curve. **It has not yet run on a real flight controller** —
it passes against a mock shaped like ours — which is what step 1 and step 3 are
for. It is per drone: a replacement airframe needs it again.

**3. Bench test, propellers off:** throttle up slightly and tilt the aircraft by
hand. The low motors should spin up and hold until you level it again. If they
do not, ANGLE is not active and there is no point flying.

**To find out which switch is the autonomy switch** (worth a minute, no drone
needed): on the transmitter open the servo monitor (Linkage menu), flip each
switch, and note which one moves **channel 9**. That one, pushed to its far end,
is the autonomy switch. `--watch` shows the same thing from the Jetson.

---

## Before you leave the bench (once, ~5 minutes)

Install the recorder so it starts on boot. This needs the drone reachable, so do
it at the bench, not on the line:

```
python3 ~/aigp/flight_recorder.py --install
```

It prints three `sudo` commands. Run them. Then confirm:

```
systemctl status aigp-recorder --no-pager
```

From then on, **every time the drone powers up it starts recording**, into
`~/flights/flight_<date>_<time>.csv`. There is nothing to start and nothing to
remember.

---

## At the flight line

| | |
|---|---|
| 1 | **Battery in.** Recording starts by itself a few seconds later |
| 2 | **Fly.** Take off, settle into a steady hover at about head height |
| 3 | **Flip a spare switch** while hovering — hold it for 5+ seconds, then flip it back |
| 4 | **Repeat at least three times.** Land and re-hover between them if that's easier |
| 5 | **Brief full-throttle climb** (≤1 s) from a hover, then recover — this is the thrust measurement |
| 6 | **Battery out.** The file is already on disk |

**Three hovers minimum.** One is a number, two is a coin toss.

### Which switch?

Any spare one. Channels 6, 8, and 12–16 are unassigned in Betaflight, and the
recorder logs **all sixteen channels**, so you don't have to decide in advance or
tell anyone which you used — the analysis finds it.

It just has to be a switch you can reach while flying, and one you don't flip for
any other reason during the flight.

---

## Afterwards

Plug in, and:

```
python3 ~/aigp/analyze_flight.py ~/flights/flight_<...>.csv --mass 1.745
```

It finds the marker switch by itself, prints every channel it considered so a
wrong guess is visible, and applies the same acceptance tests as before: armed
throughout, not tilted, not rotating, throttle steady, long enough, three windows
minimum. **Marking says where to look; it never overrides the data.**

Out comes the hover throttle from the **motor outputs** — not the stick, which
passes through Betaflight's throttle curve first and is a different number.

---

## What you get from one flight

```
python3 ~/aigp/analyze_thrust.py ~/flights/flight_<...>.csv
```

gives thrust-to-weight from the same recording. It uses **rotor speed, not the
accelerometer** — the accelerometer is biased low by about 0.37 g at full
throttle through vibration, which is a third of the signal, and would make the
aircraft look weaker than it is. Mass never enters the calculation: it cancels
between the hover and the burst.

| Measurement | From |
|---|---|
| **Hover throttle** | the marked hovers |
| **Thrust coefficient** | the same hovers — bidirectional DShot is on, so per-motor RPM is logged |
| **Thrust-to-weight** | the full-throttle climb, via RPM |
| **Hover attitude bias** | free, from the same data — it adds to the camera tilt |
| **Battery sag** | the whole recording |
| **Real telemetry rate** | the row timestamps |

---

## Things that have bitten us

**Weigh the drone first.** Race-ready, with the pack you fly. It takes 30 seconds
and everything above scales off it. `1.745 kg` is from the spec, not our scale.

**Label the airframes.** There are four and they are not interchangeable. We have
configuration from two different flight controllers and only noticed because
their chip IDs differ. A strip of tape with a number on each would save an
argument later.

**Never run the Betaflight CLI before flying.** `bf_cli.py` and
`bf_beginner.py --apply` wedge this flight controller — measured twice out of
two, and it takes a battery pull to recover. Fine at the bench, disastrous on the
line.

**Props off for anything on the ground.**

**If the aircraft never armed**, the analysis says so in as many words rather
than quietly producing a hover figure from a drone sitting on the floor. That has
happened once already and it produced a hover throttle of 0.098, which would
imply a drone that can lift ten times its own weight.
