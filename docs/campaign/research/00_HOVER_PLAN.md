# Getting to a first autonomous hover — the combined plan

19 Sep 2026. Synthesis of five research reports in this folder. Read this one
first; the others hold the evidence, sources and commands.

| Report | What it settles |
|---|---|
| `01_msp_override.md` | How the Jetson takeover works in Betaflight 4.4.3, read from the source |
| `02_hover_strategy.md` | How to hover with no usable barometer; the seven-rung ladder; gains |
| `03_connecting_to_the_drone.md` | Getting connected in minutes; new-drone checklist and draft script |
| `04_betaflight_issues.md` | What our logs and settings dump actually show; props-off checklist |
| `05_hover_code_review.md` | Defects in the hover code, demonstrated in simulation; bench checks |
| `06_georgia_tech_reference.md` | **The setup and method of the team that is through five gates on this airframe — now our reference design** |

**Decision, 19 Sep: Plan A (the trained policy) is deferred — there is no
realistic chance of running it. Plan B, the slow self-levelled fallback, is the
target.** Everything below serves Plan B: hover first, then one gate, then a lap
at walking pace. Scoring is most gates passed, so a slow run that finishes beats
anything that stops at gate four. Where Georgia Tech have a working answer on
this airframe (report 06), we use it.

**Nothing here has been tried on a drone.** Source-verified means read from the
stock Betaflight 4.4.3 code; ours is a vendor build, so the bench checks below
still have to confirm it behaves the same.

---

## The five things that are actually stopping us

1. **There is no height sensor.** The hover script flies on the barometer, and the
   barometer reads metres low as soon as the props load up — and lower the more
   throttle is applied. More throttle looks like falling, so the controller adds
   throttle. Simulated with the real script and a barometer model fitted to our
   own logs: throttle goes to its cap and the drone climbs into the net within a
   few seconds, with none of the script's guards firing in time. (The two reports
   fitted the size of the error differently, about 20 and about 40 metres per unit
   of throttle. The direction and the outcome are the same.)
2. **Flipping to the Jetson with nothing valid being sent kills the drone.** The
   flight controller's buffer for Jetson stick values starts at zero, and zero
   counts as a broken receiver: 0.3 s later it raises `RX_FAILSAFE`, 5 s later it
   drops and disarms. This is what "switch to ANGLE and it dies" was, and why
   `RX_FAILSAFE` kept appearing. **Rule: the Jetson must already be streaming
   valid values before anyone touches that switch.**
3. **If the Jetson stops sending, the flight controller holds the last sticks
   forever.** No timeout exists in 4.4.3. So a script that "stops sending" on a
   fault leaves the throttle frozen wherever it was. Every fault path must keep
   streaming a gentle descent instead.
4. **Hand-flying has been in ACRO with too little throttle.** The logs show the
   aircraft is healthy — it rotates at the rate commanded, which rules out wrong
   motor order or prop direction — but throttle never passed about 26 % of stick,
   so it has never left ground effect, and every tip-over began with a right-stick
   input that ACRO then held. The self-disarm in the log was Betaflight's
   runaway-takeoff protection firing on the ground strike: a result of the
   tip-over, not its cause.
5. **Sign and setup errors waiting behind those.** Pitch is inverted in the
   position hold (`xy_hold.py`) and in our own `flight/control_adapter.py`; the
   position hold has no way to line up the camera's "north" with the flight
   controller's heading; and the `--beginner` ANGLE setup switches ANGLE off
   exactly when the hover script needs it on.

---

## The ladder — one rung at a time, do not skip

Each rung has a pass check. If it fails, stop and fix that rung.

**Rung 0 — get connected (report 03).** From Windows PowerShell, not WSL: give
the Jetson's USB adapter a fixed address (the teammate's `fix-usb-network.ps1`),
clear the old SSH key for that address, log in. On every new board, once: run
`setup_jetson_uart.sh`, stop the recorder service if present, confirm the flight
controller answers. *Pass:* `msp_bench.py info` prints the firmware version.
A board that shows up as `APX` is in recovery mode: check header W7 pins 3–4,
one full battery pull, otherwise hand it back.

**Rung 1 — know the aircraft, props off.** Read the settings over the flight
controller's own USB port with Betaflight Configurator 10.9 (never the text
command line over the Jetson's serial port: it takes the port until reboot).
Save a full backup. Note the override mask, the mode table and the active PID
profile. *Pass:* a saved `dump all` for this drone, mask known.

**Rung 2 — ANGLE for the pilot, props off.** Assign ANGLE so it is on for a
human *and* stays on when override is engaged (see "One decision needed").
*Pass:* tilt the drone by hand with a little throttle; the low motors push back.

**Rung 3 — prove the takeover, props off (reports 01 and 05).** Stream valid
sticks from the Jetson first, then flip override: `MSP_RC` should follow the
Jetson, no `RX_FAILSAFE`. Then kill the script, and separately pull the serial
cable, and switch the transmitter off: write down exactly what the motors do
and for how long. *Pass:* behaviour matches the source reading, or we now know
how this vendor build differs. This rung replaces a guess with a measurement.

**Rung 4 — pilot hovers in ANGLE, recorder running.** A decisive climb out of
ground effect to 0.7 m or higher, hold, land. Three times. *Pass:* a hover
throttle read from above 0.5 m height, and a pilot who can hold a hover for ten
seconds. Fix the recorder's disk-sync and RPM decoding first or this data is
partly lost.

**Rung 5 — height from the camera, drone held in the hand.** Print one ArUco
marker (a square barcode OpenCV can read without any trained detector), put it
on a stand 4–6 m ahead, compute height and sideways position from it, correcting
for pitch. Walk the powered, props-off drone up, down and sideways. *Pass:*
height tracks a tape measure within about 10 cm, at 20 Hz or better. First check
the Jetson's OpenCV has the ArUco module (one line).

**Rung 6 — throttle-only hover, handed over in the air.** Override mask 4, so
the pilot keeps roll, pitch and yaw in ANGLE throughout. The pilot hovers at
0.7–0.8 m facing the marker, then flips override; the Jetson starts from the
pilot's average throttle over the last second or two and may move it only ±0.03.
Gentle gains (report 02). Marker lost: hold that throttle, never cut. Any fault:
keep streaming a slow descent. Pilot keeps the throttle stick at hover so taking
back is smooth. *Pass:* ten seconds within ±0.2 m.

**Rung 7 — add sideways hold, then floor takeoff.** Only after rung 6 is boring.

**Rung 8 — Plan B proper.** The Jetson takes all four sticks (override mask 15),
still in ANGLE: it commands lean angles, yaw rate and throttle, never rotation
rates. Height and position come from gate fixes (`flight/pnp.py` +
`flight/pose_filter.py`, already built and tested in simulation); the gate
counter is `flight/gate_tracker.py`; the guidance in `flight/controller.py`
already computes the lean it wants and only needs to hand that to the sticks as
an angle instead of converting it to a rate. A scored run allows no pilot input,
so rungs 6–7 (pilot on the sticks) are *test* configurations, not race ones.
Still missing entirely: **a gate detector** — nothing on the Jetson finds gate
corners in a real image yet.

---

## Code that has to change before rung 6

- `hover_10s.py`: height from the marker, not the barometer; fault path streams a
  descent; start from handed-over throttle; smaller gains; request timing.
- `xy_hold.py`: flip the pitch sign; its test model has the same wrong sign.
- `flight/control_adapter.py`: `sign_pitch` to −1, and the bench script's motor
  expectations.
- `tools/flight_recorder.py`: sync to disk about once a second; fix RPM decoding
  (13 bytes per motor).
- `tools/setup_angle_mode.py`: see below.

## One decision needed — now settled by report 06

Georgia Tech fly with ANGLE **always on** and override mask **4**. Do the same.
The original wording is kept below for the record.

The hover needs ANGLE **on** while the Jetson has control; the trained policy
will need it **off**. Proposal: make `--beginner` assign ANGLE permanently on
(human flying and hover tests), and keep today's behaviour as a separate
`--race` option for when a policy exists. Switching between them already needs
a settings change, because the override mask differs too (4 versus 15).

## Ask the organizers today

1. May we wire a small downward distance sensor to the Jetson? It would be the
   best height source by a wide margin.
2. May we assign flight modes and change the override mask?
3. What are the switches on the supplied transmitter meant to do, channel by
   channel — especially the one labelled ACRO / ANGLE?
4. Is the vendor firmware's override behaviour stock 4.4.3, or does it have the
   newer 300 ms freshness check?
5. Spare boards for Jetsons that come up in recovery mode; is an Ethernet cable
   to the Jetson permitted at the bench?

The longer lists are in reports 02, 03 and 04.
