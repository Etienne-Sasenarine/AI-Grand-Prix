# 04 — "Betaflight issues": troubleshooting report for our airframe

Written 19 Sep 2026. Aircraft: 1745 g, 8-inch, 6S, Betaflight 4.4.3 vendor build `BF_BLOCK2 (SH74)`,
STM32H743, DShot300 bidirectional, CRSF receiver, Futaba T16IZ. Dump read: `logs/orin/bf_dump_before_beginner_20260918.txt`
(mcu_id `003a0031…`). Logs analysed: `hover_20260918_003956.csv`, `hover_20260918_004725.csv`.

Tags: **[CITED: url]** · **[SOURCE 4.4.3: file/function]** = stock Betaflight source at tag 4.4.3, read directly ·
**[FROM-DUMP: line]** · **[FROM-LOG]** · **[MY ESTIMATE]** = reasoning, not proof. Every [SOURCE] claim assumes
Neros did not change that file in their build.

**The headline, because it changes what you do next:** the logs show the flight controller (FC) is healthy.
Motor order, prop direction and gyro orientation are **right** — the aircraft followed the sticks at the commanded
rotation rate to within ~7 %. Every tip-over in the logs was *commanded by the pilot's right stick* in ACRO mode at
a throttle too low to leave the ground. The self-disarm was Betaflight's crash safety reacting to the ground
impact, not the cause of it. This is a piloting-mode problem (no self-levelling), not a broken drone.

---

## 1. Top 5 things most likely stopping you hovering, in order

**1. You are in ACRO with the throttle parked at the lift-off threshold.** In ACRO ("rate" mode) the stick sets
*how fast it rotates*, not *how far it leans*; centre the stick and it **stays tilted**. In both logs the throttle
never went above **1294 µs (≈26 % stick)** and sat at 1190–1220 µs for seconds — exactly where the skids are
half-unloaded and the aircraft pivots on them [FROM-LOG]. *2-minute check:* read section 3 — each tilt starts with a
stick movement. *Fix:* item 2 and section 5 (ANGLE, 25–30° limit), then a
**decisive** lift: raise throttle smoothly but without pausing to roughly 1300–1350 µs until it is ~1 m up, then
ease back to hover (≈1230–1270 µs is my estimate from the vertical accelerometer reaching 1.17 g at 1196–1220 µs;
[SAURON]'s eyeball "lift-off 0.18, hover 0.25" agrees) [MY ESTIMATE]. Hands **off the right stick** until it is
airborne. Ground effect (extra lift and turbulence within about one prop-diameter of the floor) is why hovering
at 10 cm is harder than at 1 m.

**2. ANGLE mode is on no switch.** The only mode lines are ARM, USER1, USER2, USER4 and MSP OVERRIDE
[FROM-DUMP: 449–453]. ANGLE is permanent box id 1 [SOURCE 4.4.3: msp/msp_box.c] and is absent. The accelerometer
is present and calibrated (`acc_calibration = -72,-37,2,1`, last field 1 = calibrated) [FROM-DUMP: 651], so ANGLE
is available. *2-minute check, props off:* after assigning ANGLE (section 5), arm, raise throttle a little, tilt the
frame by hand: the **low-side motors speed up and stay sped up until you level it**. In ACRO they only react while
you are moving it.

**3. The switch labelled ACRO/ANGLE most probably engages MSP OVERRIDE, which with no Jetson sending is a
receiver failsafe.** Full mechanism in section 2.3 — it also explains the RX_FAILSAFE flag. *2-minute check, props
off, disarmed:* run `setup_angle_mode.py --watch` (or Configurator → Receiver tab) and flip that switch through all
positions; see which channel moves and whether it passes 1700 on **channel 9 (AUX5)**. Until assigned otherwise,
**tape that switch** in the position that reads ~1094.

**4. Arming refusals that look like faults but are not.** In `004725` the arm switch was high for **7.9 s
(t = 72.3–80.2 s) with throttle at 1113–1294 µs** and again at 82.3–83.3 s (1198 µs); it would not arm because
throttle must be below `min_check = 1050` [FROM-LOG][FROM-DUMP: 668]. After the self-disarm the RUNAWAY flag stays
latched until the arm switch is put **off** [SOURCE 4.4.3: fc/core.c `updateArmingStatus`, lines ~396–401].
*2-minute check:* read the arming-disable flags (section 4, step 4) before every attempt. Throttle fully down,
arm switch off-then-on.

**5. Physical state after the impacts, and the level reference.** `003956` ends with a **3.3 g** ground strike and a
471 °/s yaw kick [FROM-LOG]. *2-minute check, battery out:* spin each prop by hand (no rub, no bent blade, nuts
tight, motor bells not loose); confirm the prop layout still matches section 4 step 2. At rest the FC reads roll
**+1.5…+2.0°, pitch −2.0…−2.7°** [FROM-LOG]. If that floor was truly level, ANGLE mode will hold that small lean
and drift slowly (2.5° of lean ≈ 0.4 m/s² sideways [MY ESTIMATE]) — recalibrate the accelerometer on a level
surface (Configurator → Setup → *Calibrate Accelerometer*, or MSP command 205). If the floor or the skids were not
level, leave it alone.

Not on the list: motor order, prop direction, board alignment, PID tune. The logs rule them out (section 3).

---

## 2. Plain-language reading of the dump

### 2.1 What is switched on

| Item | Value | Meaning | Tag |
|---|---|---|---|
| Features | RX_SERIAL, SERVO_TILT, GPS, TELEMETRY, OSD, AIRMODE, ANTI_GRAVITY | AIRMODE = PID loop keeps full authority at low throttle (good in the air; on the ground it lets a held stick tip the aircraft). MOTOR_STOP is off → props spin as soon as you arm | [FROM-DUMP: 277–283] |
| Serial ports | `20`=USB → MSP · UART1 → GPS · **UART2 and UART4 → MSP** · UART6 → VTX SmartAudio (2048) · UART7 → receiver (64) | Two hardware MSP UARTs; one is the Jetson's. USB is a third, independent MSP port — this is the safe way in | [FROM-DUMP: 286–291] |
| Receiver | CRSF, map `AETR1234`, `rxrange` all 1000–2000 | Standard. Our stick bottom reads 997 µs, below `min_check` 1050 — fine | [FROM-DUMP: 344, 503–506, 686] |
| Motors | DSHOT300, `dshot_bidir ON`, 14 poles, `dshot_idle_value 550` (5.5 %), `dyn_idle_min_rpm 29` on profile 2 | Dynamic idle holds ≥2900 rpm using RPM telemetry. `min_throttle 1070` is ignored under DShot | [FROM-DUMP: 724–737, profile 2] |
| Orientation | `gyro_1_sensor_align CW270`, `align_board_* 0`, `yaw_motors_reversed OFF`, mixer QUADX | Confirmed correct by the logs, see section 3 | [FROM-DUMP: 749–751, 784, 1079] |
| `small_angle` | 180 | Arms at any tilt, even upside-down. No ANGLE arming flag will ever appear | [FROM-DUMP: 804] |
| Runaway takeoff | ON, deactivate at 20 % throttle for 500 ms | See 2.4 | [FROM-DUMP: 852–854] |
| `crash_recovery` | OFF (all profiles) | So the CRASH flag cannot be the self-disarm | [FROM-DUMP: 1409] |
| PID profiles | 0 `10"`, 1 `5"`, **2 `8"` (active)**, 3 `8" Fiber`, 4/5 unnamed | Never fly another profile: those are other airframes' gains | [FROM-DUMP: 1174–1802] |
| ANGLE parameters, profile 2 | `level_limit 55`, `angle_level_strength 50`, `thrust_linear 40`, `tpa_rate 80 @1300 (D only)` | 55° at full stick is a racing number | [FROM-DUMP: profile 2] |
| Rate profiles | 4 identical: rc_rate 55/55/57, srate 75/75/70, expo 0, `thr_mid 54`, `thr_expo 68`, no throttle limit; rateprofile 0 active | Throttle expo flattens the curve around **54 %** stick — but we hover near 25 %, where the curve is *steeper* | [FROM-DUMP: 1804–1905] |
| `msp_override_channels_mask` | 11 in this dump (team has since set 15) | Bits are in **receiver order** (AETR): 1=roll, 2=pitch, **4=throttle**, 8=yaw. So 11 = roll+pitch+yaw to the Jetson, *throttle stays with the pilot*. 15 = all four | [SOURCE 4.4.3: rx/rx.c `readRxChannelsApplyRanges` passes `rcmap[channel]`; rx/msp_override.c] |

Consequence of that last row worth checking once: `MSP_SET_RAW_RC` values are stored by raw index and read back
through the channel map, so on this AETR aircraft the frame order is **roll, pitch, throttle, yaw**, while `MSP_RC`
reports roll, pitch, yaw, throttle [SOURCE 4.4.3: rx/msp.c `rxMspFrameReceive`, rx/rx.c]. Confirm the organizers'
`RCTransmitter` sends in that order before trusting it with throttle.

### 2.2 Channel by channel — what each transmitter control most probably does

Values are what the logs show. "AUXn" is Betaflight's name; radio channel = n + 4.

| Ch | Seen in logs | FC function | Confidence |
|---|---|---|---|
| 1–4 | sticks | roll, pitch, yaw, throttle | certain |
| 5 (AUX1) | 1000 / 2000 | **ARM** at ≥1600 (`aux 0 0 0 1600 2100`) | certain [FROM-DUMP: 449] |
| 6 (AUX2) | 1000, never moved | No Betaflight mode. Possibly the vendor's 3-position `neros_sensor_channel = 5` (camera/sensor select) if that setting counts from 0 | low |
| 7 (AUX3) | 1094, never moved | **Camera-tilt servo**: `servo 0 … 6` forwards channel index 6 (= ch 7) to SERVO1 on pin E05. Also **USER2 → PINIO2 (pin D10)** at ≥1600 | servo: high · what D10 powers: unknown [FROM-DUMP: 33, 300, 451] |
| 8 (AUX4) | 1992, never moved | Nothing in Betaflight | — |
| **9 (AUX5)** | **1094 ↔ 1520** (moved in `004725` at 31.2, 33.5, 35.4 s, armed, no effect) | **MSP OVERRIDE at ≥1700** (`aux 4 50 4 1700 2100`). 1094/1520 look like the low and middle of a 3-position switch; the third position (~1950) has not been logged and would be override | override on ch 9: certain. That this is the "ACRO/ANGLE" switch: **probable, unproven** |
| 10 (AUX6) | 1094, never moved | **USER4 → PINIO4 (pin D11)** at ≥1600 | unknown load |
| 11 (AUX7) | **1031 disarmed, 1079 armed**, follows the arm switch within 0.1 s | **USER1 → PINIO1 (pin C02)**, range 1050–2100, so it is ON whenever armed. The vendor setting `vtx_off_threshold = 1050` is the same number: I read this as "video transmitter enabled when armed", with the VTX POWER switch moving the same channel higher | medium [FROM-LOG][FROM-DUMP: 450, 1138–1139] |
| 12 | 1118 constant | Possibly vendor `rx_band_switch_channel` / `neros_drone_type_channel` (= 11 if counted from 0; threshold 1500). TAG TEAM may live here | low |
| 13–14 | 1500 | not transmitted | — |
| 15, 16 | 2012; 1716–2012 varying | Not switches: ExpressLRS puts link quality on ch 15 and signal strength on ch 16 [CITED: https://www.expresslrs.org/software/switch-config/] | high |

**What the USER boxes physically switch.** `pinio_box = 40,41,244,43,255` with `resource PINIO 1..5 = C02, D10, A08,
D11, C03` and `pinio_config = 1` (push-pull output, not inverted) [FROM-DUMP: 128–132, 1066–1067][SOURCE 4.4.3:
drivers/pinio.h, io/piniobox.c]. So: **USER1 (id 40) drives pin C02, USER2 (41) drives D10, USER4 (43) drives D11**;
PINIO3 (A08) is bound to id **244, which is not a stock Betaflight box** (stock firmware would leave that pin
idle; Neros may have added a box); PINIO5 (C03) is unbound (255). The dump names MCU pins only — it cannot say
what is soldered to them. Typical loads are VTX power, lights, a fan or a payload relay. Only Neros can say;
props off, you can flip each and watch what changes on the aircraft.

### 2.3 Why "ANGLE" made the drone "just die" — and why RX_FAILSAFE appears

With override active, Betaflight takes the masked channels from the last `MSP_SET_RAW_RC` frame. If the Jetson has
**never sent one, those values are 0** [SOURCE 4.4.3: rx/msp.c — static `mspFrame[]`, zero-initialised]. Zero is
below `rx_min_usec = 885`, so the channel is "invalid": held for 300 ms, then Betaflight **declares the receiver
signal lost** [SOURCE 4.4.3: rx/rx.c `detectAndApplySignalLossBehaviour`, `MAX_INVALID_PULSE_TIME_MS 300`]. That:

1. sets the **RX_FAILSAFE ("RXLOSS") arming-disable flag** at once [SOURCE 4.4.3: flight/failsafe.c
   `failsafeOnValidDataFailed`] — observed problem (3);
2. puts the sticks on stage-1 failsafe values: roll/pitch/yaw centred, **throttle held** (`rxfail 3 h`, a vendor
   change from the default "auto = low") [FROM-DUMP: 603–606];
3. after `failsafe_delay = 50` (**5 s**) runs `failsafe_procedure = DROP` = motors off and disarm; or disarms
   *immediately* if throttle had been low for the previous 10 s (`failsafe_throttle_low_delay = 100`), which is the
   case sitting on the ground [SOURCE 4.4.3: flight/failsafe.c `failsafeUpdateState`][FROM-DUMP: 741–746].

On the ground that is "armed, flip switch, motors stop, will not re-arm" — "it just died". So the team's hypothesis
is **mechanically confirmed**; what is *not* proven is that the labelled switch is the one on channel 9. Alternatives
from the dump: USER1/2/4 only toggle output pins and cannot stop motors; no FAILSAFE, PARALYZE or PREARM box is
assigned; ch 7 only moves a servo. Nothing else on any channel can produce that symptom. Two corollaries:

- The same thing happens in the air: 5 s of frozen throttle and centred sticks (in ACRO centred ≠ level), then a
  drop. Once one valid frame has been sent, a stopped script holds the **last** values with no timeout at all
  [SOURCE 4.4.3: rx/msp_override.c — no timer anywhere]. The `[NO SOURCE]` row in `SPEC.md` §2 can now be cited.
- **Transmitter off = the same 5 s hold then drop.** Know this before the first hover.

### 2.4 Runaway takeoff prevention, in one paragraph

It disarms if, while armed with motors running, any axis's PID output exceeds 60 % **and** the gyro shows real
rotation (>15 °/s roll/pitch or >50 °/s yaw) for **75 ms** [SOURCE 4.4.3: fc/core.c, `RUNAWAY_TAKEOFF_*` defines and
the block after `pidController`]. It stops watching only after **0.5 s accumulated** of throttle ≥20 % with a stick
≥15 % deflected (or throttle ≥40 %) and all PID outputs under 10 % — i.e. after a clean take-off
[CITED: https://betaflight.com/docs/wiki/guides/current/Runaway-Takeoff-Prevention]. A beginner who creeps the
throttle, skids on the floor, or bumps down never satisfies that, so it stays live and any hard ground contact
trips it — the docs warn that "severe bouncing/bumping the ground on takeoff may cause a disarm". **Leave it ON.**
It is temporarily disabled while Configurator is connected, so bench behaviour differs from the field.

---

## 3. Log analysis [FROM-LOG]

Both files: ~23.6 Hz, 120 s, no gaps > 0.06 s, battery 24.3 → 22.9 V. Gyro divided by 16.4. "Commanded rate" is
the Betaflight rates formula `200·rc_rate·x / (1 − |x|·srate)` applied to the logged stick.

**Tracking test (armed, throttle > 1150 µs).** `004725`: roll correlation 0.94, slope **1.02**; pitch 0.94, slope
**0.94**. `003956`: pitch 0.93 / 0.93; roll 0.33 only because the largest command happened against the floor. A quad
with a wrong motor order or a reversed prop cannot do this — it diverges in a fraction of a second whatever the
stick says. **Motor order, prop direction, gyro alignment and PID sign are correct** on this airframe.

| File | t (s) | Event | Numbers |
|---|---|---|---|
| 003956 | 0 → 5.8 | already armed; switch disarm | throttle median 1098, max 1125 µs |
| 003956 | 27.6 → 72.6 | armed 45 s; switch disarm | throttle max 1189 µs |
| 003956 | 67.4–69.2 | **tip to −40° roll** | roll stick held left at 1271–1308 µs **with throttle at 997**; throttle then raised to 1078–1093 and the aircraft rolled at −53…−70 °/s — commanded was −63 °/s. It did what it was told |
| 003956 | 87.47 → **90.02** | armed, **self-disarm, arm switch still 2000** (lowered at 90.54) | detail below |
| 004725 | 22.5 → 65.4 | armed 43 s; **switch disarm at throttle 1190, pitch −17°** | pitch stick pulled to 1358 at 64.0–64.6 s → −28…−33 °/s (commanded ≈ −40) → −18°; stick centred and it **stayed at −18°** (ACRO), pilot disarmed |
| 004725 | 72.3–80.2, 82.3–83.3 | **arm switch high, FC refused** | throttle 1113–1294 µs > `min_check` 1050 → THROTTLE flag |
| 004725 | 88.9 → 103.8 | armed 15 s; switch disarm at throttle 1220 | roll peaked +15.4° at 102.4 s |
| 004725 | 116.7 → end | armed, idle | — |

Aux channels that moved: **ch 5** (arm), **ch 9** (`004725` only, 1520↔1094, three times, harmless), **ch 11**
(1031↔1079 with every arm/disarm). Nothing else moved. Two earlier 2–4 s logs (`002355`, `002428`) show every
channel at 1500 with throttle 885 — that is what "no receiver link" looks like over MSP, and is an RXLOSS state.

**The t ≈ 90 s self-disarm, blow by blow.** 88.24–88.66 throttle 997 → 1196 in 0.4 s while the roll stick drifts
left to 1390: aircraft rolls at −28…−32 °/s (commanded −29) to −10.6°; vertical accel reaches 1.17 g — it is light
on the skids. 89.26–89.86: pilot corrects with **right roll to 1896 and forward pitch to 1760 while *lowering*
throttle to 1148**: rates reach +198 °/s roll (commanded ≈ 207) and +91 °/s pitch (commanded ≈ 85); vertical accel
falls to 0.47–0.6 g (descending). 89.98: roll +25°, pitch +28°, accel **−2.8 g / −1.4 g / +3.3 g**, gyro −310 roll,
−471 yaw = ground strike. **90.02: disarmed**, one sample (≤ 85 ms) later.

**Diagnosis.** Not prop/motor order (tracking is accurate up to the strike). Not failsafe (channels update
smoothly). Not crash recovery (OFF). Not the pilot (switch high for another 0.5 s). By elimination, and with the
timing matching its 75 ms trigger, this is **runaway takeoff prevention firing on the ground strike** — protection
never having been deactivated because throttle barely reached 20 % — confidence high [MY ESTIMATE]. The *tip* that
preceded it was an ACRO over-correction: holding the stick over commands continuous rotation. In ANGLE with a 25°
limit the same stick movement would have produced a 20° lean and self-recovery.

As far as these files show the aircraft has **never been above ground effect**; `logs/orin/README.md` already
says not to use their hover numbers, and that stands.

---

## 4. Props-off verification checklist (≈15 minutes, do it once per airframe)

**Props physically removed. Battery in only when a step needs motors. Aircraft restrained or held by the frame.**
Use Configurator over the FC's USB port (section 5). While connected the FC shows the MSP arming flag and will not
arm from the radio; motor tests use the Motors tab.

1. **Orientation.** Setup tab: tilt nose down, roll right, yaw right — the 3-D model must copy you
   [CITED: https://oscarliang.com/quad-flip-over-issue-fix/]. (Logs already say yes.)
2. **Motor order and direction.** Motors tab → tick "I understand the risks" → raise **one** slider to ~1100.
   Betaflight QUADX numbering: **1 rear-right, 2 front-right, 3 rear-left, 4 front-left**. With
   `yaw_motors_reversed = OFF` ("props in") motors 1 and 4 turn clockwise, 2 and 3 counter-clockwise, seen from
   above — check with a strip of tape on the bell or a fingertip on the side of the bell
   [CITED: https://oscarliang.com/test-motor-spin-direction/]
   [CITED: https://oscarliang.com/reversed-motor-prop-rotation-quadcopter/]. Do **not** use the "Motor Direction"
   wizard to change anything — vendor setup, and the logs show it is right.
3. **Props match the motors** (battery out): each prop's raised leading edge must face its motor's direction of
   spin; front props throw air toward the camera side on a props-in build. After any crash, re-check this first —
   a prop refitted on the wrong corner is the classic flip-on-take-off.
4. **Arming flags.** Configurator Setup tab shows "Arming Disable Flags"; from the Jetson read `MSP_STATUS_EX`
   (150): byte 15 = N (extra mode-flag bytes), byte 16+N = flag count, **bytes 17+N … 20+N = 32-bit little-endian
   flag word** [SOURCE 4.4.3: msp/msp.c `MSP_STATUS_EX`]. Bits [SOURCE 4.4.3: fc/runtime_config.h]:
   0 NOGYRO · 1 FAILSAFE · **2 RXLOSS** · 3 BADRX · 4 BOXFAILSAFE · **5 RUNAWAY** · 6 CRASH · **7 THROTTLE** · 8 ANGLE ·
   9 BOOTGRACE · 10 NOPREARM · 11 LOAD · 12 CALIB · **13 CLI** · 14 CMS · 15 BST · **16 MSP** · 17 PARALYZE · 18 GPS ·
   19 RESCUE_SW · 20 RPMFILTER · 21 REBOOT_REQD · 22 DSHOT_BBANG · 23 NO_ACC_CAL · 24 MOTOR_PROTO · 25 ARMSWITCH.
   Meanings and beep codes: [CITED: https://betaflight.com/docs/wiki/guides/current/Arming-Sequence-And-Safety],
   [CITED: https://oscarliang.com/quad-arming-issue-fix/]. Expected on the bench with USB in: MSP only. BADRX =
   link came back with the arm switch already on → switch off. RPMFILTER = an ESC is not returning RPM.
5. **Switch map.** Receiver tab: flip every transmitter switch through every position, write down channel and
   values. Fill in the "unproven" cells of table 2.2. Modes tab: the yellow marker shows which ranges each hits.
6. **ANGLE response** (after section 5, USB unplugged, battery in, armed by radio, low throttle): tilt by hand →
   low-side motors rise and **hold** until levelled; release → equalise. Flip to ACRO position → they react only
   while moving. If ANGLE behaves like ACRO, the mode is not active.
7. **Failsafe on TX off** (armed, low throttle, props off): switch the transmitter off. Expected with this dump:
   motors **keep running up to 5 s, then stop**, RXLOSS shows. If you had been at idle for > 10 s they stop at once.
   Transmitter back on: arm switch **off** then on to re-arm. Decide as a team whether 5 s of held throttle is
   acceptable for beginner hovers (section 6, question 4).
8. **Override switch with no Jetson script** (armed, props off): flip ch 9 high; expect the behaviour in 2.3.

---

## 5. Beginner settings that are safe to change — without the UART CLI

### 5.1 What to change (and what not to)

| Change | Value | Why it is safe for the race tune |
|---|---|---|
| Assign **ANGLE** (box id 1) to a channel | team plan: AUX5, 900–1700 (everything below override). Alternative: a spare channel the pilot controls, e.g. ch 8 | Adds a mode range in an empty slot (5–19 are free). ACRO and the policy are untouched when the switch is in the ACRO/override position |
| `level_limit` on **profile 2** | 25–30 (from 55) | Exists only in ANGLE/HORIZON. In 4.4 the target lean is simply `level_limit × stick deflection`, linear, then `error × angle_level_strength/10` becomes the rate setpoint [SOURCE 4.4.3: flight/pid.c `pidLevel`, flight/pid_init.c]. Full stick = 25° |
| `angle_level_strength` | leave at 50 | default-ish; lower feels mushy on a heavy quad [MY ESTIMATE] |
| Throttle curve on **rate profile 1** (not 0) | `thr_mid ≈ 25–30`, `thr_expo` 60–70 | Moves the flat, gentle part of the curve to where we actually hover. Full stick is still full power. Rate profile 0 — which the policy's stick conversion assumes — is not touched |
| Yaw rate on rate profile 1 | `yaw_rc_rate` ~35, `yaw_srate` ~30 | In ANGLE, roll/pitch **rates are not used at all** (see `pidLevel`), so only yaw and throttle are worth changing |
| Accelerometer calibration | only if step 5 of section 1 says so | — |

Do **not** change: PIDs, filters, `thrust_linear`, `dshot_idle_value`, dynamic idle, `runaway_takeoff_prevention`,
`small_angle`, motor direction, anything under Ports, anything in the Firmware Flasher. Do not select PID profile
0/1/3. If a human rate profile is used, make the autonomy code read `MSP_STATUS_EX` byte 14 (active rate profile)
and refuse to fly unless it is 0.

### 5.2 Route A (preferred): Betaflight Configurator 10.9.0 over the FC's own USB

- **Version:** 10.9.0 is the release that pairs with firmware 4.4.x; newer web/app versions may refuse or
  mis-handle 4.4 [CITED: https://github.com/betaflight/betaflight-configurator/releases/tag/10.9.0]. Windows file:
  `betaflight-configurator_10.9.0_win64-portable.zip` (no install needed) or `…_win64-installer.exe`, from that page.
- **Driver:** an H743 board appears as a normal "COM" port (STM32 virtual COM port) — Windows 10/11 usually needs
  nothing. If no port appears: try a known **data** cable first, then the ImpulseRC Driver Fixer
  (https://github.com/ImpulseRC/ImpulseRC_Driver_Fixer). **Zadig is only for DFU/bootloader mode, which we never
  need (no flashing) — pointing Zadig at the COM device breaks it** [CITED: https://oscarliang.com/fc-driver-issues/].
- **First action, every airframe: back up.** CLI tab → type `dump all` → *Save to File*; name it with the
  mcu_id. Then `diff all` likewise for completeness. Restore = CLI tab → *Load from file* (or paste) → it ends with
  `save`. Restore only onto the **same** airframe. The CLI tab over USB is safe: Configurator sends `exit` for you
  and USB re-enumerates after the reboot.
- **Modes tab:** *Add Range* under ANGLE → pick the AUX channel (or AUTO and flip the switch) → drag the range →
  **Save**. **Receiver tab** to see live channels. **PID Tuning tab:** profile selector must read 2; "Angle Limit"
  is under the Angle/Horizon box; *Rateprofile* selector → 1 for the throttle/yaw edits → **Save**.
- Ignore any pop-up offering to update firmware or apply presets/defaults.

### 5.3 Route B: binary MSP from the Jetson (what `setup_angle_mode.py` does)

Verified against source at 4.4.3, API 1.45 [SOURCE 4.4.3: msp/msp.c, msp/msp_protocol.h]:

- `MSP_MODE_RANGES` 34 → 20 × (permanent id, aux index, start step, end step); step = (µs − 900)/25.
  `MSP_SET_MODE_RANGE` **35** ← slot, permanent id, aux index (0 = AUX1), start step, end step, [mode logic, linked id].
  Unknown id or slot ≥ 20 → error reply.
- `MSP_SELECT_SETTING` 210 ← one byte: `0–5` = PID profile (ignored while armed); `0x80 | n` = rate profile n.
- `MSP_PID_ADVANCED` 94 / `MSP_SET_PID_ADVANCED` 95, acts on the **current** profile. Layout: 3×U16 zero (0–5),
  U8, U8, `feedforward_transition` (8), 4×U8 (9–12), `rateAccelLimit` U16 (13–14), `yawRateAccelLimit` U16 (15–16),
  **`levelAngleLimit` U8 at byte offset 17**, then a dead byte (18), … The tool's empirical answer, 17, **matches the
  source**. `angle_level_strength` is not here; it is P of the LEVEL entry in `MSP_PID` 112 / `MSP_SET_PID` 202.
- `MSP_SET_RC_TUNING` 204 edits the current rate profile (needs ≥ 10 bytes; read 111 first, modify, write back).
- `MSP_EEPROM_WRITE` 250 — refused while armed. `MSP_ACC_CALIBRATION` 205. `MSP_BOXIDS` 119.
- **`MSP_BOXNAMES` (116) times out (problem 6)** almost certainly because the reply is ≥ 255 bytes, which MSP v1
  sends as a **"jumbo" frame**: size byte = 255 followed by a 16-bit real length [SOURCE 4.4.3: msp/msp_serial.c,
  `JUMBO_FRAME_SIZE_LIMIT`]. A parser that does not know this sees a bad checksum and waits forever. You do not
  need names: `MSP_BOXIDS` gives one permanent-id byte per box, in the same order as the mode-flag bits in
  `MSP_STATUS_EX`; ANGLE is id 1.

### 5.4 Why the UART CLI wedges the FC (problems 4 and 5)

- `#` on an MSP port, after 100 ms of silence, calls `cliEnter()`; from then on that UART is a text console and
  answers **no MSP** until the FC reboots. In 4.4 both `exit` and `save` reboot [SOURCE 4.4.3: msp/msp_serial.c
  `mspEvaluateNonMspData` / `mspProcessPendingRequest`; cli/cli.c `cliExit`, `cliSave`]. If a session dies before
  `exit`, the port just looks dead. `save` writes flash *before* rebooting, so "saved but then silent" is possible.
- **Leading hypothesis for "silent at every baud rate until a battery pull" [MY ESTIMATE]:** the same function
  treats the byte **`R`** (`reboot_character = 82` [FROM-DUMP: 800]) as "reboot into the STM32 ROM bootloader". The
  ROM bootloader is silent on the UART and survives everything except a power cycle — exactly the symptom. After
  `save`/`exit` the FC prints `Rebooting`; if anything on the Jetson echoes received text back (a leftover serial
  console/getty on `/dev/ttyTHS1`, or a terminal in echo mode), the FC receives an `R` just after it restarts.
  **2-minute test next time it wedges:** before pulling the battery, plug the FC's USB into a Windows PC. If
  Device Manager shows **"STM32 BOOTLOADER"** instead of a COM port, this is confirmed. Also check
  `systemctl is-active nvgetty serial-getty@ttyTHS1` on the Jetson.
- Related trap: `MSP_SET_MODE_RANGE` is command **35 = ASCII `#`**, and RC values such as 1106, 1362, 1618,
  1874 µs have low byte 0x52 = `R`. Harmless inside a well-formed frame; if the FC joins mid-frame (just rebooted,
  bytes dropped) and the stream then pauses > 100 ms, a stray byte can be read as a command [MY ESTIMATE — low
  probability, real mechanism]. Rule: after any FC reboot wait ~3 s and get a reply to a harmless request
  (`MSP_API_VERSION`, 1) **before** sending anything else.
- **`diff all` is empty (problem 5) — normal.** `# config: YES` means vendor defaults are compiled in; `diff`
  prints differences from *those*. Use `dump all`. (`SPEC.md` already records this.) Housekeeping: the earlier
  `logs/orin/diff_all.txt` contains that airframe's receiver bind phrase **unredacted** — treat the file like the
  Wi-Fi password and keep it out of the repo.
- A `save` that "did not take": the text never reached a live CLI, or `save` was refused over an invalid value.
  Configurator shows the error text; the blind UART does not.

---

## 6. Questions for the Neros / organizer desk

1. Which transmitter switch drives **channel 9**, and is the ACRO/ANGLE label a leftover from a stock config in
   which AUX5 was ANGLE? What are its three positions meant to do on this event build?
2. **May we add an ANGLE mode range** (and lower `level_limit`)? The spec allows "rates, PIDs, filters, telemetry";
   a mode range is none of those. Another team runs `aux 5 1 4 900 2100`. Is ANGLE allowed during a scored run?
3. What do **USER1 / USER2 / USER4** (pins C02, D10, D11) and **PINIO3 / box id 244** (pin A08) switch? Is it
   expected that channel 11 moves 1031 → 1079 on arming (VTX enable)? What do VTX POWER and TAG TEAM send, on
   which channels — and can any of them affect flight?
4. Failsafe is **5 s of held throttle then DROP** (`failsafe_delay 50`, `rxfail 3 h`). Is that deliberate for the
   race? May we shorten it for beginner hover practice, and must we restore it?
5. Is `runaway_takeoff_prevention` expected to stay ON for autonomous take-offs? (A slow autonomous lift-off will
   meet the same trap we did.)
6. Does the vendor build change `msp_serial.c` / the CLI? Is there a supported way to read settings from the
   Jetson UART, and is `reboot_character = 82` live on the MSP UARTs?
7. Is a laptop with **Configurator 10.9.0 on the FC's USB port** permitted at the bench? Is there an official
   per-airframe backup `dump all` we can restore from if we break something?
8. Expected **hover throttle** for this airframe at 1745 g (stick µs), so a beginner knows the target?

---

### Sources

Betaflight 4.4.3 source, read directly under https://raw.githubusercontent.com/betaflight/betaflight/4.4.3/src/main/ :
`fc/core.c`, `fc/runtime_config.h`, `flight/failsafe.c`, `flight/pid.c`, `flight/pid_init.c`, `rx/rx.c`, `rx/msp.c`,
`rx/msp_override.c`, `msp/msp.c`, `msp/msp_box.c`, `msp/msp_serial.c`, `msp/msp_protocol.h`, `io/piniobox.c`,
`drivers/pinio.h`, `cli/cli.c`. Web pages are cited inline where used.

Not found: a citable source for the "decisive lift-off" technique (common practice, tagged [MY ESTIMATE]);
anything public on the Neros build, its `neros_*` / `vtx_ctrl_channel` settings or box 244. The Failsafe and
PinioBox doc pages exist, but the claims here rest on the source code, not on them.
