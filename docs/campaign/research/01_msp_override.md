# How Betaflight 4.4.3 MSP override actually works (read from the firmware source)

Written 19 Sep 2026. Every claim is tagged **[SOURCE-VERIFIED]** (I downloaded the file at git tag
`4.4.3` and read the function named) or **[RECALLED/UNVERIFIED]** (memory or inference, not checked).
All source links are under `https://github.com/betaflight/betaflight/blob/4.4.3/src/main/` — shortened
below to `BF/`. Caveat that applies to everything: your firmware is a vendor build ("BF_BLOCK2", commit
`8034950b5`, built Aug 2026). I read *stock* 4.4.3. The vendor could have patched anything. The bench
procedure in section 8 is how you find out.

Terms: **MSP** = the serial request/reply protocol the Jetson speaks to the flight controller (FC).
**MSP override** = a Betaflight mode that, while a switch is on, swaps chosen stick channels from the
pilot's radio for values the Jetson sent with the `MSP_SET_RAW_RC` message. **Mask** =
`msp_override_channels_mask`, the number that chooses which channels get swapped. **Failsafe** = what
the FC does when it decides it has lost its receiver: stage 1 (substitute safe-ish values), then
stage 2 (here `DROP` = disarm and fall). **rcData** = the FC's final, validated stick values in
microseconds (1000-2000), which is what `MSP_RC` reports.

## What this means for you

- **Your docs are right about the mask.** Bit n = receiver channel n+1 *in radio order* (AETR).
  4 = throttle only. 8 = yaw only. 11 = roll+pitch+yaw, *not* throttle. 15 = all four sticks.
- **There is NO timeout on MSP stick data in 4.4.3.** If the Jetson script dies with override on, the
  FC flies the last values it was sent, forever, until the pilot flips override off. This is now
  source-verified, not lore. (Upstream only added a 300 ms freshness check in release 2026.6, PR #15217.)
- **Symptom (a) "flip the switch and it dies" is explained.** Before any `MSP_SET_RAW_RC` has arrived,
  the FC's MSP stick buffer is all zeros. Zero is outside the valid pulse range (885-2115), so the FC
  treats the overridden channel as a *broken receiver channel*: holds the old value 300 ms, then
  enters failsafe stage 1, then after `failsafe_delay` (50 = 5 s) stage 2 `DROP` = disarm.
- **It is worse than that: it latches.** Once that failsafe starts, *every AUX channel freezes at its
  last value* — including the override switch itself and the ARM switch. Flipping override back off
  is not seen. The pilot has about 300 ms to undo the flip; after that the only exits are (1) valid MSP
  frames start arriving, (2) stage-2 disarm at 5 s, (3) power cycle. This also explains symptom (b),
  a stuck `RX_FAILSAFE` flag with nothing connected.
- **DO start the Jetson's RC stream first, confirm it, then flip override on. Never the reverse.**
  Send all four stick channels with sane values (1500/1500/1000-thr/1500) even if the mask is 4 —
  it costs nothing and removes the zero trap if someone changes the mask.
- **DO keep every MSP value inside 885-2115** (better: 1000-2000). One out-of-range value on a masked
  channel for more than 300 ms = failsafe + frozen switches.
- **DON'T expect the FC to save you when the script dies.** The organizer library's docstring ("it
  triggers failsafe when that stream stops") is true for `feature RX_MSP`, **false for override**. Make
  the Jetson side fail safe: a watchdog process that keeps streaming neutral sticks / low throttle, and
  a pilot briefed to flip override off.
- **DON'T lose the pilot's radio link.** Override does not replace the receiver. If CRSF drops, all
  channels go invalid and failsafe runs *even with perfect MSP data*. MSP values are only sampled
  when a CRSF packet arrives.
- **DON'T put the override switch's own channel (bit 8 = 256) or ARM (bit 4 = 16) in the mask.**
- **DON'T send `#` or a bare `R` down the MSP UART while disarmed.** `#` drops the FC into CLI mode,
  which stops *all* MSP on *all* ports until `exit` (which reboots). `R` (`reboot_character = 82`)
  reboots into the ROM bootloader, which looks exactly like "wedged until battery pull". Use the
  Configurator CLI tab over USB to read or set the mask instead.
- **`rxfail 3 h` in your dump is a vendor change you should know about**: on failsafe stage 1 the
  *throttle holds its last value* (stock default would cut it to 885). So for 5 s after a failure
  the drone keeps its throttle with roll/pitch/yaw centred, then drops.

## Findings

### 1. Mask bit semantics — bit n is the RAW receiver channel, not the internal axis
**[SOURCE-VERIFIED: `BF/rx/rx.c` `readRxChannelsApplyRanges()`; `BF/rx/msp_override.c` `rxMspOverrideReadRawRc()`]**

```c
const uint8_t rawChannel = channel < RX_MAPPABLE_CHANNEL_COUNT ? rxConfig()->rcmap[channel] : channel;
if (rxConfig()->msp_override_channels_mask) {
    sample = rxMspOverrideReadRawRc(&rxRuntimeState, rxConfig(), rawChannel);
...
bool override = (1 << chan) & rxConfig->msp_override_channels_mask;   // chan == rawChannel
if (IS_RC_MODE_ACTIVE(BOXMSPOVERRIDE) && override) return overrideSample; else return rxSample;
```

`channel` is Betaflight's internal index (ROLL=0, PITCH=1, YAW=2, THROTTLE=3). `rcmap[]` translates
it to the position in the radio's frame. With `map AETR1234`, `parseRcChannels()` (`BF/rx/rx.c`)
gives rcmap[ROLL]=0, rcmap[PITCH]=1, rcmap[THROTTLE]=2, rcmap[YAW]=3. The mask is tested against
that *raw* position, so under AETR:

| bit | value | raw channel | stick |
|---|---|---|---|
| 0 | 1 | ch1 A | roll |
| 1 | 2 | ch2 E | pitch |
| 2 | 4 | ch3 T | **throttle** |
| 3 | 8 | ch4 R | **yaw** |
| 4 | 16 | ch5 | AUX1 = ARM (keep out) |
| 8 | 256 | ch9 | AUX5 = the override switch (keep out) |

So **4 = throttle only, 8 = yaw only, 11 = roll+pitch+yaw (no throttle), 15 = all four.** The team's
docs are correct. If anyone ever changes `map`, the meaning of the mask changes with it.
Also: the "MSP OVERRIDE" mode only exists in the mode list when the mask is non-zero
(`BF/msp/msp_box.c`, `if (rxConfig()->msp_override_channels_mask) BME(BOXMSPOVERRIDE)`), and its
permanent box id is 50 (same file), so `aux 4 50 4 1700 2100` really is MSP OVERRIDE on AUX5.

### 2. No frames ever / frames stop — zeros, then hold-forever; no timeout exists
**[SOURCE-VERIFIED: `BF/rx/msp.c` whole file; `BF/rx/rx.c` `detectAndApplySignalLossBehaviour()`, `getRxfailValue()`, `isPulseValid()`; `BF/flight/failsafe.c` `failsafeOnValidDataFailed()`, `failsafeCheckDataFailurePeriod()`]**

The entire MSP stick store is `static uint16_t mspFrame[18];` — zero at boot. `rxMspFrameReceive()`
copies the channels you sent and **zeroes every channel you did not send**. `rxMspReadRawRC()` just
returns `mspFrame[chan]`. There is no timestamp, no age check, no `MSP_OVERRIDE_*` constant. The only
freshness flag (`rxMspFrameDone` / `rxMspFrameStatus`) is used solely when MSP is the *primary*
receiver (`feature RX_MSP`), which you have off. `needRxSignalMaxDelayUs` (100 ms) watches the CRSF
receiver only.

**Frames stop mid-flight:** `mspFrame` keeps the last values; they are valid numbers; nothing ever
invalidates them. **The FC holds the last MSP sticks indefinitely. No failsafe.** Answer to (c): yes.
Upstream confirms by fixing it later: PR #15217 "RX MSP Override: require fresh MSP RC data before
substituting channels" (merged for 2026.6, adds a 300 ms freshness window) —
https://github.com/betaflight/betaflight/pull/15217. Whether your Aug-2026 vendor build back-ported
it is unknown; test it (8, step 6).

**Override ON, no frame ever (or a channel you did not send):** the overridden channel reads 0.
`isPulseValid()` requires `rx_min_usec..rx_max_usec` (885..2115), so it is invalid. Then, per
`detectAndApplySignalLossBehaviour()`:
1. 0-300 ms (`MAX_INVALID_PULSE_TIME_MS`): that channel holds its previous rcData. Nothing visible.
2. After 300 ms: `rxFlightChannelsValid = false`. The bad channel takes its `rxfail` value (your
   dump: roll/pitch/yaw `a` = 1500; throttle `h` = hold last). `failsafeOnValidDataFailed()` sets
   the **`RX_FAILSAFE`** arming-disable flag.
3. Because `thisChannelValid = rxFlightChannelsValid && isPulseValid(sample)` and the flag was just
   cleared, **every channel processed after the bad one is also treated as invalid** — all AUX
   channels, which are `rxfail h` (hold). ARM and the override switch freeze where they were. With
   mask 4 (throttle = internal channel 3) roll/pitch/yaw stay live from the radio because they are
   processed first; with 11 or 15 roll goes bad first and everything after it freezes.
4. No valid data for `failsafe_delay` x 0.1 s = **5.0 s** -> stage 2 -> `DROP` -> disarm.
5. **Latch:** the frozen override switch keeps override on, which keeps the zero flowing, which
   keeps the channels invalid. Step 3 is my reading of the loop order, not something the code
   comments state — **confirm on the bench** (8, step 3).

Related upstream reports with matching symptoms: issue #13416 "MSP_OVERRIDE commands not being
executed — motors stop when enabling override after arming"; issue #13374 (override still fails
safe when the RC link is lost — confirms the radio must stay connected).

### 3. Send rate, channel order in, channel order out
**[SOURCE-VERIFIED: `BF/msp/msp.c` cases `MSP_SET_RAW_RC`, `MSP_RC`; `BF/rx/rx.c` `rxFrameCheck()`]**

- **Rate:** the FC requires none (finding 2). Practical: 50-100 Hz. `serial_update_rate_hz = 100`, so
  the FC reads the UART 100 times a second; faster gains nothing. A frame of 8 channels is 22 bytes
  plus a 6-byte ack — trivial at 115200 baud, but **read and discard the acks** or buffers fill.
- **When the values take effect:** rcData is recomputed only when a *CRSF* packet arrives
  (`rxDataProcessingRequired` is set by the receiver's frame status, not by MSP). MSP values wait
  for the next radio packet: negligible at 150-500 Hz ELRS, up to 20 ms at 50 Hz.
- **`MSP_SET_RAW_RC` order = raw radio order = AETR**: `[roll, pitch, throttle, yaw, aux1...]`.
  The handler stores `frame[i]` at index i and the reader indexes by `rawChannel`. The organizer
  library's `IDX_THROTTLE=2, IDX_YAW=3` is correct.
- **`MSP_RC` order = internal order**: it writes `rcData[i]`, and rcData is ROLL, PITCH, **YAW,
  THROTTLE**, aux... Your measurement (roll, pitch, yaw, throttle) matches. **Send order and
  read-back order differ in positions 3 and 4** — an easy bug when comparing sent vs echoed.

### 4. Arming-disable flags with an MSP client attached
**[SOURCE-VERIFIED: `BF/msp/msp.c` case `MSP_SET_ARMING_DISABLED`; `BF/fc/core.c` `updateArmingStatus()`; `BF/flight/failsafe.c`; `BF/cli/cli.c` `cliEnter()`; `BF/fc/rc_controls.c` `calculateThrottleStatus()`]**

- **`RX_FAILSAFE`**: set whenever flight channels are invalid — radio off/unbound, *or* the zero /
  out-of-range MSP trap from finding 2 (latched, so it persists with Configurator unplugged and even
  with the switch flipped back). Cleared the instant a fully valid packet is processed.
- **`MSP`**: set only when some client sends `MSP_SET_ARMING_DISABLED` with byte0 != 0. It is
  tracked *per port*; the flag clears only when every port that set it sends byte0 = 0. It also
  **disarms immediately if armed**. Configurator sends it on connect **[RECALLED]**; yanking USB
  without "Disconnect" can leave it set until reboot **[RECALLED]**. An optional byte1 = 1 on the
  *enable* message temporarily disables runaway-takeoff prevention (useful props-off, below).
- **`CLI`**: set by `cliEnter()`; only a reboot clears it.
- **`THROTTLE`**: rcData throttle must be < `min_check` (1050). If throttle is in the mask and override
  is on, that is the *Jetson's* throttle, not the pilot's.
- **`ARM_SWITCH`**: if any flag was up while the arm switch was on, switch must go off then on again.
- **`BAD_RX_RECOVERY`**: receiver came back with the arm switch already on; cycle the switch.
- `ANGLE` (tilt) cannot fire: `small_angle = 180`. `BOOT_GRACE_TIME` clears a few seconds after power-up.
- **Arming over MSP via AUX1 is impossible while bit 4 (16) is outside the mask** — the MSP AUX1
  value is simply never read. That is the right design: the pilot owns ARM. The library's `arm()` does
  nothing on this setup.
- **Recommended arm sequence:** Jetson streaming valid neutral frames with throttle 1000 -> override
  OFF -> pilot throttle low -> pilot arms -> pilot flips override ON. Engaging override swaps values
  instantly (softened only by the RC smoothing filter), so the Jetson's throttle at that instant
  should be near what the pilot is holding.
- `runaway_takeoff_prevention = ON`: props-off with throttle raised, the FC sees no response and
  disarms itself with `RUNAWAY_TAKEOFF`. Expected on the bench; not an override bug.

### 5. ANGLE mode with override on — linear, 500 us = `level_limit`, with two caveats
**[SOURCE-VERIFIED: `BF/flight/pid.c` `pidLevel()`, `getLevelModeRcDeflection()`; `BF/fc/rc.c` `updateRcCommands()`]**

`angle = levelAngleLimit * getLevelModeRcDeflection(axis)`, clamped to +/-limit; deflection =
`(us - 1500 -/+ deadband) / (500 - deadband)` with level expo mixed in. Your dump has `deadband = 0`
and `roll/pitch_level_expo = 0`, so **target angle = level_limit x (us-1500)/500, exactly linear.**
The teammate's assumption holds. Caveats: (1) `level_limit` is **per PID profile** — your dump has
30, 23, 55, 55... across profiles, so read which profile is active; (2) the deflection goes through
the RC smoothing filter (finding 6), so the angle target lags ~16 ms. Actual/rate (ACRO) curves do
not apply to roll/pitch in ANGLE; yaw is still a rate. MSP values are treated exactly like radio
values — nothing downstream knows they came from MSP. Note the dump shows **no ANGLE range on any
switch**, consistent with SPEC.md.

### 6. RC smoothing and frame-rate detection
**[SOURCE-VERIFIED: `BF/fc/rc.c` `processRcSmoothingFilter()`, `rcSmoothingSetFilterCutoffs()`; `BF/common/filter.c` `pt3FilterGain()`]** (delay figures are my arithmetic)

- Frame-rate detection (`rxGetFrameDelta`) times the **CRSF** link only. MSP frames are invisible to
  it, so a 50 Hz MSP stream neither confuses nor informs it.
- Your cutoffs are fixed, not auto: setpoint 15 Hz, feedforward 15 Hz, throttle 10 Hz — which is the
  firmware's minimum (`RC_SMOOTHING_CUTOFF_MIN_HZ 15`), i.e. as smooth/slow as it goes. The filter is
  a PT3 (three cascaded first-order low-passes). Low-frequency delay is about 3/(2*pi*1.961*fc):
  **~16 ms on roll/pitch/yaw, ~24 ms on throttle**, 10-90 % step rise roughly 35 / 50 ms. Put that in
  the simulator's latency budget. It is heavy but it is what makes stair-stepped 50 Hz commands
  harmless, so leave it for now.
- Feedforward differentiates the setpoint at the CRSF packet rate, so a 50 Hz staircase looks like
  spikes separated by zeros; the 15 Hz filter blunts that **[inference, not measured]**.

### 7. Reading or setting the mask without the CLI — and why the UART wedges
**[SOURCE-VERIFIED: `BF/msp/msp.c` case `MSP_RX_CONFIG` and a full-file grep; `BF/cli/settings.c`; `BF/msp/msp_serial.c` `mspEvaluateNonMspData()`; `BF/fc/tasks.c` `taskHandleSerial()`; `BF/cli/cli.c` `cliEnter()`, `cliExit()`]**

- **No MSP message carries the mask in 4.4.3.** `msp_override_channels_mask` appears in exactly three
  places: the struct, the default (0), and the CLI settings table. `MSP_RX_CONFIG` (API 1.45) does not
  include it, and Betaflight has no generic "get setting by name" MSP2 call (that is an INAV feature).
- **What you *can* learn over MSP:** (1) mask != 0 <=> "MSP OVERRIDE" appears in `MSP_BOXNAMES` /
  id 50 in `MSP_BOXIDS`; (2) the exact mask, empirically: stream distinctive values (e.g. 1111, 1222,
  1333, 1444), have the pilot flip override on, read `MSP_RC`; channels that echo your numbers are in
  the mask. That is a complete, CLI-free readout (remember `MSP_RC` swaps positions 3 and 4).
- **To set it: Configurator -> CLI tab over USB** (`set msp_override_channels_mask = 15`, `save`).
  Separate port, the tool handles enter/exit/reboot. Stop the Jetson's MSP script first.
- **Why the UART wedges:** any byte received outside an MSP frame while **disarmed** is inspected:
  `#` -> `cliEnter(port)`; `reboot_character` (82, `R`) -> **reboot into the ROM bootloader (DFU)**.
  In CLI mode `taskHandleSerial()` does `if (cliMode) { cliProcess(); return; }` — **MSP processing
  stops on every port**, the `CLI` arming flag is set, and the only way out is the text `exit\r`,
  which **reboots the FC** (`cliExit()` -> `cliReboot()`). An MSP library that sent `#` and then keeps
  sending binary frames never sends `exit`, so it hangs until power-cycled. If you must script it:
  send `#`, wait for the prompt, send commands ending `\r`, then `save\r` (reboots) or `exit\r`
  (reboots), wait ~5 s, reopen the port. While **armed**, non-MSP bytes are ignored
  (`MSP_SKIP_NON_MSP_DATA`), so this cannot happen in flight.
- **Accidental trigger:** a Linux console still attached to `/dev/ttyTHS1` prints prompts and text
  containing `#` and `R`. That is why `setup_jetson_uart.sh` must be run once per board. A
  desynchronised MSP stream (dropped byte -> checksum fail -> parser idle) can also expose a payload
  byte 0x23 or 0x52 as a "command" **[inference]**.

### 8. Minimal safe bench procedure — PROPS OFF, battery on, radio on
Poll `MSP_RC`, `MSP_MOTOR` and `MSP_STATUS_EX` (arming-disable flags + mode flags) at ~10 Hz throughout.
1. **Baseline.** Override off, no RC stream. Move sticks: `MSP_RC` follows the radio. Note order R,P,Y,T.
2. **Read the mask.** Stream `[1111,1222,1000,1444, 1000,1000,1000,1000]` at 50 Hz. Flip override on.
   Whichever of roll/pitch/throttle/yaw echo your numbers are in the mask. Confirm the mode flag shows.
3. **Zero trap + latch (the important one).** Disarmed. Stop the stream, **power-cycle the FC** so
   the buffer is zero, do not start the stream, flip override on. Expect: `RX_FAILSAFE` within ~0.4 s.
   Flip override off. *Does the flag clear?* If not, the latch is confirmed. Start the stream: the
   flag should clear within about a second.
4. **Out-of-range.** Override on, stream a masked channel at 800: expect the same failsafe after 300 ms.
5. **Arm path.** Stream neutral + throttle 1000, override off, pilot arms, flip override on. Raise MSP
   throttle to ~1150: `MSP_MOTOR` should rise (expect a `RUNAWAY_TAKEOFF` disarm after a moment —
   props are off; keep throttle low and brief).
6. **Kill the sender.** Armed, override on, MSP throttle ~1100, then `kill -9` the script. Watch
   `MSP_RC` from a second process or the Configurator Receiver tab over USB. Stock 4.4.3 predicts:
   **values hold forever, no flag, motors keep running.** If instead they revert to the pilot's
   sticks after ~300 ms, the vendor back-ported PR #15217 — good news; write it in SPEC.md.
7. **Pilot takes back.** From step 6, flip override off: `MSP_RC` must return to radio values at once.
   This is the real kill path; time it.
8. **Radio loss.** Override on with a good stream, switch the transmitter off: expect `RX_FAILSAFE`,
   roll/pitch/yaw -> 1500, throttle held (`rxfail 3 h`), disarm at 5 s. This proves override does not
   survive radio loss — and shows that turning the transmitter off is *not* an instant kill.

## Open questions to test on the bench
1. Does the vendor build behave like stock 4.4.3 in step 6 (hold forever) or like 2026.6 (300 ms)?
2. Is the AUX-freeze latch (finding 2, step 3 of that list) real on this firmware? It is my reading
   of loop order; nobody upstream documents it in those words.
3. On the *new* airframe: what are the mask, `rxfail 3`, `failsafe_delay`, the active PID profile's
   `level_limit`, and is there an ANGLE range? The dump I read is from the old board.
4. Was symptom (a) airborne or on the ground, and instant or ~5 s after the flip? Stock code with
   `rxfail 3 h` predicts "controls go dead at 0.3 s, motors stop at 5 s". An instant stop would point
   at something else (e.g. throttle `rxfail` = auto on that board, giving 885 at 0.3 s).
5. What CRSF packet rate is the radio running? That is the true update rate of MSP sticks (finding 3).
6. Does Configurator leave the `MSP` arming flag set after an unclean USB unplug on this build?
7. Is 5 s of held throttle with centred sticks (`failsafe_delay 50` + `rxfail 3 h`) acceptable indoors
   near a net? It is an organizer choice; ask before changing it.

Sources: Betaflight 4.4.3 tree (files named above, fetched from raw.githubusercontent.com 19 Sep);
[PR #15217](https://github.com/betaflight/betaflight/pull/15217);
[issue #13416](https://github.com/betaflight/betaflight/issues/13416);
[issue #13374](https://github.com/betaflight/betaflight/issues/13374);
[issue #14004](https://github.com/betaflight/betaflight/issues/14004) (flips when engaging override
with off-centre sticks in ANGLE — titles only read, not the threads). Local: `target/msp/msp_rc.py`,
`logs/orin/bf_dump_before_beginner_20260918.txt`, `SPEC.md` section 2.
