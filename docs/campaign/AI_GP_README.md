# AI_GP

**Team Electric Fire's flight client for the AI Grand Prix physical qualifier** (Anduril LC3, Santa Ana, 15–22 September 2026). This repository holds what ran, or was built to run, on the aircraft's Jetson Orin NX: the MSP link to the Betaflight flight controller, the runners that fly a trained policy, the classical fallback stack, the bench tools that measured the aircraft, and the documents written on site.

Two sibling repositories hold the rest of the work, and the write-up of all three is the [paper](https://github.com/bojro/aigp-sim/blob/main/paper/paper.md) ([PDF](https://github.com/bojro/aigp-sim/blob/main/paper/paper.pdf)):

- [`bojro/aigp-sim`](https://github.com/bojro/aigp-sim): the Isaac Lab simulator, the plant model, the PPO policies and their checkpoints, and the paper.
- [`bojro/aigp-perception`](https://github.com/bojro/aigp-perception): the gate-corner detector, its labelling pipeline and the two shipped models.
- [`bojro/AI_GP-archive`](https://github.com/bojro/AI_GP-archive): the four eras before the physical qualifier (June to mid-September), sixteen branches, kept whole.

## Read these first

All three are on the `hover-throttle-ramp` branch, which carries the on-site work through 21 September; `main` stops at 19 September (see Branches below).

1. **`CLAUDE.md`**: the rules that cost hardware or a competition slot if broken. Never change the Betaflight configuration and never open its command-line console (it wedges this flight controller; only a battery pull clears it). The flight controller keeps the last MSP command if the stream stops, so a runner must never exit while the override switch is on. Props-off handover is a gate, re-run after any change. A policy checkpoint is not interchangeable with its runner; read the input width off the weights.
2. **`pq/FACTS.md`**: every number we measured on the aircraft, in one place: the 40 Hz link, the gyro in raw counts at 16.384 per degree per second, the pitch sign, the override mask and switch positions, the rate profile and what it does to a linear stick map.
3. **`pq/flight/onboard/RUNNING.md`**: the runners, which weights each loads, and the switch sequence a pilot follows.

## What is here

```
pq/                      the on-site work, 17 to 22 September
  FACTS.md               measured numbers, the source of truth
  flight/                the NumPy racing client: geometry, PnP, pose filter, gate tracker, curves, flight loop
  flight/onboard/        what was on the Jetson: the three policy runners, the contract, the adapter, the .npz weights
  tools/                 bench and measurement tools: loop rate, axis signs, telemetry decode, flight recorder
  hover/                 the staged hover bring-up (props-off takeover check, throttle ramp)
  research/              what we read and worked out before flying: MSP override, Betaflight, the hover plan
  logs/orin/             telemetry from the aircraft, with a README saying what each file is evidence of
  target/                the organizers' MSP library and camera tools, as shipped on the Jetson
  organizer_docs/, reference_text/   the specification, the Orin guides, the gate coordinates
  briefing/              the 17 September engineering journal and its figures
orin-setup/              Jetson connection scripts, motor and hover tools from the bench week
models/                  gate_pose_hand497 and hand434, the detectors that flew (recipe in aigp-perception)
isaac_drone_racer/       a September snapshot of the simulator; the maintained copy is aigp-sim
```

The root Python modules, `vision/`, `ekf/`, `control/`, `planning/` and `tools/` are the virtual-qualifier client from June to August (MAVLink, HSV and YOLO detectors, Kalman planners, HG-DAgger). They are not on the aircraft's path and exist only on some branches; the archive repository has all of them with their history.

## The chain on the aircraft

Camera (IMX477, 20° up-tilt, 30 fps) to a YOLOv8n-pose detector (~30 ms on the GPU, keypoint threshold 0.25) to an observation of 55 channels by 32 frames built at 40 Hz, to a NumPy policy emitting collective thrust and three body rates, to an adapter that inverts Betaflight's rate curve and applies the measured signs, to four RC channels streamed over MSP override on AUX5. Telemetry comes back on the same 115200-baud UART in one `MSP_MULTIPLE_MSP` request, 20 ms at the 95th percentile with the stream running. The pilot holds ARM and the override switch, which are the only abort.

## What was validated, and what was not

Validated, props off: the AUX5 handover to and from the policy, radio disarm, 320 of 320 observations at 40 Hz under full load, the telemetry decode against the organizers' library, the axis signs, the gyro scale, and sixteen deployment tests of the runner.

Never validated: props-on policy flight, automatic takeoff or landing, any position or height hold. The first props-on attempt lasted 0.45 s. The fallback stack's first flight hit the ceiling. No scored autonomous run was made; by the end of the week all four aircraft were grounded. The paper's Sections 6 to 9 have the account and what we would do differently.

## Branches

`main` and `hover-throttle-ramp` diverged on 20 September and have not been reconciled: `main` took the removal of the virtual-qualifier root, `hover-throttle-ramp` took the on-site week under `pq/` and is the branch this README describes. `perception-sim` carries `stack_sim/`, the fallback stack flown in Isaac against a pessimistic sensor model; `planb` carries the onboard fallback runner; `codex/hover-10s` carries the hover bundle that was on the Jetson; `gate-autolabeling` and `perception_assisted` carry labelling and dead-reckoning experiments. None of these is merged.

## Team

We are Team Electric Fire, Cornell University: Bojro Das (College of Arts and Sciences), Geneustace Wicaksono, Etienne Sasenarine, John Apessos, Grant Lin, Aaron Legg and Narayan Topalli (College of Engineering). The MSP library and camera toolchain under `pq/target/` are the organizers'. The simulator descends from Kousheek Chakraborty's `isaac_drone_racer` (BSD-3-Clause).
