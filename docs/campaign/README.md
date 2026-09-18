# `pq/` — the Physical Qualifier working tree

Restored 18 Sep 2026. Everything in this directory was produced on site on
17–18 September and, until this commit, **existed only on the team laptop** — it
had never been committed to any branch of this repo or of `AI_GP-archive`.

It is kept under one directory for two reasons: every cross-reference between
these documents and this code is relative and stays correct, and `tools/` here
would otherwise collide with the unrelated `/tools/` at the repo root.

## Start here

| File | What it is |
|---|---|
| **`SPEC.md`** | **The consolidated source of truth.** Every hardware, course, rules and simulator number, each tagged with where it came from and how much to trust it (`[SPEC] [MEAS] [DERIV] [ASSUM] [CONFLICT] [OPEN]`). Check it before answering any factual question |
| **`EXECUTION.md`** | What we do next, in order, with numbered actionables (A1–A10, B1–B6, C1–C8, …) |
| **`HANDOFF.md`** | State of the flight software, the six bugs found and why they matter |
| **`SIM_MEASUREMENTS.md`** | The field measurement card. Measure four things; do not measure inertia |
| **`EXPERIMENTS.md`** | The bench and flight running order, with the commands |
| `FIELD_PLAN.md` | Reference detail behind the above: safety architecture, bring-up ladder, contingencies |
| `GATE_IDENTITY.md` | Why the gate index matters and what happens without it |
| `PLAN_B.md` | The slow, reliable fallback controller |
| `INSTALL_PLAN.md` | Earlier draft — check dates before trusting it |
| `REPO_RESTORE.md` | The audit this commit acts on |

`CLAUDE.md` is the workspace briefing for coding agents. Its paths are written
relative to the laptop workspace root, which is this directory.

## Code

| Path | What it is | Tested |
|---|---|---|
| `flight/` | The onboard stack: policy runtime, geometry, observation assembly, gate tracker, PnP, pose filter, Betaflight curves, control adapter, flight loop | `cd flight && python run_all.py` — 13 suites |
| `tools/` | Orin bench and measurement tools: MSP logger, hover and rate-step analysis, Betaflight CLI bridge, RPM telemetry, plus a flight-model mock so all of it runs with no hardware | `cd tools && python run_all.py` — 8 pass, 1 skipped (camera, needs the drone) |
| `experiments/tools/` | Isaac stress and retrain harnesses. Newer than the copies under `/analysis/2026-09-16/` |

### Running `flight/test_parity.py` from inside the repo

It is the most important test in the package: it checks that what the network
sees in flight is what it saw in training. It therefore has to compare against
the *same camera the checkpoint was trained on*, and the two copies of
`isaac_drone_racer` differ on exactly that. This repo's copy carries the
corrected PQ camera (`FX = 423.6`); the checkpoint in `flight/models/` was
trained against the virtual-qualifier camera (`FX = 320`). Comparing across them
does not error — it reports about 100 px of parity failure that reads like a
geometry bug.

The test now selects a copy by `FX` and refuses to guess, so from inside the
repo point it at a matching checkout:

```
AIGP_TRAIN_DIR=/path/to/a/FX-320/isaac_drone_racer python test_parity.py
```

On the team laptop it finds the right copy on its own. This is a live
discrepancy, not a test problem: **the committed checkpoint is trained on the
wrong camera** and is superseded once the retrain lands.

**Pure NumPy and OpenCV.** The Jetson image ships no deep-learning framework and
installing one there is a multi-hour job that breaks on venue Wi-Fi. Never
`pip install opencv-python` on the board — it silently breaks the GStreamer
camera path.

`flight/` has **never touched hardware.** Nothing in it is validated against a
real drone, camera or flight controller.

## Reference documents

`organizer_docs/` holds the three official documents that were in no repo:
the Orin NX quick start, the Orin NX software guide, and the gate-coordinate
table. `reference_text/` holds their plain-text extracts, which matter because
there is no `pdftotext` on the team laptop — re-extracting needs `pypdf` in the
Isaac venv.

The two technical specifications are at the repo root in `/reference/`:
`260902_PQ_Technical_Spec_0002.pdf` is VADR-TS-005, the **Physical** Qualifier —
the source of truth here. `AI Grand Prix Tech Specs.pdf` is VADR-TS-001, the
**Virtual** Qualifier: a different race, a different simulated camera and a
MAVLink interface. Several bugs have come from mixing the two up.

**The event Welcome Packet is deliberately absent.** It contains the team Wi-Fi
password and must not be committed here or anywhere else.

## Measured evidence

`logs/orin/` holds four hover logs from 18 Sep and the flight controller's
`diff all` dump. **The hover logs are invalid as hover measurements** and
`logs/orin/README.md` explains why in detail — they are kept because they are the
evidence behind three findings recorded in `SPEC.md`: the barometer is unusable
with props turning, raw gyro is in counts rather than degrees per second (this
closes an open `[CONFLICT]`), and the logger never recorded motor outputs.

`experiments/results/ctx/` holds the context-mode Isaac runs from 17 Sep that
`GATE_IDENTITY.md` rests on.

## Related work elsewhere in this repo

`/orin-setup/` holds a teammate's connection scripts and props-off motor test,
with live-measured facts worth folding into `SPEC.md`: BTFL 4.4.3 / API 1.45,
board SH74, four DShot300 motors, and peak RPM per motor at output 1050.
`/isaac_drone_racer/tools/physical_constants.py` covers similar ground to
`tools/`; the two should be reconciled rather than maintained separately.
