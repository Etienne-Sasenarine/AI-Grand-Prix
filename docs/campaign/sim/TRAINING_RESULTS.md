# Training results, with the checkpoint and source for each number

Every figure below names the checkpoint it was measured on and the commit or
document it comes from. Numbers without a checkpoint name are not
reproducible and are marked as such. Conversions used by the pod scripts:
hover settled % = `Episode_Reward/settled` / 3.0; racing gates per episode =
`Episode_Reward/gate_passed` × 2.667 at 40 Hz.

The course is the organizer's ten gates; gate 9 is a stacked double gate
flown through twice per lap (upper opening southbound at 4.05 m, lower
northbound at 1.35 m), so a lap is **11 crossings** and a two-lap run is 22.
The simulator's 11 track entries are those 11 crossings.

## The plant had to be fixed before any number meant anything

First live smoke run on an RTX 6000 Ada, 20 Sep: 10 of 14 checks passed. After
the fixes, **15 of 15** (`docs/FINDINGS_2026-09-20.md`, commits `6a6c91b`,
`2fca817`, `159bf23`, `444a8a3`, `1a4259a`):

| finding | before | after |
|---|---|---|
| total mass (5 rigid bodies × 1.745 kg) | 8.725 kg, TWR 0.2 | 1.745 kg |
| inertia products from a 7.16° COM tilt | 4.3e-4 (12% of roll moment) | 0 |
| envs killed by `flyaway` on step one (stale gate pose) | 48 of 64 | 1 of 64 (a spawn inside a gate, left open) |

Reward weights had been tuned against an aircraft that could not fly, in
episodes that ended on step one.

## Start distribution

* **Before**: the best racing policy averaged 9.5 gates per episode from the
  training spawn mix yet passed G1 from the real start in **0 of 456**
  attempts; started at G6 it flew seven gates deep and passed G1 with 7.4%
  failure (`1348d67`).
* Two bugs made the fix inert: the spawn pose tensor was copied before the pad,
  scatter and floor-height blocks ran (0 of 1024 spawns within 1.5 m of the
  pad, none below 0.67 m, `56c9b22`), and a ±0.7 m vertical jitter applied
  after height selection put 15% of floor spawns underground (`5a7f7f3`).
* **After**, `scripts/diag/diag_takeoff.py` on 256 pad starts with a
  checkpoint that had never trained on the corrected distribution
  (`7c73f14`): 7.88 of 8 corners visible at spawn, spawn height
  0.122 / 0.293 / 0.497 m (min / median / max), 0 died within 0.5 s,
  **221 passed G1 (86.3%)**, median 2.15 s to G1; the aircraft popped to
  2.98 m at 1 s against G1 at 1.35 m.
* Training mix since `5a7f7f3`: 35% competition pad targeting G1, 35% floor
  run-in at a random gate, 30% in motion 1 m past a random gate.
* Pad vs mix on `race40_leg3`: **9.18** gates from the pad against 8.98 from
  the mix over 15 s (`a783ecd`).

## Speed cap

The first corrected-plant racing policy flew at 14.0 m/s median, 21.3 m/s p95,
and **73%** of its crashes were gate strikes. Two laps of 106.2 m need 5.3 m/s
to finish inside the 40 s episode; `over_speed` now penalises (|v| − 8)² above
8 m/s, and `vel_gate_passage` is zeroed (`4fe43ef`, `rewards.py`).

## Hover: its own PPO config

Hover shared racing's agent config until 20 Sep. Measured over the same window
(runs 20-53-55 hover vs 20-53-52 racing, `c1de10a`):

| | hover | racing |
|---|---|---|
| policy std | 0.6094 → 0.6604 (**+8.4%**) | 0.5854 → 0.5801 (−0.9%) |
| learning rate | 8.8e-5 → 2.2e-4 (**2.4×**) | 7.0e-5 → 9.4e-5 (1.3×) |
| settled | 67.9% → 61.8% | |

Cause: a converged station keeper has no task gradient left, so
`entropy_loss_scale` grows `log_std`, amplified by `KLAdaptiveLR` whose default
`max_lr` is 100× base. Hover now has `entropy_loss_scale: 0.0` and
`max_lr: 1e-4`; racing's file is unchanged. After the split: std −4.1%,
settled 68.3% peak → 77.4% → **87.0%** (`SESSION_2026-09-20.md`, `d3065d0`).

Shared trunk vs separate policy/value networks, 2×2 at 6k timesteps
(`skrl_cfg.yaml`): shared 13.38 reward / 0.110 gates vs separate 8.56 / 0.043
with lookahead; 12.79 / 0.088 vs 8.06 / 0.037 without. A full 50k separate run
went NaN. Default is shared.

## Hover stress sweep

`hover_leg2`-era best checkpoint, `scripts/diag/stress_hover.py`, 15 s,
scored after settling (`SESSION_2026-09-20.md`):

| condition | survived | settled |
|---|---|---|
| nominal | 95.3% | **75.8%** |
| rate tau = 0.060 s | 94.1% | 73.9% |
| mass × 0.90 | 95.7% | 73.2% |
| wind 0.6 N | 89.8% | 36.5% |
| mass × 1.15 | 77.2% | 20.4% |
| delay = 4 steps (66 ms) | 70.1% | 13.0% |
| hover stick pinned 0.19 | 76.4% | 7.0% |
| hover stick pinned 0.31 | 55.7% | 1.9% |

Robust to how the aircraft *responds*; fragile to how much thrust it *needs*.
The two hover rows are outside the trained 0.21–0.29 band. Note this sweep
predates the checkpoint-in-log fix (`d3065d0`); the exact checkpoint is not
recorded.

## 40 Hz vs 60 Hz

The Jetson runner ticks at 40 Hz (the UART cannot carry telemetry and the RC
stream faster; measured in the flight repo, `pq/flight/FINDINGS_phase_b.md`).
Policies trained at 60.

**Hover**, `hover_leg2_2353.pt`, 15 s, 2×2 (`d3065d0`):

| | delay 0–2 (trained) | delay pinned 0 |
|---|---|---|
| 60 Hz | 87.0% settled, p95 0.37 m | 88.7%, 0.36 m |
| 40 Hz | 52.6%, 0.93 m | 66.1%, 0.55 m |

34 points lost at 40 Hz: 22.6 from the rate itself (actions held 1.5× longer,
history spanning 0.8 s instead of 0.53 s), 13.5 from the extra latency.

**Racing**, `race_leg1_2146.pt`, 15 s, training spawn mix (`c28dc54`): 8.246
gates at 60 Hz vs 3.086 at 40, i.e. racing keeps **37%** where hover keeps
60%. A ratio only: the respawning mid-course population inflates the absolute.

**Retrained at 40 Hz** (`AIGP_HOVER_HZ=40` / `AIGP_POLICY_HZ=40`): the 40 Hz
hover scores **85.5% settled with p95 0.331 m**, a better tail than the 60 Hz
policy at its own rate, and 89.0% at 60 Hz. That measurement is recorded in
the flight repo (`pq/flight/FINDINGS_phase_b.md`, `CLAUDE.md`), not here.

## Detector dropout

As an evaluation-only perturbation, `kp_dropout` (30% uniform) cost **71% of
baseline gates** on a policy that had never trained against it (`b0c44e0`).
Training against the calibrated clustered model (`AIGP_KP_DROP=0.13`,
`AIGP_KP_STICKY=0.81`) is the `race40drop` chain below. Writing dropped
corners as 0 instead of `NOT_SEEN` collapsed racing from 12.80 to 0.13 gates
per episode (`a783ecd`).

## The chains, 20–21 Sep (pod, RTX 6000 Ada)

Racing on 20 Sep reached **10.52 gates per episode** (40 s, exploration
noise), up from a previous best of 9.53, on the harder start distribution
(`SESSION_2026-09-20.md`, archived as `race_leg1_2146.pt`).

Best checkpoints pulled 21 Sep (`~/dev/aigp-checkpoints/2026-09-21/PROVENANCE.md`,
not in this repo):

| chain | file | metric | scored under |
|---|---|---|---|
| hover, 40 Hz | `hover40_best_agent.pt` | **82.7%** settled (peak 2.480) | |
| racing, 40 Hz | `race40_best_agent.pt` | **15.27** gates/ep (peak 5.724) | perfect corners |
| racing, 40 Hz, dropout | `race40drop_best_agent.pt` | **10.23** gates/ep (peak 3.835), still improving when stopped | corners dropping |

Three notions of "best" and none evaluated head to head: skrl's
`best_agent.pt` (mean total reward), the periodic snapshot nearest our
metric's peak (checkpoints land every 960 steps and the peaks did not), and the
leg-end score the ratchet trusted. The `race40` vs `race40drop` comparison
that matters, both under the same measured dropout, was never run.

**The ratchet cycled.** The forever-chains revert to their best after two
regressions and re-run a leg; with a fixed seed and config the re-run scores
the same value bit-identically, the next leg scores lower, and it reverts
again. `rate40` alternated 1.9345 / 1.5106 and `race40` 4.6209 / 3.5406 from
about 04:10 on 21 Sep, roughly eight GPU-hours producing nothing new. Each leg
restart also costs ~12% because skrl restores policy weights but not the
optimiser state.

## Throughput and cost (RTX 6000 Ada 48 GB)

`scripts/bench_throughput.py`, `daf63f8`, `1ae8acd`, `0989447`:

| | |
|---|---|
| `Isaac-Drone-Hover-v0`, 4096 envs | ~17.5 it/s, **~71,700 env-steps/s**, 7.6 GB GPU, 62–83% utilisation |
| saturation | ~90k env-steps/s whatever the slicing; one job gets 61k, two concurrent jobs share 90k; 4× envs bought 1.47× throughput |
| standing tax | the idle Kit streaming app holds ~3.5 GB and ~300% of one CPU; usable GPU memory ~45 GB |
| a full 50k-timestep racing run at 4096 envs | 48 min, 205 M samples, 2083 updates, **≈ $0.70** |

Compute was never the constraint. The plant being wrong was, and it does not
announce itself.
