# aigp-sim

Isaac Sim training stack for the AI Grand Prix racing policy. The flight client
— YOLO, MSP, `main.py` — stays in [`AI_GP`](https://github.com/Code-Red-Cables/AI_GP).

A PPO policy learns to fly a 5-inch quadcopter through eleven gates from
projected gate keypoints plus IMU. It is deployed to a Jetson Orin NX which
sends collective thrust and body rates to a Betaflight flight controller over
MSP.

## Why this is a separate repo

`AI_GP` had grown 1.5 GB of run logs and 129 MB of checkpoints around a 250 KB
simulator package. A rented GPU box should clone the thing it is going to run,
not the flight client, the vision tooling and the competition PDFs.

The split has one real hazard, and `contract/` exists to manage it. See below.

## The observation contract

`utils/aigp_obs.py` here and `race_obs.py` in the flight repo are two
implementations of one vector — torch across 4096 environments, numpy on a
single Orin at 60 Hz. Keeping them separate is deliberate; forcing both through
one implementation would make both worse.

What must never differ is the *definition*. If the two disagree by a channel
order or a clip, **nothing raises**. The policy receives a vector whose
channels mean slightly different things than the ones it trained on, and flies
into a gate with complete confidence.

So `contract/` defines it once — plain Python, no dependencies — and hashes
every value both ends must agree on:

```python
from contract import observation, verify

observation.observation_dim("v2")   # 1760
verify.short_hash("v2")             # 'a20c14d6a335'
verify.check_run_dir(run_dir)       # raises ContractMismatch on drift
```

`train.py` writes `contract.json` into every run directory beside `params/`,
and `play.py` refuses a checkpoint whose stamp disagrees with the running code.
To change the contract deliberately: change it here, let the hash move, retrain.
That cost is the point — it is what stops it moving by accident.

Randomised plant parameters are deliberately **not** hashed. The policy is
meant to see a spread of hover points and thrust curves; hashing those would
make every run incompatible with every other.

### Versions

| | frame | flat | |
|---|---|---|---|
| `v1` | 51 | 1632 | what every checkpoint through `pq_speed_best` trained on |
| `v2` | 55 | 1760 | `v1` plus the four action channels |

`v2` matters once the plant has latency. With no delay a policy can infer what
it commanded from what happened next, so action history is redundant. With
~40 ms of real delay it is not: the policy commands roll, sees nothing for two
or three steps, commands more, and the accumulated commands land together and
overshoot. Eschmann's ablation measured trajectory tracking going from 10/10 to
0/10 when action history was removed with delay still simulated.

`v2` is a strict append, so a `v1` reader sees exactly `v1` in the first 51
channels.

## Layout

```
contract/     the observation vector, camera and plant constants, hashed
tasks/        Isaac Lab env: rewards, terminations, commands, the PQ track
dynamics/     action decode, thrust curve, rate loop
utils/        observation construction, gate counter, logging
assets/       drone and gate USD (Git LFS)
scripts/rl/   train.py, play.py
tests/        96 run without Isaac; the rest need the training box
```

Package names are unchanged from `AI_GP/isaac_drone_racer/` on purpose — Isaac
Lab resolves task entry points by string, so renaming buys nothing and breaks
`gym.register`.

## Install

Needs Git LFS **before** the first clone, or the 98 MB drone USD arrives as a
pointer file and Isaac fails to load it with an unhelpful error.

```bash
git lfs install
git clone <url> && cd aigp-sim
```

Isaac Sim 4.5 + Isaac Lab 2.1.0 + **skrl 1.4.2** (not 2.x — Isaac Lab 2.1 uses
the 1.x runner API). Full pinned steps in [`RUNNING.md`](RUNNING.md).

**Renting a GPU box?** [`DEPLOY_PROCEDURE.md`](DEPLOY_PROCEDURE.md) is the
step-by-step that works; [`RUNPOD_SETUP.md`](RUNPOD_SETUP.md) is the catalogue
of what goes wrong and why.

**Read one of them first.** It lists
every failure from a real attempt, in the order they bite. The first one is
fatal and silent: Isaac Sim needs a working Vulkan device even headless, and a
container started without the `graphics` driver capability cannot provide one —
you find out only after installing 20 GB. Check `vulkaninfo --summary` before
anything else.

```bash
python -m pytest -q          # 96 pass without Isaac installed
```

On Linux, export `LD_LIBRARY_PATH` before training or PhysX silently falls back
to the CPU solver and runs ~100x slower.

## Train

```bash
python -u scripts/rl/train.py --task Isaac-Drone-Racer-v0 --headless --num_envs 4096
```

Rendering is off unless `--video`. That matters when choosing a GPU: **the A100
is the wrong card** — GA100 is the one Ampere die without RT cores and is not
on Isaac Sim's supported list. L40S, RTX 6000 Ada, A10G or a 4090 all work.

## Checkpoints

Not in this repo — large, reproducible, and meaningless without the contract
hash they trained under. Fetch the keeper from `AI_GP/isaac_drone_racer/models/`
when warm-starting.

## State

Known gaps, honestly:

- **Latency is not modelled.** Nothing in the env delays an observation or an
  action. Two independent measurements put the cliff between 30 and 50 ms, and
  Betaflight's own command→actuation is ~40 ms.
- **`rc_smoothing_setpoint_cutoff` is 15 Hz on our FC** (measured). That is a
  bandwidth limit on the rate setpoint, not just a delay, and the sim's rate PD
  responds instantly.
- **Detector error is a guess.** Keypoints are projected perfectly and then
  given i.i.d. Gaussian noise. Real detector failure is temporally correlated —
  a blurred turn drops many frames in a row, not scattered singles.
- **Inertia is estimated, not measured.** See `contract/plant.py`.
- **The randomisation knobs live in the eval harness, not training.**
- **Three asset files are broken LFS pointers** inherited from `AI_GP`, with
  their content missing from that remote. Nothing loads them, so training is
  unaffected — see [`assets/BROKEN_LFS_OBJECTS.md`](assets/BROKEN_LFS_OBJECTS.md).
