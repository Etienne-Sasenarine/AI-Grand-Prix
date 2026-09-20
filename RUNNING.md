# Running this on *this* machine

Canonical start/install for the vendored tree is
[`../ISAACSIM.md`](../ISAACSIM.md) at the AI_GP repo root. This file is the
machine-specific note (Vulkan ICD, sitecustomize, WSL `libcuda`).

The short version: **train in WSL, watch in Windows.** They are two separate
Isaac Sim installs sharing one source tree on `D:`.

| | Training | Visualising |
|---|---|---|
| Runs in | WSL (Ubuntu) | Windows |
| Python | conda `env_isaaclab` | `D:\isaacsim_venv` |
| Mode | headless | GUI |
| Task id | `Isaac-Drone-Racer-v0` | `Isaac-Drone-Racer-Play-v0` |

The split exists because headless training in WSL is fast and stable, while
Isaac Sim's GUI under WSL falls back to software rendering. The Windows install
drives the display directly.

---

## Train

From WSL:

```bash
conda activate env_isaaclab
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
cd /mnt/d/Code/Competitions/AIGP/isaac_drone_racer
python scripts/rl/train.py --task Isaac-Drone-Racer-v0 --headless --num_envs 4096
```

`LD_LIBRARY_PATH` is not optional. Without it PhysX cannot find `libcuda.so`,
silently falls back to the CPU solver, and training runs roughly two orders of
magnitude slower while still appearing to work. If you see
`PhysX error: Could not load libcuda.so`, that is this.

4096 environments fits in the 8 GB on the RTX 4070 Laptop. Drop to 1024 if you
hit an OOM after changing the observation size or history length.

For a long run, detach it so a closed terminal does not kill it:

```bash
nohup python scripts/rl/train.py --task Isaac-Drone-Racer-v0 --headless \
  --num_envs 4096 > /tmp/train.log 2>&1 &
```

Checkpoints land in
`logs/skrl/drone_racer/<timestamp>_ppo_torch/checkpoints/`, and progress is
readable with `tensorboard --logdir logs/skrl/drone_racer`.

## Watch a trained policy

From Windows PowerShell:

```powershell
cd D:\Code\Competitions\AIGP\isaac_drone_racer
$env:OMNI_KIT_ACCEPT_EULA="YES"
$env:ENABLE_CAMERAS="1"
$env:VK_DRIVER_FILES="C:\WINDOWS\System32\DriverStore\FileRepository\nvami.inf_amd64_07a5b3dbac82d20b\nv-vk64.json"
$env:VK_ICD_FILENAMES=$env:VK_DRIVER_FILES

D:\isaacsim_venv\Scripts\python.exe scripts\rl\play.py `
  --task Isaac-Drone-Racer-Play-v0 `
  --num_envs 1 --real-time --enable_cameras `
  --checkpoint D:\Code\Competitions\AIGP\isaac_drone_racer\models\pq_speed_best.pt
```

Point `--checkpoint` at whichever run you want; omit it to load the latest.

**Use `Isaac-Drone-Racer-Play-v0`, not `Isaac-Drone-Racer-v0`.** They are both
registered and both launch happily, but the plain `-v0` id is the *training*
config: it disables the FPV camera, randomises which gate you start at, and
frames the viewer on the whole course. Launching the wrong one looks like the
viewer features are broken rather than absent.

First launch with `--enable_cameras` spends a few minutes compiling shaders
before anything appears. This is normal and only happens once.

### While it is running

| Key | Effect |
|---|---|
| `R` | restart the run from gate 1 |
| `V` | toggle drone camera / free chase view |
| middle mouse | pan |
| alt + left mouse | orbit |
| `F` | frame the selection |

The viewport starts in the drone's own camera. A **Telemetry** panel shows speed
in m/s and km/h, the current gate, altitude and vertical speed.

The free view is a chase camera anchored to the drone (`origin_type =
"asset_root"`), so orbiting and `F` stay useful. Earlier it was a fixed camera
190 m back framing all 269 m of track, where a 280 mm airframe is about one
pixel — which is why the sim looked empty.

## Tests

```bash
cd /mnt/d/Code/Competitions/AIGP/isaac_drone_racer
.venv/bin/python -m pytest tests/test_aigp_obs.py -q
```

`.venv` is a plain CPU-torch environment for unit tests, deliberately separate
from Isaac Sim so the observation contract can be checked in seconds without
booting a simulator. `tests/test_dynamics.py` currently fails to import against
a pre-existing upstream mismatch and is unrelated to the racing stack.

---

## Things that will bite

**Kit swallows Python `stdout`.** Anything `print`ed from `play.py` is invisible
in the GUI, which makes a working feature look dead. Viewer events are mirrored
to `%TEMP%\play_events.txt`; read that file before concluding a hotkey is
broken.

**Vulkan ICD conflict.** Without `VK_DRIVER_FILES` pinned to the NVIDIA ICD, the
loader finds multiple drivers (including a ghost Intel entry) and the GUI dies
at startup. The path above is machine-specific and changes when the NVIDIA
driver updates; if the GUI stops launching, re-find it under
`C:\WINDOWS\System32\DriverStore\FileRepository\nv*\nv-vk64.json`.

**`sitecustomize.py` in `D:\isaacsim_venv\Lib\site-packages\`.** This preloads
`osqp` and `h5py` at interpreter startup so they bind their own DLLs before Kit
loads conflicting copies. Without it the GUI crashes with an access violation or
`0xc0000139`. It is easy to lose when rebuilding the venv — do not delete it.

**`skrl` is pinned to 1.4.2.** Isaac Lab 2.1.0 is built against the 1.x runner
API; 2.x fails with `AttributeError: 'NoneType' object has no attribute 'shape'`
during agent construction.
