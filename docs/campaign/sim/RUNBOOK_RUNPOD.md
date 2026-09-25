# Training on a rented GPU box (RunPod)

Part 1 is the sequence that works, written from the attempt that got Isaac
Sim running on 20 Sep 2026 and verified end to end: both tasks pass 15 of 15
smoke checks on an RTX 6000 Ada and training launches from the same box.
Part 2 is how to size a run before paying for it. Part 3 is the catalogue of
everything that went wrong on the way, in the order it bit, so the next
attempt can skip the evening. What the first real smoke run found once Isaac
was up is in [`FINDINGS_2026-09-20.md`](FINDINGS_2026-09-20.md).

**The short version:** deploy from the Isaac Lab container, not a PyTorch
template; raise the file-descriptor limit before anything launches Isaac; and
never trust an exit code from anything in this stack.

---

# Part 1: The sequence that works

## 1. Deploy the pod

| Field | Value |
|---|---|
| Container image | `nvcr.io/nvidia/isaac-lab:2.1.0` (anonymously pullable, no NGC login) |
| GPU | RTX 6000 Ada or A6000: any Ada/Ampere **with RT cores**. The A100 is the wrong card: GA100 is the one Ampere die without them and is not on Isaac Sim's supported list. L40S, RTX 6000 Ada, A10G or a 4090 all work |
| Container disk | **200 GB** (the default 30 GB does not hold Isaac Sim) |
| Persistent storage | none (see the billing note in §8) |
| SSH terminal access | checked |
| Exposed TCP port | 22 |

Environment variables:

```
ACCEPT_EULA=Y
PRIVACY_CONSENT=Y
OMNI_KIT_ACCEPT_EULA=YES
```

**Do not** set `NVIDIA_DRIVER_CAPABILITIES`: RunPod strips that key from
user-supplied env vars, and you do not need to, because this image declares it
in its own manifest (a path RunPod does not filter) and it arrives as `all`.

### About the start command

RunPod's "Container start command" sets **CMD**, and Docker *appends* CMD to
the image's ENTRYPOINT rather than replacing it. This image's entrypoint is
`/isaac-sim/runheadless.sh`, so whatever you type becomes arguments to it:

```
PID 1: /bin/sh -c /isaac-sim/runheadless.sh <your command here>
```

A start command like `bash -c "sleep infinity"` therefore does not do what it
looks like, and anything after a `;` runs only once `runheadless.sh` exits,
which it does not, because it launches the streaming app and blocks.

**Simplest: leave the start command empty** and install what you need
afterwards (§2). The streaming app runs and idles; it wastes ~3.5 GB of GPU but
keeps PID 1 alive.

**If you do set one**, end it with `; sleep infinity`, not `&& sleep infinity`.
With `&&`, any failing command earlier in the chain (most easily a second
`sshd` when one is already bound to port 22) short-circuits the chain,
`sleep infinity` never runs, PID 1 exits and the container dies. The start
command that folds in everything learned, if you want one:

```
bash -c 'ulimit -n 65535; apt-get update -qq && apt-get install -y -qq openssh-server tmux && mkdir -p ~/.ssh && echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys && ssh-keygen -A && mkdir -p /run/sshd && /usr/sbin/sshd; sleep infinity'
```

## 2. Install sshd, through RunPod's proxy

The image ships no SSH daemon, and RunPod's own SSH setup is part of the start
script the entrypoint displaces. Until sshd exists you can only reach the pod
via `ssh.runpod.io`, which gives an interactive shell and no `scp`/`rsync`.

Two things about that proxy, both of which cost hours:

* `ssh <pod>@ssh.runpod.io 'command'` returns `Error: Your SSH client doesn't
  support PTY`. That means **authentication succeeded**; it just needs `-tt`.
* Anything you background there dies when the session closes. `nohup`,
  `setsid` and `&` all fail. Keep the session open for the whole job.

The pattern that works, with the trailing `sleep` holding stdin open so bash
does not see EOF and exit, killing whatever is mid-flight:

```bash
( printf 'apt-get install -y -qq openssh-server tmux; \
mkdir -p /root/.ssh /run/sshd; \
echo "%s" >> /root/.ssh/authorized_keys; \
chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys; \
ssh-keygen -A; /usr/sbin/sshd; sleep 3; \
pgrep -x sshd >/dev/null && echo SSHD_UP || echo SSHD_DOWN\n' \
  "$(cat ~/.ssh/runpod_aigp.pub)"; sleep 200 ) \
| ssh -tt -i ~/.ssh/runpod_aigp -o StrictHostKeyChecking=no \
      <podid>@ssh.runpod.io 2>&1 | grep -oE "SSHD_(UP|DOWN)"
```

Install **tmux** in the same breath. Without it there is no way to run anything
longer than a session, and installing it later needs the very thing you lack.

> **Never `pkill` the streaming Kit app.** It looks like idle waste and it is
> load-bearing: killing it makes `runheadless.sh` exit, the shell advances
> into the rest of PID 1's chain, that chain fails, and the container exits.
> This killed two pods before the mechanism was clear.

## 3. Get the repo across

`rsync` is not in the image. Use tar over ssh, which needs nothing at either
end:

```bash
cd ~/dev/aigp-sim
tar czf - --exclude='__pycache__' --exclude='.pytest_cache' --exclude='.git' . \
  | ssh -i ~/.ssh/runpod_aigp -p <PORT> root@<HOST> \
      'mkdir -p /workspace/aigp-sim && cd /workspace/aigp-sim && tar xzf -'
```

Excluding `.git` halves the transfer; the pod does not need history. This also
carries the Git LFS payload already resolved, so no `git lfs pull` and no
chance of the 98 MB drone USD arriving as a pointer. A `git clone` of the
private repo would hang waiting for credentials (§3 below). Verify:

```bash
head -c 12 assets/5_in_drone/configuration/5_in_drone_base.usd | grep -q git-lfs \
  && echo "POINTER, broken" || echo "real USD"
du -h assets/5_in_drone/configuration/5_in_drone_base.usd   # expect ~98M
```

## 4. Raise the file-descriptor limit

**The single most important line in this document.** Omniverse opens thousands
of descriptors loading its extension set, and a low limit segfaults Isaac Sim
inside `SimulationApp.__init__`, which reads as a mysterious crash with no
useful error. SSH sessions inherit sshd's limit, usually the 1024 default even
where the container's hard limit is 524288.

```bash
ulimit -n 65535
```

Put it at the top of every script that launches Isaac, **and before the first
`tmux` invocation** (§3 of the catalogue explains why the tmux server matters).
`scripts/pod/train_launch.sh` checks both and refuses to start otherwise.

## 5. Verify Isaac Sim actually launches

Cheap, and the real gate. Do it before installing anything else.

```bash
cat > /tmp/isaac_test.py <<'PY'
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
print("ISAAC_APP_LAUNCHED_OK", flush=True)
from isaacsim.core.api import World
w = World(stage_units_in_meters=1.0)
w.reset()
for _ in range(20):
    w.step(render=False)
print("ISAAC_PHYSX_STEPPED_OK", flush=True)
app.close()
print("ISAAC_CLEAN_EXIT", flush=True)
PY

cat > /root/run_test.sh <<'SH'
#!/bin/bash
ulimit -n 65535
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
cd /workspace/isaaclab && ./isaaclab.sh -p /tmp/isaac_test.py
echo "EXIT_RC=$?"
SH
chmod +x /root/run_test.sh
tmux kill-server 2>/dev/null; ulimit -n 65535
tmux new-session -d -s it "bash /root/run_test.sh > /tmp/it.log 2>&1"
```

Watch `/tmp/it.log`. First launch compiles shaders and takes several minutes.
Noise you can ignore: `Failed to create NGX context` and anything about NGX
(DLSS, unused headless), and `IOMMU is enabled` (a warning, not the cause of a
crash; the fd limit is).

## 6. Install skrl

Isaac Sim 4.5 and Isaac Lab 2.1 are in the image. Only skrl is missing, and it
must be **1.4.2**: Isaac Lab 2.1 uses the 1.x runner API, and 2.x fails with
`AttributeError: 'NoneType' object has no attribute 'shape'`.

```bash
cd /workspace/isaaclab
./isaaclab.sh -p -m pip install "skrl==1.4.2" pandas pytest
cd /workspace/aigp-sim && /workspace/isaaclab/isaaclab.sh -p -m pip install -e . --no-deps
python -c "import tasks"     # must succeed; pip's exit code is not enough
```

On Linux outside the container, export `LD_LIBRARY_PATH` (for WSL:
`/usr/lib/wsl/lib`) before training, or PhysX cannot find `libcuda.so`,
silently falls back to the CPU solver, and runs ~100× slower while appearing
to work. `PhysX error: Could not load libcuda.so` is that.

## 7. Smoke test, then train

Use the launcher rather than calling `train.py` directly. It runs the smoke
test as a gate, refuses to continue unless it passes, checks the fd limit and
the tmux server, and exports `PYTHONUNBUFFERED` and `PYTHONFAULTHANDLER`:

```bash
cd /workspace/aigp-sim
bash scripts/pod/train_launch.sh Isaac-Drone-Hover-v0 hover01 --max_iterations 300
```

Hover first: it is the cage-test artifact and the racing seed. **Stop it well
short of convergence**: a converged station keeper has learned to damp all
motion, which is the opposite of racing. Then racing, warm-started from it
(the policy weights transfer; the value head, being task-specific, does not):

```bash
bash scripts/pod/train_launch.sh Isaac-Drone-Racer-v0 race01 \
    --checkpoint /workspace/logs/skrl/drone_racer/<hover-run>/checkpoints/best_agent.pt
```

`scripts/pod/pipeline.sh` chains hover → racing → hover-from-racing → stress
sweep; `hover_forever.sh` / `race_forever.sh` resume legs and archive each
leg's best (skrl does not restore `checkpoint_best_modules` on load, so a leg's
best is only the best within that leg).

### Do not trust the exit code of anything here

Two independent mechanisms throw it away, and they compound: `isaaclab.sh`
swallows the exit code of the script it runs, and `SimulationApp.close()`
hard-exits, so `sys.exit(status)` never runs *and* Python's buffered stdout is
discarded. `smoke_test.py` therefore prints `SMOKE_RESULT=PASS|FAIL` on stdout,
flushed. Grep that. The launcher does, and treats a *missing* verdict as
failure, because a missing verdict means it died before finishing.

### Pull checkpoints off the pod, continuously

From your own machine (the pod is behind NAT and cannot reach you):

```bash
export POD_HOST=<ip> POD_PORT=<port> POD_KEY=~/.ssh/runpod_aigp
bash scripts/pod/pull_checkpoints.sh          # every 15 min until stopped
```

A pod with **no network volume is deleted** when the balance hits zero, and
the disk goes with it. A checkpoint that exists only on the pod is a
checkpoint you can lose entirely.

---

# Part 2: Sizing a run

On a rented box "how many environments", "how long" and "how much" are one
question. The arithmetic, so a run can be costed before it is started rather
than watched and hoped over.

## The units, which are easy to get wrong

skrl's progress bar counts **timesteps**, and one timestep advances *every*
environment once:

```
samples          = timesteps x num_envs
wall clock (s)   = timesteps / (it/s)
policy updates   = timesteps / rollouts          (rollouts = 24 here)
```

Doubling `num_envs` doubles the samples per timestep; it does **not** halve the
timesteps needed (it improves each gradient estimate and widens the
domain-randomisation coverage per update). Nor does it halve `it/s`, until the
GPU saturates: below the knee it is close to free, above it a straight loss.
The figure of merit is **env-steps/s**, and the right `num_envs` is where that
flattens. `scripts/bench_throughput.py --headless` measures it and names the
knee; run it with nothing else on the card.

## Measured, RTX 6000 Ada 48 GB

| | |
|---|---|
| `Isaac-Drone-Hover-v0`, 4096 envs | ~17.5 it/s, ~71,700 env-steps/s, 7.6 GB GPU, 62–83% utilisation |
| saturation | ~90k env-steps/s however the work is sliced |
| standing tax | the idle Kit streaming app holds ~3.5 GB of GPU and ~300% of one CPU; usable memory is ~45 GB, not 49 |

## Worked example

A full default racing run is `timesteps: 50000` in `skrl_cfg.yaml`:

```
at 4096 envs, 17.5 it/s:
    wall clock = 50000 / 17.5      = 2857 s   = 48 min
    samples    = 50000 x 4096      = 205 M
    updates    = 50000 / 24        = 2083
    cost       = 0.8 h x $0.87/h   = $0.70
```

A complete racing run is under an hour and under a dollar. The constraint on
this project was never compute; it was the plant being wrong, which is far
more expensive because it does not announce itself.

Compute wall clock from `timesteps / (it/s at that size)` and write it down
before starting: a run whose finish time was never estimated is a run nobody
notices has hung.

---

# Part 3: Failure catalogue

Everything that went wrong on 20 Sep, in the order it bit. Each entry appears
once; the working sequence above already routes around all of them.

## 1. Vulkan: `vkCreateInstance: Found no drivers`

Isaac Sim needs a working Vulkan device **even headless**: Kit initialises an
RTX renderer at startup whether or not you ask it to draw. RunPod's PyTorch
template starts the container with compute capabilities only. The graphics
libraries are all present (`libGLX_nvidia.so.0` even exports
`vk_icdGetInstanceProcAddr`) but the driver cannot initialise a graphics
context. Not fixable from inside: writing an ICD file, `mknod` on the device
(unprivileged container), and setting `NVIDIA_DRIVER_CAPABILITIES=all` in the
shell (read at container *creation*) all fail. You find out only after
installing 20 GB. Check first:

```bash
vulkaninfo --summary 2>&1 | grep -iE "deviceName|ERROR"
```

A `deviceName` line means you are fine. `Found no drivers` means stop and
redeploy from `nvcr.io/nvidia/isaac-lab:2.1.0`, whose own image ENV carries
`NVIDIA_DRIVER_CAPABILITIES=all` and `VK_DRIVER_FILES`.

## 2. Python 3.12 has no Isaac Sim 4.5

The PyTorch template ships Python 3.12; Isaac Sim 4.5's wheels are cp310 only
(`pip index versions isaacsim` lists 6.x only). Isaac Sim 6.x is not a free
upgrade: this repo targets Isaac Lab 2.1 APIs. If you must build outside the
container: conda-forge `python=3.10` (Anaconda's default channels need a
Terms-of-Service acceptance and have murky commercial terms), then
`torch==2.5.1` cu118, `isaacsim[all,extscache]==4.5.0` from
`pypi.nvidia.com`, Isaac Lab `v2.1.0` with `./isaaclab.sh --install none`.

## 3. `import isaacsim` hangs forever on the EULA

Looks like a slow download: 0.0% CPU, 12 MB resident, sleeping in
`wait_woken`, blocked reading stdin for `Do you accept the EULA? (Yes/No):`.
Set `OMNI_KIT_ACCEPT_EULA=YES` (persist it in `.bashrc`) and run installers
with `< /dev/null` so anything else wanting stdin fails loudly. To tell a hang
from work, check CPU time, not elapsed: `ps -o time=,etime= -p $PID`.

## 4. A private repo makes `git clone` hang, not fail

No credentials means blocking on a username prompt at 0% CPU. Push the working
tree from the machine that has it (Part 1 §3).

## 5. The repo must be importable

`ModuleNotFoundError: No module named 'tasks'`: `pip install -e . --no-deps`
or `PYTHONPATH=/workspace/aigp-sim`. pip may print `No module named 'omni.kit'`
while still succeeding; check `python -c "import tasks"` rather than pip's
exit.

## 6. RunPod injects SSH keys at pod start only

A key added to a running pod does not reach `authorized_keys`: the proxy works,
direct TCP refuses. Pipe the key in through the proxy (Part 1 §2 does this)
rather than restarting and losing a scarce GPU slot.

## 7. No sshd in the Isaac Lab image, and installing one after boot is fragile

Overriding the start command replaces RunPod's own startup script, so the pod
is reachable only through the proxy. Installing sshd afterwards failed three
ways for one reason: `nohup setsid ... &` dies with the proxy session; `printf`
into `ssh` closes stdin and bash exits, killing apt; the proxy pty mangles
scripted input. Either do it in the start command, where PID 1 owns it, or use
the held-open pipe in Part 1 §2. Anything that must outlive a shell belongs in
the start command.

## 8. Segfault in `SimulationApp.__init__`: the file-descriptor limit

The container's soft `ulimit -n` is 1024 (hard 524288). Isaac dies ~7 s in,
during extension load, with no traceback. IOMMU was the first suspect and a
red herring. `ulimit -n 65535` fixes it, **but not from inside a script under
tmux**: the tmux server outlives the session that started it and hands its own
limits to every pane, so a server started from a 1024-fd SSH shell caps the
run no matter what the script asks for, and the symptom is indistinguishable.
Raise the limit before the first `tmux`, or `tmux kill-server` first. Verify:

```bash
tmux new-window -d "bash -c 'ulimit -n > /tmp/fd.txt'"; sleep 1; cat /tmp/fd.txt
```

Also worth checking: `df -h /dev/shm` (ours was 88 GB; a 64 MB default would
be a problem).

## 9. Isaac's hard exit eats both your output and your status

`SimulationApp.close()` terminates the process without unwinding. `sys.exit`
after it never runs, so the shell sees 0 whatever happened; and Python's
block-buffered stdout is discarded, so a script that ran perfectly can leave a
log of only C++ warnings (native, unbuffered) followed by nothing. That reads
exactly like §8. Before concluding a run crashed:

```bash
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
```

The first makes the output appear; the second prints a native traceback if it
really is a segfault. Both belong in every script that launches Isaac. And
every script whose result matters prints a flushed verdict line; `$?` is not
evidence.

## 10. Mistakes that were mine, not the platform's

* `pkill -f "import isaacsim"` matched the shell running that very command and
  killed its own SSH session, silently. Use `pkill -f "[i]mport isaacsim"`.
* `rsync --info=progress2` does not exist in macOS's rsync 2.6.9.
* Talking ourselves out of the Isaac Sim container at 1 am because it "needed
  NGC credentials" (it does not) traded three setup steps for five
  compatibility problems and a pod that could not run the simulator at all.

## 11. Billing traps

* Credit presets start at $150 and **$200 is preselected**; click "Other".
* **Auto-Pay must read `Disabled`** before loading credit.
* Container disk defaults to 30 GB; Isaac Sim alone is ~10 GB of wheels. Use
  100 GB+ (about $0.013/h).
* A pod with no network volume is terminated at zero balance and its data is
  unrecoverable. Sync checkpoints off-box on a timer.

## Quick reference

| Symptom | Cause |
|---|---|
| `vkCreateInstance: Found no drivers` | wrong image; use `isaac-lab:2.1.0` (§1) |
| dies ~7 s in, no traceback, `RC=0` | fd limit, or a stale tmux server capping it (§8) |
| log has only C++ warnings, then nothing | hard exit discarded the stdout buffer (§9) |
| script "passed" but nothing was checked | `$?` is meaningless; grep the printed verdict (§9) |
| `ulimit -n 65535` in the script had no effect | tmux server started from a 1024-fd shell (§8) |
| direct SSH refused, proxy works | no sshd in image, or key injected after start (§6, §7) |
| `Your SSH client doesn't support PTY` | auth succeeded; add `-tt` |
| container dies after you kill something | PID 1's `&&` chain broke; never kill Kit |
| `git clone` hangs at 0% CPU | private repo, no credentials; use tar |
| `import isaacsim` hangs at 0% CPU | EULA prompt on stdin (§3) |
| `rsync: command not found` | not in the image; use tar |
| PhysX runs ~100× slow, no error | `LD_LIBRARY_PATH` missing `libcuda.so`, CPU solver fallback |
