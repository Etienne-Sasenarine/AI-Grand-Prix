# Deploying to a RunPod box, step by step

The reproducible path, written from the attempt that actually got Isaac Sim
running. [`RUNPOD_SETUP.md`](RUNPOD_SETUP.md) is the catalogue of everything
that went wrong and why; this is just the sequence that works.

**Verified end to end.** Both tasks pass 15 of 15 smoke checks on an RTX 6000
Ada, and training launches from the same box. What the first real smoke run
found, and what each failure turned out to be, is in
[`FINDINGS_2026-09-20_resolved.md`](FINDINGS_2026-09-20_resolved.md) — two of
the four were not what they looked like.

---

## 1. Deploy the pod

| Field | Value |
|---|---|
| Container image | `nvcr.io/nvidia/isaac-lab:2.1.0` |
| Registry auth | **leave empty** — it pulls anonymously |
| GPU | RTX 6000 Ada or A6000 (any Ada/Ampere with RT cores) |
| Container disk | **200 GB** |
| Persistent storage | none |
| SSH terminal access | checked |
| Exposed TCP port | 22 |

Environment variables:

```
ACCEPT_EULA=Y
PRIVACY_CONSENT=Y
OMNI_KIT_ACCEPT_EULA=YES
```

**Do not** set `NVIDIA_DRIVER_CAPABILITIES` — RunPod strips that key from
user-supplied env vars. You do not need to: this image declares it in its own
manifest, which is a path RunPod does not filter, and it arrives as `all`.

### About the start command

RunPod's "Container start command" sets **CMD**, and Docker *appends* CMD to the
image's ENTRYPOINT rather than replacing it. This image's entrypoint is
`/isaac-sim/runheadless.sh`, so whatever you type becomes **arguments to it**:

```
PID 1: /bin/sh -c /isaac-sim/runheadless.sh <your command here>
```

That is why a start command like `bash -c "sleep infinity"` does not do what it
looks like. It also means anything after a `;` in your command runs only once
`runheadless.sh` exits — which it does not, because it launches the streaming
app and blocks.

**Simplest approach: leave the start command empty** and install what you need
afterwards (step 2). The streaming app will run and idle; it wastes a little
GPU but keeps PID 1 alive, and the card has room.

**If you do set one**, end it with `; sleep infinity` and not
`&& sleep infinity`. With `&&`, any failing command earlier in the chain — most
easily a second `sshd` when one is already bound to port 22 — short-circuits the
chain, `sleep infinity` never runs, PID 1 exits and **the container dies**.

## 2. Install sshd, through RunPod's proxy

The image ships no SSH daemon, and RunPod's own SSH setup is part of the start
script the entrypoint displaces. Until sshd exists you can only reach the pod
via `ssh.runpod.io`, which gives an interactive shell and no `scp`/`rsync`.

Two things about that proxy, both of which cost hours to learn:

* `ssh <pod>@ssh.runpod.io 'command'` returns `Error: Your SSH client doesn't
  support PTY`. That means **authentication succeeded** — it just needs `-tt`.
* Anything you background there dies when the session closes. `nohup`, `setsid`
  and `&` all fail. Keep the session open for the whole job instead.

The pattern that works — note the trailing `sleep`, which holds stdin open so
bash does not see EOF and exit, killing whatever is mid-flight:

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
> load-bearing: killing it makes `runheadless.sh` exit, the shell advances into
> the rest of PID 1's chain, that chain fails, and the container exits. This
> killed two pods before the mechanism was clear.

## 3. Get the repo across

`rsync` is not installed in the image. Use tar over ssh, which needs nothing at
either end:

```bash
cd ~/dev/aigp-sim
tar czf - --exclude='__pycache__' --exclude='.pytest_cache' --exclude='.git' . \
  | ssh -i ~/.ssh/runpod_aigp -p <PORT> root@<HOST> \
      'mkdir -p /workspace/aigp-sim && cd /workspace/aigp-sim && tar xzf -'
```

Excluding `.git` halves the transfer and the pod does not need history.

**Verify the drone asset is real, not a Git LFS pointer.** Transferring the
working tree carries the LFS payload already resolved, which is the point — a
`git clone` on the pod would need credentials for a private repo and hangs
waiting for them:

```bash
head -c 12 assets/5_in_drone/configuration/5_in_drone_base.usd | grep -q git-lfs \
  && echo "POINTER — broken" || echo "real USD"
du -h assets/5_in_drone/configuration/5_in_drone_base.usd   # expect ~98M
```

## 4. Raise the file-descriptor limit

**The single most important line in this document.** Omniverse opens thousands
of descriptors loading its extension set, and a low limit segfaults Isaac Sim
inside `SimulationApp.__init__` — which reads as a mysterious crash with no
useful error.

SSH sessions inherit sshd's limit, which is usually the 1024 default even where
the container's hard limit is far higher. Put this at the top of **every script
that launches Isaac**, not just in your shell:

```bash
ulimit -n 65535
```

Check the hard ceiling first if it fails: `ulimit -Hn`.

## 5. Verify Isaac Sim actually launches

Do this before installing anything else. It is cheap and it is the real gate.

```bash
cat > /tmp/isaac_test.py <<'EOF'
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
EOF

cat > /root/run_test.sh <<'EOF'
#!/bin/bash
ulimit -n 65535
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
cd /workspace/isaaclab && ./isaaclab.sh -p /tmp/isaac_test.py
echo "EXIT_RC=$?"
EOF
chmod +x /root/run_test.sh
tmux new-session -d -s it "bash /root/run_test.sh > /tmp/it.log 2>&1"
```

Watch `/tmp/it.log`. First launch compiles shaders and takes several minutes.

Errors that are **noise** and can be ignored: `Failed to create NGX context`
and anything about NGX — that is DLSS, which headless training does not use.
`IOMMU is enabled` is also a warning, not the cause of a crash; the file
descriptor limit is.

## 6. Install skrl

Isaac Sim 4.5 and Isaac Lab 2.1 are already in the image. Only skrl is missing,
and it must be **1.4.2** — Isaac Lab 2.1 uses the 1.x runner API, and 2.x fails
with `AttributeError: 'NoneType' object has no attribute 'shape'`.

```bash
cd /workspace/isaaclab
./isaaclab.sh -p -m pip install "skrl==1.4.2" pandas pytest
cd /workspace/aigp-sim && /workspace/isaaclab/isaaclab.sh -p -m pip install -e . --no-deps
python -c "import tasks"     # must succeed; pip's exit code is not enough
```

## 7. Smoke test, then train

Use the launcher rather than calling `train.py` directly. It runs the smoke
test as a gate, refuses to continue unless it passes, and sets the three
environment things whose absence looks like a crash:

```bash
cd /workspace/aigp-sim
bash scripts/pod/train_launch.sh Isaac-Drone-Hover-v0 hover01 --max_iterations 300
```

Hover first: it is the cage-test artifact and the racing seed. **Stop it well
short of convergence** — a converged station keeper has learned to damp all
motion, which is the opposite of racing. Then racing, warm-started from it:

```bash
bash scripts/pod/train_launch.sh Isaac-Drone-Racer-v0 race01 \
    --checkpoint /workspace/logs/skrl/drone_racer/<hover-run>/checkpoints/best_agent.pt
```

The policy weights transfer; the value head does not, being task-specific.

### Do not trust the exit code — of anything here

Two independent mechanisms throw it away, and they compound:

* `isaaclab.sh` swallows the exit code of the script it runs.
* `SimulationApp.close()` hard-exits, so `sys.exit(status)` never runs **and**
  Python's buffered stdout is discarded. A run that completed can leave a log
  containing only C++ warnings and a clean `0`, which is indistinguishable
  from the §8 startup segfault.

`smoke_test.py` therefore prints `SMOKE_RESULT=PASS|FAIL` on stdout, flushed.
Grep that. The launcher does, and treats a *missing* verdict as failure —
because a missing verdict means it died before finishing.

### Pull checkpoints off the pod, continuously

From your own machine, not the pod — the pod is behind NAT and cannot reach
you:

```bash
export POD_HOST=<ip> POD_PORT=<port> POD_KEY=~/.ssh/runpod_aigp
bash scripts/pod/pull_checkpoints.sh          # every 15 min until stopped
```

A pod with **no network volume is deleted** when the balance hits zero, and
the disk goes with it. There is no snapshot and no recovery. A checkpoint that
exists only on the pod is a checkpoint you can lose entirely.

---

## Quick reference

| Symptom | Cause |
|---|---|
| `vkCreateInstance: Found no drivers` | Wrong image — use `isaac-lab:2.1.0` |
| Segfault in `SimulationApp.__init__` | `ulimit -n` too low |
| Direct SSH refused, proxy works | No sshd in image (step 2) |
| `Your SSH client doesn't support PTY` | Auth succeeded; add `-tt` |
| Container dies after you kill something | PID 1's `&&` chain broke; never kill Kit |
| `git clone` hangs at 0% CPU | Private repo, no credentials — use tar (step 3) |
| `import isaacsim` hangs at 0% CPU | EULA prompt on stdin; set the env vars |
| `rsync: command not found` | Not in the image — use tar |
