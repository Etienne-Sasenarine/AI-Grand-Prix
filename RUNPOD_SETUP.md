# Running this on a rented GPU box

Written after a first attempt on RunPod that got as far as a fully installed
Isaac Sim and then could not launch it. Every heading below is something that
actually went wrong on 20 Sep 2026, in the order it bit, so the next attempt
can skip the evening.

**The short version:** deploy from the Isaac Sim container, not a PyTorch
template. Almost everything below is a downstream consequence of not doing
that.

---

## 0. The one that killed the run: `NVIDIA_DRIVER_CAPABILITIES`

Isaac Sim needs a working Vulkan device **even headless**. Kit initialises an
RTX renderer at startup whether or not you ask it to draw anything.

RunPod's PyTorch template starts the container with compute capabilities only.
The NVIDIA graphics libraries are all present — `libGLX_nvidia.so.0`,
`libnvidia-glcore`, `libnvidia-rtcore`, `libnvoptix` — and `libGLX_nvidia.so.0`
even exports `vk_icdGetInstanceProcAddr`. But the driver cannot initialise a
graphics context, and Vulkan reports:

```
ERROR: vkCreateInstance: Found no drivers!
ERROR: vkCreateInstance failed with ERROR_INCOMPATIBLE_DRIVER
```

**This is not fixable from inside the container.** Things that do not work:

| Attempt | Result |
|---|---|
| Write `/usr/share/vulkan/icd.d/nvidia_icd.json` | ICD is found, driver still fails to init |
| `mknod /dev/nvidia0 c 195 6` (GPU was `/dev/nvidia6`, enumerates as index 0) | `Operation not permitted` — container is unprivileged |
| Set `NVIDIA_DRIVER_CAPABILITIES=all` in the shell | Read by nvidia-container-runtime at **creation**, not at runtime |

**Check this first, before installing anything:**

```bash
vulkaninfo --summary 2>&1 | grep -iE "deviceName|ERROR"
echo "caps=${NVIDIA_DRIVER_CAPABILITIES:-unset}"
```

A `deviceName` line means you are fine. `Found no drivers` means stop and
redeploy — nothing installed afterwards will run.

**Fix:** deploy from `nvcr.io/nvidia/isaac-sim:4.5.0` (needs a free NGC account
and its API key as registry credentials), or any template that sets
`NVIDIA_DRIVER_CAPABILITIES` to include `graphics`.

---

## 1. Python 3.12 has no Isaac Sim 4.5

RunPod's PyTorch template ships Python 3.12 on Ubuntu 24.04. Isaac Sim 4.5's
wheels are **cp310 only**:

```
$ pip index versions isaacsim --extra-index-url https://pypi.nvidia.com
Available versions: 6.1.0.0, 6.0.1.0, 6.0.0.1, 6.0.0.0     # no 4.5.0
```

Taking Isaac Sim 6.x instead is not a free upgrade: this repo targets Isaac Lab
2.1 APIs, and 6.x wants a newer Isaac Lab whose manager APIs differ.

**Fix:** Miniconda with `python=3.10`, or use the Isaac Sim container which
already has the right interpreter.

## 2. Conda's default channels need a Terms-of-Service acceptance

```
CondaToSNonInteractiveError: Terms of Service have not been accepted for
the following channels: https://repo.anaconda.com/pkgs/main ...
```

**Fix:** use conda-forge and skip Anaconda's channels entirely. This also
sidesteps Anaconda's commercial-use terms, which are murky for a university
team:

```bash
conda config --system --add channels conda-forge
conda config --system --remove channels defaults
conda config --system --set channel_priority strict
conda create -y -n aigp python=3.10 -c conda-forge --override-channels
```

## 3. `import isaacsim` hangs forever on the EULA

The worst failure of the night because it looks like a slow download. The
process sat at **0.0% CPU, 0:00.01 total CPU time, 12 MB resident, sleeping in
`wait_woken`** for ten minutes. It was blocked reading stdin:

```
Do you accept the EULA? (Yes/No):
```

`ISAACSIM.md` warns about this and the install script did not set it.

**Fix**, and persist it so every tmux session and training run inherits it:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
echo 'export OMNI_KIT_ACCEPT_EULA=YES' >> /root/.bashrc
```

Also run installers with `< /dev/null` so anything else that wants stdin fails
loudly instead of hanging.

**How to tell a hang from work:** check CPU time, not elapsed time.

```bash
PID=$(pgrep -f "[i]mport isaacsim"); ps -o time=,etime= -p $PID; cat /proc/$PID/wchan
```

## 4. A private repo makes `git clone` hang, not fail

`git clone` of a private GitHub repo with no credentials blocks waiting for a
username. It sat at 0 CPU for nearly four minutes with 84 KB in `.git`.

**Fix:** skip credentials entirely and push the working tree from the machine
that already has it:

```bash
rsync -az --exclude='__pycache__' --exclude='.pytest_cache' \
  -e "ssh -i ~/.ssh/runpod_aigp -p <PORT>" \
  ~/dev/aigp-sim/ root@<HOST>:/workspace/aigp-sim/
```

This also carries the Git LFS payload already resolved, so no `git lfs pull` and
no chance of the 98 MB drone USD arriving as a pointer. Verify:

```bash
head -c 12 assets/5_in_drone/configuration/5_in_drone_base.usd | grep -q git-lfs \
  && echo "POINTER - broken" || echo "real USD"
```

## 5. The repo must be importable

```
ModuleNotFoundError: No module named 'tasks'
```

**Fix:** `pip install -e . --no-deps`, or set `PYTHONPATH=/workspace/aigp-sim`.
Note `pip install -e .` may print `No module named 'omni.kit'` while still
succeeding — check `python -c "import tasks"` rather than trusting pip's exit.

## 6. RunPod injects SSH keys at pod **start**

A key added to a pod that is already running does not reach its
`authorized_keys`. Symptom: `ssh.runpod.io` (the proxy, which authenticates
against account keys) works, while the direct TCP route refuses the same key.

The proxy only gives an interactive shell — `ssh ... 'command'` returns
`Error: Your SSH client doesn't support PTY`, which confusingly means
**authentication succeeded**.

**Fix without restarting** (restarting risks losing a scarce GPU slot) — pipe
the key in through the proxy:

```bash
printf 'mkdir -p ~/.ssh && echo "%s" >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys\nexit\n' \
  "$(cat ~/.ssh/runpod_aigp.pub)" \
  | ssh -tt -i ~/.ssh/runpod_aigp <podid>@ssh.runpod.io
```

Then the direct TCP route works, and with it `scp` and `rsync`.

---

## Mistakes that were mine, not the platform's

- **`pkill -f "import isaacsim"`** matched the shell running the command that
  contained that string, and killed its own SSH session. Output came back empty
  and the step silently did nothing. Use a bracket to break self-matching:
  `pkill -f "[i]mport isaacsim"`.
- **`rsync --info=progress2`** does not exist in macOS's rsync 2.6.9.
- **Talking ourselves out of the Isaac Sim container.** It was the original
  recommendation, dropped because it needed NGC credentials — "three extra
  steps at 1am". That traded three setup steps for five compatibility problems
  and, ultimately, a pod that could not run the simulator at all.

## Billing traps

- RunPod's credit presets start at **$150** and **$200 is preselected**. Click
  "Other" and type the amount, or you buy $200 by reflex.
- **Auto-Pay must read `Disabled`** before loading credit. It is off by default
  but check it, because it is the only path to exceeding a prepaid balance.
- **Container disk defaults to 30 GB.** Isaac Sim alone is ~10 GB of wheels and
  more once extracted; use **100 GB+**. The cost is about $0.013/hr, roughly 13
  cents a night.
- A pod with **no network volume is terminated** at zero balance and its data is
  unrecoverable. Sync checkpoints off-box on a timer.

---

## The order that should work

```bash
# 0. FIRST, before anything else
vulkaninfo --summary 2>&1 | grep -iE "deviceName|ERROR"   # must show a device

# 1. system bits the template lacks
apt-get update -qq && apt-get install -y git-lfs libvulkan1 vulkan-tools libglu1-mesa
export OMNI_KIT_ACCEPT_EULA=YES
echo 'export OMNI_KIT_ACCEPT_EULA=YES' >> /root/.bashrc

# 2. python 3.10 (skip if using the Isaac Sim container)
# ... conda-forge env as in section 2 ...

# 3. the stack, pinned
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
pip install "isaacsim[all,extscache]==4.5.0" --extra-index-url https://pypi.nvidia.com
python -c "import isaacsim; print('ok')" < /dev/null      # must return in seconds
git clone --depth 1 --branch v2.1.0 https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab && ./isaaclab.sh --install none < /dev/null
pip install "skrl==1.4.2" pandas pytest

# 4. this repo, by rsync (private)
#    then:
pip install -e . --no-deps && python -c "import tasks; print('ok')"

# 5. the gate
python scripts/smoke_test.py --headless          # must exit 0
```
