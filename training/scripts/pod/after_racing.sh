#!/bin/bash
# Wait for the racing A/B to finish, then converge hover and stress it.
#
#     nohup bash scripts/pod/after_racing.sh > /workspace/logs/after_racing.log 2>&1 &
#
# Runs unattended. Each stage checks the previous one actually produced
# something rather than trusting an exit code -- Isaac hard-exits 0 whatever
# happened, so a chain that trusted $? would train the next stage on a
# checkpoint that was never written.
#
# Hover converges *from racing* here, which is the ordering the pair could not
# have: racing now exists, and it has seen far more of the randomised plant
# than the seven-minute seed policy did. Whether that actually transfers is
# untested -- it is a hypothesis, and the stress sweep afterwards is what would
# show it up.

set -u

REPO="${REPO:-/workspace/aigp-sim}"
ISAAC_PY="${ISAAC_PY:-/workspace/isaaclab/_isaac_sim/python.sh}"
NUM_ENVS="${NUM_ENVS:-4096}"
HOVER_ITERS="${HOVER_ITERS:-1200}"
LOGS="/workspace/logs"
RUNS="${REPO}/logs/skrl/drone_racer"

ulimit -n 65535
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
cd "$REPO" || exit 1

say() { echo "[$(date +%H:%M:%S)] $*"; }

say "waiting for race_warm and race_scratch"
while tmux has-session -t race_warm 2>/dev/null || tmux has-session -t race_scratch 2>/dev/null; do
    sleep 60
done
say "both racing runs finished"

# Pick the better arm on final mean reward. This is also the A/B result, so it
# gets printed rather than just used -- with one seed per arm it is indicative
# and not conclusive, and a small gap should be read as no result.
say "comparing the two arms"
BETTER="$("$ISAAC_PY" - <<'PY' 2>/dev/null
from tensorboard.backend.event_processing import event_accumulator
import glob, os

best, best_val = None, None
for p in glob.glob("logs/skrl/drone_racer/*/events.out.tfevents.*"):
    run = os.path.dirname(p)
    ea = event_accumulator.EventAccumulator(p, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags()["scalars"]
    key = next((t for t in tags if "Total reward (mean)" in t), None)
    if not key:
        continue
    vals = ea.Scalars(key)
    if len(vals) < 20:          # skip the short hover seed run
        continue
    final = sum(v.value for v in vals[-10:]) / 10.0
    print(f"#  {os.path.basename(run):<40} final_mean_reward={final:9.3f}", file=__import__("sys").stderr)
    if best_val is None or final > best_val:
        best, best_val = run, final

if best:
    ckpt = os.path.join(best, "checkpoints", "best_agent.pt")
    if os.path.exists(ckpt):
        print(ckpt)
PY
)"

if [ -z "$BETTER" ]; then
    say "!! could not identify a racing checkpoint -- stopping"
    exit 1
fi
say "warm-starting hover from: $BETTER"

say "=== hover_converged: ${HOVER_ITERS} iters"
"$ISAAC_PY" scripts/rl/train.py --task Isaac-Drone-Hover-v0 \
    --headless --num_envs "$NUM_ENVS" --max_iterations "$HOVER_ITERS" \
    --checkpoint "$BETTER" > "${LOGS}/hover_converged.log" 2>&1

HOVER_CKPT="$(find "$RUNS" -name 'best_agent.pt' -newer "$BETTER" -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn | head -1 | cut -d' ' -f2-)"
if [ -z "$HOVER_CKPT" ]; then
    say "!! hover produced no checkpoint -- stopping"
    tr '\r' '\n' < "${LOGS}/hover_converged.log" | grep -iE "error|traceback|out of memory" | tail -5
    exit 1
fi
say "hover converged -> $HOVER_CKPT"

say "=== stress sweep"
CKPT="$HOVER_CKPT" bash scripts/pod/stress_sweep.sh > "${LOGS}/stress.log" 2>&1
say "stress results:"
grep -E "^  [a-z]" "${LOGS}/stress.log" | sed 's/^/  /'

say "AFTER_RACING_DONE"
say "  racing: $BETTER"
say "  hover:  $HOVER_CKPT"
