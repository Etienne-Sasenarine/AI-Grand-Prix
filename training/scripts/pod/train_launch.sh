#!/bin/bash
# Launch a training run on a rented box, with every trap we have hit folded in.
#
# Usage, from anywhere on the pod:
#
#     bash scripts/pod/train_launch.sh Isaac-Drone-Hover-v0 hover01 [extra args...]
#     bash scripts/pod/train_launch.sh Isaac-Drone-Racer-v0 race01 \
#          --checkpoint /workspace/runs/hover01/.../checkpoints/best_agent.pt
#
# Deliberately not a Python entry point: the things that go wrong here go wrong
# before Python starts.
#
# See RUNPOD_SETUP.md for why each line is here. In short:
#   - a tmux server started from a 1024-fd shell caps every pane it spawns,
#     and Isaac segfaults in SimulationApp.__init__ on a low limit
#   - Isaac's SimulationApp.close() hard-exits, discarding buffered stdout and
#     any exit status, so nothing here may trust $?
#   - a pod with no network volume is deleted at zero balance, so a checkpoint
#     that exists only on the pod is a checkpoint you can lose

set -u

TASK="${1:?usage: train_launch.sh <task-id> <run-name> [train.py args...]}"
RUN_NAME="${2:?usage: train_launch.sh <task-id> <run-name> [train.py args...]}"
shift 2

REPO="${REPO:-/workspace/aigp-sim}"
ISAAC_PY="${ISAAC_PY:-/workspace/isaaclab/_isaac_sim/python.sh}"
NUM_ENVS="${NUM_ENVS:-4096}"
LOG="/workspace/logs/${RUN_NAME}.log"

mkdir -p /workspace/logs

# --- the checks that are cheaper than a wasted run --------------------------

if [ ! -x "$ISAAC_PY" ]; then
    echo "FATAL: no Isaac python at $ISAAC_PY" >&2
    exit 1
fi

HARD_LIMIT="$(ulimit -Hn)"
if [ "$HARD_LIMIT" != "unlimited" ] && [ "$HARD_LIMIT" -lt 65535 ]; then
    echo "FATAL: fd hard limit is $HARD_LIMIT, below the 65535 Isaac needs." >&2
    echo "       Raise it on the container, not here: a script cannot exceed it." >&2
    exit 1
fi
ulimit -n 65535

# A tmux server outlives the session that started it and hands its own limits
# to every pane. One started from an ordinary 1024-fd SSH session will cap this
# run no matter what the line above says, and the failure looks exactly like
# the startup segfault. Kill a stale server before the limit is raised, never
# after -- and never kill the streaming Kit app, which is PID 1's child and
# takes the container with it.
if tmux has-session 2>/dev/null; then
    echo "note: a tmux server is already running." >&2
    echo "      If this run dies ~7s in with no traceback, that server was" >&2
    echo "      started from a low-fd shell. 'tmux kill-server' then retry." >&2
fi

# --- environment ------------------------------------------------------------

export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
# Without these a crash is indistinguishable from a clean run that printed
# nothing, because the hard exit throws away whatever is still in the buffer.
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

cd "$REPO" || exit 1

# --- the gate ---------------------------------------------------------------
# Never start a multi-hour run on a plant that has not been checked. The exit
# status is not evidence -- Isaac hard-exits 0 regardless -- so grep the
# verdict the smoke test prints for exactly this reason.

echo "=== smoke test ==="
SMOKE_LOG="/workspace/logs/${RUN_NAME}_smoke.log"
"$ISAAC_PY" scripts/smoke_test.py --headless --num_envs 64 > "$SMOKE_LOG" 2>&1
VERDICT="$(grep -oE 'SMOKE_RESULT=(PASS|FAIL)' "$SMOKE_LOG" | tail -1)"

if [ "$VERDICT" != "SMOKE_RESULT=PASS" ]; then
    echo "FATAL: smoke test did not pass (got '${VERDICT:-no verdict at all}')." >&2
    echo "       A missing verdict means it died before finishing; see $SMOKE_LOG" >&2
    grep -E '\[FAIL\]' "$SMOKE_LOG" >&2
    exit 1
fi
echo "smoke test passed"

# --- launch -----------------------------------------------------------------

echo "=== training: $TASK as $RUN_NAME ==="
echo "log: $LOG"

tmux new-session -d -s "$RUN_NAME" \
    "cd '$REPO' && '$ISAAC_PY' scripts/rl/train.py \
        --task '$TASK' --headless --num_envs $NUM_ENVS $* \
        > '$LOG' 2>&1; echo TRAIN_EXITED >> '$LOG'"

sleep 5
if tmux has-session -t "$RUN_NAME" 2>/dev/null; then
    echo "launched in tmux session '$RUN_NAME'"
    echo "  watch:  tail -f $LOG"
    echo "  attach: tmux attach -t $RUN_NAME"
else
    echo "FATAL: session exited within 5s; see $LOG" >&2
    tail -20 "$LOG" >&2
    exit 1
fi
