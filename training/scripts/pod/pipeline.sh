#!/bin/bash
# The full training sequence, unattended.
#
#     NUM_ENVS=8192 bash scripts/pod/pipeline.sh
#
# Three artifacts, in an order that matters:
#
#   1. racing, warm-started from the short hover policy
#   2. hover-converged, warm-started from *racing*
#   3. a stress sweep over the converged hover
#
# Why hover appears twice. A converged station keeper has learned to damp all
# motion, which is the opposite of what racing needs -- so the policy racing
# seeds from must be stopped early, while it is still plastic. That objection
# only applies to the checkpoint racing inherits. Once racing exists, a
# converged hover costs it nothing: it is a separate artifact for a separate
# job, the cage deployment test. Training it *from* racing rather than from
# scratch starts it off a policy that has already seen far more of the
# randomised plant.
#
# Each stage checks the previous one produced something before proceeding, so
# a failure stops the chain rather than quietly training the next stage on a
# checkpoint that was never written.

set -u

REPO="${REPO:-/workspace/aigp-sim}"
ISAAC_PY="${ISAAC_PY:-/workspace/isaaclab/_isaac_sim/python.sh}"
NUM_ENVS="${NUM_ENVS:-4096}"
RACE_ITERS="${RACE_ITERS:-2083}"      # 50000 timesteps / 24 rollouts, the config default
HOVER_ITERS="${HOVER_ITERS:-1200}"    # converged: 4x the seed run
LOGS="/workspace/logs"
RUNS="${REPO}/logs/skrl/drone_racer"

mkdir -p "$LOGS"
ulimit -n 65535
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
cd "$REPO" || exit 1

say() { echo "[$(date +%H:%M:%S)] $*"; }

newest_checkpoint() {
    # Most recently modified best_agent.pt under the run root. Runs are named
    # by timestamp, but sorting by name breaks the moment a run is resumed, so
    # sort by mtime instead.
    find "$RUNS" -name 'best_agent.pt' -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn | head -1 | cut -d' ' -f2-
}

run_stage() {
    local task="$1" name="$2" iters="$3" ckpt="${4:-}"
    local log="${LOGS}/${name}.log"
    local args=(--task "$task" --headless --num_envs "$NUM_ENVS" --max_iterations "$iters")
    [ -n "$ckpt" ] && args+=(--checkpoint "$ckpt")

    say "=== ${name}: ${task}, ${iters} iters, ${NUM_ENVS} envs"
    [ -n "$ckpt" ] && say "    warm start: ${ckpt}"

    "$ISAAC_PY" scripts/rl/train.py "${args[@]}" > "$log" 2>&1

    # Never trust the exit code: Isaac hard-exits 0 whatever happened. Judge by
    # whether a checkpoint newer than this stage's start actually exists.
    local produced
    produced="$(newest_checkpoint)"
    if [ -z "$produced" ]; then
        say "!! ${name} produced no checkpoint -- stopping the chain"
        tr '\r' '\n' < "$log" | grep -iE "error|traceback|out of memory" | tail -5
        return 1
    fi
    say "    ${name} done -> ${produced}"
    printf '%s' "$produced"
}

say "pipeline starting, ${NUM_ENVS} envs"

SEED="$(newest_checkpoint)"
if [ -z "$SEED" ]; then
    say "!! no hover seed checkpoint found under $RUNS"
    exit 1
fi
say "hover seed: $SEED"

RACE_CKPT="$(run_stage Isaac-Drone-Racer-v0 race01 "$RACE_ITERS" "$SEED" | tail -1)" || exit 1
HOVER_CKPT="$(run_stage Isaac-Drone-Hover-v0 hover_converged "$HOVER_ITERS" "$RACE_CKPT" | tail -1)" || exit 1

say "=== stress sweep over the converged hover"
CKPT="$HOVER_CKPT" bash scripts/pod/stress_sweep.sh > "${LOGS}/stress.log" 2>&1
say "    stress done -- results:"
grep -E "^  [a-z]" "${LOGS}/stress.log" | sed 's/^/    /'

say "PIPELINE_DONE"
say "  racing:  $RACE_CKPT"
say "  hover:   $HOVER_CKPT"
