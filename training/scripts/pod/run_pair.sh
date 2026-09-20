#!/bin/bash
# Racing and converged hover at the same time, on one card.
#
#     SEED_CKPT=/path/to/hover/best_agent.pt bash scripts/pod/run_pair.sh
#
# Both are deliverables: racing is the competition policy, converged hover is
# the cage deployment test. Neither needs the other to exist first, so they can
# share the card.
#
# The cost of running them together is that hover can no longer warm-start from
# racing -- racing does not exist yet. It continues from the short hover policy
# instead, which is simply "train hover longer". What that gives up is the
# untested hypothesis that racing's wider exposure to the randomised plant
# would transfer; cheap to give up, since it was never measured.
#
# Why two and not four. A single 4096-env job sits at 62-83% GPU utilisation
# and adds ~6.3 GB on top of the 3.5 GB the Kit streaming app holds, so memory
# is nowhere near the constraint -- 16 GB of 47 for the pair. Compute is: the
# env-count sweep showed throughput saturating near 90k env-steps/s, so a
# second job pushes the card toward full while a third would mostly slow the
# first two down. Expect each to run slower than it would alone; the pair still
# finishes sooner than running them back to back.

set -u

REPO="${REPO:-/workspace/aigp-sim}"
ISAAC_PY="${ISAAC_PY:-/workspace/isaaclab/_isaac_sim/python.sh}"
NUM_ENVS="${NUM_ENVS:-4096}"
RACE_ITERS="${RACE_ITERS:-2083}"     # 50000 timesteps / 24 rollouts, config default
HOVER_ITERS="${HOVER_ITERS:-1200}"   # 4x the seed run
SEED_CKPT="${SEED_CKPT:?set SEED_CKPT to the short hover policy}"
LOGS="/workspace/logs"
# Refuse to launch without this much VRAM free. The cost of being wrong here is
# a dead container on a pod with no volume, so the margin is deliberately fat.
MIN_FREE_MB="${MIN_FREE_MB:-15000}"

mkdir -p "$LOGS"
ulimit -n 65535

if [ ! -f "$SEED_CKPT" ]; then
    echo "FATAL: no seed checkpoint at $SEED_CKPT" >&2
    exit 1
fi

launch() {
    local name="$1" task="$2" iters="$3"; shift 3
    local free_mb
    free_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)"
    if [ "$free_mb" -lt "$MIN_FREE_MB" ]; then
        echo "  ${name}: REFUSED, only ${free_mb} MB free (need ${MIN_FREE_MB})"
        return 1
    fi

    tmux new-session -d -s "$name" \
        "cd '$REPO' && \
         OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y \
         PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 PYTHONPATH='$REPO' \
         '$ISAAC_PY' scripts/rl/train.py --task '$task' \
            --headless --num_envs $NUM_ENVS --max_iterations $iters $* \
            > '${LOGS}/${name}.log' 2>&1; echo TRAIN_EXITED >> '${LOGS}/${name}.log'"
    echo "  ${name}: launched (${free_mb} MB was free)"
}

echo "pair: ${NUM_ENVS} envs each"
launch race01 Isaac-Drone-Racer-v0 "$RACE_ITERS" --checkpoint "$SEED_CKPT" || exit 1

# Stagger. Two Isaac processes building a 4096-env scene simultaneously contend
# hard on the CPU side and both take longer than either would alone. The
# training phases overlap fine; the construction phases do not.
echo "  waiting 120s before the second launch (scene construction contends)"
sleep 120

launch hover_converged Isaac-Drone-Hover-v0 "$HOVER_ITERS" --checkpoint "$SEED_CKPT" || {
    echo "  hover refused; racing continues alone"
}

sleep 15
echo ""
for s in race01 hover_converged; do
    tmux has-session -t "$s" 2>/dev/null && echo "  ${s}: alive" || echo "  ${s}: NOT RUNNING"
done
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
echo "PAIR_LAUNCHED"
