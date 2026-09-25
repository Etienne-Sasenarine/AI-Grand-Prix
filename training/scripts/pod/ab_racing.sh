#!/bin/bash
# Does the hover warm-start actually help? Two racing runs, concurrently.
#
#     SEED_CKPT=/path/to/hover/best_agent.pt bash scripts/pod/ab_racing.sh
#
# The premise came from another team: train hover first, then racing, and the
# racing policy learns faster. Plausible -- a policy that already knows how to
# hold attitude is not starting from noise -- but it has never been tested on
# this stack, and warm-starting is not free. The value head does not transfer,
# so racing begins with a critic that is confidently wrong about a task it has
# never seen, and that can be worse than starting clean.
#
# The two runs differ in exactly one thing: whether --checkpoint is passed. Same
# environment count, same iteration budget, same config.
#
# Run concurrently, because the sweep showed throughput saturating near 90k
# env-steps/s while a single 4096-env job sits at 62-83% utilisation and 9.8 GB
# of 47. Two jobs fit with room to spare and push the card closer to full,
# which is a better use of it than one job with twice the environments: four
# times the envs bought only 1.47x the throughput and cost 2.7x the wall clock
# per gradient update.

set -u

REPO="${REPO:-/workspace/aigp-sim}"
ISAAC_PY="${ISAAC_PY:-/workspace/isaaclab/_isaac_sim/python.sh}"
NUM_ENVS="${NUM_ENVS:-4096}"
ITERS="${ITERS:-2083}"          # 50000 timesteps / 24 rollouts, the config default
SEED_CKPT="${SEED_CKPT:?set SEED_CKPT to the short hover policy}"
LOGS="/workspace/logs"

mkdir -p "$LOGS"
ulimit -n 65535

launch() {
    local name="$1"; shift
    tmux new-session -d -s "$name" \
        "cd '$REPO' && \
         OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y \
         PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 \
         PYTHONPATH='$REPO' \
         '$ISAAC_PY' scripts/rl/train.py --task Isaac-Drone-Racer-v0 \
            --headless --num_envs $NUM_ENVS --max_iterations $ITERS $* \
            > '${LOGS}/${name}.log' 2>&1; echo TRAIN_EXITED >> '${LOGS}/${name}.log'"
}

echo "A/B racing: ${ITERS} iters, ${NUM_ENVS} envs each"
echo "  race_warm    warm-started from ${SEED_CKPT}"
echo "  race_scratch no checkpoint"

launch race_warm --checkpoint "$SEED_CKPT"

# Stagger the second launch. Two Isaac processes building a 4096-env scene at
# the same moment contend hard on the CPU side and both take longer than either
# would alone; the training phase overlaps fine, the construction phase does
# not.
sleep 90
launch race_scratch

sleep 10
for s in race_warm race_scratch; do
    tmux has-session -t "$s" 2>/dev/null && echo "  ${s}: launched" \
        || echo "  ${s}: FAILED TO LAUNCH -- see ${LOGS}/${s}.log"
done
echo "AB_LAUNCHED"
