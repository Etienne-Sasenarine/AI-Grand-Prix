#!/bin/bash
# Fly a trained hover policy against plants it did not train on.
#
#     CKPT=/path/to/best_agent.pt bash scripts/pod/stress_sweep.sh
#
# One process per condition. Building a second environment in the same
# interpreter does not give a second clean measurement -- Isaac keeps working
# on a stage that still holds the first, which showed up in the env-count
# sweep as a build going from 19 s to over three minutes with GPU memory never
# reflecting the new size. Paying an Isaac startup per condition is the cost of
# numbers that mean something.
#
# The conditions are deliberately plausible rather than extreme. The question
# is "will the cage test go well", not "what arbitrary number breaks it".

set -u

REPO="${REPO:-/workspace/aigp-sim}"
ISAAC_PY="${ISAAC_PY:-/workspace/isaaclab/_isaac_sim/python.sh}"
CKPT="${CKPT:?set CKPT to a trained hover checkpoint}"
NUM_ENVS="${NUM_ENVS:-512}"
OUT="${OUT:-/workspace/logs/stress}"

# nominal is the control and must come first: every other row is only readable
# against it. A policy that scores 0.4 m median everywhere has not been
# perturbed, it is just mediocre.
CONDITIONS="${CONDITIONS:-nominal mass=1.15 mass=0.90 hover=0.19 hover=0.31 delay=4 tau=0.060 wind=0.6}"

mkdir -p "$OUT"
ulimit -n 65535
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
cd "$REPO" || exit 1

echo "stress sweep over: $CKPT"
printf '\n  %-16s %9s %9s %11s %9s\n' condition survived settled median_err p95_err
echo "  -------------------------------------------------------------"

for cond in $CONDITIONS; do
    log="${OUT}/$(echo "$cond" | tr '=.,' '___').log"
    "$ISAAC_PY" scripts/diag/stress_hover.py --headless \
        --checkpoint "$CKPT" --condition "$cond" --num_envs "$NUM_ENVS" \
        > "$log" 2>&1

    line="$(grep -oE 'STRESS .*survived=[0-9.]+ settled=[0-9.]+ median_err=[0-9.]+ p95_err=[0-9.]+' "$log" | tail -1)"
    if [ -z "$line" ]; then
        printf '  %-16s %9s  (no result -- see %s)\n' "$cond" FAILED "$log"
        grep -iE "error|traceback|out of memory" "$log" | tail -2
        continue
    fi

    printf '  %-16s %9s %9s %11s %9s\n' "$cond" \
        "$(echo "$line" | grep -oE 'survived=[0-9.]+' | cut -d= -f2)" \
        "$(echo "$line" | grep -oE 'settled=[0-9.]+'  | cut -d= -f2)" \
        "$(echo "$line" | grep -oE 'median_err=[0-9.]+' | cut -d= -f2)" \
        "$(echo "$line" | grep -oE 'p95_err=[0-9.]+' | cut -d= -f2)"
    sleep 5
done

echo ""
echo "STRESS_SWEEP_DONE"
