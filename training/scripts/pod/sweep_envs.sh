#!/bin/bash
# Sweep environment counts, one process per size, with a memory guard.
#
#     bash scripts/pod/sweep_envs.sh                    # 4096 8192 16384
#     SIZES="4096 8192" bash scripts/pod/sweep_envs.sh
#
# Two things this exists to avoid.
#
# **A second gym.make in one process is not a second measurement.** Building
# 8192 envs after 4096 in the same interpreter took over three minutes of CPU
# against nineteen seconds for the first, with GPU memory never reflecting the
# new size -- Isaac was working on a stage that still held the old one. Any
# number that came out of it would have been measured on a dirty card. One
# process per size costs an Isaac startup each (~40 s) and is worth it.
#
# **An out-of-memory in PhysX can take the container with it**, and on a
# volumeless pod that means losing the disk. So the sweep checks free VRAM
# before each size, refuses one it cannot afford, and stops rather than
# pushing into the next size after a failure.

set -u

REPO="${REPO:-/workspace/aigp-sim}"
ISAAC_PY="${ISAAC_PY:-/workspace/isaaclab/_isaac_sim/python.sh}"
TASK="${TASK:-Isaac-Drone-Racer-v0}"
SIZES="${SIZES:-4096 8192 16384}"
# Refuse a size unless this much VRAM is free. Generous on purpose: the cost of
# being wrong is the container, not a retry.
MIN_FREE_MB="${MIN_FREE_MB:-12000}"

ulimit -n 65535
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
cd "$REPO" || exit 1

echo "sweep: $TASK over [$SIZES]"
printf '\n  %6s  %8s  %13s  %8s\n' envs it/s env-steps/s "GPU MB"
echo "  ---------------------------------------------------"

prev_rate=0
for n in $SIZES; do
    free_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)"
    if [ "$free_mb" -lt "$MIN_FREE_MB" ]; then
        echo "  ${n}: SKIPPED, only ${free_mb} MB free (need ${MIN_FREE_MB})"
        break
    fi

    log="/tmp/sweep_${n}.log"
    "$ISAAC_PY" scripts/bench_throughput.py --headless \
        --task "$TASK" --counts "$n" > "$log" 2>&1

    row="$(tr '\r' '\n' < "$log" | grep -E "^ +${n} +[0-9.]+ +[0-9,]+" | tail -1)"
    if [ -z "$row" ]; then
        echo "  ${n}: FAILED or produced no row -- stopping. See $log"
        tr '\r' '\n' < "$log" | grep -iE "out of memory|CUDA error|FAILED|Error" | tail -3
        break
    fi

    rate="$(echo "$row" | awk '{gsub(/,/,"",$3); print $3}')"
    peak_mb="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)"
    gain=""
    [ "$prev_rate" != "0" ] && gain="$(awk -v a="$rate" -v b="$prev_rate" 'BEGIN{printf "%.2fx", a/b}')"
    printf '  %6s  %8s  %13s  %8s  %s\n' \
        "$n" "$(echo "$row" | awk '{print $2}')" "$(echo "$row" | awk '{print $3}')" \
        "$peak_mb" "$gain"
    prev_rate="$rate"

    # Let the driver fully release the card before the next build. Without the
    # pause the next process can see stale free-memory and start a size it
    # cannot actually fit.
    sleep 8
done

echo ""
echo "SWEEP_DONE"
