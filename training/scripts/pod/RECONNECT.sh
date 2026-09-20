#!/bin/bash
# Run this when the laptop is back online. One command, no arguments.
#
#     bash ~/dev/aigp-sim/scripts/pod/RECONNECT.sh
#
# Prints what happened while you were away, then retrieves the results bundle.
# Safe to run repeatedly, and safe to run before the pipeline has finished --
# it will say so rather than pretending.

set -u

POD_HOST="${POD_HOST:-195.26.233.93}"
POD_PORT="${POD_PORT:-35432}"
POD_KEY="${POD_KEY:-$HOME/.ssh/runpod_aigp}"
LOCAL_DIR="${LOCAL_DIR:-$HOME/dev/aigp-runs}"

SSH=(ssh -i "$POD_KEY" -p "$POD_PORT" -o StrictHostKeyChecking=no
     -o ConnectTimeout=20 "root@${POD_HOST}")

echo "== pod reachable? =="
if ! "${SSH[@]}" true 2>/dev/null; then
    echo "  NO. Either the network is still down, or the pod was terminated"
    echo "  (which happens at zero balance, and takes the disk with it)."
    echo "  Anything already in ${LOCAL_DIR} is still yours:"
    find "$LOCAL_DIR" -name '*.pt' 2>/dev/null | wc -l | xargs echo "   checkpoints:"
    exit 1
fi
echo "  yes"

echo
echo "== what is still running =="
"${SSH[@]}" 'tmux ls 2>/dev/null || echo "  nothing"'

echo
echo "== summary =="
"${SSH[@]}" 'cat /workspace/SUMMARY.txt 2>/dev/null || echo "  not generated yet -- pipeline still running"'

echo
echo "== retrieving =="
mkdir -p "$LOCAL_DIR"

if "${SSH[@]}" 'test -f /workspace/RESULTS.tar.gz' 2>/dev/null; then
    # The bundle is small and complete; prefer it over walking the log tree.
    "${SSH[@]}" 'cat /workspace/RESULTS.tar.gz' 2>/dev/null \
        | tar xzf - -C "$LOCAL_DIR" 2>/dev/null \
        && echo "  bundle extracted to ${LOCAL_DIR}" \
        || echo "  bundle transfer failed"
else
    echo "  no bundle yet; pulling checkpoints directly instead"
    "${SSH[@]}" "tar czf - -C /workspace/aigp-sim/logs/skrl --exclude='events.out.tfevents.*' ." \
        2>/dev/null | tar xzf - -C "$LOCAL_DIR" 2>/dev/null \
        && echo "  checkpoints pulled" || echo "  pull failed"
fi

echo
echo "  total local: $(find "$LOCAL_DIR" -name '*.pt' 2>/dev/null | wc -l | tr -d ' ') checkpoints, $(du -sh "$LOCAL_DIR" 2>/dev/null | cut -f1)"
echo
echo "The pod bills whether or not it is doing anything. If the pipeline has"
echo "finished and the results are down, terminate it from the RunPod console."
