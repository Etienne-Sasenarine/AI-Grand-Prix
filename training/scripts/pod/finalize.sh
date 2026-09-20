#!/bin/bash
# Wait for the pipeline to finish, then bundle everything worth keeping.
#
#     nohup bash scripts/pod/finalize.sh > /workspace/logs/finalize.log 2>&1 &
#
# Exists because the laptop that pulls checkpoints is not always connected, and
# a pod with no network volume is deleted at zero balance with its disk. The
# full log tree is a few hundred MB of tensorboard events; this reduces the
# irreplaceable part to one small tarball retrievable over a bad connection in
# a single command.
#
# In:  the best checkpoint of every run, the params needed to reload them, the
#      A/B comparison, the stress results, and a plain-text summary.
# Out: intermediate checkpoints and tensorboard events -- large, and
#      reproducible from the checkpoints that are kept.

set -u

REPO="${REPO:-/workspace/aigp-sim}"
RUNS="logs/skrl/drone_racer"          # relative to REPO, so tar paths stay tidy
LOGS="/workspace/logs"
OUT="/workspace/RESULTS.tar.gz"
SUMMARY="${LOGS}/SUMMARY.txt"

cd "$REPO" || exit 1

say() { echo "[$(date -u +%H:%M)] $*"; }

say "waiting for the pipeline"
while tmux has-session -t race_warm 2>/dev/null \
   || tmux has-session -t race_scratch 2>/dev/null \
   || tmux has-session -t chain 2>/dev/null; do
    sleep 60
done
say "pipeline finished"

write_summary() {
    echo "AIGP training results"
    echo "generated $(date -u '+%Y-%m-%d %H:%M UTC')"
    echo
    echo "== runs =="
    for d in "${RUNS}"/*/; do
        [ -d "${d}checkpoints" ] || continue
        printf '  %-34s %s checkpoints\n' \
            "$(basename "$d")" "$(find "${d}checkpoints" -name '*.pt' | wc -l | tr -d ' ')"
    done
    echo
    echo "== A/B: racing warm-start vs scratch =="
    grep -aE "final_mean_reward|warm-starting hover" "${LOGS}/after_racing.log" 2>/dev/null \
        | sed 's/^/  /' || echo "  (not available)"
    echo
    echo "== stress sweep: converged hover =="
    if [ -f "${LOGS}/stress.log" ]; then
        grep -aE "condition|^  [a-z]" "${LOGS}/stress.log" | sed 's/^/  /'
    else
        echo "  (not run)"
    fi
    echo
    echo "== errors found in logs =="
    local found=0
    for f in "${LOGS}"/*.log; do
        [ -f "$f" ] || continue
        c="$(grep -acE 'Traceback|out of memory|CUDA error' "$f" 2>/dev/null)"
        if [ "${c:-0}" -gt 0 ]; then
            printf '  %-28s %s error lines\n' "$(basename "$f")" "$c"
            found=1
        fi
    done
    [ "$found" = 0 ] && echo "  none"
}

write_summary > "$SUMMARY"
cp "$SUMMARY" /workspace/SUMMARY.txt

say "bundling"
LIST="$(mktemp)"
trap 'rm -f "$LIST"' EXIT

# Best checkpoint per run, plus the params directory each one needs to reload.
find "${RUNS}" -name 'best_agent.pt' >> "$LIST"
find "${RUNS}" -type d -name 'params' >> "$LIST"

if [ ! -s "$LIST" ]; then
    say "!! nothing to bundle -- no checkpoints found under ${RUNS}"
    exit 1
fi

# The summary lives outside REPO, so it is added by absolute path separately.
tar czf "$OUT" -C "$REPO" -T "$LIST" \
                -C /workspace SUMMARY.txt 2>/dev/null

if gzip -t "$OUT" 2>/dev/null; then
    say "bundle ok: $(du -h "$OUT" | cut -f1), $(wc -l < "$LIST" | tr -d ' ') paths"
else
    say "!! bundle failed verification"
    exit 1
fi

say "FINALIZE_DONE"
say "  bundle:  $OUT"
say "  summary: /workspace/SUMMARY.txt"
