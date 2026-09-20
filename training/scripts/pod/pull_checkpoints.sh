#!/bin/bash
# Pull training checkpoints off a rented pod, on a loop, from your own machine.
#
#     scripts/pod/pull_checkpoints.sh                  # loop until interrupted
#     ONCE=1 scripts/pod/pull_checkpoints.sh           # single pass
#
# Configure by environment, so no host or key ever lands in the repo:
#
#     export POD_HOST=195.26.233.93 POD_PORT=35432
#     export POD_KEY=~/.ssh/runpod_aigp
#     export LOCAL_DIR=~/dev/aigp-runs
#
# Why this direction. A pod with no network volume is **deleted** when the
# balance reaches zero, and its disk goes with it -- there is no snapshot and
# no recovery. A checkpoint that exists only on the pod is a checkpoint you can
# lose, so the copy has to be somewhere the pod cannot take with it.
#
# It pulls rather than the pod pushing because the pod cannot reach your
# machine: it is behind NAT with no inbound route. The laptop has to do the
# asking.
#
# tar over ssh rather than rsync: rsync is not in the Isaac Lab image, and
# installing it is one more thing to go wrong at 2am.

set -u

POD_HOST="${POD_HOST:?set POD_HOST to the pod's address}"
POD_PORT="${POD_PORT:-22}"
POD_KEY="${POD_KEY:-$HOME/.ssh/runpod_aigp}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/aigp-sim/logs}"
LOCAL_DIR="${LOCAL_DIR:-$HOME/dev/aigp-runs}"
INTERVAL_S="${INTERVAL_S:-900}"

SSH=(ssh -i "$POD_KEY" -p "$POD_PORT"
     -o StrictHostKeyChecking=no
     -o ConnectTimeout=15
     -o ServerAliveInterval=20
     "root@${POD_HOST}")

mkdir -p "$LOCAL_DIR"

pull_once() {
    local stamp
    stamp="$(date '+%Y-%m-%d %H:%M:%S')"

    if ! "${SSH[@]}" "test -d '$REMOTE_DIR'" 2>/dev/null; then
        echo "[$stamp] pod unreachable or $REMOTE_DIR missing"
        return 1
    fi

    # Checkpoints only. The rest of a run directory is tensorboard event files
    # that grow without bound and are not what you would be sad to lose.
    local before after
    before="$(find "$LOCAL_DIR" -name '*.pt' 2>/dev/null | wc -l | tr -d ' ')"

    if "${SSH[@]}" "cd '$REMOTE_DIR' && tar czf - \
            \$(find . -name '*.pt' -o -name 'params' -type d) 2>/dev/null" \
            2>/dev/null | tar xzf - -C "$LOCAL_DIR" 2>/dev/null; then
        after="$(find "$LOCAL_DIR" -name '*.pt' 2>/dev/null | wc -l | tr -d ' ')"
        local newest
        newest="$(find "$LOCAL_DIR" -name '*.pt' -exec ls -t {} + 2>/dev/null | head -1)"
        echo "[$stamp] ok: $after checkpoints locally (+$((after - before)))"
        [ -n "$newest" ] && echo "            newest: ${newest#"$LOCAL_DIR"/}"
        return 0
    fi

    echo "[$stamp] transfer failed"
    return 1
}

echo "pulling ${POD_HOST}:${REMOTE_DIR} -> ${LOCAL_DIR}"
if [ -n "${ONCE:-}" ]; then
    pull_once
    exit $?
fi

echo "every ${INTERVAL_S}s; Ctrl-C to stop"
fails=0
while true; do
    if pull_once; then
        fails=0
    else
        fails=$((fails + 1))
        # Say it plainly rather than letting failures scroll past. Repeated
        # failures are how you find out the pod is already gone.
        if [ "$fails" -ge 3 ]; then
            echo "  !! $fails consecutive failures -- is the pod still alive?"
        fi
    fi
    sleep "$INTERVAL_S"
done
