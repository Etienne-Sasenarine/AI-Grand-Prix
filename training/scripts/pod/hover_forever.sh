#!/bin/bash
# Keep hover training going, resuming from its own best checkpoint each time.
#
# Stop with:   touch /workspace/STOP_HOVER
#
# Mirror of the racing chain, with the run-type test inverted: a hover run's
# dumped config contains "settled" and a racing one does not. Picking by mtime
# alone already handed a racing resume a hover checkpoint once today, so the
# type is read rather than assumed -- in both directions.
set -u

REPO=/workspace/aigp-sim
ISAAC_PY=/workspace/isaaclab/_isaac_sim/python.sh
ITERS="${ITERS:-800}"
STOP=/workspace/STOP_HOVER
LOGS=/workspace/logs

rm -f "$STOP"
ulimit -n 65535
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 PYTHONPATH=$REPO
cd "$REPO" || exit 1

say(){ echo "[$(date -u +%H:%M)] $*"; }

newest_hover_ckpt(){
  for d in $(ls -dt logs/skrl/drone_racer/*/ 2>/dev/null); do
    [ -f "${d}checkpoints/best_agent.pt" ] || continue
    grep -qa "settled" "${d}params/env.yaml" 2>/dev/null || continue   # must BE hover
    echo "${d}checkpoints/best_agent.pt"; return 0
  done
  return 1
}

for leg in $(seq 1 9999); do
  [ -f "$STOP" ] && { say "STOP file present; ending after $((leg-1)) legs"; break; }
  while pgrep -f "[t]rain.py --task Isaac-Drone-Hover" >/dev/null; do sleep 30; done

  CKPT="$(newest_hover_ckpt)" || { say "no hover checkpoint; stopping"; break; }
  say "=== hover leg $leg, resuming from $CKPT"

  "$ISAAC_PY" scripts/rl/train.py --task Isaac-Drone-Hover-v0 --headless \
      --num_envs 4096 --max_iterations "$ITERS" --checkpoint "$CKPT" \
      > "${LOGS}/hover_leg${leg}.log" 2>&1

  NEW="$(newest_hover_ckpt)"
  if [ "$NEW" = "$CKPT" ]; then
    say "!! leg $leg wrote no new checkpoint -- stopping"
    tr '\r' '\n' < "${LOGS}/hover_leg${leg}.log" | grep -aiE "error|traceback" | tail -3
    break
  fi

  # Keep every leg's best, because resuming does not.
  #
  # skrl initialises checkpoint_best_modules["reward"] to -2**31 in
  # Agent.__init__ and load() never restores it, so each leg's best_agent.pt
  # is the best *within that leg* with no memory of what the parent scored.
  # The chain resumes from the newest run, not the best one -- so a leg that
  # comes out uniformly worse is still what the next leg builds on. That is a
  # random walk, not a ratchet. Archiving costs 9 MB a leg and means a good
  # policy can always be recovered by hand.
  mkdir -p /workspace/archive
  cp "$NEW" "/workspace/archive/hover_leg${leg}_$(date -u +%H%M).pt" 2>/dev/null || true
  say "    hover leg $leg done -> $NEW"
done
say "HOVER_CHAIN_ENDED"
