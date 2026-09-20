#!/bin/bash
# Keep racing training going, resuming from its own best checkpoint each time,
# until told to stop.
#
# Stop it cleanly with:   touch /workspace/STOP_RACING
# That lets the current leg finish and write its checkpoints rather than
# killing training mid-update.
#
# Each leg resumes from the *racing* run with the newest checkpoint, verified
# by reading its config -- a hover run finishing in between would otherwise be
# the newest directory, and the chain would silently continue racing from a
# station-keeping policy. That has already happened once today.
#
# Note what a resume is and is not: skrl restarts the timestep counter, so the
# KL-adaptive learning rate begins fresh each leg rather than continuing to
# anneal. That keeps the policy plastic, which is what we want while it is
# still improving, but it means N legs are not the same as one run of N times
# the length.
set -u

REPO=/workspace/aigp-sim
ISAAC_PY=/workspace/isaaclab/_isaac_sim/python.sh
ITERS="${ITERS:-2083}"
MAX_LEGS="${MAX_LEGS:-9999}"          # no ceiling: runs until STOP_RACING
STOP=/workspace/STOP_RACING
LOGS=/workspace/logs

rm -f "$STOP"
ulimit -n 65535
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 PYTHONPATH=$REPO
cd "$REPO" || exit 1

say(){ echo "[$(date -u +%H:%M)] $*"; }

newest_racing_ckpt(){
  # newest best_agent.pt whose run is a RACING run
  for d in $(ls -dt logs/skrl/drone_racer/*/ 2>/dev/null); do
    [ -f "${d}checkpoints/best_agent.pt" ] || continue
    grep -qa "settled" "${d}params/env.yaml" 2>/dev/null && continue   # hover
    echo "${d}checkpoints/best_agent.pt"; return 0
  done
  return 1
}

for leg in $(seq 1 "$MAX_LEGS"); do
  if [ -f "$STOP" ]; then say "STOP file present; chain ending after $((leg-1)) legs"; break; fi

  # wait for any training already on the card to finish
  while pgrep -f "kit/python/bin/python3 scripts/rl/train.py --task Isaac-Drone-Racer" >/dev/null; do sleep 30; done

  CKPT="$(newest_racing_ckpt)" || { say "no racing checkpoint found; stopping"; break; }
  say "=== leg $leg/$MAX_LEGS, resuming from $CKPT"

  "$ISAAC_PY" scripts/rl/train.py --task Isaac-Drone-Racer-v0 --headless \
      --num_envs 4096 --max_iterations "$ITERS" --checkpoint "$CKPT" \
      > "${LOGS}/race_leg${leg}.log" 2>&1

  NEW="$(newest_racing_ckpt)"
  if [ "$NEW" = "$CKPT" ]; then
    say "!! leg $leg wrote no new checkpoint -- stopping rather than looping on a failure"
    tr '\r' '\n' < "${LOGS}/race_leg${leg}.log" | grep -aiE "error|traceback" | tail -3
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
  cp "$NEW" "/workspace/archive/race_leg${leg}_$(date -u +%H%M).pt" 2>/dev/null || true
  say "    leg $leg done -> $NEW"
done
say "RACE_CHAIN_ENDED"
