#!/usr/bin/env bash
# Fetch the large assets that are deliberately NOT stored in git history.
#
# What IS in git (do not fetch here): the trained model weights are small
# enough (6-13 MB) to live in the tree and are versioned with the code:
#   - perception/models/*.pt, *.onnx   (YOLO gate-pose detectors)
#   - training/checkpoints/*.pt         (race40 / race40drop PPO policies)
#   - deploy/runtime/models/*.npz       (onboard NumPy policy)
#
# What this script fetches (too big / not source, kept in GitHub Releases):
#   1. Isaac Sim drone + gate USD assets  -> training/assets/  (currently
#      git-LFS pointers; the 98 MB base USD was never uploaded to LFS)
#   2. Roboflow gate-pose dataset          -> perception/frames/
#   3. Flight telemetry bags               -> data/bags/
#
# Fill in the TODO URLs once the assets are attached to a GitHub Release, e.g.
#   https://github.com/Etienne-Sasenarine/AI-Grand-Prix/releases/download/v0.1-assets/<file>
#
# Usage:
#   bash tools/download_models.sh            # fetch everything with a URL set
#   bash tools/download_models.sh assets     # fetch a single group
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# fetch <url> <dest-path>   — skips if url is a TODO or the file already exists.
fetch() {
  local url="$1" dest="$2"
  if [[ "$url" == TODO* || -z "$url" ]]; then
    echo "SKIP  $dest  (no URL yet — set it in tools/download_models.sh)"
    return 0
  fi
  if [[ -e "$ROOT/$dest" && -s "$ROOT/$dest" ]]; then
    echo "HAVE  $dest"
    return 0
  fi
  echo "GET   $dest"
  mkdir -p "$ROOT/$(dirname "$dest")"
  curl -fL --retry 3 -o "$ROOT/$dest" "$url"
}

# unpack <url> <dest-dir>   — for tarball/zip archives (datasets, bags).
unpack() {
  local url="$1" dest="$2"
  if [[ "$url" == TODO* || -z "$url" ]]; then
    echo "SKIP  $dest/  (no URL yet — set it in tools/download_models.sh)"
    return 0
  fi
  echo "GET   $dest/  (archive)"
  mkdir -p "$ROOT/$dest"
  local tmp; tmp="$(mktemp)"
  curl -fL --retry 3 -o "$tmp" "$url"
  case "$url" in
    *.zip) unzip -o "$tmp" -d "$ROOT/$dest" ;;
    *)     tar -xf "$tmp" -C "$ROOT/$dest" ;;
  esac
  rm -f "$tmp"
}

group="${1:-all}"

# ------------------------------------------------------------------ 1. assets
# The USD assets in training/assets are git-LFS pointers. If you have access to
# the LFS remote that holds them, the simplest path is:  git lfs pull
# Otherwise fetch the packaged copies from a Release:
if [[ "$group" == "all" || "$group" == "assets" ]]; then
  fetch "TODO_5_in_drone_base_usd_url"  "training/assets/5_in_drone/configuration/5_in_drone_base.usd"
  fetch "TODO_gate_aigp_usd_url"        "training/assets/gate/gate_aigp.usd"
fi

# ---------------------------------------------------------------- 2. dataset
if [[ "$group" == "all" || "$group" == "dataset" ]]; then
  unpack "TODO_roboflow_gate_pose_dataset_zip_url"  "perception/frames"
fi

# ------------------------------------------------------------------- 3. bags
if [[ "$group" == "all" || "$group" == "bags" ]]; then
  unpack "TODO_flight_bags_tar_url"  "data/bags"
fi

echo "Done. Set any TODO_* URLs above to enable those downloads."
