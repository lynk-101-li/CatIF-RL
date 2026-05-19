#!/usr/bin/env bash
# Supervised CatIF training: from scratch on the 6,034 activity-positive
# variants emitted by 03_run_gdc.sh. Mirrors Cat-IF2/src/run_grdif0924.sh
# (--Date Sep24) but driven by catif_rl/config/catif.yaml.

set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/lib_env.sh"
activate_env catif

CONFIG="${CONFIG:-$CONFIG_DIR/catif.yaml}"
DEVICE="${DEVICE:-cuda:0}"

cd "$REPO_ROOT"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -m catif_rl.training.train_supervised \
  --config "$CONFIG"

echo "[train_catif] Finished. Checkpoints under runs/catif/"
