#!/usr/bin/env bash
# Supervised pretraining of EnzymeIF on the DLKcat-BRENDA train split plus
# CATH v4.2.0 regularizers. Mirrors EnzymeIF-main/src/run_grid0601.sh
# (--Date Jul01) but driven by catif_rl/config/enzymeif.yaml.

set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/lib_env.sh"
activate_env catif

CONFIG="${CONFIG:-$CONFIG_DIR/enzymeif.yaml}"
DEVICE="${DEVICE:-cuda:0}"

cd "$REPO_ROOT"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -m catif_rl.training.train_supervised \
  --config "$CONFIG"

echo "[train_enzymeif] Finished. Checkpoints under runs/enzymeif/"
