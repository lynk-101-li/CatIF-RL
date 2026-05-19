#!/usr/bin/env bash
# Reconstruct the DLKcat-BRENDA + CATH dataset from raw inputs.
# Outputs: data/process/{train,valid,test}/*.pt graph tensors.

set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/lib_env.sh"
activate_env catif

BRENDA_RAW="${BRENDA_RAW:-$DATA_DIR/brenda/dlkcat_brenda_records.csv}"
CATH_BACKBONES_DIR="${CATH_BACKBONES_DIR:-$DATA_DIR/cath_v4_2_0}"
WORK_DIR="${WORK_DIR:-$DATA_DIR}"

cd "$REPO_ROOT"

# (1) Sequence-level filtering: 16,839 records -> 7,713 distinct enzymes.
python -m catif_rl.data.brenda \
  --records "$BRENDA_RAW" \
  --output "$WORK_DIR/brenda_filtered.csv" \
  --max-length 1180

# (2) ESMFold backbone prediction for each distinct sequence.
python -m catif_rl.data.esmfold_backbones \
  --fasta "$WORK_DIR/brenda_filtered.fasta" \
  --output-dir "$WORK_DIR/raw"

# (3) Train / validation / test split following the DLKcat strategy.
python -m catif_rl.data.splits \
  --distinct-enzymes "$WORK_DIR/brenda_filtered.csv" \
  --cath-backbones-dir "$CATH_BACKBONES_DIR" \
  --output-dir "$WORK_DIR/process" \
  --seed 1234 \
  --train-ratio 0.9

# (4) PDB -> graph .pt conversion for every split.
python -m catif_rl.data.graph_construction \
  --raw "$WORK_DIR/raw" \
  --out "$WORK_DIR/process"

echo "[build_dataset] Dataset ready under: $WORK_DIR/process"
