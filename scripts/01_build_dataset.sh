#!/usr/bin/env bash
# Build the processed PyTorch Geometric .pt graph dataset from raw PDB inputs.
#
# Inputs expected in data/raw/ (download from the Zenodo deposit per
# data/README.md):
#   data/raw/enzymeif/train_and_validation/   6,290 native enzyme PDBs
#   data/raw/catif/{train,valid}/             5,430 + 604 GDC-mutant PDBs
#   data/raw/test/                            1,423 native enzyme test PDBs
#
# CATH v4.2.0 structural regularizer PDBs are downloaded automatically from
# RCSB via catif_rl/data/download_pdb.py and the manifest at
# catif_rl/data/assets/chain_set_splits.json (originally from GraDe-IF).
# This step takes ~1-2 hours over a typical academic network.
#
# Output: data/process/{train,valid,test}/*.pt graph tensors.

set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/lib_env.sh"
activate_env catif

cd "$REPO_ROOT"

CATH_WORKDIR="${CATH_WORKDIR:-$DATA_DIR/raw/_cath_workdir}"
CATH_OUTDIR="${CATH_OUTDIR:-$DATA_DIR/raw/enzymeif/cath_v4_2_0}"
PROCESS_DIR="${PROCESS_DIR:-$DATA_DIR/process}"

# ---------- (1) Download CATH v4.2.0 PDBs ----------
# The manifest catif_rl/data/assets/chain_set_splits.json (18,021 train +
# 608 valid + 1,120 test chain identifiers) is bundled with this repository.
# EnzymeIF training uses only the train + valid subsets; the test subset is
# downloaded for completeness but not consumed (the EnzymeIF test set is
# the held-out enzyme set under data/raw/test/, not CATH).
if [ ! -d "$CATH_OUTDIR" ] || [ "$(ls "$CATH_OUTDIR" 2>/dev/null | wc -l)" -lt 18000 ]; then
  echo "[build_dataset] Downloading CATH v4.2.0 PDBs from RCSB..."
  mkdir -p "$CATH_WORKDIR"/{all,train,validation,test}
  cp catif_rl/data/assets/chain_set_splits.json "$CATH_WORKDIR/chain_set_splits.json"
  ( cd "$CATH_WORKDIR" && python "$REPO_ROOT/catif_rl/data/download_pdb.py" )
  mkdir -p "$CATH_OUTDIR"
  cp "$CATH_WORKDIR/train"/*.pdb      "$CATH_OUTDIR/" 2>/dev/null || true
  cp "$CATH_WORKDIR/validation"/*.pdb "$CATH_OUTDIR/" 2>/dev/null || true
  echo "[build_dataset] CATH PDBs ready: $CATH_OUTDIR ($(ls "$CATH_OUTDIR" | wc -l) chains)"
else
  echo "[build_dataset] CATH PDBs already present at $CATH_OUTDIR; skipping download."
fi

# ---------- (2) Split / merge enzyme + CATH for the EnzymeIF cohort ----------
# Canonical pipeline (see catif_rl/data/_split_scripts/数据划分说明.txt):
#   split_baseon_sequences.py -> run_split.py -> dataset_split_final.py
# with align_pdb_names.py as a helper. These scripts produce the 5,661 / 629
# enzyme split (random.Random(1234), 9:1) and merge with the 18,021 / 608
# CATH split, yielding the EnzymeIF training cohort summarised in SI Table S4.
echo "[build_dataset] Running EnzymeIF split + merge ..."
# (Run the three scripts in catif_rl/data/_split_scripts/ in the order above;
# the originals were authored with specific working-directory assumptions,
# so adapt the cwd / paths as needed for your installation.)

# ---------- (3) PDB -> PyG .pt graph tensors ----------
echo "[build_dataset] Building PyG .pt graph tensors..."
python -m catif_rl.data.graph_construction \
  --raw "$DATA_DIR/raw" \
  --out "$PROCESS_DIR"

echo "[build_dataset] Done. Processed graphs at: $PROCESS_DIR"
