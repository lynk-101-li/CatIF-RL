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

# ---------- (2) PDB -> PyG .pt graph tensors (per-cohort) ----------
# `catif_rl.data.graph_construction` operates one directory at a time. We run
# it once per raw PDB source; the .pt tensors mirror the raw layout under
# data/process/. After this step the on-disk structure is:
#
#   data/process/enzymeif/train_and_validation/  6,290 .pt  <-- RL/GDC cond pool
#   data/process/enzymeif/cath_v4_2_0/           18,629 .pt <-- CATH regularizers
#   data/process/catif/train/                    5,430 .pt  <-- GDC train mutants
#   data/process/catif/valid/                    604 .pt    <-- GDC valid mutants
#   data/process/test/                           1,423 .pt  <-- held-out test
#
# The split step below (3) then writes the supervised final partition to
#   data/process/{train,valid}/    (EnzymeIF cohort 9:1 split + CATH merge)
# leaving the per-cohort directories above untouched as stable RL/GDC inputs.
echo "[build_dataset] Step 2: PDB -> .pt graph tensors per cohort"

for src_subdir in \
    "enzymeif/train_and_validation" \
    "enzymeif/cath_v4_2_0" \
    "catif/train" \
    "catif/valid" \
    "test"; do
  in_dir="$DATA_DIR/raw/$src_subdir"
  out_dir="$PROCESS_DIR/$src_subdir"
  if [ ! -d "$in_dir" ]; then
    echo "[build_dataset]   skipping $src_subdir (no $in_dir)"; continue
  fi
  echo "[build_dataset]   $src_subdir: $in_dir -> $out_dir"
  python -m catif_rl.data.graph_construction \
    --input-dir  "$in_dir" \
    --output-dir "$out_dir"
done

# ---------- (3) Split / merge enzyme + CATH for the EnzymeIF cohort ----------
# Canonical pipeline (see catif_rl/data/_split_scripts/README_split_workflow.txt):
#   align_pdb_names.py -> split_baseon_sequences.py -> run_split.py
#   -> dataset_split_final.py
# Produces the 5,661 / 629 enzyme split (random.Random(1234), 9:1) and merges
# with the 18,021 / 608 CATH split, yielding the EnzymeIF training cohort
# summarised in SI Table S4.
#
# The four legacy scripts have working-directory assumptions baked into their
# defaults, so we expose them through catif_rl.data.splits which calls each
# subprocess in order. Tune --pdb-dir / --train-csv / --dev-csv / --test-csv
# to match wherever your BRENDA pre-processing put those files (defaults from
# DLKcat live under data/brenda/).
echo "[build_dataset] Step 3: EnzymeIF split + merge"
python -m catif_rl.data.splits \
  --pdb-dir        "$DATA_DIR/raw/enzymeif/train_and_validation" \
  --train-csv      "$DATA_DIR/brenda/Brenda_dataset_split/train_set.csv" \
  --dev-csv        "$DATA_DIR/brenda/Brenda_dataset_split/dev_set.csv" \
  --test-csv       "$DATA_DIR/brenda/Brenda_dataset_split/test_set.csv" \
  --cath-train-dir "$PROCESS_DIR/enzymeif/cath_v4_2_0" \
  --cath-valid-dir "$PROCESS_DIR/enzymeif/cath_v4_2_0" \
  --output-dir     "$PROCESS_DIR"

echo "[build_dataset] Done. Processed graphs at: $PROCESS_DIR"
