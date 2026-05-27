#!/usr/bin/env bash
# Thin runner shim around upstream CataPro's `inference/predict.py`.
#
# This shim is shipped by the CatIF-RL package (i.e. tracked here, not
# in the upstream clone) so that the same interface is available across
# clones / forks and so that the output filename matches the manuscript
# convention <prefix>_kcatpred_catapro.csv (shared with DLKcat and UniKP).
#
# Usage:
#   bash _run_catapro.sh \
#     --input  <ProID,ProSeq,ProSeq',SMILES>.csv \
#     --prefix <output_stem> \
#     --repo   <path-to-CataPro-master> \
#     --mode   benchmark|rl
#
# On success:
#   $repo/inference/results/<prefix>_kcatpred_catapro.csv
#
# Implementation notes:
#   - The two preprocess passes generate WT and MUT views of the input
#     via upstream `data_src/table_trans_input.py`.
#   - predict.py is invoked twice (WT, MUT) with the upstream's standard
#     hyperparameters (batch 64, fp16, cuda:0 -- override by editing
#     this shim if you are running on CPU).
#   - The two predictions are then merged via the upstream's
#     `table_trans_out.py` (benchmark) or `table_trans_out_rl.py` (rl);
#     the only difference is whether the substrate column is grouped
#     against the manuscript's WT-aggregated reference.

set -euo pipefail

INPUT=""
PREFIX=""
REPO=""
MODE="benchmark"
DEVICE="cuda:0"
BATCH=64

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)  INPUT="$2";  shift 2;;
    --prefix) PREFIX="$2"; shift 2;;
    --repo)   REPO="$2";   shift 2;;
    --mode)   MODE="$2";   shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --batch)  BATCH="$2";  shift 2;;
    -h|--help)
      sed -n '1,30p' "$0"
      exit 0
      ;;
    *)  echo "[_run_catapro][ERROR] unknown arg: $1" >&2; exit 2;;
  esac
done
[[ -z "$INPUT"  ]] && { echo "[_run_catapro][ERROR] --input is required"  >&2; exit 2; }
[[ -z "$PREFIX" ]] && { echo "[_run_catapro][ERROR] --prefix is required" >&2; exit 2; }
[[ -z "$REPO"   ]] && { echo "[_run_catapro][ERROR] --repo is required"   >&2; exit 2; }
[[ ! -d "$REPO" ]] && { echo "[_run_catapro][ERROR] --repo not a dir: $REPO" >&2; exit 2; }
[[ ! -f "$INPUT" ]] && { echo "[_run_catapro][ERROR] --input not a file: $INPUT" >&2; exit 2; }

case "$MODE" in
  benchmark) MERGE_SCRIPT="data_src/table_trans_out.py"    ; MERGE_EXTRA="--base_no_group" ;;
  rl)        MERGE_SCRIPT="data_src/table_trans_out_rl.py" ; MERGE_EXTRA="" ;;
  *)         echo "[_run_catapro][ERROR] --mode must be benchmark|rl, got: $MODE" >&2; exit 2;;
esac

INPUT_ABS="$(cd "$(dirname "$INPUT")" && pwd)/$(basename "$INPUT")"

cd "$REPO/inference"
mkdir -p dataset/process results

WT_INP="dataset/process/${PREFIX}_wt_inp.csv"
MUT_INP="dataset/process/${PREFIX}_mut_inp.csv"
WT_OUT="results/${PREFIX}_wt_pred.csv"
MUT_OUT="results/${PREFIX}_mut_pred.csv"
FINAL_OUT="results/${PREFIX}_kcatpred_catapro.csv"

echo "[_run_catapro] [1/5] preprocess WT  -> $WT_INP"
python data_src/table_trans_input.py \
  --in_csv  "$INPUT_ABS" \
  --out_csv "$WT_INP" \
  --seq_col "ProSeq" \
  --no_group

echo "[_run_catapro] [2/5] preprocess MUT -> $MUT_INP"
python data_src/table_trans_input.py \
  --in_csv  "$INPUT_ABS" \
  --out_csv "$MUT_INP" \
  --no_group

echo "[_run_catapro] [3/5] predict WT -> $WT_OUT"
python predict.py \
  -inp_fpath   "$WT_INP" \
  -model_dpath "../models" \
  -batch_size  "$BATCH" \
  -device      "$DEVICE" \
  --pin_memory --num_workers 4 --fp16 \
  -out_fpath   "$WT_OUT"

echo "[_run_catapro] [4/5] predict MUT -> $MUT_OUT"
python predict.py \
  -inp_fpath   "$MUT_INP" \
  -model_dpath "../models" \
  -batch_size  "$BATCH" \
  -device      "$DEVICE" \
  --pin_memory --num_workers 4 --fp16 \
  -out_fpath   "$MUT_OUT"

echo "[_run_catapro] [5/5] merge -> $FINAL_OUT"
python "$MERGE_SCRIPT" \
  --orig_csv "$INPUT_ABS" \
  --wt_csv   "$WT_OUT" \
  --mut_csv  "$MUT_OUT" \
  --out_csv  "$FINAL_OUT" \
  $MERGE_EXTRA

echo "[_run_catapro] done: $FINAL_OUT"
