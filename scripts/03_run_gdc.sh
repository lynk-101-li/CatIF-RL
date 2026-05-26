#!/usr/bin/env bash
# Generative Dataset Curation (manuscript §2.3): take ~62,900 EnzymeIF-sampled
# candidates, apply the ESMFold structural gate (RMSD<4 A, pLDDT>90), then the
# three-predictor activity ensemble. Retain ~6,034 activity-positive variants
# for downstream CatIF supervised training (the exact count varies by random
# seed; the paper reports 15,188 structure-valid -> 6,034 activity-positive).

set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/lib_env.sh"

ENZYMEIF_CKPT="${ENZYMEIF_CKPT:-$CKPT_DIR/enzymeif_Jul01_epoch467.pt}"
PAIRS_CSV="${PAIRS_CSV:-$DATA_DIR/brenda/brenda_train_and_dev_set.csv}"
WORK_DIR="${WORK_DIR:-$RUNS_DIR/gdc}"
SAMPLE_SEED="${SAMPLE_SEED:-11}"
GROUP_SIZE="${GROUP_SIZE:-10}"                 # K=10 per backbone (paper)
STEP="${STEP:-100}"
RMSD_THRESHOLD="${RMSD_THRESHOLD:-4.0}"
PLDDT_THRESHOLD="${PLDDT_THRESHOLD:-90.0}"
ENSEMBLE_THRESHOLD="${ENSEMBLE_THRESHOLD:-0.0}"

CANDIDATES="$WORK_DIR/gdc_candidates.csv"
STRUCTURAL_METRICS="$WORK_DIR/gdc_structural_metrics.csv"
REFOLD_DIR="$WORK_DIR/refold"
PRED_DIR="$WORK_DIR/predictions"

cd "$REPO_ROOT"
mkdir -p "$WORK_DIR" "$REFOLD_DIR" "$PRED_DIR"

# Stage 1: bulk sampling from EnzymeIF (K samples per training backbone).
activate_env catif
python -m catif_rl.sampling.batch \
  --condition-dirs "$DATA_DIR/process/train" \
  --pairs-csv      "$PAIRS_CSV" \
  --ckpt-path      "$ENZYMEIF_CKPT" \
  --group-size     "$GROUP_SIZE" \
  --step           "$STEP" \
  --seed           "$SAMPLE_SEED" \
  --diverse \
  --out-csv        "$CANDIDATES"

# Stage 2: ESMFold refold every candidate, compute pLDDT and backbone RMSD.
activate_env esmfold
python -m catif_rl.data.esmfold_backbones \
  --fasta      "$CANDIDATES" \
  --output-dir "$REFOLD_DIR"

activate_env catif
python -m catif_rl.evaluation.structural \
  --ref-dir  "$DATA_DIR/raw/enzymeif/train_and_validation" \
  --pred-dir "$REFOLD_DIR" \
  --csv-out  "$STRUCTURAL_METRICS" \
  --metrics  rmsd,plddt

# Stage 3: score with each predictor (DLKcat, UniKP, CataPro). The wrappers
# write to $PRED_DIR by default (see catif_rl.reward.predictors.*); a
# subsequent commit will guarantee this even when an upstream wrapper would
# otherwise dump under external/<repo>/...
activate_env dlkcat
python -c "from catif_rl.reward.predictors import dlkcat; dlkcat.predict('$CANDIDATES', mode='benchmark', output_dir='$PRED_DIR')"
activate_env unikp
python -c "from catif_rl.reward.predictors import unikp; unikp.predict('$CANDIDATES', mode='benchmark', output_dir='$PRED_DIR')"
activate_env catapro
python -c "from catif_rl.reward.predictors import catapro; catapro.predict('$CANDIDATES', mode='benchmark', output_dir='$PRED_DIR')"

# Stage 4: run the GDC funnel (structural filter + ensemble normalizer).
activate_env catif
python -m catif_rl.reward.gdc \
  --candidates         "$CANDIDATES" \
  --structural-metrics "$STRUCTURAL_METRICS" \
  --dlkcat-csv         "$PRED_DIR/dlkcat_pred.csv" \
  --unikp-csv          "$PRED_DIR/unikp_pred.csv" \
  --catapro-csv        "$PRED_DIR/catapro_pred.csv" \
  --output-dir         "$WORK_DIR" \
  --rmsd-threshold     "$RMSD_THRESHOLD" \
  --plddt-threshold    "$PLDDT_THRESHOLD" \
  --ensemble-threshold "$ENSEMBLE_THRESHOLD"

echo "[run_gdc] Activity-positive variants under: $WORK_DIR/activity_positive.csv"
