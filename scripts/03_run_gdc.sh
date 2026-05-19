#!/usr/bin/env bash
# Generative Dataset Curation: take 62,900 EnzymeIF-sampled candidates,
# apply ESMFold structural filter (RMSD<4 A, pLDDT>90), then the three-
# predictor activity ensemble. Retain the activity-positive set (~6,034
# variants) for downstream CatIF supervised training.

set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/lib_env.sh"

CANDIDATES="${CANDIDATES:-$RUNS_DIR/enzymeif/gdc_candidates.csv}"
STRUCTURAL_METRICS="${STRUCTURAL_METRICS:-$RUNS_DIR/enzymeif/gdc_structural_metrics.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUNS_DIR/gdc}"

cd "$REPO_ROOT"

# Stage 1: bulk sampling from EnzymeIF (10 per training backbone).
activate_env catif
python -m catif_rl.sampling.batch \
  --condition-dir "$DATA_DIR/process/train" \
  --ckpt "$CKPT_DIR/enzymeif_Jul01_epoch467.pt" \
  --group-size 10 \
  --seed 11 \
  --diverse \
  --output "$CANDIDATES"

# Stage 2: ESMFold refold every candidate, compute pLDDT and backbone RMSD.
activate_env esmfold
python -m catif_rl.data.esmfold_backbones \
  --fasta "$CANDIDATES" \
  --output-dir "$RUNS_DIR/gdc/refold"

activate_env catif
python -m catif_rl.evaluation.structural \
  --ref-dir "$DATA_DIR/raw/train" \
  --pred-dir "$RUNS_DIR/gdc/refold" \
  --csv-out "$STRUCTURAL_METRICS" \
  --metrics rmsd,plddt

# Stage 3: score with each predictor (DLKcat, UniKP, CataPro), normalise, ensemble.
activate_env dlkcat
python -c "from catif_rl.reward.predictors import dlkcat; dlkcat.predict('$CANDIDATES', mode='benchmark')"
activate_env unikp
python -c "from catif_rl.reward.predictors import unikp; unikp.predict('$CANDIDATES', mode='benchmark')"
activate_env catapro
python -c "from catif_rl.reward.predictors import catapro; catapro.predict('$CANDIDATES', mode='benchmark')"

# Stage 4: run the GDC funnel (structural filter + ensemble normaliser).
activate_env catif
python -m catif_rl.reward.gdc \
  --candidates "$CANDIDATES" \
  --structural-metrics "$STRUCTURAL_METRICS" \
  --output-dir "$OUTPUT_DIR" \
  --rmsd-threshold 4.0 \
  --plddt-threshold 90.0 \
  --ensemble-threshold 0.0

echo "[run_gdc] Activity-positive variants under: $OUTPUT_DIR/activity_positive.csv"
