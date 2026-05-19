#!/usr/bin/env bash
# Score the benchmark FASTA outputs from 06_sample_benchmark.sh:
#   1. Match each mutant to its substrate (catif_rl.reward.substrate_match).
#   2. Run DLKcat to obtain Δlog k_cat per (mutant, substrate).
#   3. ESMFold-refold a single-seed structural-evaluation subset.
#   4. Compute Recovery, pLDDT, backbone RMSD, ΔlgKcat, SR@δ.
#   5. Run paired Wilcoxon + BH-FDR + bootstrap statistics.

set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/lib_env.sh"

BENCH_DIR="${BENCH_DIR:-$RUNS_DIR/benchmark}"
TEST_REF_DIR="${TEST_REF_DIR:-$DATA_DIR/raw/test}"
TEMPLATE_CSV="${TEMPLATE_CSV:-$DATA_DIR/brenda/test_mut_substrate_template.csv}"
SCORE_DIR="${SCORE_DIR:-$RUNS_DIR/benchmark_scores}"
STRUCT_SEED=1   # SI Section 2.7: structural metrics on a single pre-specified seed

cd "$REPO_ROOT"
mkdir -p "$SCORE_DIR"

for method_dir in "$BENCH_DIR"/*/; do
  method=$(basename "$method_dir")
  echo "[score_benchmark] method=$method"
  method_score_dir="$SCORE_DIR/$method"
  mkdir -p "$method_score_dir"

  # (1) substrate match + DLKcat scoring for every seed.
  for seed in "${BENCHMARK_SEEDS[@]}"; do
    activate_env catif
    python -m catif_rl.reward.substrate_match \
      "$TEMPLATE_CSV" \
      "$method_dir/seed_${seed}" \
      "$method_score_dir/test_mut_substrate_seed${seed}.csv"

    activate_env dlkcat
    python -c "from catif_rl.reward.predictors import dlkcat; dlkcat.predict('$method_score_dir/test_mut_substrate_seed${seed}.csv', mode='benchmark')"
  done

  # (2) Structural evaluation on the first benchmark seed (per SI §2.7).
  activate_env esmfold
  python -m catif_rl.data.esmfold_backbones \
    --fasta "$method_dir/seed_${BENCHMARK_SEEDS[0]}" \
    --output-dir "$method_score_dir/refold_seed${BENCHMARK_SEEDS[0]}"

  activate_env catif
  python -m catif_rl.evaluation.structural \
    --ref-dir "$TEST_REF_DIR" \
    --pred-dir "$method_score_dir/refold_seed${BENCHMARK_SEEDS[0]}" \
    --csv-out "$method_score_dir/rmsd_plddt.csv" \
    --metrics rmsd,plddt
done

# (3) Build master_per_protein.csv across all methods.
activate_env catif
python -m catif_rl.evaluation.build_master \
  --score-dir "$SCORE_DIR" \
  --output "$SCORE_DIR/master_per_protein.csv"

# (4) Compute Tables S7 / S8 / S9 (CI + paired Wilcoxon + threshold sensitivity).
python -m catif_rl.evaluation.statistics \
  --master "$SCORE_DIR/master_per_protein.csv" \
  --output-dir "$SCORE_DIR/tables"

echo "[score_benchmark] Scores + statistics under: $SCORE_DIR"
