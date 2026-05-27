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
# Per SI Section 2.7: the structural-evaluation subset is computed on a
# single pre-specified seed (the first entry of the benchmark seed list,
# i.e. 1111). Exposed as STRUCT_SEED so users can override deliberately;
# the published default matches the seed used in SI Tables S7 / S8.
STRUCT_SEED="${STRUCT_SEED:-${BENCHMARK_SEEDS[0]}}"

cd "$REPO_ROOT"
mkdir -p "$SCORE_DIR"

for method_dir in "$BENCH_DIR"/*/; do
  method=$(basename "$method_dir")
  echo "[score_benchmark] method=$method"
  method_score_dir="$SCORE_DIR/$method"
  mkdir -p "$method_score_dir"

  # (1) substrate match + DLKcat scoring for every seed.
  for seed in "${BENCHMARK_SEEDS[@]}"; do
    seed_score_dir="$method_score_dir/seed_${seed}"
    mkdir -p "$seed_score_dir"

    activate_env catif
    python -m catif_rl.reward.substrate_match \
      "$TEMPLATE_CSV" \
      "$method_dir/seed_${seed}" \
      "$seed_score_dir/test_mut_substrate_seed${seed}.csv"

    # The wrapper copies the upstream output to $seed_score_dir/dlkcat_pred.csv,
    # which is the canonical filename build_master --score-dir mode expects.
    activate_env dlkcat
    python -c "from catif_rl.reward.predictors import dlkcat; dlkcat.predict('$seed_score_dir/test_mut_substrate_seed${seed}.csv', mode='benchmark', output_dir='$seed_score_dir')"
  done

  # (2) Structural evaluation on the pre-specified structural seed (per SI §2.7).
  activate_env esmfold
  python -m catif_rl.data.esmfold_backbones \
    --fasta "$method_dir/seed_${STRUCT_SEED}" \
    --output-dir "$method_score_dir/refold_seed${STRUCT_SEED}"

  activate_env catif
  python -m catif_rl.evaluation.structural \
    --ref-dir "$TEST_REF_DIR" \
    --pred-dir "$method_score_dir/refold_seed${STRUCT_SEED}" \
    --csv-out "$method_score_dir/rmsd_plddt.csv" \
    --metrics rmsd,plddt
done

# (3) Build master_per_protein.csv across all methods.
activate_env catif
python -m catif_rl.evaluation.build_master \
  --score-dir "$SCORE_DIR" \
  --output    "$SCORE_DIR/master_per_protein.csv"

# (4) Compute Tables S7 / S8 / S9 / S10 (CI + paired Wilcoxon + BH-FDR + SR@delta).
mkdir -p "$SCORE_DIR/tables"
python -m catif_rl.evaluation.statistics \
  --master     "$SCORE_DIR/master_per_protein.csv" \
  --output-dir "$SCORE_DIR/tables"

echo "[score_benchmark] Scores + statistics under: $SCORE_DIR"
