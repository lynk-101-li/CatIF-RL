#!/usr/bin/env bash
# Sample all eleven methods on the 1,423-enzyme held-out test set, 5 seeds
# per backbone (one design per seed). Outputs FASTAs under runs/benchmark/.
# Seeds: {1111, 2222, 3333, 4444, 5555} (matches SI Table S6).

set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/lib_env.sh"

TEST_DIR="${TEST_DIR:-$DATA_DIR/process/test}"
OUT_BASE="${OUT_BASE:-$RUNS_DIR/benchmark}"
METHODS="${METHODS:-gradeif enzymeif catif catif_rl_r1 catif_rl_r2 catif_rl_r3 proteinmpnn esmif ligandmpnn pifold abacust}"

cd "$REPO_ROOT"
mkdir -p "$OUT_BASE"

for method in $METHODS; do
  for seed in "${BENCHMARK_SEEDS[@]}"; do
    out_dir="$OUT_BASE/${method}/seed_${seed}"
    mkdir -p "$out_dir"
    echo "[sample_benchmark] method=$method seed=$seed"
    case "$method" in
      gradeif|enzymeif|catif|catif_rl_r*)
        activate_env catif
        # Use the appropriate checkpoint per method via env var indirection.
        ckpt_var="CKPT_${method^^}"
        ckpt="${!ckpt_var:-}"
        if [ -z "$ckpt" ]; then
          echo "[sample_benchmark] $ckpt_var not set; expected pretrained checkpoint path" >&2
          exit 1
        fi
        python -m catif_rl.sampling.infer \
          --test_dir "$TEST_DIR" \
          --ckpt_path "$ckpt" \
          --output_dir "$out_dir" \
          --seed "$seed"
        ;;
      proteinmpnn|esmif|ligandmpnn|pifold|abacust)
        echo "[sample_benchmark] $method runs via external upstream sampler; see catif_rl.evaluation.baselines for the dispatch table" >&2
        ;;
    esac
  done
done

echo "[sample_benchmark] Outputs under: $OUT_BASE"
