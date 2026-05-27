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

# The five external baselines (ProteinMPNN, ESM-IF, LigandMPNN, PiFold,
# ABACUS-T) are NOT re-sampled in this repo -- their fold-and-design
# pipelines each require their own conda env and upstream weights. We
# instead ship the archived per-seed FASTA outputs through Zenodo (see
# data/README.md "Benchmark baseline FASTA archives") and stage them
# into $BASELINE_ARCHIVE_DIR; this script symlinks the right seed dir
# into $OUT_BASE for each external method so downstream scoring
# (scripts/07_score_benchmark.sh) sees a uniform layout.
BASELINE_ARCHIVE_DIR="${BASELINE_ARCHIVE_DIR:-$DATA_DIR/benchmark_baselines}"

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
        # External baseline: re-use the archived FASTA outputs that
        # accompany the manuscript (see data/benchmark_baselines/ and
        # the Zenodo deposit linked in data/README.md). We symlink the
        # right per-seed dir into $OUT_BASE/<method>/seed_<seed>/ so
        # downstream scoring sees the same layout the in-repo methods
        # produce.
        archive_src="$BASELINE_ARCHIVE_DIR/$method/seed_${seed}"
        if [ ! -d "$archive_src" ]; then
          echo "[sample_benchmark][ERROR] $method seed=$seed: archive not staged" >&2
          echo "[sample_benchmark][ERROR]   expected: $archive_src" >&2
          echo "[sample_benchmark][ERROR]   fix: download the baseline tarball per data/README.md" >&2
          exit 1
        fi
        n_fa=$(find "$archive_src" -maxdepth 1 \( -name '*.fa' -o -name '*.fasta' \) 2>/dev/null | wc -l | tr -d ' ')
        if [ "$n_fa" -eq 0 ]; then
          echo "[sample_benchmark][ERROR] $method seed=$seed: $archive_src has no .fa/.fasta files" >&2
          exit 1
        fi
        # Stage via symlink so 07_score_benchmark.sh sees the standard
        # $OUT_BASE/<method>/seed_<seed>/<*.fa> layout without us
        # duplicating ~712 MB per seed.
        for f in "$archive_src"/*.fa "$archive_src"/*.fasta; do
          [ -e "$f" ] || continue
          ln -sf "$f" "$out_dir/$(basename "$f")"
        done
        echo "[sample_benchmark] $method seed=$seed: linked $n_fa archived FASTA(s) from ${archive_src#$REPO_ROOT/}"
        ;;
    esac
  done
done

echo "[sample_benchmark] Outputs under: $OUT_BASE"
