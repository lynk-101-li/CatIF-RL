#!/usr/bin/env bash
# Shared helpers for the pipeline entry points under scripts/.
# Source this from each top-level script. Never invoke directly.

# Resolve the repo root from the script's location.
_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$_LIB_DIR/.." && pwd)"
export REPO_ROOT

# Common subdirectories.
EXTERNAL_DIR="$REPO_ROOT/external"
DATA_DIR="$REPO_ROOT/data"
CKPT_DIR="$REPO_ROOT/checkpoints"
RUNS_DIR="$REPO_ROOT/runs"
CONFIG_DIR="$REPO_ROOT/catif_rl/config"
export EXTERNAL_DIR DATA_DIR CKPT_DIR RUNS_DIR CONFIG_DIR

# Switch into the named conda environment. Sources the conda init file from
# whichever conda installation is on PATH.
activate_env() {
  local env_name="$1"
  if ! command -v conda >/dev/null 2>&1; then
    echo "[lib_env] error: conda not on PATH" >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$env_name"
}

# Standard 5-seed sweep used across all sampling on the held-out benchmark.
BENCHMARK_SEEDS=(1111 2222 3333 4444 5555)
export BENCHMARK_SEEDS

# Single seed used for case studies (manuscript Section 3.5).
CASE_STUDY_SEED=12345
export CASE_STUDY_SEED

# Ensure the runs/ directory exists.
mkdir -p "$RUNS_DIR"
