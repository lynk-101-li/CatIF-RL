#!/usr/bin/env bash
# CatIF-RL Round 1: sample CatIF policy on train+valid backbones, score with
# DLKcat/UniKP/CataPro, normalise into a reward CSV, then run GRPO update
# (E=2 inner epochs, training-run length flag 50 per SI Table S3).

set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/lib_env.sh"

CATIF_CKPT="${CATIF_CKPT:-$CKPT_DIR/catif_Sep24_epoch228.pt}"
CONFIG="${CONFIG:-$CONFIG_DIR/grpo_round1.yaml}"
ROUND_DIR="${ROUND_DIR:-$RUNS_DIR/grpo_round1}"
SAMPLE_SEED="${SAMPLE_SEED:-11}"
DEVICE="${DEVICE:-cuda:0}"

cd "$REPO_ROOT"
mkdir -p "$ROUND_DIR"
RAW_SAMPLE="$ROUND_DIR/round1_raw_samp.csv"
REWARD_CSV="$ROUND_DIR/round1_reward_data.csv"

# Stage 1: bulk sample G=5 candidates per (enzyme, substrate).
activate_env catif
python -m catif_rl.sampling.batch \
  --condition-dirs "$DATA_DIR/process/enzymeif/train_and_validation" \
  --pairs-csv "$DATA_DIR/brenda/brenda_train_and_dev_set.csv" \
  --ckpt-path "$CATIF_CKPT" \
  --group-size 5 \
  --step 100 \
  --seed "$SAMPLE_SEED" \
  --diverse \
  --out-csv "$RAW_SAMPLE"

# Stage 2: ensemble scoring with the three predictors. Each wrapper copies
# its output to "$ROUND_DIR" under the canonical filenames dlkcat_pred.csv /
# unikp_pred.csv / catapro_pred.csv, which is where catif_rl.reward.ensemble_rl
# expects to find them.
activate_env dlkcat
python -c "from catif_rl.reward.predictors import dlkcat; dlkcat.predict('$RAW_SAMPLE', mode='rl', output_dir='$ROUND_DIR')"
activate_env unikp
python -c "from catif_rl.reward.predictors import unikp; unikp.predict('$RAW_SAMPLE', mode='rl', output_dir='$ROUND_DIR')"
activate_env catapro
python -c "from catif_rl.reward.predictors import catapro; catapro.predict('$RAW_SAMPLE', mode='rl', output_dir='$ROUND_DIR')"

# Stage 3: normalise and ensemble.
activate_env catif
# SI Table S5 calibration: reuse the q90-q10 scales frozen by GDC
# (runs/gdc/normalizer.json) for every RL round, so reward magnitudes are
# directly comparable across R1 / R2 / R3.
NORMALIZER_JSON="${NORMALIZER_JSON:-$RUNS_DIR/gdc/normalizer.json}"
python -m catif_rl.reward.ensemble_rl \
  --in-dir      "$ROUND_DIR" \
  --normalizer  "$NORMALIZER_JSON" \
  --reward-file "$REWARD_CSV"

# Stage 4: GRPO inner loop (2 epochs operational; --epochs flag = 50 per Table S3).
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -m catif_rl.training.grpo \
  --policy-ckpt "$CATIF_CKPT" \
  --ref-ckpt "$CATIF_CKPT" \
  --condition-dir "$DATA_DIR/process/enzymeif/train_and_validation" \
  --scored-csv "$REWARD_CSV" \
  --output-dir "$ROUND_DIR" \
  --device "$DEVICE" \
  --epochs 50 \
  --warmup-epochs 0 \
  --reward-mode lin_sym \
  --reward-tau 0.40 \
  --mutation-penalty 0.10 \
  --mutation-free-frac 0.30 \
  --sample-step 100 \
  --lr 5e-6 \
  --weight-decay 1e-2 \
  --kl-target 0.01 \
  --accum-steps 1 \
  --grad-clip 2.0 \
  --max-beta 0.5 \
  --min-reward-distinct 3 \
  --save-every 1 \
  --seed 42 \
  --plot-every 1

echo "[rl_round1] Round 1 finished. Epoch-02 checkpoint: $ROUND_DIR/policy_epoch02.pt"
