#!/usr/bin/env bash
# CatIF-RL Round 2: sample Round-1 policy on train+valid backbones, score
# with the three predictors, build the Round 2 reward CSV, run GRPO update
# (E=2 inner epochs, training-run length flag 25 per SI Table S3).

set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/lib_env.sh"

CATIF_REF="${CATIF_REF:-$CKPT_DIR/catif_Sep24_epoch228.pt}"
ROUND1_CKPT="${ROUND1_CKPT:-$RUNS_DIR/grpo_round1/policy_epoch02.pt}"
CONFIG="${CONFIG:-$CONFIG_DIR/grpo_round2.yaml}"
ROUND_DIR="${ROUND_DIR:-$RUNS_DIR/grpo_round2}"
SAMPLE_SEED="${SAMPLE_SEED:-11}"
DEVICE="${DEVICE:-cuda:0}"

cd "$REPO_ROOT"
mkdir -p "$ROUND_DIR"
RAW_SAMPLE="$ROUND_DIR/round2_raw_samp.csv"
REWARD_CSV="$ROUND_DIR/round2_reward_data.csv"

activate_env catif
python -m catif_rl.sampling.batch \
  --condition-dirs "$DATA_DIR/process/train" "$DATA_DIR/process/valid" \
  --pairs-csv "$DATA_DIR/brenda/brenda_train_and_dev_set.csv" \
  --ckpt-path "$ROUND1_CKPT" \
  --group-size 5 \
  --step 100 \
  --seed "$SAMPLE_SEED" \
  --diverse \
  --out-csv "$RAW_SAMPLE"

activate_env dlkcat
python -c "from catif_rl.reward.predictors import dlkcat; dlkcat.predict('$RAW_SAMPLE', mode='rl')"
activate_env unikp
python -c "from catif_rl.reward.predictors import unikp; unikp.predict('$RAW_SAMPLE', mode='rl')"
activate_env catapro
python -c "from catif_rl.reward.predictors import catapro; catapro.predict('$RAW_SAMPLE', mode='rl', rl_round_tag='round2', rl_subset_tag='r1_epoch02')"

activate_env catif
python -m catif_rl.reward.ensemble_rl \
  --in-dir "$ROUND_DIR" \
  --reward-file "$REWARD_CSV"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -m catif_rl.training.grpo \
  --policy-ckpt "$ROUND1_CKPT" \
  --ref-ckpt "$CATIF_REF" \
  --condition-dir "$DATA_DIR/process/train" \
  --scored-csv "$REWARD_CSV" \
  --output-dir "$ROUND_DIR" \
  --device "$DEVICE" \
  --epochs 25 \
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

echo "[rl_round2] Round 2 finished. Epoch-02 checkpoint: $ROUND_DIR/policy_epoch02.pt"
