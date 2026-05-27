#!/usr/bin/env bash
# CatIF-RL Round 3: sample Round-2 policy on train+valid backbones, score,
# build the Round 3 reward CSV, run final GRPO update (E=2 inner epochs,
# training-run length flag 25 per SI Table S3).

set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/lib_env.sh"

CATIF_REF="${CATIF_REF:-$CKPT_DIR/catif_Sep24_epoch228.pt}"
ROUND2_CKPT="${ROUND2_CKPT:-$RUNS_DIR/grpo_round2/policy_epoch02.pt}"
CONFIG="${CONFIG:-$CONFIG_DIR/grpo_round3.yaml}"
ROUND_DIR="${ROUND_DIR:-$RUNS_DIR/grpo_round3}"
SAMPLE_SEED="${SAMPLE_SEED:-11}"
DEVICE="${DEVICE:-cuda:0}"

cd "$REPO_ROOT"
mkdir -p "$ROUND_DIR"
RAW_SAMPLE="$ROUND_DIR/round3_raw_samp.csv"
REWARD_CSV="$ROUND_DIR/round3_reward_data.csv"

activate_env catif
python -m catif_rl.sampling.batch \
  --condition-dirs "$DATA_DIR/process/enzymeif/train_and_validation" \
  --pairs-csv "$DATA_DIR/brenda/brenda_train_and_dev_set.csv" \
  --ckpt-path "$ROUND2_CKPT" \
  --group-size 5 \
  --step 100 \
  --seed "$SAMPLE_SEED" \
  --diverse \
  --out-csv "$RAW_SAMPLE"

activate_env dlkcat
python -c "from catif_rl.reward.predictors import dlkcat; dlkcat.predict('$RAW_SAMPLE', mode='rl', output_dir='$ROUND_DIR')"
activate_env unikp
python -c "from catif_rl.reward.predictors import unikp; unikp.predict('$RAW_SAMPLE', mode='rl', output_dir='$ROUND_DIR')"
activate_env catapro
python -c "from catif_rl.reward.predictors import catapro; catapro.predict('$RAW_SAMPLE', mode='rl', rl_round_tag='round3', rl_subset_tag='r2_epoch02', output_dir='$ROUND_DIR')"

activate_env catif
# SI Table S5 calibration: reuse the q90-q10 scales frozen by GDC, so the
# Round-3 reward scale matches Rounds 1 / 2 (and the published numbers).
NORMALIZER_JSON="${NORMALIZER_JSON:-$RUNS_DIR/gdc/normalizer.json}"
python -m catif_rl.reward.ensemble_rl \
  --in-dir      "$ROUND_DIR" \
  --normalizer  "$NORMALIZER_JSON" \
  --reward-file "$REWARD_CSV"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -m catif_rl.training.grpo \
  --policy-ckpt "$ROUND2_CKPT" \
  --ref-ckpt "$CATIF_REF" \
  --condition-dir "$DATA_DIR/process/enzymeif/train_and_validation" \
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

echo "[rl_round3] Round 3 finished. Final CatIF-RL checkpoint: $ROUND_DIR/policy_epoch02.pt"
