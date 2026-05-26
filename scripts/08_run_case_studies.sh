#!/usr/bin/env bash
# Run the four case studies from manuscript Section 3.5:
#   - Three global redesigns (EC 1.4.1.20, EC 2.4.2.1, EC 5.3.1.1)
#   - One motif-preserving SalR inpainting (EC 1.1.1.248)
# All cases use the fixed seed 12345 (CASE_STUDY_SEED).

set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/lib_env.sh"

CATIF_RL_CKPT="${CATIF_RL_CKPT:-$CKPT_DIR/catif_rl_R3_epoch02.pt}"
OUT_BASE="${OUT_BASE:-$RUNS_DIR/case_studies}"

cd "$REPO_ROOT"
mkdir -p "$OUT_BASE"
activate_env catif

# ----- Global redesigns -----
for case in EC1.4.1.20_Lsphaericus EC2.4.2.1_Hsapiens EC5.3.1.1_Tbrucei; do
  echo "[case_studies] global redesign: $case"
  out_dir="$OUT_BASE/$case"
  mkdir -p "$out_dir"
  python -m catif_rl.sampling.infer \
    --input-pdb "$REPO_ROOT/case_study/$case/native.pdb" \
    --ckpt_path "$CATIF_RL_CKPT" \
    --output_dir "$out_dir" \
    --seed "$CASE_STUDY_SEED"
done

# ----- Motif-preserving inpainting (SalR) -----
SALR_DIR="$OUT_BASE/EC1.1.1.248_SalR"
mkdir -p "$SALR_DIR"
echo "[case_studies] motif-preserving inpainting: SalR"

# Fix the four catalytic residues Asn152, Ser180, Tyr236, Lys240 to their
# native identities and redesign every other residue. Indices are 0-based.
python -m catif_rl.sampling.inpaint \
  --pdb "$REPO_ROOT/case_study/EC1.1.1.248_SalR/native.pdb" \
  --fix 151,179,235,239 \
  --ckpt "$CATIF_RL_CKPT" \
  --u 5 \
  --seed "$CASE_STUDY_SEED" \
  --output "$SALR_DIR/designed.fasta"

echo "[case_studies] Outputs under: $OUT_BASE"
