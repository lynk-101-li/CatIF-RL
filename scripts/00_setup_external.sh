#!/usr/bin/env bash
# Clone the four upstream repositories that CatIF-RL depends on.
# This script does NOT download pretrained model weights or datasets --
# follow the per-repo README inside each cloned checkout for those steps.

set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/lib_env.sh"

mkdir -p "$EXTERNAL_DIR"
cd "$EXTERNAL_DIR"

clone_or_skip() {
  local url="$1"
  local target="$2"
  if [ -d "$target/.git" ]; then
    echo "[setup_external] $target already cloned, skipping"
  else
    echo "[setup_external] cloning $url -> $target"
    git clone "$url" "$target"
  fi
}

# Graph denoising diffusion backbone (Yi et al., 2023).
clone_or_skip "https://github.com/ykiiiiii/GraDe_IF.git" "GraDe_IF"

# Three kinetic predictors used as the activity reward ensemble.
clone_or_skip "https://github.com/SysBioChalmers/DLKcat.git"     "DLKcat5"
clone_or_skip "https://github.com/Luo-SynBioLab/UniKP.git"       "UniKP"
clone_or_skip "https://github.com/zchwang/CataPro.git"           "CataPro-master"

cat <<'EOF'

[setup_external] All upstream repositories cloned under external/.

Next steps (manual; see docs/external_dependencies.md):

  1. Create one conda environment per upstream tool. Templates are provided
     under model_environments_yml_database/ for reference but NOT shipped
     by this script (each environment can be hundreds of MB):

       conda env create -f environment.yml                          # catif (this repo)
       conda env create -f external/DLKcat5/environment.yml         # dlkcat
       conda env create -f external/UniKP/environment_unikp.yml     # unikp
       conda env create -f external/CataPro-master/environment_catapro.yml  # catapro
       conda env create -n esmfold ...                              # esmfold

  2. Download the model weights for each upstream tool following its own
     README. CatIF-RL deliberately does NOT redistribute upstream weights.

  3. Download the processed graph dataset and pretrained CatIF-RL checkpoints
     into data/ and checkpoints/ respectively -- see data/README.md and
     checkpoints/README.md.

EOF
