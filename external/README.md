# `external/`

This directory is **gitignored**. After running
``scripts/00_setup_external.sh`` it will contain the four upstream
repositories that CatIF-RL depends on:

| Subdirectory | Upstream | Purpose | License |
|---|---|---|---|
| `GraDe_IF/` | <https://github.com/ykiiiiii/GraDe_IF> | Graph denoising diffusion backbone (`diffusion.utils`, `diffusion.model.egnn_pytorch`, original `diffusion.gradeif_app`). Loaded onto `sys.path` by `catif_rl.models.gradeif_adapter`. | upstream repo currently has no LICENSE file; see "License compatibility" below |
| `DLKcat5/` | <https://github.com/SysBioChalmers/DLKcat> | Predictor #1 in the activity reward ensemble. Wrapped by `catif_rl.reward.predictors.dlkcat`. | GPL v3 |
| `UniKP/` | <https://github.com/Luo-SynBioLab/UniKP> | Predictor #2 in the ensemble. Wrapped by `catif_rl.reward.predictors.unikp`. | upstream repo currently has no LICENSE file; treat as "all rights reserved" |
| `CataPro-master/` | <https://github.com/zchwang/CataPro> | Predictor #3 in the ensemble. Wrapped by `catif_rl.reward.predictors.catapro`. | MIT |

## Why not vendored

CatIF-RL follows a *depend, don't redistribute* pattern: we ship our own
training / sampling / orchestration code and ask the user to clone each
upstream repository at install time. This keeps each project under its own
upstream licence and version. No upstream code is mirrored or rewritten
into this repository.

## License compatibility notes

- **GraDe-IF and UniKP** -- neither upstream currently publishes a LICENSE
  file. In this state, cloning and running them for personal / research use
  is the most that can be done with confidence. If you intend to use this
  pipeline beyond that, contact the upstream maintainers and request that
  they add an explicit LICENSE.

- **DLKcat (GPL v3)** -- our subprocess wrapper does not statically link to
  DLKcat code, so the GPL "viral" provision does not propagate to CatIF-RL.
  Redistribution of the *unmodified* DLKcat checkout retains its GPL terms.

- **CataPro (MIT)** -- compatible with the CatIF-RL MIT licence; no further
  action required.

## Manual setup (after `00_setup_external.sh`)

```bash
# Per-repo conda environments (full pin sets):
conda env create -f environment.yml                       # catif (this repo)
conda env create -f external/DLKcat5/environment.yml      # dlkcat
conda env create -f external/UniKP/environment_unikp.yml  # unikp
conda env create -f external/CataPro-master/environment_catapro.yml  # catapro
# ESMFold uses the `esm` package; install in its own env or pip-install fair-esm.
```

Each upstream repository's README explains how to obtain its model weights.
This project deliberately does not redistribute those weights or provide a
single-shot download script for them.
