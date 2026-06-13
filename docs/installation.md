# Installation

CatIF-RL was developed on Linux with CUDA 11.7 and a single RTX-class GPU
(EnzymeIF pretraining used an RTX 4090 24 GB cloud node; CatIF supervised and
all CatIF-RL refinement ran on a local RTX 4060 Ti 16 GB).

## 1. Clone the repository

```bash
git clone https://github.com/lynk-101-li/CatIF-RL.git
cd CatIF-RL
```

## 2. Create the main conda environment

```bash
conda env create -f environment.yml
conda activate catif
```

The pip-only path (`pip install -r requirements.txt`) is possible but
requires pre-existing CUDA 11.7 / OpenMM / DSSP installs; the conda path is
recommended.

## 3. Clone the upstream dependencies

```bash
bash scripts/00_setup_external.sh
```

This clones the GraDe-IF backbone and the three reward predictors (DLKcat,
UniKP, CataPro) into `external/`. The script does **not** download pretrained
weights or datasets -- see `external/README.md` for the per-upstream weight
download instructions.

## 4. Per-predictor conda environments

Each reward predictor runs in its own conda environment because their pinned
dependency sets are mutually incompatible:

```bash
conda env create -f external/DLKcat5/environment.yml         # dlkcat
conda env create -f external/UniKP/environment_unikp.yml     # unikp
conda env create -f external/CataPro-master/environment_catapro.yml  # catapro
# ESMFold:
conda create -n esmfold python=3.10 pip
conda activate esmfold && pip install "fair-esm[esmfold]"
```

## 5. Download the data and checkpoints

The processed graph dataset and pretrained CatIF-RL checkpoints are too
large for the git repository -- they are archived separately. Follow the
instructions in [`data/README.md`](../data/README.md) and
[`checkpoints/README.md`](../checkpoints/README.md).

## 6. Smoke test

```bash
python -c "import catif_rl; print(catif_rl.__version__)"
python -m pytest tests/ -q
```

A passing smoke test confirms that the `catif_rl` package imports and that
the GraDe-IF backbone is correctly wired through `sys.path` injection.

## Troubleshooting

- **`ImportError: GraDe-IF backbone not found at external/GraDe_IF`** --
  `scripts/00_setup_external.sh` was not run or did not complete. Re-run it
  and confirm the directory exists.
- **CUDA OOM during GRPO training** -- reduce `--accum-steps` is rarely
  enough; the more reliable lever is to reduce `--sample-step` (DDIM
  steps used to compute the policy log-prob during the inner loop), or to
  move the supervised stages to a higher-VRAM card.
- **DLKcat / UniKP / CataPro report a `conda activate` failure inside the
  subprocess wrapper** -- make sure your shell's profile is loaded
  non-interactively; the wrappers source `conda info --base`'s
  `etc/profile.d/conda.sh` explicitly.
