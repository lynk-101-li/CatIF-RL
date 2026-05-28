# `scripts/`

Shell entry points, ordered by pipeline stage. Each script sources
`lib_env.sh` for the shared `REPO_ROOT` / conda-activation helpers and the
benchmark seed list.

| Script | Stage | Notes |
|--------|-------|-------|
| `00_setup_external.sh` | Setup | Clones GraDe-IF, DLKcat, UniKP, CataPro into `external/`. Does **not** download model weights or datasets. |
| `01_build_dataset.sh` | §2.1 | CATH PDB fetch from RCSB → graph `.pt` per cohort → 9:1 enzyme split + CATH merge. Assumes Zenodo raw PDBs are already in `data/raw/`. |
| `02_train_enzymeif.sh` | §2.2 | Supervised pretrain (config: `catif_rl/config/enzymeif.yaml`). |
| `03_run_gdc.sh` | §2.3 | Sample → ESMFold refold → 3-predictor ensemble → activity-positive variants. |
| `04_train_catif.sh` | §2.4 | Supervised CatIF (config: `catif_rl/config/catif.yaml`). |
| `05a_rl_round1.sh` | §2.5 | Sample → score → GRPO Round 1 (training-run length flag 50). |
| `05b_rl_round2.sh` | §2.5 | Same, Round 2 (training-run length flag 25). |
| `05c_rl_round3.sh` | §2.5 | Same, Round 3 (final CatIF-RL checkpoint). |
| `06_sample_benchmark.sh` | §2.7 | 11 methods × 5 seeds × 1 design = 7,115 sequences per method on the 1,423-enzyme test set. Seeds: `{1111, 2222, 3333, 4444, 5555}`. |
| `07_score_benchmark.sh` | §2.7 | Substrate match → DLKcat → ESMFold refold (1 seed) → Recovery / pLDDT / RMSD / SR@δ → paired statistics. |
| `08_run_case_studies.sh` | §3.5 | Four cases (3 global + 1 motif inpaint), `seed=12345`. |

Every script accepts environment variables for paths and overrides (see the
top of each file). Defaults assume the repo-relative layout:
`checkpoints/...`, `data/...`, `runs/...`. The shared helper file
`lib_env.sh` is sourced first; do not invoke it directly.

## Conda environments referenced

| Env | Used by |
|-----|---------|
| `catif` | The main pipeline (sampling, GRPO, evaluation, dataset construction). |
| `dlkcat` | DLKcat scoring. |
| `unikp` | UniKP scoring. |
| `catapro` | CataPro scoring. |
| `esmfold` | ESMFold backbone prediction. |

YAML environment files for each are provided alongside the upstream repos
they came from (`external/.../environment.yml`) and as reference copies
under `model_environments_yml_database/` (in the project root, **not** in
the repo; see `docs/external_dependencies.md`).
