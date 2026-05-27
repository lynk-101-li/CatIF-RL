# External dependencies

CatIF-RL depends on four upstream repositories that are cloned at install
time rather than vendored into this checkout. The full list also documents
the model weights that each upstream tool requires separately.

## Inverse-folding backbone

**GraDe-IF** -- <https://github.com/ykiiiiii/GraDe_IF>

The discrete-diffusion graph backbone. Provides:

- `diffusion.utils.PredefinedNoiseScheduleDiscrete`
- `diffusion.model.egnn_pytorch.*` (EGNN graph layers)
- Reference `diffusion.gradeif_app` (the CatIF-RL repo ships a locally
  modified variant under `catif_rl/models/gradeif_app.py`)

Loaded via `sys.path` injection from `catif_rl/models/gradeif_adapter.py`.

## *k*<sub>cat</sub> reward predictors (SI Table S5)

Three independent predictors are averaged (after 10th/90th quantile
normalization) to produce the activity reward used during GDC and GRPO.

| Predictor | Upstream | Conda env | Inference entry | Notes |
|-----------|----------|-----------|-----------------|-------|
| DLKcat | <https://github.com/SysBioChalmers/DLKcat> | `dlkcat` | `DeeplearningApproach/Code/example/{run_test_input,run_rl_input}.py` | Also doubles as the per-protein evaluator on the held-out test set. |
| UniKP | <https://github.com/Luo-SynBioLab/UniKP> | `unikp` | `{run_prediction,run_rl_pred}.py` | ProtT5 + SMILES transformer + ExtraTrees regressor. |
| CataPro | <https://github.com/zchwang/CataPro> | `catapro` | `inference/predict.py`, driven by `catif_rl/reward/predictors/_run_catapro.sh` (this repo's shim, called from `catapro.py`) | Dual-branch deep regressor. The upstream `run_catapro_{test,rl}.sh` example shells are not used: their hard-coded paths would break public reproduction, so the shim invokes `predict.py` directly with `--input` / `--prefix` and stays in sync with `dlkcat.py` and `unikp.py`. |

Each predictor is wrapped by a thin subprocess interface under
`catif_rl/reward/predictors/`.

## Structural plausibility predictor

**ESMFold** (Lin et al., 2023) -- distributed via the `esm` PyPI package.

Used for backbone prediction during dataset construction and for refolding
generated sequences during evaluation (single-seed refolded set, per SI
Section 2.7).

## External baselines (manuscript §3.3)

Five external inverse-folding baselines are sampled on the held-out
benchmark for comparison:

- ProteinMPNN (`dauparas/ProteinMPNN`)
- ESM-IF1 (provided through the `esm` package or `facebookresearch/esm`)
- LigandMPNN (`dauparas/LigandMPNN`)
- PiFold (`A4Bio/PiFold`)
- ABACUS-T (Liu et al., 2025)

`catif_rl.evaluation.baselines` exposes a dispatch table mapping baseline
name to invocation pattern. The sample shells used to produce the SI Table
S6 outputs are archived alongside the SI workspace (not in this repo).

## License compatibility

See `external/README.md` for a per-upstream licence table and the
implications for redistribution.
