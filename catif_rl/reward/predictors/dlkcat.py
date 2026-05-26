"""Subprocess wrapper for the DLKcat *k*<sub>cat</sub> predictor.

The DLKcat implementation (Li et al., 2022) is cloned to ``external/DLKcat5/``
by ``scripts/00_setup_external.sh``. It runs in its own conda environment
(``dlkcat``) because its pinned PyTorch / DGL set conflicts with the main
``catif`` environment.

The downstream entry points used here are:

- ``external/DLKcat5/DeeplearningApproach/Code/example/run_test_input.py``
  Used on the held-out benchmark (manuscript §2.7).
- ``external/DLKcat5/DeeplearningApproach/Code/example/run_rl_input.py``
  Used on the RL-stage candidate pool (manuscript §2.5).

Both scripts take a CSV with (ProID, ProSeq, SMILES) columns and emit
``<input>_kcatpred_dlkcat.csv`` containing the per-row log10(k_cat) prediction.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Literal, Optional


_DEFAULT_REPO = Path("external/DLKcat5")
_CONDA_ENV = "dlkcat"


class DLKcatNotInstalledError(RuntimeError):
    """Raised when the DLKcat upstream checkout cannot be located."""


def _resolve_repo(repo_root: Optional[Path]) -> Path:
    repo = (repo_root or _DEFAULT_REPO).resolve()
    if not repo.is_dir():
        raise DLKcatNotInstalledError(
            "DLKcat not found at " + str(repo) + ". "
            "Run scripts/00_setup_external.sh to clone it."
        )
    return repo


def predict(
    input_csv: Path,
    repo_root: Optional[Path] = None,
    mode: Literal["benchmark", "rl"] = "benchmark",
    conda_env: str = _CONDA_ENV,
    output_dir: Optional[Path] = None,
) -> Path:
    """Run DLKcat on a (ProID, ProSeq, SMILES) CSV.

    Returns the path to the generated ``*_kcatpred_dlkcat.csv``.

    By default the upstream entry script writes its output into
    ``external/DLKcat5/DeeplearningApproach/Code/example/output_data/``,
    which is outside the per-round / per-benchmark working directory that
    downstream scoring (``catif_rl.reward.gdc`` and
    ``catif_rl.reward.ensemble_rl``) reads from. To bridge that gap, when
    ``output_dir`` is supplied the wrapper additionally copies the output
    to ``<output_dir>/dlkcat_pred.csv`` (the canonical filename consumed
    downstream) and returns *that* path.
    """

    if shutil.which("conda") is None:
        raise DLKcatNotInstalledError("conda not found on PATH")

    repo = _resolve_repo(repo_root)
    example_dir = repo / "DeeplearningApproach" / "Code" / "example"
    script = "run_rl_input.py" if mode == "rl" else "run_test_input.py"

    input_csv = Path(input_csv).resolve()
    upstream_output = example_dir / "output_data" / (input_csv.stem + "_kcatpred_dlkcat.csv")

    cmd = (
        "source $(conda info --base)/etc/profile.d/conda.sh && "
        "conda activate " + conda_env + " && "
        "cd " + str(example_dir) + " && "
        "python " + script + " " + str(input_csv)
    )
    subprocess.run(["bash", "-c", cmd], check=True)

    if output_dir is not None:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        canonical = output_dir / "dlkcat_pred.csv"
        shutil.copyfile(upstream_output, canonical)
        return canonical
    return upstream_output
