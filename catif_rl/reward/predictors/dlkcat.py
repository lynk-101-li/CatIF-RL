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
) -> Path:
    """Run DLKcat on a (ProID, ProSeq, SMILES) CSV.

    Returns the path to the generated ``*_kcatpred_dlkcat.csv``.
    """

    if shutil.which("conda") is None:
        raise DLKcatNotInstalledError("conda not found on PATH")

    repo = _resolve_repo(repo_root)
    example_dir = repo / "DeeplearningApproach" / "Code" / "example"
    script = "run_rl_input.py" if mode == "rl" else "run_test_input.py"

    input_csv = input_csv.resolve()
    output_csv = example_dir / "output_data" / (input_csv.stem + "_kcatpred_dlkcat.csv")

    cmd = (
        "source $(conda info --base)/etc/profile.d/conda.sh && "
        "conda activate " + conda_env + " && "
        "cd " + str(example_dir) + " && "
        "python " + script + " " + str(input_csv)
    )
    subprocess.run(["bash", "-c", cmd], check=True)
    return output_csv
