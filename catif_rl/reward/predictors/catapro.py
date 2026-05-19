"""Subprocess wrapper for the CataPro *k*<sub>cat</sub> predictor.

CataPro (Wang et al., 2025) is cloned to ``external/CataPro-master/`` by
``scripts/00_setup_external.sh``. It runs in its own conda environment
(``catapro``).

The downstream entry points used here live under
``external/CataPro-master/inference/``:

- ``run_catapro_test.sh``  -- held-out benchmark scoring
- ``run_catapro_rl.sh <round_tag> <subset_tag>`` -- RL-stage scoring
- ``predict.py`` -- the underlying Python entry called by the shells

Both shells emit ``results/<input_stem>_kcatpred_catapro.csv``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Literal, Optional, Tuple


_DEFAULT_REPO = Path("external/CataPro-master")
_CONDA_ENV = "catapro"


class CataProNotInstalledError(RuntimeError):
    """Raised when the CataPro upstream checkout cannot be located."""


def _resolve_repo(repo_root: Optional[Path]) -> Path:
    repo = (repo_root or _DEFAULT_REPO).resolve()
    if not repo.is_dir():
        raise CataProNotInstalledError(
            "CataPro not found at " + str(repo) + ". "
            "Run scripts/00_setup_external.sh to clone it."
        )
    return repo


def predict(
    input_csv: Path,
    repo_root: Optional[Path] = None,
    mode: Literal["benchmark", "rl"] = "benchmark",
    rl_round_tag: Optional[str] = None,
    rl_subset_tag: Optional[str] = None,
    conda_env: str = _CONDA_ENV,
) -> Path:
    """Run CataPro on a (ProID, ProSeq, SMILES) CSV.

    In ``rl`` mode the caller must also supply ``rl_round_tag`` and
    ``rl_subset_tag`` (these become positional arguments to
    ``run_catapro_rl.sh``).

    Returns the path to the generated ``*_kcatpred_catapro.csv``.
    """

    if shutil.which("conda") is None:
        raise CataProNotInstalledError("conda not found on PATH")

    repo = _resolve_repo(repo_root)
    inference_dir = repo / "inference"
    results_dir = inference_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    input_csv = input_csv.resolve()
    output_csv = results_dir / (input_csv.stem + "_kcatpred_catapro.csv")

    if mode == "rl":
        if rl_round_tag is None or rl_subset_tag is None:
            raise ValueError("rl mode requires rl_round_tag and rl_subset_tag")
        shell = "run_catapro_rl.sh " + rl_round_tag + " " + rl_subset_tag
    else:
        shell = "run_catapro_test.sh"

    cmd = (
        "source $(conda info --base)/etc/profile.d/conda.sh && "
        "conda activate " + conda_env + " && "
        "cd " + str(inference_dir) + " && "
        "bash " + shell
    )
    subprocess.run(["bash", "-c", cmd], check=True)
    return output_csv
