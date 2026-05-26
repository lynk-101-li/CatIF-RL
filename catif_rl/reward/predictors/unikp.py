"""Subprocess wrapper for the UniKP *k*<sub>cat</sub> predictor.

UniKP (Yu et al., 2023) is cloned to ``external/UniKP/`` by
``scripts/00_setup_external.sh``. It runs in its own conda environment
(``unikp``) because of a ProtT5 / Transformer dependency that conflicts with
the main ``catif`` environment.

The downstream entry points used here are:

- ``external/UniKP/run_prediction.py``
  Used on the held-out benchmark and at EnzymeIF / CatIF stages.
- ``external/UniKP/run_rl_pred.py``
  Used on the RL-stage candidate pool.

Both scripts take a CSV with (ProID, ProSeq, SMILES) columns and emit
``<input>_kcatpred_unikp.csv`` into ``dataset4prediction/output_data/``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Literal, Optional


_DEFAULT_REPO = Path("external/UniKP")
_CONDA_ENV = "unikp"


class UniKPNotInstalledError(RuntimeError):
    """Raised when the UniKP upstream checkout cannot be located."""


def _resolve_repo(repo_root: Optional[Path]) -> Path:
    repo = (repo_root or _DEFAULT_REPO).resolve()
    if not repo.is_dir():
        raise UniKPNotInstalledError(
            "UniKP not found at " + str(repo) + ". "
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
    """Run UniKP on a (ProID, ProSeq, SMILES) CSV.

    Returns the path to the generated ``*_kcatpred_unikp.csv``.

    When ``output_dir`` is supplied, the upstream output is additionally
    copied to ``<output_dir>/unikp_pred.csv`` (the canonical filename
    consumed by ``catif_rl.reward.gdc`` and ``catif_rl.reward.ensemble_rl``)
    and that path is returned.
    """

    if shutil.which("conda") is None:
        raise UniKPNotInstalledError("conda not found on PATH")

    repo = _resolve_repo(repo_root)
    script = "run_rl_pred.py" if mode == "rl" else "run_prediction.py"
    out_dir = repo / "dataset4prediction" / "output_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    input_csv = Path(input_csv).resolve()
    upstream_output = out_dir / (input_csv.stem + "_kcatpred_unikp.csv")

    cmd = (
        "source $(conda info --base)/etc/profile.d/conda.sh && "
        "conda activate " + conda_env + " && "
        "cd " + str(repo) + " && "
        "python " + script + " " + str(input_csv) + " --out_dir " + str(out_dir)
    )
    subprocess.run(["bash", "-c", cmd], check=True)

    if output_dir is not None:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        canonical = output_dir / "unikp_pred.csv"
        shutil.copyfile(upstream_output, canonical)
        return canonical
    return upstream_output
