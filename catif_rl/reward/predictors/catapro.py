"""Subprocess wrapper for the CataPro *k*<sub>cat</sub> predictor.

CataPro (Wang et al., 2025) is cloned to ``external/CataPro-master/`` by
``scripts/00_setup_external.sh``. It runs in its own conda environment
(``catapro``).

Unlike DLKcat and UniKP, upstream CataPro does not expose a single
Python entry point: its example shells (``inference/run_catapro_*.sh``)
hardcode workstation-specific paths. To keep the wrapper interface
symmetric with DLKcat / UniKP, this module ships its own thin runner
shell ``_run_catapro.sh`` (next to this file) that:

1. Drives the upstream ``predict.py`` directly with --input / --prefix.
2. Emits the canonical filename ``<input_stem>_kcatpred_catapro.csv``
   that the manuscript pipeline expects, matching the convention used
   by DLKcat and UniKP.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Literal, Optional


_DEFAULT_REPO = Path("external/CataPro-master")
_CONDA_ENV = "catapro"
_RUNNER = Path(__file__).resolve().parent / "_run_catapro.sh"


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
    conda_env: str = _CONDA_ENV,
    output_dir: Optional[Path] = None,
    device: str = "cuda:0",
    batch_size: int = 64,
) -> Path:
    """Run CataPro on a (ProID, ProSeq, ProSeq', SMILES) CSV.

    Returns the path to the generated ``*_kcatpred_catapro.csv``.

    The prefix passed to the runner is the input CSV's stem, so the
    upstream output lands at
    ``<repo>/inference/results/<input_stem>_kcatpred_catapro.csv``.
    When ``output_dir`` is supplied the file is additionally copied to
    ``<output_dir>/<input_stem>_kcatpred_catapro.csv`` (the manuscript
    convention; ``catif_rl.reward.gdc``, ``catif_rl.reward.ensemble_rl``,
    and ``catif_rl.evaluation.build_master`` all glob for
    ``*_kcatpred_catapro.csv``) and that path is returned.
    """

    if shutil.which("conda") is None:
        raise CataProNotInstalledError("conda not found on PATH")

    if not _RUNNER.is_file():
        raise CataProNotInstalledError(
            "CataPro runner shell missing at " + str(_RUNNER) + " "
            "(this file ships with the catif_rl package; reinstall it)."
        )

    repo = _resolve_repo(repo_root)
    input_csv = Path(input_csv).resolve()
    prefix = input_csv.stem
    upstream_output = repo / "inference" / "results" / (prefix + "_kcatpred_catapro.csv")

    cmd = (
        "source $(conda info --base)/etc/profile.d/conda.sh && "
        "conda activate " + conda_env + " && "
        "bash " + str(_RUNNER) +
        " --input "  + str(input_csv) +
        " --prefix " + prefix +
        " --repo "   + str(repo) +
        " --mode "   + mode +
        " --device " + device +
        " --batch "  + str(batch_size)
    )
    subprocess.run(["bash", "-c", cmd], check=True)

    if output_dir is not None:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        dst = output_dir / upstream_output.name
        shutil.copyfile(upstream_output, dst)
        return dst
    return upstream_output
