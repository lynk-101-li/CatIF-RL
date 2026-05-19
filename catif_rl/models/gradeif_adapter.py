"""Make the upstream GraDe-IF source importable as ``diffusion.*``.

The GraDe-IF reference implementation (Yi et al., 2023) is cloned into
``external/GraDe_IF/`` by ``scripts/00_setup_external.sh``. Several of our
modules (``catif_rl.models.gradeif_app``, ``catif_rl.training.grpo``, etc.)
reuse the upstream ``diffusion.utils.PredefinedNoiseScheduleDiscrete`` and
``diffusion.model.egnn_pytorch.*`` symbols verbatim. Importing this module
inserts the external directory at the front of ``sys.path`` so that the
``diffusion.*`` and ``diffusion.model.*`` namespaces resolve to the cloned
upstream code.

The user-modified application wrapper :mod:`catif_rl.models.gradeif_app` then
overrides the upstream module of the same name (``diffusion.gradeif_app``)
within this repository -- import it via the fully qualified
``catif_rl.models.gradeif_app`` to access the modifications.

This module has only side effects; nothing is exported.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Repo root is three levels above this file: catif_rl/models/gradeif_adapter.py
_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXTERNAL_GRADEIF = _REPO_ROOT / "external" / "GraDe_IF"


def _inject_sys_path() -> None:
    if not _EXTERNAL_GRADEIF.is_dir():
        # Upstream not yet cloned. Module-level imports below will raise an
        # ImportError with a clearer message than the default traceback.
        raise ImportError(
            "GraDe-IF backbone not found at " + str(_EXTERNAL_GRADEIF) + ". "
            "Run scripts/00_setup_external.sh to clone the upstream sources."
        )
    path_str = str(_EXTERNAL_GRADEIF)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


_inject_sys_path()
