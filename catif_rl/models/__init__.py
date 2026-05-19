"""GraDe-IF backbone adapter and EGNN-based diffusion policy wrappers.

The upstream GraDe-IF source (``diffusion.gradeif``, ``diffusion.utils``, and
``diffusion.model.egnn_pytorch``) is cloned into ``external/GraDe_IF/`` by
``scripts/00_setup_external.sh``. ``gradeif_adapter`` injects that directory
into ``sys.path`` so the upstream modules become importable as if they were
part of this repository.
"""

from catif_rl.models import gradeif_adapter  # noqa: F401  (path injection side effect)
