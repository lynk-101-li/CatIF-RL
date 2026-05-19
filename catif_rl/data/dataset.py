"""High-level dataset interface used by training and sampling.

The underlying ``Cath`` PyTorch-Geometric :class:`~torch_geometric.data.Dataset`
implementation lives in :mod:`catif_rl.data.large_dataset` (preserved verbatim
from the original code base for compatibility). This module re-exports it
under a more descriptive name and may grow specialised subclasses in the
future (for example, separate ``EnzymeGraphDataset`` and
``CATHRegularizerDataset`` wrappers if their loading needs diverge).
"""

from catif_rl.data.large_dataset import Cath

# Aliases used elsewhere in the codebase. Currently both point at the same
# underlying class; their behaviour is selected by the directory the loader
# is pointed at (``train/`` vs ``valid/`` vs ``test/``).
EnzymeGraphDataset = Cath
CATHRegularizerDataset = Cath

__all__ = ["Cath", "EnzymeGraphDataset", "CATHRegularizerDataset"]
