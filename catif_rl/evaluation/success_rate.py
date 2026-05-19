"""Joint structure-function Success Rate (SR) at multiple Δlog10 k_cat thresholds.

SR is defined in manuscript §2.7 as the fraction of generated variants
satisfying a joint criterion:

    pLDDT > 90
    backbone RMSD < 4 Å
    Δlog10 k_cat > δ

The benchmark reports SR at δ ∈ {0, 0.25, 0.5, 1.0, 1.5, 2.0}; see SI Table S9
for the threshold-sensitivity analysis.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_DELTAS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)
PLDDT_THRESHOLD = 90.0
RMSD_THRESHOLD = 4.0


def compute_sr(
    per_variant: pd.DataFrame,
    deltas: Iterable[float] = DEFAULT_DELTAS,
    plddt_threshold: float = PLDDT_THRESHOLD,
    rmsd_threshold: float = RMSD_THRESHOLD,
) -> pd.DataFrame:
    """Compute SR(δ) for one or more δ thresholds.

    Parameters
    ----------
    per_variant
        Frame with one row per generated variant. Required columns:
        ``delta_lgkcat``, ``plddt``, ``rmsd``.
    deltas
        δ thresholds to evaluate. Default is the SI Table S9 set.

    Returns
    -------
    pandas.DataFrame
        Two-column frame ``[delta, sr]``.
    """

    pl_ok = per_variant["plddt"] > plddt_threshold
    rm_ok = per_variant["rmsd"] < rmsd_threshold
    structural_ok = pl_ok & rm_ok

    rows = []
    n_total = len(per_variant)
    if n_total == 0:
        return pd.DataFrame(columns=["delta", "sr"])

    for d in deltas:
        kcat_ok = per_variant["delta_lgkcat"] > d
        sr = float(np.mean(structural_ok & kcat_ok))
        rows.append({"delta": d, "sr": sr})
    return pd.DataFrame(rows)
