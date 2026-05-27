"""Joint structure-function Success Rate (SR) at multiple Δlog10 k_cat thresholds.

SR is defined in manuscript §2.7 as the fraction of generated variants
satisfying a joint criterion:

    pLDDT > 90
    backbone RMSD < 4 Å
    Δlog10 k_cat > δ

The benchmark reports SR at δ ∈ {0, 0.25, 0.5, 1.0, 1.5, 2.0}; see SI Table S9
for the threshold-sensitivity analysis.

CLI
---

    python -m catif_rl.evaluation.success_rate \
      --in-csv  master_per_variant.csv \
      --out-csv sr_per_delta.csv

The input CSV must contain at least ``delta_lgkcat``, ``plddt``, ``rmsd``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
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


def _parse_deltas(arg: str) -> list[float]:
    """Parse a comma-separated δ list, e.g. '0,0.25,0.5,1.0,1.5,2.0'."""
    try:
        return [float(x) for x in arg.split(",") if x.strip()]
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            "--deltas must be a comma-separated list of floats, got " + repr(arg)
        ) from e


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m catif_rl.evaluation.success_rate",
        description=(
            "Compute SR(δ) = fraction of variants with pLDDT > T_pLDDT, "
            "backbone RMSD < T_RMSD, and Δlog10 k_cat > δ, for one or more δ "
            "thresholds. Reproduces the manuscript's §2.7 / SI Table S9 metric."
        ),
    )
    p.add_argument("--in-csv", type=Path, required=True,
                   help="Per-variant CSV with columns delta_lgkcat, plddt, rmsd.")
    p.add_argument("--out-csv", type=Path, required=True,
                   help="Output CSV with columns [delta, sr] (one row per δ).")
    p.add_argument("--deltas", type=_parse_deltas,
                   default=list(DEFAULT_DELTAS),
                   help=("Comma-separated δ thresholds (default: SI Table S9 set "
                         + ",".join(str(d) for d in DEFAULT_DELTAS) + ")."))
    p.add_argument("--plddt-threshold", type=float, default=PLDDT_THRESHOLD,
                   help="pLDDT threshold (default: %(default)s).")
    p.add_argument("--rmsd-threshold", type=float, default=RMSD_THRESHOLD,
                   help="Backbone-RMSD threshold in angstroms (default: %(default)s).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.in_csv.is_file():
        print(f"[success_rate][ERROR] input CSV not found: {args.in_csv}", file=sys.stderr)
        return 2
    df = pd.read_csv(args.in_csv)
    missing = {"delta_lgkcat", "plddt", "rmsd"} - set(df.columns)
    if missing:
        print(
            "[success_rate][ERROR] input CSV is missing required columns: "
            + ", ".join(sorted(missing)),
            file=sys.stderr,
        )
        return 2
    out = compute_sr(
        df,
        deltas=args.deltas,
        plddt_threshold=args.plddt_threshold,
        rmsd_threshold=args.rmsd_threshold,
    )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    # Echo to stdout so the script log captures the table.
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
