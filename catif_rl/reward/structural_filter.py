"""
Apply the manuscript's structural plausibility gate to a CSV of predicted
backbone-and-pLDDT metrics, producing two CSVs: the variants that pass and
the ones that are rejected.

Gate (manuscript Section 2.3):

    PASS  =  (Backbone_RMSD < rmsd_threshold) & (Avg_pLDDT > plddt_threshold)
    FAIL  =  NOT PASS

i.e. a variant is kept only if BOTH the backbone fit is tight AND the
ESMFold confidence is high. Defaults: rmsd_threshold=4.0 Å, plddt_threshold=90.

Input CSV must contain at least the columns ``CA_RMSD``, ``Backbone_RMSD``,
``Avg_pLDDT``; any other columns are preserved verbatim and copied into
both output CSVs.

Usage (as invoked from ``scripts/03_run_gdc.sh``)::

    python -m catif_rl.reward.structural_filter \\
      --input   runs/gdc/rmsd_plddt.csv \\
      --good    runs/gdc/structure_valid.csv \\
      --bad     runs/gdc/structure_invalid.csv

Override the thresholds with ``--rmsd 4.0 --plddt 90``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def apply_structural_gate(df: pd.DataFrame,
                          rmsd_threshold: float = 4.0,
                          plddt_threshold: float = 90.0) -> pd.Series:
    """Return a boolean Series: True = variant passes the structural gate.

    The gate is ``(Backbone_RMSD < rmsd_threshold) & (Avg_pLDDT > plddt_threshold)``.
    Rows with missing values in either column fail by default.
    """
    required = {"Backbone_RMSD", "Avg_pLDDT"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"input CSV is missing required columns: {sorted(missing)}")
    good = (df["Backbone_RMSD"] < rmsd_threshold) & (df["Avg_pLDDT"] > plddt_threshold)
    # Coerce NaN-in-mask to False (don't accidentally pass NaN rows)
    return good.fillna(False).astype(bool)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Apply the manuscript's structural plausibility gate "
                    "(backbone RMSD + ESMFold pLDDT) to a metrics CSV."
    )
    p.add_argument("--input", required=True, type=Path,
                   help="input CSV with CA_RMSD / Backbone_RMSD / Avg_pLDDT columns")
    p.add_argument("--good", required=True, type=Path,
                   help="output CSV for rows that PASS the gate")
    p.add_argument("--bad", required=True, type=Path,
                   help="output CSV for rows that FAIL the gate")
    p.add_argument("--rmsd", dest="rmsd_threshold", type=float, default=4.0,
                   help="backbone RMSD upper bound, exclusive (default: 4.0 Angstrom)")
    p.add_argument("--plddt", dest="plddt_threshold", type=float, default=90.0,
                   help="mean pLDDT lower bound, exclusive (default: 90)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    df = pd.read_csv(args.input)
    good_mask = apply_structural_gate(df, args.rmsd_threshold, args.plddt_threshold)
    good = df[good_mask].reset_index(drop=True)
    bad = df[~good_mask].reset_index(drop=True)
    args.good.parent.mkdir(parents=True, exist_ok=True)
    args.bad.parent.mkdir(parents=True, exist_ok=True)
    good.to_csv(args.good, index=False)
    bad.to_csv(args.bad, index=False)
    print(f"[OK] structural gate: {len(good)} passed, {len(bad)} rejected "
          f"(RMSD < {args.rmsd_threshold}, pLDDT > {args.plddt_threshold})")
    print(f"  good -> {args.good}")
    print(f"  bad  -> {args.bad}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
