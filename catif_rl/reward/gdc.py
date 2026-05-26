"""Generative Dataset Curation (GDC) funnel orchestration.

Implements the two-stage candidate filter described in manuscript §2.3:

1. Structural plausibility -- backbone RMSD < 4 Angstrom AND mean pLDDT > 90
   after ESMFold refolding (delegated to
   :mod:`catif_rl.reward.structural_filter`).
2. Activity ensemble -- normalize each of DLKcat / UniKP / CataPro
   delta_lgKcat onto a comparable scale via 10th/90th quantile range
   (fall back to std, then 1.0), then take the mean. Variants with
   ``S_ensemble > 0`` are retained as activity-positive.

The ensemble quantile parameters are frozen at the start of GDC and reused
through all CatIF supervised training, GRPO rounds 1-3, and final test
evaluation (SI Table S5 note 3).

Usage (as invoked from ``scripts/03_run_gdc.sh``)::

    python -m catif_rl.reward.gdc \\
      --candidates           runs/enzymeif/gdc_candidates.csv \\
      --structural-metrics   runs/enzymeif/gdc_structural_metrics.csv \\
      --dlkcat-csv           runs/gdc/predictions/dlkcat_pred.csv \\
      --unikp-csv            runs/gdc/predictions/unikp_pred.csv \\
      --catapro-csv          runs/gdc/predictions/catapro_pred.csv \\
      --output-dir           runs/gdc \\
      --rmsd-threshold       4.0 \\
      --plddt-threshold      90.0 \\
      --ensemble-threshold   0.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Join keys used across the candidate / structural / predictor CSVs.
JOIN_KEYS = ["ProID", "ProSeq'"]
DELTA_COL = "delta_lgKcat"


def _scale_only_norm(x: pd.Series) -> pd.Series:
    """Scale-only normalize a Series: x / scale, with q90-q10 -> std -> 1 fallback.

    Matches the formulation used in catif_rl.reward.ensemble_gdc and
    ensemble_rl. Sign is preserved (no centering).
    """
    x = pd.to_numeric(x, errors="coerce")
    q10, q90 = x.quantile(0.10), x.quantile(0.90)
    scale = (q90 - q10) if pd.notna(q10) and pd.notna(q90) else float("nan")
    if not pd.notna(scale) or scale == 0:
        sd = x.std(ddof=0)
        scale = sd if (pd.notna(sd) and sd > 0) else 1.0
    return x / scale


def _read_predictor_csv(path: Path, tag: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = set(JOIN_KEYS + [DELTA_COL])
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"{path} is missing required columns: {sorted(missing)}")
    df = df[JOIN_KEYS + [DELTA_COL]].copy()
    df = df.rename(columns={DELTA_COL: f"delta_{tag}"})
    for k in JOIN_KEYS:
        df[k] = df[k].astype(str)
    return df.drop_duplicates(subset=JOIN_KEYS, keep="first")


def run_gdc_funnel(
    candidate_csv: Path,
    structural_metrics_csv: Path,
    dlkcat_csv: Path,
    unikp_csv: Path,
    catapro_csv: Path,
    output_dir: Path,
    rmsd_threshold: float = 4.0,
    plddt_threshold: float = 90.0,
    ensemble_threshold: float = 0.0,
) -> dict:
    """Run the two-stage GDC filter end-to-end.

    Returns
    -------
    summary
        Dict with keys ``n_candidates``, ``n_structure_valid``,
        ``n_activity_positive``, plus the output file paths.
    """
    from catif_rl.reward.structural_filter import apply_structural_gate

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Stage 1: structural plausibility -------------------------------------
    cands = pd.read_csv(candidate_csv)
    structs = pd.read_csv(structural_metrics_csv)

    # Stash original count for the summary
    n_cands = len(cands)

    # Apply the gate on the structural metrics CSV; carry forward only the
    # passing variants by joining back to the candidate pool on (ProID, ProSeq').
    for k in JOIN_KEYS:
        if k not in cands.columns or k not in structs.columns:
            raise KeyError(
                f"both candidate and structural CSV must contain {k}; "
                f"cands has {sorted(cands.columns)}, structs has {sorted(structs.columns)}"
            )
        cands[k] = cands[k].astype(str)
        structs[k] = structs[k].astype(str)
    good_mask = apply_structural_gate(structs, rmsd_threshold, plddt_threshold)
    valid_keys = structs.loc[good_mask, JOIN_KEYS].drop_duplicates()
    structure_valid = cands.merge(valid_keys, on=JOIN_KEYS, how="inner")

    struct_path = output_dir / "structure_valid.csv"
    structure_valid.to_csv(struct_path, index=False)
    n_struct_valid = len(structure_valid)
    print(f"[gdc] structural gate: {n_struct_valid} of {n_cands} variants pass "
          f"(RMSD < {rmsd_threshold} AND pLDDT > {plddt_threshold})")

    if n_struct_valid == 0:
        print("[gdc] no structure-valid variants; skipping activity stage")
        return {
            "n_candidates": n_cands,
            "n_structure_valid": 0,
            "n_activity_positive": 0,
            "structure_valid_csv": str(struct_path),
            "activity_positive_csv": None,
        }

    # --- Stage 2: three-predictor activity ensemble ---------------------------
    d_dlk = _read_predictor_csv(dlkcat_csv, "dlkcat")
    d_uni = _read_predictor_csv(unikp_csv, "unikp")
    d_cat = _read_predictor_csv(catapro_csv, "catapro")

    for df in (structure_valid,):
        for k in JOIN_KEYS:
            df[k] = df[k].astype(str)

    # Inner-join structure-valid candidates with all three predictors.
    merged = (structure_valid
              .merge(d_dlk, on=JOIN_KEYS, how="inner")
              .merge(d_uni, on=JOIN_KEYS, how="inner")
              .merge(d_cat, on=JOIN_KEYS, how="inner"))

    merged["s_dlkcat"]  = _scale_only_norm(merged["delta_dlkcat"])
    merged["s_unikp"]   = _scale_only_norm(merged["delta_unikp"])
    merged["s_catapro"] = _scale_only_norm(merged["delta_catapro"])
    merged["mean3"]     = (merged["s_dlkcat"] + merged["s_unikp"] + merged["s_catapro"]) / 3.0

    activity_positive = merged.loc[merged["mean3"] > ensemble_threshold].reset_index(drop=True)
    n_pos = len(activity_positive)

    pos_path = output_dir / "activity_positive.csv"
    activity_positive.to_csv(pos_path, index=False)
    print(f"[gdc] activity ensemble: {n_pos} of {len(merged)} structure-valid variants "
          f"have mean3 > {ensemble_threshold}")
    print(f"[gdc] wrote: {pos_path}")

    return {
        "n_candidates": n_cands,
        "n_structure_valid": n_struct_valid,
        "n_activity_positive": n_pos,
        "structure_valid_csv": str(struct_path),
        "activity_positive_csv": str(pos_path),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidates",         type=Path, required=True,
                   help="raw candidate pool (typically 62,900 rows from EnzymeIF sampling)")
    p.add_argument("--structural-metrics", type=Path, required=True,
                   help="ESMFold refold metrics with Backbone_RMSD + Avg_pLDDT per variant")
    p.add_argument("--dlkcat-csv",         type=Path, required=True,
                   help="DLKcat per-variant predictions (must contain delta_lgKcat)")
    p.add_argument("--unikp-csv",          type=Path, required=True)
    p.add_argument("--catapro-csv",        type=Path, required=True)
    p.add_argument("--output-dir",         type=Path, required=True,
                   help="destination for structure_valid.csv and activity_positive.csv")
    p.add_argument("--rmsd-threshold",     type=float, default=4.0)
    p.add_argument("--plddt-threshold",    type=float, default=90.0)
    p.add_argument("--ensemble-threshold", type=float, default=0.0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    summary = run_gdc_funnel(
        candidate_csv=args.candidates,
        structural_metrics_csv=args.structural_metrics,
        dlkcat_csv=args.dlkcat_csv,
        unikp_csv=args.unikp_csv,
        catapro_csv=args.catapro_csv,
        output_dir=args.output_dir,
        rmsd_threshold=args.rmsd_threshold,
        plddt_threshold=args.plddt_threshold,
        ensemble_threshold=args.ensemble_threshold,
    )
    print(f"[gdc] summary: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
