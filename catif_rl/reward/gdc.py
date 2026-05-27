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
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Join keys used across the candidate / structural / predictor CSVs.
JOIN_KEYS = ["ProID", "ProSeq'"]
DELTA_COL = "delta_lgKcat"
NORMALIZER_FILENAME = "normalizer.json"   # frozen scale calibration, see M1 below


def _compute_one_normalizer(x: pd.Series) -> dict:
    """Compute the scale-only normalizer parameters for one predictor's deltas.

    Mirrors the formulation in ``catif_rl.reward.ensemble_gdc`` / ``ensemble_rl``:
    scale = q90 - q10; if degenerate, fall back to std(x); then to 1.

    Returns
    -------
    dict with keys ``q10``, ``q90``, ``fallback`` (one of ``"q90-q10"``,
    ``"std"``, ``"fallback_1"``), and ``scale`` (the divisor actually used).
    """
    x = pd.to_numeric(x, errors="coerce")
    q10 = x.quantile(0.10)
    q90 = x.quantile(0.90)
    fallback = "q90-q10"
    scale = (q90 - q10) if pd.notna(q10) and pd.notna(q90) else float("nan")
    if not pd.notna(scale) or scale == 0:
        sd = x.std(ddof=0)
        if pd.notna(sd) and sd > 0:
            scale, fallback = float(sd), "std"
        else:
            scale, fallback = 1.0, "fallback_1"
    return {
        "q10": float(q10) if pd.notna(q10) else None,
        "q90": float(q90) if pd.notna(q90) else None,
        "fallback": fallback,
        "scale": float(scale),
    }


def _apply_normalizer(x: pd.Series, scale: float) -> pd.Series:
    """Divide by a saved scale (no centering, sign preserved)."""
    return pd.to_numeric(x, errors="coerce") / float(scale)


def _scale_only_norm(x: pd.Series) -> pd.Series:
    """Back-compat shim: compute the GDC-style normalizer on-the-fly and apply it.

    Used by the in-process path inside ``run_gdc_funnel`` when no pre-computed
    normalizer is being threaded through. New callers should prefer
    ``_compute_one_normalizer`` + ``_apply_normalizer`` so the calibration can
    be persisted to ``normalizer.json`` and reused across rounds.
    """
    return _apply_normalizer(x, _compute_one_normalizer(x)["scale"])


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

    # Compute the q10 / q90 / std-fallback normalizer ONCE on the structure-
    # valid GDC pool and persist it to normalizer.json. Subsequent RL rounds
    # load this calibration via `catif_rl.reward.ensemble_rl --normalizer ...`
    # instead of recomputing quantiles on every round's distribution, which
    # is what SI Table S5 calls the "frozen" reward scale.
    normalizer = {
        "version": "1.0",
        "frozen_at": _dt.datetime.utcnow().isoformat() + "Z",
        "frozen_from": str(output_dir.resolve()),
        "n_samples_used": int(len(merged)),
        "rmsd_threshold": float(rmsd_threshold),
        "plddt_threshold": float(plddt_threshold),
        "predictors": {
            "dlkcat":  _compute_one_normalizer(merged["delta_dlkcat"]),
            "unikp":   _compute_one_normalizer(merged["delta_unikp"]),
            "catapro": _compute_one_normalizer(merged["delta_catapro"]),
        },
    }
    normalizer_path = output_dir / NORMALIZER_FILENAME
    with normalizer_path.open("w") as f:
        json.dump(normalizer, f, indent=2)
    print(f"[gdc] froze normalizer over {normalizer['n_samples_used']:,} structure-valid variants:")
    for tag, params in normalizer["predictors"].items():
        print(f"  {tag:8s} scale={params['scale']:.6g}  (fallback={params['fallback']}, "
              f"q10={params['q10']!s:>8}, q90={params['q90']!s:>8})")
    print(f"[gdc] wrote: {normalizer_path}")

    # Apply the frozen scales to compute mean3 (equivalent to the old
    # on-the-fly _scale_only_norm path, but with the saved scale on record).
    merged["s_dlkcat"]  = _apply_normalizer(merged["delta_dlkcat"],
                                            normalizer["predictors"]["dlkcat"]["scale"])
    merged["s_unikp"]   = _apply_normalizer(merged["delta_unikp"],
                                            normalizer["predictors"]["unikp"]["scale"])
    merged["s_catapro"] = _apply_normalizer(merged["delta_catapro"],
                                            normalizer["predictors"]["catapro"]["scale"])
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
        "normalizer_json": str(normalizer_path),
    }


def _auto_discover_predictor(predictions_dir: Path, tag: str) -> Path:
    """Locate the single ``*_kcatpred_<tag>.csv`` under ``predictions_dir``.

    Matches the manuscript filename convention emitted by
    ``catif_rl.reward.predictors.<tag>``.
    """
    import glob as _glob
    matches = sorted(_glob.glob(str(predictions_dir / f"*_kcatpred_{tag}.csv")))
    if not matches:
        raise SystemExit(
            f"no *_kcatpred_{tag}.csv under {predictions_dir}; "
            f"pass --{tag}-csv <path> or check that the wrapper ran with output_dir pointing here."
        )
    if len(matches) > 1:
        raise SystemExit(
            f"multiple *_kcatpred_{tag}.csv under {predictions_dir}: "
            f"{[Path(m).name for m in matches]}; pass --{tag}-csv to disambiguate."
        )
    return Path(matches[0])


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidates",         type=Path, required=True,
                   help="raw candidate pool (typically 62,900 rows from EnzymeIF sampling)")
    p.add_argument("--structural-metrics", type=Path, required=True,
                   help="ESMFold refold metrics with Backbone_RMSD + Avg_pLDDT per variant")
    # When --predictions-dir is set and any of the three per-predictor CSVs
    # is omitted, the wrapper auto-discovers <stem>_kcatpred_<tag>.csv there.
    p.add_argument("--predictions-dir",    type=Path, default=None,
                   help="directory holding *_kcatpred_{dlkcat,unikp,catapro}.csv "
                        "(used to auto-discover any predictor CSV not given explicitly)")
    p.add_argument("--dlkcat-csv",         type=Path, default=None,
                   help="DLKcat per-variant predictions; if omitted, auto-discovered under --predictions-dir")
    p.add_argument("--unikp-csv",          type=Path, default=None)
    p.add_argument("--catapro-csv",        type=Path, default=None)
    p.add_argument("--output-dir",         type=Path, required=True,
                   help="destination for structure_valid.csv and activity_positive.csv")
    p.add_argument("--rmsd-threshold",     type=float, default=4.0)
    p.add_argument("--plddt-threshold",    type=float, default=90.0)
    p.add_argument("--ensemble-threshold", type=float, default=0.0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    # Fill in any predictor CSVs that weren't explicitly given.
    if (args.dlkcat_csv is None or args.unikp_csv is None or args.catapro_csv is None):
        if args.predictions_dir is None:
            raise SystemExit(
                "must pass either --predictions-dir or all three of "
                "--dlkcat-csv / --unikp-csv / --catapro-csv"
            )
        if args.dlkcat_csv is None:
            args.dlkcat_csv = _auto_discover_predictor(args.predictions_dir, "dlkcat")
        if args.unikp_csv is None:
            args.unikp_csv = _auto_discover_predictor(args.predictions_dir, "unikp")
        if args.catapro_csv is None:
            args.catapro_csv = _auto_discover_predictor(args.predictions_dir, "catapro")
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
