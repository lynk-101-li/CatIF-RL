#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_master_per_protein.py — Master per-protein aggregator for SI Step 6.

Reads per-baseline (per-seed-per-substrate) DLKcat output CSVs from
materials/05_baseline_outputs/ and emits ONE wide table indexed by ProID:

  master_per_protein.csv:  rows = ProID, columns = {model}__{metric}
  metrics: delta_lgKcat, recovery, Backbone_RMSD, Avg_pLDDT, n_obs

Aggregation rule:
  - kcat CSVs: each row is one (protein, substrate, seed) prediction.
    * Every baseline contributes 5 individual seed CSVs.  The catif and
      enzymeif folders use a non-standard cumulative-style label
      (_1, _12, _123, _1234, _12345) but each file holds one seed's output;
      proteinmpnn/esmif/etc. simply use _1.._5.
    * Within-protein delta_lgKcat / recovery are averaged over substrates × seeds.
  - GraDe-IF special case (matching the user's existing comparison notebook):
    * delta_lgKcat is computed from the "test_mut_substrate_gradeif_*"-pattern
      files (5 CSVs).
    * recovery is computed from the "test_mut_substrate_*_gradeif"-pattern
      files (5 CSVs).
    * For SI documentation purposes, both can be reported as coming from the
      kcat-pattern set.
  - rmsd CSVs are already per-protein (≈1,423 rows, single seed except
    enzymeif which is 5-seed averaged); merged on ProID.

This is the ONE intermediate that Tables S7/S8/S9 and Figures S5/S6/S7/S8
all consume.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------------
# Default paths (overridable via CLI)
# ----------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

# Legacy locations the user actually used to produce the published master CSV.
# In the released repo these are not present; --score-dir or --kcat-dir/--rmsd-dir
# should be used to point at runs/benchmark_scores/ instead.
DEFAULT_KCAT_DIR = ROOT / "materials/05_baseline_outputs/kcat_rr_metrics"
DEFAULT_RMSD_DIR = ROOT / "materials/05_baseline_outputs/rmsd_pldddt_metrics"
DEFAULT_OUT_CSV  = HERE / "master_per_protein.csv"

# ----------------------------------------------------------------------
# Per-baseline configuration
# ----------------------------------------------------------------------
# kcat_globs: list of relative-to-KCAT_DIR glob patterns to load + concat.
# rmsd_csv:   relative-to-RMSD_DIR exact filename (per-protein).
BASELINES = {
    "ProteinMPNN": {
        "kcat_globs": ["proteinmpnn/test_mut_substrate_proteinmpnn_*_kcatpred_dlkcat.csv"],
        "rmsd_csv":   "rmsd_plddt_proteinmpnn_test_mut_1.csv",
    },
    "ESM-IF": {
        "kcat_globs": ["esmif/test_mut_substrate_esmif_*_kcatpred_dlkcat.csv"],
        "rmsd_csv":   "rmsd_plddt_esmif_test_mut_1.csv",
    },
    "LigandMPNN": {
        "kcat_globs": ["ligandmpnn/test_mut_substrate_ligandmpnn_*_kcatpred_dlkcat.csv"],
        "rmsd_csv":   "rmsd_plddt_ligandmpnn_test_mut_1.csv",
    },
    "PiFold": {
        "kcat_globs": ["pifold/test_mut_substrate_pifold_*_kcatpred_dlkcat.csv"],
        "rmsd_csv":   "rmsd_plddt_pifold_test_mut_1.csv",
    },
    "ABACUS-T": {
        "kcat_globs": ["abacust/test_mut_substrate_abacust_*_kcatpred_dlkcat.csv"],
        "rmsd_csv":   "rmsd_plddt_abacust_test_mut_1.csv",
    },
    "GraDe-IF": {
        # Per user's existing notebook: delta_lgKcat and recovery use DIFFERENT
        # gradeif file sets.  kcat_globs is the source for delta_lgKcat;
        # recovery_globs (when present) overrides the source for recovery.
        "kcat_globs":     ["gradeif/test_mut_substrate_gradeif_*_kcatpred_dlkcat.csv"],
        "recovery_globs": ["gradeif/test_mut_substrate_*_gradeif_kcatpred_dlkcat.csv"],
        "rmsd_csv":       "rmsd_plddt_gradeif_test_mut_1.csv",
    },
    "EnzymeIF": {
        # 5 seed files, cumulative-style naming (_1, _12, _123, _1234, _12345)
        # but each file is one independent seed's output.
        "kcat_globs": ["enzymeif/test_dataset/test_mut_substrate_enzymeif_*_kcatpred_dlkcat.csv"],
        "rmsd_csv":   "rmsd_plddt_enzymeif_test_mut_12345.csv",
    },
    "CatIF": {
        # Same as EnzymeIF: cumulative-style naming for 5 independent seeds.
        "kcat_globs": ["catif/test_mut_substrate_catif_*_Sep24_kcatpred_dlkcat.csv"],
        "rmsd_csv":   "rmsd_plddt_catif_test_mut_1.csv",
    },
    "CatIF-RL R1": {
        "kcat_globs": ["catif_rl/Jan30/round1_Jan30_epoch02/"
                       "test_mut_substrate_rl_round1_Jan30_epoch02_seed*_kcatpred_dlkcat.csv"],
        "rmsd_csv":   "rmsd_plddt_round1_Jan30_epoch02_test_mut_1.csv",
    },
    "CatIF-RL R2": {
        "kcat_globs": ["catif_rl/Jan30/round2_Jan30_epoch02/"
                       "test_mut_substrate_rl_round2_Jan30_epoch02_seed*_kcatpred_dlkcat.csv"],
        "rmsd_csv":   "rmsd_plddt_round2_Jan30_epoch02_test_mut_1.csv",
    },
    "CatIF-RL R3": {
        "kcat_globs": ["catif_rl/Jan30/round3_Jan30_epoch02/"
                       "test_mut_substrate_rl_round3_Jan30_epoch02_seed*_kcatpred_dlkcat.csv"],
        "rmsd_csv":   "rmsd_plddt_round3_Jan30_epoch02_test_mut_1.csv",
    },
}

# Used as the canonical "CatIF-RL" reference column for SI tables.
CATIF_RL_LABEL = "CatIF-RL R3"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _recovery_rate(seq_a: str, seq_b: str) -> float:
    a, b = str(seq_a), str(seq_b)
    if not a or len(a) != len(b):
        return float("nan")
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def _normalise_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df


def _load_kcat(globs: list[str], kcat_dir: Path) -> pd.DataFrame:
    parts = []
    for g in globs:
        for path in sorted(glob.glob(str(kcat_dir / g))):
            df = _normalise_cols(pd.read_csv(path))
            parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _per_protein_delta(kcat_df: pd.DataFrame) -> pd.DataFrame:
    """Per-protein mean delta_lgKcat (averaged over substrates × seeds)."""
    if kcat_df.empty:
        return pd.DataFrame(columns=["ProID", "delta_lgKcat", "n_obs"])
    df = kcat_df.copy()
    df["ProID"] = df["ProID"].astype(str)
    return df.groupby("ProID").agg(
        delta_lgKcat=("delta_lgKcat", "mean"),
        n_obs=("delta_lgKcat", "count"),
    ).reset_index()


def _per_protein_recovery(rec_df: pd.DataFrame) -> pd.DataFrame:
    """Per-protein mean recovery rate, computed from ProSeq vs ProSeq'."""
    if rec_df.empty:
        return pd.DataFrame(columns=["ProID", "recovery"])
    df = rec_df.copy()
    df["ProID"] = df["ProID"].astype(str)
    df["recovery"] = [
        _recovery_rate(a, b) for a, b in zip(df["ProSeq"], df["ProSeq'"])
    ]
    return df.groupby("ProID").agg(
        recovery=("recovery", "mean"),
    ).reset_index()


def _per_protein_rmsd(rmsd_csv: Path) -> pd.DataFrame:
    if not rmsd_csv.exists():
        print(f"  [warn] missing RMSD csv: {rmsd_csv.name}")
        return pd.DataFrame(columns=["ProID", "Backbone_RMSD", "Avg_pLDDT"])
    df = _normalise_cols(pd.read_csv(rmsd_csv))
    rename = {}
    for c in df.columns:
        cl = c.lower().replace(" ", "_").replace("-", "_")
        if cl in {"backbone_rmsd", "rmsd"}:
            rename[c] = "Backbone_RMSD"
        elif "plddt" in cl:
            rename[c] = "Avg_pLDDT"
        elif cl == "proid":
            rename[c] = "ProID"
    df = df.rename(columns=rename)
    keep = [c for c in ["ProID", "Backbone_RMSD", "Avg_pLDDT"] if c in df.columns]
    out = df[keep].drop_duplicates(subset=["ProID"], keep="first").reset_index(drop=True)
    out["ProID"] = out["ProID"].astype(str)
    return out


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def _build_master_from_baselines(kcat_dir: Path, rmsd_dir: Path) -> pd.DataFrame:
    """Aggregate per-baseline outputs using the BASELINES glob table."""
    per_baseline: dict[str, pd.DataFrame] = {}
    for name, cfg in BASELINES.items():
        print(f"\n[load] {name}")
        kcat_df = _load_kcat(cfg["kcat_globs"], kcat_dir)
        if kcat_df.empty:
            print(f"  [warn] no kcat data found for {name} (globs: {cfg['kcat_globs']})")
            continue
        n_kcat = len({p for g in cfg["kcat_globs"]
                      for p in glob.glob(str(kcat_dir / g))})
        rec_globs = cfg.get("recovery_globs") or cfg["kcat_globs"]
        rec_df = _load_kcat(rec_globs, kcat_dir)
        n_rec = len({p for g in rec_globs
                     for p in glob.glob(str(kcat_dir / g))})
        if rec_globs is cfg["kcat_globs"] or rec_globs == cfg["kcat_globs"]:
            print(f"  kcat & recovery: {n_kcat} file(s), {len(kcat_df):,} rows")
        else:
            print(f"  kcat:     {n_kcat} file(s), {len(kcat_df):,} rows")
            print(f"  recovery: {n_rec} file(s), {len(rec_df):,} rows  "
                  "(GraDe-IF dual-source per user's existing pipeline)")
        per_delta = _per_protein_delta(kcat_df)
        per_rec = _per_protein_recovery(rec_df)
        per_rmsd = _per_protein_rmsd(rmsd_dir / cfg["rmsd_csv"])
        merged = per_delta.merge(per_rec, on="ProID", how="outer") \
                          .merge(per_rmsd, on="ProID", how="left")
        print(f"  proteins: {len(merged):,}  | "
              f"ΔlgKcat mean = {merged['delta_lgKcat'].mean():+.4f}  | "
              f"recovery mean = {merged['recovery'].mean():.4f}  | "
              f"pLDDT mean = {merged['Avg_pLDDT'].mean():.2f}  | "
              f"RMSD mean = {merged['Backbone_RMSD'].mean():.3f}")
        per_baseline[name] = merged

    # Pivot to wide table
    master = None
    metric_cols = ["delta_lgKcat", "recovery", "Backbone_RMSD", "Avg_pLDDT", "n_obs"]
    for name, df in per_baseline.items():
        sub = df[["ProID"] + metric_cols].copy()
        sub.columns = ["ProID"] + [f"{name}__{c}" for c in metric_cols]
        master = sub if master is None else master.merge(sub, on="ProID", how="outer")
    if master is None:
        return pd.DataFrame()
    return master.sort_values("ProID").reset_index(drop=True)


def _build_master_from_score_dir(score_dir: Path) -> pd.DataFrame:
    """Aggregate per-method outputs from the runs/benchmark_scores/ layout.

    Expected per method ``<score_dir>/<method>/``:
      - ``test_mut_substrate_seed*.csv_kcatpred_dlkcat.csv`` per seed
        (each row: ProID, ProSeq, ProSeq', SMILES, delta_lgKcat, ...)
      - ``rmsd_plddt.csv`` (per-protein, one seed -- the first benchmark seed
        per SI §2.7)

    Method names are taken from the subdirectory names.
    """
    if not score_dir.is_dir():
        raise FileNotFoundError(f"--score-dir does not exist: {score_dir}")
    method_dirs = sorted(p for p in score_dir.iterdir() if p.is_dir())
    if not method_dirs:
        raise RuntimeError(f"no method subdirectories under {score_dir}")

    per_method: dict[str, pd.DataFrame] = {}
    for md in method_dirs:
        name = md.name
        kcat_paths = sorted(glob.glob(str(md / "*_kcatpred_dlkcat.csv")))
        if not kcat_paths:
            kcat_paths = sorted(glob.glob(str(md / "seed_*/*_kcatpred_dlkcat.csv")))
        if not kcat_paths:
            print(f"[load] {name}: no DLKcat outputs; skipping")
            continue
        parts = [_normalise_cols(pd.read_csv(p)) for p in kcat_paths]
        kcat_df = pd.concat(parts, ignore_index=True)
        per_delta = _per_protein_delta(kcat_df)
        per_rec   = _per_protein_recovery(kcat_df)
        rmsd_csv  = md / "rmsd_plddt.csv"
        per_rmsd  = _per_protein_rmsd(rmsd_csv)
        merged = per_delta.merge(per_rec, on="ProID", how="outer") \
                          .merge(per_rmsd, on="ProID", how="left")
        print(f"[load] {name}: {len(kcat_paths)} kcat file(s), {len(merged):,} proteins")
        per_method[name] = merged

    master = None
    metric_cols = ["delta_lgKcat", "recovery", "Backbone_RMSD", "Avg_pLDDT", "n_obs"]
    for name, df in per_method.items():
        sub = df[["ProID"] + metric_cols].copy()
        sub.columns = ["ProID"] + [f"{name}__{c}" for c in metric_cols]
        master = sub if master is None else master.merge(sub, on="ProID", how="outer")
    if master is None:
        return pd.DataFrame()
    return master.sort_values("ProID").reset_index(drop=True)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Master per-protein aggregator. Reads per-baseline DLKcat + RMSD/pLDDT "
                    "outputs and emits one wide CSV indexed by ProID."
    )
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--score-dir", type=Path, default=None,
                     help="Path to runs/benchmark_scores/ (preferred mode; reads per-method "
                          "subdirectories produced by scripts/07_score_benchmark.sh)")
    src.add_argument("--kcat-dir", type=Path, default=DEFAULT_KCAT_DIR,
                     help="(Legacy mode) directory containing per-baseline DLKcat output "
                          "subdirectories with the historical filename patterns")
    p.add_argument("--rmsd-dir", type=Path, default=DEFAULT_RMSD_DIR,
                   help="(Legacy mode) directory containing per-baseline rmsd_plddt CSVs")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT_CSV,
                   help=f"output master CSV path (default: {DEFAULT_OUT_CSV})")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if args.score_dir is not None:
        print(f"[mode] score-dir layout: {args.score_dir}")
        master = _build_master_from_score_dir(args.score_dir)
    else:
        print(f"[mode] legacy BASELINES layout: kcat={args.kcat_dir}, rmsd={args.rmsd_dir}")
        master = _build_master_from_baselines(args.kcat_dir, args.rmsd_dir)
    if master.empty:
        print("[error] no data aggregated -- check the input paths")
        return 1
    print(f"\n[summary] master shape: {master.shape}, proteins: {master['ProID'].nunique():,}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(args.output, index=False)
    print(f"[save] {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
