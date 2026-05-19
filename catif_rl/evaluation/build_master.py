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

import glob
from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
KCAT_DIR = ROOT / "materials/05_baseline_outputs/kcat_rr_metrics"
RMSD_DIR = ROOT / "materials/05_baseline_outputs/rmsd_pldddt_metrics"

OUT_CSV = HERE / "master_per_protein.csv"

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


def _load_kcat(globs: list[str]) -> pd.DataFrame:
    parts = []
    for g in globs:
        for path in sorted(glob.glob(str(KCAT_DIR / g))):
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

def main() -> None:
    print(f"[paths] KCAT_DIR = {KCAT_DIR}")
    print(f"[paths] RMSD_DIR = {RMSD_DIR}")
    print(f"[paths] OUT_CSV  = {OUT_CSV}")

    per_baseline: dict[str, pd.DataFrame] = {}
    for name, cfg in BASELINES.items():
        print(f"\n[load] {name}")

        # ---- delta_lgKcat source (kcat_globs) ----
        kcat_df = _load_kcat(cfg["kcat_globs"])
        if kcat_df.empty:
            print(f"  [warn] no kcat data found for {name} (globs: {cfg['kcat_globs']})")
            continue
        n_kcat = len({p for g in cfg["kcat_globs"]
                      for p in glob.glob(str(KCAT_DIR / g))})

        # ---- recovery source (recovery_globs if specified, else kcat_globs) ----
        rec_globs = cfg.get("recovery_globs") or cfg["kcat_globs"]
        rec_df = _load_kcat(rec_globs)
        n_rec = len({p for g in rec_globs
                     for p in glob.glob(str(KCAT_DIR / g))})
        if rec_globs is cfg["kcat_globs"] or rec_globs == cfg["kcat_globs"]:
            print(f"  kcat & recovery: {n_kcat} file(s), {len(kcat_df):,} rows")
        else:
            print(f"  kcat:     {n_kcat} file(s), {len(kcat_df):,} rows")
            print(f"  recovery: {n_rec} file(s), {len(rec_df):,} rows  "
                  "(GraDe-IF dual-source per user's existing pipeline)")

        # ---- aggregate ----
        per_delta = _per_protein_delta(kcat_df)
        per_rec = _per_protein_recovery(rec_df)
        per_rmsd = _per_protein_rmsd(RMSD_DIR / cfg["rmsd_csv"])
        merged = per_delta.merge(per_rec, on="ProID", how="outer") \
                          .merge(per_rmsd, on="ProID", how="left")

        print(f"  proteins: {len(merged):,}  | "
              f"ΔlgKcat mean = {merged['delta_lgKcat'].mean():+.4f}  | "
              f"recovery mean = {merged['recovery'].mean():.4f}  | "
              f"pLDDT mean = {merged['Avg_pLDDT'].mean():.2f}  | "
              f"RMSD mean = {merged['Backbone_RMSD'].mean():.3f}")
        per_baseline[name] = merged

    # ---------- pivot to wide table ----------
    master = None
    metric_cols = ["delta_lgKcat", "recovery", "Backbone_RMSD", "Avg_pLDDT", "n_obs"]
    for name, df in per_baseline.items():
        sub = df[["ProID"] + metric_cols].copy()
        sub.columns = ["ProID"] + [f"{name}__{c}" for c in metric_cols]
        master = sub if master is None else master.merge(sub, on="ProID", how="outer")

    master = master.sort_values("ProID").reset_index(drop=True)

    print("\n[summary]")
    print(f"  master shape: {master.shape}")
    print(f"  proteins covered: {master['ProID'].nunique():,}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(OUT_CSV, index=False)
    print(f"\n[save] {OUT_CSV}")


if __name__ == "__main__":
    main()
