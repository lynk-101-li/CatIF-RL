#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plan B (refined): scale-only normalization + mean-based selection + intersection counts.

- For each predictor's delta_lgKcat, normalize by scale only: x_scaled = x / scale.
  ``scale`` uses (q90 - q10); if degenerate, fall back to std, then to 1.
  No centering, signs are preserved.
- Compute:
    a) number of samples whose 3-predictor mean > 0 (deduplicated by ProSeq')
    b) number of samples whose DLKcat+UniKP mean > 0 (deduplicated by ProSeq')
    c) intersection of a and b (by ProSeq')
- Also save two distribution figures (log-scale y-axis) and per-set detail
  CSVs for double-checking.
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

IN_DIR   = Path("post_prediction")
POSI_DIR = Path("posi_seq")
FIG_DIR  = Path("figures")

FILES = {
    "dlkcat": "merged_mut_substrate_enzymeif_kcatpred_dlkcat.csv",
    "catapro":"merged_mut_substrate_enzymeif_kcatpred_catapro.csv",
    "unikp":  "merged_mut_substrate_enzymeif_kcatpred_unikp.csv",
}

KEYS = ['Group','ProID',"ProSeq'"]
DELTA_COL = 'delta_lgKcat'

def scale_only_norm(x: pd.Series):
    """Scale-only normalization: x_scaled = x / scale; no centering, signs preserved."""
    x = pd.to_numeric(x, errors='coerce')
    q10, q90 = x.quantile(0.10), x.quantile(0.90)
    scale = (q90 - q10) if pd.notna(q10) and pd.notna(q90) else np.nan
    method = "q90-q10"
    if not pd.notna(scale) or scale == 0:
        sd = x.std(ddof=0)
        if pd.notna(sd) and sd > 0:
            scale, method = sd, "std"
        else:
            scale, method = 1.0, "fallback_1"
    return x / scale, scale, method

def read_and_prepare(tag, filename):
    fp = IN_DIR / filename
    if not fp.exists():
        print(f"[ERROR] file missing: {fp}", file=sys.stderr); sys.exit(1)
    df = pd.read_csv(fp)
    need = set(KEYS + [DELTA_COL])
    miss = need - set(df.columns)
    if miss:
        print(f"[ERROR] {filename} is missing columns: {miss}", file=sys.stderr); sys.exit(1)
    df = df[KEYS + [DELTA_COL]].copy()
    df['Group']   = df['Group'].astype(str)
    df['ProID']   = df['ProID'].astype(str)
    df["ProSeq'"] = df["ProSeq'"].astype(str)
    df = df.dropna(subset=["ProSeq'"]).drop_duplicates(subset=KEYS, keep='first')
    df = df.rename(columns={DELTA_COL: f"delta_{tag}"})
    return df

def main():
    POSI_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Read the three predictors and inner-join on the shared keys
    d_dlk = read_and_prepare("dlkcat", FILES["dlkcat"])
    d_cat = read_and_prepare("catapro", FILES["catapro"])
    d_uni = read_and_prepare("unikp",   FILES["unikp"])
    df = d_dlk.merge(d_cat, on=KEYS, how='inner').merge(d_uni, on=KEYS, how='inner')

    # Scale-only normalization (divide, no subtract)
    df["s_dlkcat"], s_dlk, m_dlk = scale_only_norm(df["delta_dlkcat"])
    df["s_catapro"], s_cat, m_cat = scale_only_norm(df["delta_catapro"])
    df["s_unikp"],  s_uni, m_uni = scale_only_norm(df["delta_unikp"])
    print(f"[scale] DLKcat scale={s_dlk:.6g} ({m_dlk})  CataPro scale={s_cat:.6g} ({m_cat})  UniKP scale={s_uni:.6g} ({m_uni})")

    # Equal-weight mean scores
    df["mean3"] = (df["s_dlkcat"] + df["s_catapro"] + df["s_unikp"]) / 3.0
    df["mean2"] = (df["s_dlkcat"] + df["s_unikp"]) / 2.0  # top-2: DLKcat + UniKP

    # ---- Counts (deduplicated by ProSeq') ----
    a_df = df.loc[df["mean3"] > 0, KEYS + ["mean3"]].sort_values(['Group','ProID']).drop_duplicates(subset=["ProSeq'"], keep='first')
    b_df = df.loc[df["mean2"] > 0, KEYS + ["mean2"]].sort_values(['Group','ProID']).drop_duplicates(subset=["ProSeq'"], keep='first')
    a, b = len(a_df), len(b_df)

    # Intersection (by ProSeq')
    set_a = set(a_df["ProSeq'"])
    set_b = set(b_df["ProSeq'"])
    inter = set_a & set_b

    print(f"After scale-only normalization, 3-predictor mean > 0 samples (a): {a}")
    print(f"After scale-only normalization, DLKcat+UniKP mean > 0 samples (b): {b}")
    print(f"Intersection size (positive on both): {len(inter)}")

    # Save details
    a_out = POSI_DIR / "posi_planB_scaleonly_mean3_pos.csv"
    b_out = POSI_DIR / "posi_planB_scaleonly_mean2_pos.csv"
    a_df.to_csv(a_out, index=False)
    b_df.to_csv(b_out, index=False)
    # Optional: save intersection details
    inter_out = POSI_DIR / "posi_planB_scaleonly_mean2_mean3_inter.csv"
    (a_df[a_df["ProSeq'"].isin(inter)]
        .sort_values(['Group','ProID'])
        .drop_duplicates(subset=["ProSeq'"], keep='first')
        .to_csv(inter_out, index=False))
    print(f"[save] {a_out}")
    print(f"[save] {b_out}")
    print(f"[save] {inter_out}")

    # ---- Plots ---- #
    bins = 60
    fig1, ax1 = plt.subplots(figsize=(8,5))
    ax1.hist(df["s_dlkcat"].dropna(), bins=bins, alpha=0.5, label="DLKcat")
    ax1.hist(df["s_catapro"].dropna(), bins=bins, alpha=0.5, label="Catapro")
    ax1.hist(df["s_unikp"].dropna(),  bins=bins, alpha=0.5, label="UniKP")
    ax1.set_yscale('log'); ax1.axvline(0, linestyle='--')
    ax1.set_xlabel("Scaled delta_lgKcat (divide by q90-q10 / std)")
    ax1.set_ylabel("Count (log scale)")
    ax1.set_title("Scaled distributions of three predictors (no centering)")
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(FIG_DIR / "planB_scaleonly_three_predictors.png", dpi=600)

    fig2, ax2 = plt.subplots(figsize=(8,5))
    # ax2.hist(df["mean2"].dropna(), bins=bins, alpha=0.6, label="Mean(DLKcat, UniKP)")
    ax2.hist(df["mean3"].dropna(), bins=bins, alpha=0.6, label="Mean(DLKcat, UniKP, Catapro)")
    ax2.set_yscale('log'); ax2.axvline(0, linestyle='--')
    ax2.set_xlabel("Mean of scaled scores (no centering)")
    ax2.set_ylabel("Count (log scale)")
    ax2.set_title("Ensemble score distributions")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / "Ensemble_scaleonly_means.png", dpi=1200)

    print(f"[save] {FIG_DIR / 'Ensemble_scaleonly_three_predictors.png'}")
    print(f"[save] {FIG_DIR / 'Ensemble_scaleonly_means.png'}")

if __name__ == "__main__":
    main()
