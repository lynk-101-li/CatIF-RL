#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[Update] Dedup and alignment are now done on [ProID, ProSeq', SMILES].
Scale-only normalization, plus per-sample
 - sum/3
 - dataset-level average = sum(mean3) / N
 - keep a/b detail tables and two original distribution figures, plus a new
   sum/3 distribution figure (log y-axis)
 - per-file raw delta_lgKcat mean (mean_raw) before normalization
 - per-file raw q10_raw and q90_raw before normalization
 - one row each closest to q10_raw / q90_raw is exported to
   *_q10_q90_samples.csv
 - summary table raw_stats_per_predictor.csv (mean_raw / q10_raw / q90_raw)

[New] Export the reward data table:
 - columns: epoch, data_index, cond_name, group, ProID, SMILES, ProSeq,
   ProSeq', sample_idx, seed, step, ckpt_path, mean_delta
 - mean_delta = mean3

[Usage]
 1. Run with default arguments:
    python3 delta_lgkcat_nrmlz_3_rl.py

 2. Run with explicit directories and filenames:
    python3 delta_lgkcat_nrmlz_3_rl.py \
      --in_dir "3_kcat_predictor_output_table/catif_rl/epoch1" \
      --posi_dir "kcat_mean_table/catif_rl/epoch1" \
      --fig_dir "figures/catif_rl/epoch1" \
      --dlkcat "epoch1_Nov16_kcatpred_dlkcat.csv" \
      --catapro "epoch1_Nov16_kcatpred_catapro.csv" \
      --unikp "epoch1_Nov16_kcatpred_unikp.csv" \
      --reward_file "epoch1_Nov16_reward_data.csv"
"""

import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Delta lgKcat normalization + reward-data export")

    # Path arguments
    parser.add_argument("--in_dir", type=str, default="3_kcat_predictor_output_table/catif_rl/epoch2", help="input directory")
    parser.add_argument("--posi_dir", type=str, default="kcat_mean_table/catif_rl/epoch2", help="output directory for stats")
    parser.add_argument("--fig_dir", type=str, default="figures/catif_rl/epoch2", help="output directory for figures")

    # File-name arguments
    parser.add_argument("--dlkcat", type=str, default="epoch2_Nov18_kcatpred_dlkcat.csv", help="DLKcat prediction filename")
    parser.add_argument("--catapro", type=str, default="epoch2_Nov18_kcatpred_catapro.csv", help="CataPro prediction filename")
    parser.add_argument("--unikp", type=str, default="epoch2_Nov18_kcatpred_unikp.csv", help="UniKP prediction filename")
    parser.add_argument("--reward_file", type=str, default="epoch2_Nov18_reward_data.csv", help="output reward-data filename")

    return parser.parse_args()

# Key change: dedup / alignment on three columns
KEYS = ['ProID', "ProSeq'", 'SMILES']
DELTA_COL = 'delta_lgKcat'

def scale_only_norm(x: pd.Series):
    """Scale-only normalization: x_scaled = x / scale; no centering, signs preserved.
       scale prefers (q90 - q10); falls back to std, then to 1.
    """
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

def read_and_prepare(tag, filename, in_dir):
    fp = Path(in_dir) / filename
    if not fp.exists():
        print(f"[ERROR] file missing: {fp}", file=sys.stderr); sys.exit(1)
    df = pd.read_csv(fp)

    need = set(KEYS + [DELTA_COL])
    miss = need - set(df.columns)
    if miss:
        print(f"[ERROR] {filename} is missing columns: {miss}", file=sys.stderr); sys.exit(1)

    df = df[KEYS + [DELTA_COL]].copy()

    # Key change: cast all three columns to str, then dedup on all three
    df['ProID']   = df['ProID'].astype(str)
    df["ProSeq'"] = df["ProSeq'"].astype(str)
    df['SMILES']  = df['SMILES'].astype(str)

    # Avoid "fake duplicates" from missing values by filtering NA across all three
    df = df.dropna(subset=KEYS).drop_duplicates(subset=KEYS, keep='first')

    df = df.rename(columns={DELTA_COL: f"delta_{tag}"})
    return df

def nearest_quantile_rows(df: pd.DataFrame, col: str, q_vals=(0.10, 0.90)):
    """
    Returns: stats dict (mean / q10 / q90) plus a DataFrame with 1 row per
    quantile (the row whose value is closest to that quantile).
    """
    s = pd.to_numeric(df[col], errors='coerce')
    mean_raw = s.mean()
    q_raws = {}
    for q in q_vals:
        q_raws[q] = s.quantile(q)
    # Find the row whose value is closest to each q10 / q90 (1 row each)
    rows = []
    for q in q_vals:
        target = q_raws[q]
        idx = (s - target).abs().idxmin()
        row = df.loc[[idx], KEYS + [col]].copy()
        row.insert(len(KEYS), "q", q)
        row.insert(len(KEYS)+1, "q_value", float(target))
        rows.append(row)
    near_df = pd.concat(rows, axis=0, ignore_index=True)
    stats = {
        "mean_raw": mean_raw,
        "q10_raw": float(q_raws[0.10]),
        "q90_raw": float(q_raws[0.90]),
    }
    return stats, near_df

def main():
    args = parse_args()

    in_dir   = Path(args.in_dir)
    posi_dir = Path(args.posi_dir)
    fig_dir  = Path(args.fig_dir)

    files = {
        "dlkcat": args.dlkcat,
        "catapro": args.catapro,
        "unikp": args.unikp,
    }

    reward_file = args.reward_file

    posi_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Read three predictors and inner-join on the three-column key
    d_dlk = read_and_prepare("dlkcat", files["dlkcat"], in_dir)
    d_cat = read_and_prepare("catapro", files["catapro"], in_dir)
    d_uni = read_and_prepare("unikp",   files["unikp"], in_dir)

    # ---- Per-file raw stats + q10/q90 samples ---- #
    raw_stats_rows = []
    qpick_outputs = []

    for tag, d in [("dlkcat", d_dlk), ("catapro", d_cat), ("unikp", d_uni)]:
        col = f"delta_{tag}"
        stats, qdf = nearest_quantile_rows(d, col)
        raw_stats_rows.append({
            "predictor": tag,
            "mean_raw": stats["mean_raw"],
            "q10_raw":  stats["q10_raw"],
            "q90_raw":  stats["q90_raw"],
        })
        qout = posi_dir / f"{tag}_q10_q90_samples.csv"
        qdf.to_csv(qout, index=False)
        qpick_outputs.append((tag, qout))
        print(f"[raw stats] {tag}: mean={stats['mean_raw']:.6g}, q10={stats['q10_raw']:.6g}, q90={stats['q90_raw']:.6g}")
        print(f"[save] {qout}")

    raw_stats_df = pd.DataFrame(raw_stats_rows)
    raw_stats_out = posi_dir / "rl_raw_stats_per_predictor.csv"
    raw_stats_df.to_csv(raw_stats_out, index=False)
    print(f"[save] {raw_stats_out}")

    # ---- Merge three tables, then enter scale-only normalization ---- #
    df = d_dlk.merge(d_cat, on=KEYS, how='inner').merge(d_uni, on=KEYS, how='inner')

    # Scale-only normalization (divide, no subtract; sign preserved)
    df["s_dlkcat"], s_dlk, m_dlk = scale_only_norm(df["delta_dlkcat"])
    df["s_catapro"], s_cat, m_cat = scale_only_norm(df["delta_catapro"])
    df["s_unikp"],  s_uni, m_uni = scale_only_norm(df["delta_unikp"])
    print(f"[scale] DLKcat scale={s_dlk:.6g} ({m_dlk})  CataPro scale={s_cat:.6g} ({m_cat})  UniKP scale={s_uni:.6g} ({m_uni})")

    # Equal-weight sum / mean
    df["sum3"]  = (df["s_dlkcat"] + df["s_catapro"] + df["s_unikp"])
    df["mean3"] = df["sum3"] / 3.0
    df["mean2"] = (df["s_dlkcat"] + df["s_unikp"]) / 2.0

    # ---- Global stats ("dataset-level mean of the 3-sample mean") ----
    N = df["mean3"].notna().sum()
    sum_of_mean3 = df["mean3"].sum(skipna=True)
    overall_mean3 = sum_of_mean3 / N if N > 0 else np.nan
    print(f"[stats] N={N}  sum(mean3)={sum_of_mean3:.6g}  mean(mean3)=sum(mean3)/N={overall_mean3:.6g}")

    # ---- Counts (dedup by ProSeq'; SMILES is kept in the output) ----
    a_df = (df.loc[df["mean3"] > 0, KEYS + ["mean3"]]
              .sort_values(['ProID'])
              .drop_duplicates(subset=["ProSeq'"], keep='first'))
    b_df = (df.loc[df["mean2"] > 0, KEYS + ["mean2"]]
              .sort_values(['ProID'])
              .drop_duplicates(subset=["ProSeq'"], keep='first'))
    a, b = len(a_df), len(b_df)

    # Intersection (by ProSeq')
    set_a = set(a_df["ProSeq'"])
    set_b = set(b_df["ProSeq'"])
    inter = set_a & set_b

    print(f"After scale-only normalization, 3-predictor mean > 0 samples (a): {a}")
    print(f"After scale-only normalization, DLKcat+UniKP mean > 0 samples (b): {b}")
    print(f"Intersection size (positive on both): {len(inter)}")

    # Save details
    a_out = posi_dir / "rl_posi_planB_scaleonly_mean3_pos.csv"
    b_out = posi_dir / "rl_posi_planB_scaleonly_mean2_pos.csv"
    inter_out = posi_dir / "rl_posi_planB_scaleonly_mean2_mean3_inter.csv"
    a_df.to_csv(a_out, index=False)
    b_df.to_csv(b_out, index=False)
    (a_df[a_df["ProSeq'"].isin(inter)]
        .sort_values(['ProID'])
        .drop_duplicates(subset=["ProSeq'"], keep='first')
        .to_csv(inter_out, index=False))

    # Key stats summary to txt
    stats_out = posi_dir / "rl_planB_scaleonly_stats.txt"
    with open(stats_out, "w", encoding="utf-8") as f:
        f.write(
            "=== Scale-only Normalization Stats ===\n"
            f"DLKcat scale={s_dlk:.6g} ({m_dlk})\n"
            f"CataPro scale={s_cat:.6g} ({m_cat})\n"
            f"UniKP  scale={s_uni:.6g} ({m_uni})\n"
            f"N={N}\nsum(mean3)={sum_of_mean3:.6g}\nmean(mean3)=sum(mean3)/N={overall_mean3:.6g}\n"
            f"a (mean3>0, distinct ProSeq')={a}\n"
            f"b (mean2>0, distinct ProSeq')={b}\n"
            f"intersect(a,b on ProSeq')={len(inter)}\n"
        )
        f.write("\n=== Raw delta_lgKcat stats per predictor ===\n")
        for r in raw_stats_rows:
            f.write(f"{r['predictor']}: mean_raw={r['mean_raw']:.6g}, "
                    f"q10_raw={r['q10_raw']:.6g}, q90_raw={r['q90_raw']:.6g}\n")

    print(f"[save] {a_out}")
    print(f"[save] {b_out}")
    print(f"[save] {inter_out}")
    print(f"[save] {stats_out}")

    # ---- Plots ---- #
    bins = 60

    # 1) Three predictors
    fig1, ax1 = plt.subplots(figsize=(8,5))
    ax1.hist(df["s_dlkcat"].dropna(), bins=bins, alpha=0.5, label="DLKcat")
    ax1.hist(df["s_catapro"].dropna(), bins=bins, alpha=0.5, label="CataPro")
    ax1.hist(df["s_unikp"].dropna(),  bins=bins, alpha=0.5, label="UniKP")
    ax1.set_yscale('log'); ax1.axvline(0, linestyle='--')
    ax1.set_xlabel("Scaled delta_lgKcat")
    ax1.set_ylabel("Count (log)")
    ax1.set_title("Scaled distributions")
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(fig_dir / "rl_planB_scaleonly_three_predictors.png", dpi=600)

    # 2) mean2 and mean3
    fig2, ax2 = plt.subplots(figsize=(8,5))
    ax2.hist(df["mean2"].dropna(), bins=bins, alpha=0.6, label="Mean2")
    ax2.hist(df["mean3"].dropna(), bins=bins, alpha=0.6, label="Mean3")
    ax2.set_yscale('log'); ax2.axvline(0, linestyle='--')
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(fig_dir / "rl_planB_scaleonly_means.png", dpi=600)

    # 3) sum/3
    fig3, ax3 = plt.subplots(figsize=(8,5))
    ax3.hist((df["sum3"]/3.0).dropna(), bins=bins, alpha=0.75)
    ax3.set_yscale('log'); ax3.axvline(0, linestyle='--')
    fig3.tight_layout()
    fig3.savefig(fig_dir / "rl_planB_scaleonly_sum_over_3.png", dpi=600)

    # ========= New: export reward data table ========= #
    meta_fp = in_dir / files["dlkcat"]
    meta_df = pd.read_csv(meta_fp)

    wanted_meta_cols = [
        "epoch", "data_index", "cond_name", "group",
        "ProID", "SMILES", "ProSeq", "ProSeq'",
        "sample_idx", "seed", "step", "ckpt_path",
    ]

    missing_meta = [c for c in wanted_meta_cols if c not in meta_df.columns]
    if missing_meta:
        print(f"[WARN] missing columns: {missing_meta}", file=sys.stderr)

    meta_cols_present = [c for c in wanted_meta_cols if c in meta_df.columns]
    meta_df = meta_df[meta_cols_present].copy()
    for col in KEYS:
        if col in meta_df.columns:
            meta_df[col] = meta_df[col].astype(str)

    meta_df = meta_df.drop_duplicates(subset=KEYS, keep="first")
    merged_reward = df.merge(meta_df, on=KEYS, how="inner")

    reward_cols_final = [c for c in wanted_meta_cols if c in merged_reward.columns]
    reward_df = merged_reward[reward_cols_final + ["mean3"]].copy()
    reward_df = reward_df.rename(columns={"mean3": "mean_delta"})

    reward_out = posi_dir / reward_file
    reward_df.to_csv(reward_out, index=False)
    print(f"[save] {reward_out}")

if __name__ == "__main__":
    main()
