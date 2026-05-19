#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plan B（改进版）：仅缩放不平移 + 均值筛选 + 交集统计

- 对每个预测器的 delta_lgKcat 仅做尺度标准化：x_scaled = x / scale
  scale 使用 (q90 - q10)，若退化则用 std，再退化用 1；不做平移，不改变正负。
- 计算：
    a) 三模型均值 > 0 的样本数（按 ProSeq' 去重）
    b) DLKcat+UniKP 均值 > 0 的样本数（按 ProSeq' 去重）
    c) a ∩ b 的交集数量（按 ProSeq'）
- 仍输出两张分布图（y 轴 log），并保存 a/b 明细方便复核。
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
    """仅做尺度标准化：x_scaled = x / scale；不平移，不改正负。"""
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
        print(f"[错误] 文件缺失：{fp}", file=sys.stderr); sys.exit(1)
    df = pd.read_csv(fp)
    need = set(KEYS + [DELTA_COL])
    miss = need - set(df.columns)
    if miss:
        print(f"[错误] {filename} 缺列：{miss}", file=sys.stderr); sys.exit(1)
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

    # 读取三模型并三表内连接对齐
    d_dlk = read_and_prepare("dlkcat", FILES["dlkcat"])
    d_cat = read_and_prepare("catapro", FILES["catapro"])
    d_uni = read_and_prepare("unikp",   FILES["unikp"])
    df = d_dlk.merge(d_cat, on=KEYS, how='inner').merge(d_uni, on=KEYS, how='inner')

    # 尺度标准化（仅除不减）
    df["s_dlkcat"], s_dlk, m_dlk = scale_only_norm(df["delta_dlkcat"])
    df["s_catapro"], s_cat, m_cat = scale_only_norm(df["delta_catapro"])
    df["s_unikp"],  s_uni, m_uni = scale_only_norm(df["delta_unikp"])
    print(f"[缩放] DLKcat scale={s_dlk:.6g} ({m_dlk})  CataPro scale={s_cat:.6g} ({m_cat})  UniKP scale={s_uni:.6g} ({m_uni})")

    # 均值得分（等权）
    df["mean3"] = (df["s_dlkcat"] + df["s_catapro"] + df["s_unikp"]) / 3.0
    df["mean2"] = (df["s_dlkcat"] + df["s_unikp"]) / 2.0  # 前两：DLKcat + UniKP

    # —— 计数（按 ProSeq' 去重口径）——
    a_df = df.loc[df["mean3"] > 0, KEYS + ["mean3"]].sort_values(['Group','ProID']).drop_duplicates(subset=["ProSeq'"], keep='first')
    b_df = df.loc[df["mean2"] > 0, KEYS + ["mean2"]].sort_values(['Group','ProID']).drop_duplicates(subset=["ProSeq'"], keep='first')
    a, b = len(a_df), len(b_df)

    # 交集（按 ProSeq'）
    set_a = set(a_df["ProSeq'"])
    set_b = set(b_df["ProSeq'"])
    inter = set_a & set_b

    print(f"归一化（仅缩放）后三模型均值 > 0 的样本数（a）：{a}")
    print(f"归一化（仅缩放）后 DLKcat+UniKP 均值 > 0 的样本数（b）：{b}")
    print(f"交集数量（两者同时为正）：{len(inter)}")

    # 保存明细（可注释）
    a_out = POSI_DIR / "posi_planB_scaleonly_mean3_pos.csv"
    b_out = POSI_DIR / "posi_planB_scaleonly_mean2_pos.csv"
    a_df.to_csv(a_out, index=False)
    b_df.to_csv(b_out, index=False)
    # 可选保存交集明细
    inter_out = POSI_DIR / "posi_planB_scaleonly_mean2_mean3_inter.csv"
    (a_df[a_df["ProSeq'"].isin(inter)]
        .sort_values(['Group','ProID'])
        .drop_duplicates(subset=["ProSeq'"], keep='first')
        .to_csv(inter_out, index=False))
    print(f"[保存] {a_out}")
    print(f"[保存] {b_out}")
    print(f"[保存] {inter_out}")

    # —— 绘图 —— #
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

    print(f"[保存] {FIG_DIR / 'Ensemble_scaleonly_three_predictors.png'}")
    print(f"[保存] {FIG_DIR / 'Ensemble_scaleonly_means.png'}")

if __name__ == "__main__":
    main()
