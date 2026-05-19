#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【更新】去重与对齐均按 [ProID, ProSeq', SMILES]
仅“尺度标准化”、计算每个样本的 
 - sum/3
 - 再计算所有样本的平均值 = Σ(mean3)/N
 - 保留 a/b 明细、两张原分布图 + 新增一张 sum/3 分布图（y 轴 log）
 - 三个文件“标准化前”各自 delta_lgKcat 的均值 mean_raw
 - 三个文件“标准化前”的 q10_raw 和 q90_raw
 - 将最接近 q10_raw / q90_raw 的样本（各 1 条）导出到 *_q10_q90_samples.csv
 - 汇总表 raw_stats_per_predictor.csv（mean_raw / q10_raw / q90_raw）

【新增】导出奖励数据表：
 - 列：epoch, data_index, cond_name, group, ProID, SMILES, ProSeq, ProSeq', sample_idx, seed, step, ckpt_path, mean_delta
 - 其中 mean_delta = mean3

【使用范例】
 1. 使用默认参数运行：
    python3 delta_lgkcat_nrmlz_3_rl.py

 2. 指定特定的目录和文件名运行：
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
    parser = argparse.ArgumentParser(description="Delta lgKcat 标准化与奖励数据导出脚本")
    
    # 路径参数
    parser.add_argument("--in_dir", type=str, default="3_kcat_predictor_output_table/catif_rl/epoch2", help="输入目录")
    parser.add_argument("--posi_dir", type=str, default="kcat_mean_table/catif_rl/epoch2", help="统计结果输出目录")
    parser.add_argument("--fig_dir", type=str, default="figures/catif_rl/epoch2", help="图片输出目录")
    
    # 文件名参数
    parser.add_argument("--dlkcat", type=str, default="epoch2_Nov18_kcatpred_dlkcat.csv", help="DLKcat 结果文件名")
    parser.add_argument("--catapro", type=str, default="epoch2_Nov18_kcatpred_catapro.csv", help="CataPro 结果文件名")
    parser.add_argument("--unikp", type=str, default="epoch2_Nov18_kcatpred_unikp.csv", help="UniKP 结果文件名")
    parser.add_argument("--reward_file", type=str, default="epoch2_Nov18_reward_data.csv", help="输出奖励数据文件名")
    
    return parser.parse_args()

# ❶ 关键变更：按三列去重/对齐
KEYS = ['ProID', "ProSeq'", 'SMILES']
DELTA_COL = 'delta_lgKcat'

def scale_only_norm(x: pd.Series):
    """仅做尺度标准化：x_scaled = x / scale；不平移，不改正负。
       scale 优先用 (q90-q10)，退化用 std，再退化用 1。
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
        print(f"[错误] 文件缺失：{fp}", file=sys.stderr); sys.exit(1)
    df = pd.read_csv(fp)

    need = set(KEYS + [DELTA_COL])
    miss = need - set(df.columns)
    if miss:
        print(f"[错误] {filename} 缺列：{miss}", file=sys.stderr); sys.exit(1)

    df = df[KEYS + [DELTA_COL]].copy()

    # ❷ 关键变更：三列都转为字符串，并按三列去重
    df['ProID']   = df['ProID'].astype(str)
    df["ProSeq'"] = df["ProSeq'"].astype(str)
    df['SMILES']  = df['SMILES'].astype(str)

    # 为避免空值导致“虚假重复”，这里对三列同时做非空过滤
    df = df.dropna(subset=KEYS).drop_duplicates(subset=KEYS, keep='first')

    df = df.rename(columns={DELTA_COL: f"delta_{tag}"})
    return df

def nearest_quantile_rows(df: pd.DataFrame, col: str, q_vals=(0.10, 0.90)):
    """
    返回：统计 dict（mean/q10/q90）与 DataFrame（各 q 的最近样本 1 条/行）
    """
    s = pd.to_numeric(df[col], errors='coerce')
    mean_raw = s.mean()
    q_raws = {}
    for q in q_vals:
        q_raws[q] = s.quantile(q)
    # 找到最接近 q10 / q90 的样本（各 1 条）
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

    # 读取三模型并三表按三列键内连接对齐
    d_dlk = read_and_prepare("dlkcat", files["dlkcat"], in_dir)
    d_cat = read_and_prepare("catapro", files["catapro"], in_dir)
    d_uni = read_and_prepare("unikp",   files["unikp"], in_dir)

    # —— 各文件“标准化前”的统计 & q10/q90 样本 —— #
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
        print(f"[原始统计] {tag}: mean={stats['mean_raw']:.6g}, q10={stats['q10_raw']:.6g}, q90={stats['q90_raw']:.6g}")
        print(f"[保存] {qout}")

    raw_stats_df = pd.DataFrame(raw_stats_rows)
    raw_stats_out = posi_dir / "rl_raw_stats_per_predictor.csv"
    raw_stats_df.to_csv(raw_stats_out, index=False)
    print(f"[保存] {raw_stats_out}")

    # —— 合并三表，进入尺度标准化流程 —— #
    df = d_dlk.merge(d_cat, on=KEYS, how='inner').merge(d_uni, on=KEYS, how='inner')

    # 尺度标准化（仅除不减；不改变正负）
    df["s_dlkcat"], s_dlk, m_dlk = scale_only_norm(df["delta_dlkcat"])
    df["s_catapro"], s_cat, m_cat = scale_only_norm(df["delta_catapro"])
    df["s_unikp"],  s_uni, m_uni = scale_only_norm(df["delta_unikp"])
    print(f"[缩放] DLKcat scale={s_dlk:.6g} ({m_dlk})  CataPro scale={s_cat:.6g} ({m_cat})  UniKP scale={s_uni:.6g} ({m_uni})")

    # 等权求和/均值
    df["sum3"]  = (df["s_dlkcat"] + df["s_catapro"] + df["s_unikp"])
    df["mean3"] = df["sum3"] / 3.0
    df["mean2"] = (df["s_dlkcat"] + df["s_unikp"]) / 2.0

    # —— 全局统计（“三样本平均值的平均值/N”）——
    N = df["mean3"].notna().sum()
    sum_of_mean3 = df["mean3"].sum(skipna=True)
    overall_mean3 = sum_of_mean3 / N if N > 0 else np.nan
    print(f"[统计] N={N}  Σ(mean3)={sum_of_mean3:.6g}  平均( mean3 )=Σ(mean3)/N={overall_mean3:.6g}")

    # —— 计数（按 ProSeq' 去重口径；此处保持原有逻辑，仅说明会把 SMILES 一并写出）——
    a_df = (df.loc[df["mean3"] > 0, KEYS + ["mean3"]]
              .sort_values(['ProID'])
              .drop_duplicates(subset=["ProSeq'"], keep='first'))
    b_df = (df.loc[df["mean2"] > 0, KEYS + ["mean2"]]
              .sort_values(['ProID'])
              .drop_duplicates(subset=["ProSeq'"], keep='first'))
    a, b = len(a_df), len(b_df)

    # 交集（按 ProSeq'）
    set_a = set(a_df["ProSeq'"])
    set_b = set(b_df["ProSeq'"])
    inter = set_a & set_b

    print(f"归一化（仅缩放）后三模型均值 > 0 的样本数（a）：{a}")
    print(f"归一化（仅缩放）后 DLKcat+UniKP 均值 > 0 的样本数（b）：{b}")
    print(f"交集数量（两者同时为正）：{len(inter)}")

    # 保存明细
    a_out = posi_dir / "rl_posi_planB_scaleonly_mean3_pos.csv"
    b_out = posi_dir / "rl_posi_planB_scaleonly_mean2_pos.csv"
    inter_out = posi_dir / "rl_posi_planB_scaleonly_mean2_mean3_inter.csv"
    a_df.to_csv(a_out, index=False)
    b_df.to_csv(b_out, index=False)
    (a_df[a_df["ProSeq'"].isin(inter)]
        .sort_values(['ProID'])
        .drop_duplicates(subset=["ProSeq'"], keep='first')
        .to_csv(inter_out, index=False))

    # 关键统计到 txt
    stats_out = posi_dir / "rl_planB_scaleonly_stats.txt"
    with open(stats_out, "w", encoding="utf-8") as f:
        f.write(
            "=== Scale-only Normalization Stats ===\n"
            f"DLKcat scale={s_dlk:.6g} ({m_dlk})\n"
            f"CataPro scale={s_cat:.6g} ({m_cat})\n"
            f"UniKP  scale={s_uni:.6g} ({m_uni})\n"
            f"N={N}\nΣ(mean3)={sum_of_mean3:.6g}\nmean(mean3)=Σ(mean3)/N={overall_mean3:.6g}\n"
            f"a (mean3>0, distinct ProSeq')={a}\n"
            f"b (mean2>0, distinct ProSeq')={b}\n"
            f"intersect(a,b on ProSeq')={len(inter)}\n"
        )
        f.write("\n=== Raw delta_lgKcat stats per predictor ===\n")
        for r in raw_stats_rows:
            f.write(f"{r['predictor']}: mean_raw={r['mean_raw']:.6g}, "
                    f"q10_raw={r['q10_raw']:.6g}, q90_raw={r['q90_raw']:.6g}\n")

    print(f"[保存] {a_out}")
    print(f"[保存] {b_out}")
    print(f"[保存] {inter_out}")
    print(f"[保存] {stats_out}")

    # —— 绘图 —— #
    bins = 60

    # 1) 三个预测器
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

    # 2) mean2 与 mean3
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

    # ========= 新增部分：导出奖励数据表 ========= #
    meta_fp = in_dir / files["dlkcat"]
    meta_df = pd.read_csv(meta_fp)

    wanted_meta_cols = [
        "epoch", "data_index", "cond_name", "group",
        "ProID", "SMILES", "ProSeq", "ProSeq'",
        "sample_idx", "seed", "step", "ckpt_path",
    ]

    missing_meta = [c for c in wanted_meta_cols if c not in meta_df.columns]
    if missing_meta:
        print(f"[警告] 缺失列：{missing_meta}", file=sys.stderr)

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
    print(f"[保存] {reward_out}")

if __name__ == "__main__":
    main()
