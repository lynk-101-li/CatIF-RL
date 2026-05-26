#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_tables_S7_S8_S9.py — generate four SI stat-tables from master_per_protein.csv.

Inputs:
  ./master_per_protein.csv  (produced by build_master_per_protein.py)

Outputs (csv + markdown):
  ./table_S7_full_baseline.{csv,md}
      All baselines vs CatIF-RL R3 — mean ± 95% CI and paired Wilcoxon p / BH-q
      across **all four metrics** (ΔlgKcat, Recovery, pLDDT, Backbone RMSD).
  ./table_S8_round_wise.{csv,md}
      CatIF / R1 / R2 / R3 across the same 4 metrics + SR@δ ∈ {0, 0.5, 1.0, 1.5}.
  ./table_S9_threshold_sensitivity.{csv,md}
      SR@δ for δ ∈ {0, 0.25, 0.5, 1.0, 1.5, 2.0} for every baseline.
  ./table_S10_ablation_pairwise.{csv,md}
      Six paired-comparison ablation tests across 4 metrics:
        Static pipeline:  GraDe-IF → EnzymeIF; EnzymeIF → CatIF; GraDe-IF → CatIF.
        RL chain:         R1↔R2; R2↔R3; R1↔R3.
      Each metric is BH-FDR-adjusted within the family of 6 pairs.

Conventions:
  - 95 % CI from non-parametric percentile bootstrap (10,000 resamples).
  - Wilcoxon paired signed-rank test on the 1,423 proteins paired by ProID.
  - BH (Benjamini–Hochberg) FDR correction within each metric.
  - SR (success rate) = fraction of proteins satisfying jointly:
       Δlog10 k_cat > δ  AND  pLDDT > 90  AND  Backbone_RMSD < 4 Å.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
DEFAULT_MASTER = HERE / "master_per_protein.csv"
DEFAULT_OUTPUT_DIR = HERE

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
BASELINE_ORDER = [
    "ProteinMPNN", "ESM-IF", "LigandMPNN", "PiFold", "ABACUS-T",
    "GraDe-IF", "EnzymeIF", "CatIF",
    "CatIF-RL R1", "CatIF-RL R2", "CatIF-RL R3",
]
CATIF_RL = "CatIF-RL R3"  # Reference for paired comparisons.

PLDDT_TAU = 90.0
RMSD_TAU = 4.0
DELTA_THRESHOLDS_S8 = [0.0, 0.5, 1.0, 1.5]
DELTA_THRESHOLDS_S9 = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]

N_BOOT = 10_000
RNG_SEED = 42

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def col(model: str, metric: str) -> str:
    return f"{model}__{metric}"


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOT,
                 seed: int = RNG_SEED, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    n = v.size
    for i in range(n_boot):
        means[i] = rng.choice(v, size=n, replace=True).mean()
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def wilcoxon_paired(x: np.ndarray, y: np.ndarray) -> float:
    """Paired Wilcoxon signed-rank p-value, NaN-safe."""
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 5:
        return float("nan")
    diff = a[mask] - b[mask]
    if np.allclose(diff, 0):
        return float("nan")
    try:
        res = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
        return float(res.pvalue)
    except ValueError:
        return float("nan")


def bh_adjust(pvals: list[float]) -> list[float]:
    """Benjamini–Hochberg FDR adjustment, returns q-values aligned with input."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = np.arange(1, n + 1)
    q = np.empty(n)
    q_sorted = p[order] * n / ranked
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q[order] = np.clip(q_sorted, 0, 1)
    return q.tolist()


def fmt_mean_ci(values: np.ndarray, decimals: int = 3) -> str:
    """e.g. '0.605 [0.578, 0.633]'"""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return "NA"
    m = v.mean()
    lo, hi = bootstrap_ci(v)
    return f"{m:.{decimals}f} [{lo:.{decimals}f}, {hi:.{decimals}f}]"


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    if p < 1e-300:
        return "<1e-300"
    if p < 1e-3:
        return f"{p:.2e}"
    return f"{p:.3f}"


def stars(q: float) -> str:
    if not np.isfinite(q):
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return "ns"


def success_rate(df: pd.DataFrame, model: str, delta: float) -> float:
    d = df[col(model, "delta_lgKcat")]
    p = df[col(model, "Avg_pLDDT")]
    r = df[col(model, "Backbone_RMSD")]
    mask = (~d.isna()) & (~p.isna()) & (~r.isna())
    sub = df[mask]
    if len(sub) == 0:
        return float("nan")
    pass_ = ((sub[col(model, "delta_lgKcat")] > delta) &
             (sub[col(model, "Avg_pLDDT")] > PLDDT_TAU) &
             (sub[col(model, "Backbone_RMSD")] < RMSD_TAU))
    return float(pass_.mean())


def df_to_md(df: pd.DataFrame, title: str) -> str:
    lines = [f"## {title}", "", df.to_markdown(index=False, floatfmt=".4f"), ""]
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Main driver (parametrized via CLI)
# ----------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build SI tables S7 / S8 / S9 / S10 from a master per-protein CSV "
                    "produced by catif_rl.evaluation.build_master."
    )
    p.add_argument("--master", type=Path, default=DEFAULT_MASTER,
                   help=f"input master CSV (default: {DEFAULT_MASTER})")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help=f"directory to write the table_S*.{{csv,md}} files "
                        f"(default: {DEFAULT_OUTPUT_DIR})")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.master}")
    master = pd.read_csv(args.master)
    print(f"  shape: {master.shape}")

    # ======================================================================
    # Table S7 — all baselines vs CatIF-RL R3 across ALL 4 metrics
    # ======================================================================
    print("\n[Table S7] Multi-metric significance vs CatIF-RL R3")
    METRICS_S7 = [
        ("delta_lgKcat", "Δlog10 k_cat", 3),
        ("recovery",     "Recovery",     3),
        ("Avg_pLDDT",    "pLDDT",        2),
        ("Backbone_RMSD","Backbone RMSD (Å)", 3),
    ]

    # Pre-compute per-metric p-values (vs R3) for all baselines, then BH within metric.
    ref_vals = {m_key: master[col(CATIF_RL, m_key)].to_numpy() for m_key, _, _ in METRICS_S7}
    pvals_per_metric: dict[str, list[float]] = {m_key: [] for m_key, _, _ in METRICS_S7}
    for m in BASELINE_ORDER:
        for m_key, _, _ in METRICS_S7:
            if m == CATIF_RL:
                pvals_per_metric[m_key].append(np.nan)
            else:
                pvals_per_metric[m_key].append(
                    wilcoxon_paired(ref_vals[m_key], master[col(m, m_key)].to_numpy())
                )
    qvals_per_metric = {
        m_key: bh_adjust([p if np.isfinite(p) else 1.0 for p in plist])
        for m_key, plist in pvals_per_metric.items()
    }

    rows_S7 = []
    for i, m in enumerate(BASELINE_ORDER):
        row: dict[str, object] = {"Model": m}
        for m_key, m_label, dec in METRICS_S7:
            v = master[col(m, m_key)].to_numpy()
            p = pvals_per_metric[m_key][i]
            q = qvals_per_metric[m_key][i]
            row[f"{m_label} (mean [95% CI])"] = fmt_mean_ci(v, decimals=dec)
            if m == CATIF_RL:
                row[f"{m_label} p (Wilcoxon)"] = "—"
                row[f"{m_label} q (BH)"] = "—"
                row[f"{m_label} sig"] = "(reference)"
            else:
                row[f"{m_label} p (Wilcoxon)"] = fmt_p(p)
                row[f"{m_label} q (BH)"] = fmt_p(q)
                row[f"{m_label} sig"] = stars(q)
        rows_S7.append(row)

    table_S7 = pd.DataFrame(rows_S7)
    table_S7.to_csv(out_dir / "table_S7_full_baseline.csv", index=False)
    (out_dir / "table_S7_full_baseline.md").write_text(df_to_md(table_S7,
        f"Table S7. Per-protein metrics on the 1,423-enzyme test set; mean [95 % bootstrap CI] and "
        f"paired Wilcoxon p (with BH-FDR adj. q) vs {CATIF_RL} for all four metrics. "
        f"BH adjustment within each metric (8 baseline comparisons per metric)."))
    print(f"  wrote {out_dir/'table_S7_full_baseline.csv'}")
    # Compact preview: model + significance flag per metric
    sig_cols = ["Model"] + [f"{m_label} sig" for _, m_label, _ in METRICS_S7]
    print(table_S7[sig_cols].to_string(index=False))

    # ======================================================================
    # Table S8 — Round-wise (CatIF / R1 / R2 / R3) full metrics + SR@δ
    # ======================================================================
    print("\n[Table S8] Round-wise full metrics")
    ROUND_MODELS = ["CatIF", "CatIF-RL R1", "CatIF-RL R2", "CatIF-RL R3"]
    rows_S8 = []
    for m in ROUND_MODELS:
        delta = master[col(m, "delta_lgKcat")].to_numpy()
        recov = master[col(m, "recovery")].to_numpy()
        plddt = master[col(m, "Avg_pLDDT")].to_numpy()
        rmsd = master[col(m, "Backbone_RMSD")].to_numpy()
        row = {
            "Round": m,
            "Δlog10 k_cat": fmt_mean_ci(delta),
            "Recovery": fmt_mean_ci(recov),
            "pLDDT": fmt_mean_ci(plddt, decimals=2),
            "Backbone RMSD (Å)": fmt_mean_ci(rmsd),
        }
        for d in DELTA_THRESHOLDS_S8:
            row[f"SR@δ={d}"] = f"{success_rate(master, m, d):.3f}"
        rows_S8.append(row)

    table_S8 = pd.DataFrame(rows_S8)
    table_S8.to_csv(out_dir / "table_S8_round_wise.csv", index=False)
    (out_dir / "table_S8_round_wise.md").write_text(df_to_md(table_S8,
        "Table S8. CatIF and the three GRPO rounds — per-protein metrics with 95 % CI and joint success rate at multiple δ."))
    print(f"  wrote {out_dir/'table_S8_round_wise.csv'}")
    print(table_S8.to_string(index=False))

    # ======================================================================
    # Table S9 — Threshold sensitivity SR@δ across all baselines
    # ======================================================================
    print("\n[Table S9] Threshold sensitivity SR@δ")
    rows_S9 = []
    for m in BASELINE_ORDER:
        row = {"Model": m}
        for d in DELTA_THRESHOLDS_S9:
            row[f"SR@δ={d}"] = f"{success_rate(master, m, d):.3f}"
        rows_S9.append(row)

    table_S9 = pd.DataFrame(rows_S9)
    table_S9.to_csv(out_dir / "table_S9_threshold_sensitivity.csv", index=False)
    (out_dir / "table_S9_threshold_sensitivity.md").write_text(df_to_md(table_S9,
        "Table S9. Joint success rate sensitivity to the Δlog10 k_cat threshold δ "
        "(pLDDT > 90 and backbone RMSD < 4 Å held fixed)."))
    print(f"  wrote {out_dir/'table_S9_threshold_sensitivity.csv'}")
    print(table_S9.to_string(index=False))


    # ======================================================================
    # Table S10 — Ablation & round-wise pairwise significance
    # ======================================================================
    print("\n[Table S10] Ablation & round-wise pairwise (paired Wilcoxon, BH within metric)")
    # (label, model_A, model_B)  — the test asks "is model_B different from model_A?"
    ABLATION_PAIRS = [
        # Static-pipeline ablation: each row = "did this stage of the pipeline help?"
        ("Static: GraDe-IF → EnzymeIF",  "GraDe-IF", "EnzymeIF"),
        ("Static: EnzymeIF → CatIF",     "EnzymeIF", "CatIF"),
        ("Static: GraDe-IF → CatIF",     "GraDe-IF", "CatIF"),
        # RL-chain ablation: do successive RL rounds add value?
        ("RL: R1 → R2",                  "CatIF-RL R1", "CatIF-RL R2"),
        ("RL: R2 → R3",                  "CatIF-RL R2", "CatIF-RL R3"),
        ("RL: R1 → R3",                  "CatIF-RL R1", "CatIF-RL R3"),
    ]
    METRICS_S10 = METRICS_S7  # same 4 metrics

    # Compute p-values per metric across all 6 pairs, then BH within metric.
    pvals_S10: dict[str, list[float]] = {m_key: [] for m_key, _, _ in METRICS_S10}
    mean_diffs: dict[str, list[float]] = {m_key: [] for m_key, _, _ in METRICS_S10}
    for label, A, B in ABLATION_PAIRS:
        for m_key, _, _ in METRICS_S10:
            a = master[col(A, m_key)].to_numpy()
            b = master[col(B, m_key)].to_numpy()
            pvals_S10[m_key].append(wilcoxon_paired(a, b))
            mean_diffs[m_key].append(float(np.nanmean(b) - np.nanmean(a)))
    qvals_S10 = {
        m_key: bh_adjust([p if np.isfinite(p) else 1.0 for p in plist])
        for m_key, plist in pvals_S10.items()
    }

    rows_S10 = []
    for i, (label, A, B) in enumerate(ABLATION_PAIRS):
        row: dict[str, object] = {"Comparison (A → B)": label}
        for m_key, m_label, dec in METRICS_S10:
            diff = mean_diffs[m_key][i]
            p = pvals_S10[m_key][i]
            q = qvals_S10[m_key][i]
            # diff sign convention: B − A (i.e., later − earlier)
            row[f"{m_label} Δ(B−A)"] = f"{diff:+.{dec}f}"
            row[f"{m_label} p (Wilcoxon)"] = fmt_p(p)
            row[f"{m_label} q (BH)"] = fmt_p(q)
            row[f"{m_label} sig"] = stars(q)
        rows_S10.append(row)

    table_S10 = pd.DataFrame(rows_S10)
    table_S10.to_csv(out_dir / "table_S10_ablation_pairwise.csv", index=False)
    (out_dir / "table_S10_ablation_pairwise.md").write_text(df_to_md(table_S10,
        "Table S10. Pairwise ablation significance — paired Wilcoxon signed-rank tests with "
        "Benjamini–Hochberg FDR correction within each metric (6 paired comparisons per metric). "
        "Δ(B−A) is the per-protein mean difference (later − earlier stage). "
        "Static-pipeline rows test the supervised-stage chain "
        "(GraDe-IF → EnzymeIF → CatIF); RL rows test consecutive and overall RL rounds."))
    print(f"  wrote {out_dir/'table_S10_ablation_pairwise.csv'}")
    sig_cols_S10 = ["Comparison (A → B)"] + [f"{m_label} sig" for _, m_label, _ in METRICS_S10]
    print(table_S10[sig_cols_S10].to_string(index=False))

    print("\n[done] Tables S7, S8, S9, S10 saved to:", HERE)

    return 0


if __name__ == "__main__":
    sys.exit(main())
