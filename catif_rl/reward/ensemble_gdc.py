#!/usr/bin/env python3
"""GDC-stage ensemble scoring: scale-only normalization + intersection counts.

This is the GDC counterpart of ``catif_rl.reward.ensemble_rl``. It:

- reads the three per-predictor delta-lgKcat CSVs (DLKcat, UniKP, CataPro);
- scales each column by (q90 - q10) (falling back to std, then to 1) so that
  reward magnitudes are comparable across predictors without removing sign;
- counts variants whose 3-predictor mean > 0 (set ``a``);
- counts variants whose DLKcat+UniKP mean > 0 (set ``b``);
- reports |a|, |b|, |a ∩ b| (deduplicated by mutated sequence ``ProSeq'``);
- saves per-set detail CSVs and two log-scale distribution figures.

CLI
---

    python -m catif_rl.reward.ensemble_gdc \\
      --in-dir   runs/gdc/predictions \\
      --out-dir  runs/gdc/positive_sets \\
      --fig-dir  runs/gdc/figures

By default the wrapper looks for the three manuscript-convention filenames
``merged_mut_substrate_enzymeif_kcatpred_{dlkcat,catapro,unikp}.csv``;
override them with ``--dlkcat-csv`` / ``--catapro-csv`` / ``--unikp-csv``
when the GDC stage emitted different stems.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["Group", "ProID", "ProSeq'"]
DELTA_COL = "delta_lgKcat"

DEFAULT_FILENAMES = {
    "dlkcat":  "merged_mut_substrate_enzymeif_kcatpred_dlkcat.csv",
    "catapro": "merged_mut_substrate_enzymeif_kcatpred_catapro.csv",
    "unikp":   "merged_mut_substrate_enzymeif_kcatpred_unikp.csv",
}


def scale_only_norm(x: pd.Series) -> tuple[pd.Series, float, str]:
    """Scale-only normalization: x_scaled = x / scale; no centering, sign preserved."""
    x = pd.to_numeric(x, errors="coerce")
    q10, q90 = x.quantile(0.10), x.quantile(0.90)
    scale = (q90 - q10) if pd.notna(q10) and pd.notna(q90) else np.nan
    method = "q90-q10"
    if not pd.notna(scale) or scale == 0:
        sd = x.std(ddof=0)
        if pd.notna(sd) and sd > 0:
            scale, method = sd, "std"
        else:
            scale, method = 1.0, "fallback_1"
    return x / scale, float(scale), method


def read_and_prepare(tag: str, fp: Path, delta_col: str) -> pd.DataFrame:
    if not fp.exists():
        print(f"[ensemble_gdc][ERROR] file missing: {fp}", file=sys.stderr)
        sys.exit(2)
    df = pd.read_csv(fp)
    need = set(KEYS + [delta_col])
    miss = need - set(df.columns)
    if miss:
        print(
            f"[ensemble_gdc][ERROR] {fp.name} is missing columns: {sorted(miss)}",
            file=sys.stderr,
        )
        sys.exit(2)
    df = df[KEYS + [delta_col]].copy()
    df["Group"]   = df["Group"].astype(str)
    df["ProID"]   = df["ProID"].astype(str)
    df["ProSeq'"] = df["ProSeq'"].astype(str)
    df = df.dropna(subset=["ProSeq'"]).drop_duplicates(subset=KEYS, keep="first")
    df = df.rename(columns={delta_col: f"delta_{tag}"})
    return df


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m catif_rl.reward.ensemble_gdc",
        description=(
            "Ensemble three k_cat predictors at the GDC stage with scale-only "
            "(q90-q10) normalization, count positive-mean variants, and emit "
            "per-set CSVs + distribution figures."
        ),
    )
    p.add_argument("--in-dir", type=Path, default=Path("post_prediction"),
                   help="Directory containing the three predictor CSVs (default: %(default)s).")
    p.add_argument("--out-dir", type=Path, default=Path("posi_seq"),
                   help="Directory for the per-set detail CSVs (default: %(default)s).")
    p.add_argument("--fig-dir", type=Path, default=Path("figures"),
                   help="Directory for the distribution figures (default: %(default)s).")
    p.add_argument("--dlkcat-csv",  default=DEFAULT_FILENAMES["dlkcat"],
                   help="DLKcat CSV filename inside --in-dir (default: %(default)s).")
    p.add_argument("--catapro-csv", default=DEFAULT_FILENAMES["catapro"],
                   help="CataPro CSV filename inside --in-dir (default: %(default)s).")
    p.add_argument("--unikp-csv",   default=DEFAULT_FILENAMES["unikp"],
                   help="UniKP CSV filename inside --in-dir (default: %(default)s).")
    p.add_argument("--delta-col", default=DELTA_COL,
                   help="Per-predictor delta-lgKcat column name (default: %(default)s).")
    p.add_argument("--dpi", type=int, default=600,
                   help="Output figure DPI (default: %(default)s).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    d_dlk = read_and_prepare("dlkcat",  args.in_dir / args.dlkcat_csv,  args.delta_col)
    d_cat = read_and_prepare("catapro", args.in_dir / args.catapro_csv, args.delta_col)
    d_uni = read_and_prepare("unikp",   args.in_dir / args.unikp_csv,   args.delta_col)
    df = d_dlk.merge(d_cat, on=KEYS, how="inner").merge(d_uni, on=KEYS, how="inner")

    df["s_dlkcat"],  s_dlk, m_dlk = scale_only_norm(df["delta_dlkcat"])
    df["s_catapro"], s_cat, m_cat = scale_only_norm(df["delta_catapro"])
    df["s_unikp"],   s_uni, m_uni = scale_only_norm(df["delta_unikp"])
    print(
        f"[scale] DLKcat={s_dlk:.6g} ({m_dlk})  "
        f"CataPro={s_cat:.6g} ({m_cat})  "
        f"UniKP={s_uni:.6g} ({m_uni})"
    )

    df["mean3"] = (df["s_dlkcat"] + df["s_catapro"] + df["s_unikp"]) / 3.0
    df["mean2"] = (df["s_dlkcat"] + df["s_unikp"]) / 2.0  # top-2: DLKcat + UniKP

    a_df = (
        df.loc[df["mean3"] > 0, KEYS + ["mean3"]]
        .sort_values(["Group", "ProID"])
        .drop_duplicates(subset=["ProSeq'"], keep="first")
    )
    b_df = (
        df.loc[df["mean2"] > 0, KEYS + ["mean2"]]
        .sort_values(["Group", "ProID"])
        .drop_duplicates(subset=["ProSeq'"], keep="first")
    )
    a, b = len(a_df), len(b_df)
    set_a, set_b = set(a_df["ProSeq'"]), set(b_df["ProSeq'"])
    inter = set_a & set_b

    print(f"3-predictor mean > 0 (a): {a}")
    print(f"DLKcat+UniKP mean > 0 (b): {b}")
    print(f"Intersection |a ∩ b|: {len(inter)}")

    a_out = args.out_dir / "posi_planB_scaleonly_mean3_pos.csv"
    b_out = args.out_dir / "posi_planB_scaleonly_mean2_pos.csv"
    inter_out = args.out_dir / "posi_planB_scaleonly_mean2_mean3_inter.csv"
    a_df.to_csv(a_out, index=False)
    b_df.to_csv(b_out, index=False)
    (a_df[a_df["ProSeq'"].isin(inter)]
        .sort_values(["Group", "ProID"])
        .drop_duplicates(subset=["ProSeq'"], keep="first")
        .to_csv(inter_out, index=False))
    print(f"[save] {a_out}")
    print(f"[save] {b_out}")
    print(f"[save] {inter_out}")

    # Plots are imported lazily so --help works on systems without matplotlib.
    import matplotlib.pyplot as plt  # noqa: WPS433  (lazy import is intentional)

    bins = 60
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.hist(df["s_dlkcat"].dropna(),  bins=bins, alpha=0.5, label="DLKcat")
    ax1.hist(df["s_catapro"].dropna(), bins=bins, alpha=0.5, label="CataPro")
    ax1.hist(df["s_unikp"].dropna(),   bins=bins, alpha=0.5, label="UniKP")
    ax1.set_yscale("log"); ax1.axvline(0, linestyle="--")
    ax1.set_xlabel("Scaled delta_lgKcat (divide by q90-q10 / std)")
    ax1.set_ylabel("Count (log scale)")
    ax1.set_title("Scaled distributions of three predictors (no centering)")
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(args.fig_dir / "planB_scaleonly_three_predictors.png", dpi=args.dpi)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.hist(df["mean3"].dropna(), bins=bins, alpha=0.6,
             label="Mean(DLKcat, UniKP, CataPro)")
    ax2.set_yscale("log"); ax2.axvline(0, linestyle="--")
    ax2.set_xlabel("Mean of scaled scores (no centering)")
    ax2.set_ylabel("Count (log scale)")
    ax2.set_title("Ensemble score distributions")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(args.fig_dir / "Ensemble_scaleonly_means.png", dpi=args.dpi)

    print(f"[save] {args.fig_dir / 'planB_scaleonly_three_predictors.png'}")
    print(f"[save] {args.fig_dir / 'Ensemble_scaleonly_means.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
