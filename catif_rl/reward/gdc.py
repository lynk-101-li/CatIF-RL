"""Generative Dataset Curation (GDC) funnel orchestration.

Implements the two-stage candidate filter described in manuscript §2.3:

1. Structural plausibility -- backbone RMSD < 4 Å and mean pLDDT > 90 after
   ESMFold refolding (delegated to :mod:`catif_rl.reward.structural_filter`).
2. Activity ensemble -- normalize each of DLKcat / UniKP / CataPro Δlog k_cat
   onto a comparable scale via 10th/90th quantile range, then take the mean
   (delegated to :mod:`catif_rl.reward.ensemble_gdc`). Variants with
   ``S_ensemble > 0`` are retained as activity-positive.

The ensemble quantile parameters are frozen at the start of GDC and reused
through all CatIF supervised training, GRPO rounds 1-3, and final test
evaluation (SI Table S5 note 3).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def run_gdc_funnel(
    candidate_csv: Path,
    structural_metrics_csv: Path,
    output_dir: Path,
    rmsd_threshold: float = 4.0,
    plddt_threshold: float = 90.0,
    ensemble_threshold: float = 0.0,
) -> None:
    """Run the two-stage GDC filter end-to-end.

    Parameters
    ----------
    candidate_csv
        Raw candidate pool (typically 62,900 variants from EnzymeIF sampling
        on 6,290 training backbones with G=10).
    structural_metrics_csv
        Pre-computed ESMFold refolding metrics
        (``ideal_rmsf_plddt_enzymeif_merged.csv``); produced upstream of
        this script.
    output_dir
        Where to write ``structure_valid.csv`` (after stage 1) and
        ``activity_positive.csv`` (after stage 2).
    """

    # The actual implementation delegates to the two stage modules. This
    # function is the orchestration entry point invoked by
    # ``scripts/03_run_gdc.sh``.
    from catif_rl.reward import ensemble_gdc, structural_filter  # noqa: F401

    output_dir.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError(
        "GDC funnel orchestration scaffold -- delegate to "
        "structural_filter.filter() then ensemble_gdc.score(); "
        "wire-up pending."
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--structural-metrics", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--rmsd-threshold", type=float, default=4.0)
    p.add_argument("--plddt-threshold", type=float, default=90.0)
    p.add_argument("--ensemble-threshold", type=float, default=0.0)
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    run_gdc_funnel(
        candidate_csv=args.candidates,
        structural_metrics_csv=args.structural_metrics,
        output_dir=args.output_dir,
        rmsd_threshold=args.rmsd_threshold,
        plddt_threshold=args.plddt_threshold,
        ensemble_threshold=args.ensemble_threshold,
    )
