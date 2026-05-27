"""End-to-end smoke test for the benchmark aggregator + method-id registry.

Runs ``catif_rl.evaluation.build_master --score-dir`` on a 2-method x 2-seed
fixture under ``tests/fixtures/benchmark_smoke/`` and verifies that:

1. The DLKcat outputs are picked up via the manuscript filename convention
   (``*_kcatpred_dlkcat.csv``).
2. The per-method subdirectory names (``catif``, ``catif_rl_r3``) are
   translated through ``catif_rl.evaluation._method_registry`` into the
   manuscript display names (``CatIF``, ``CatIF-RL R3``) used by Tables
   S7-S10.
3. The expected column set appears in the master CSV.
4. The Δlog10 k_cat aggregation is non-degenerate (CatIF-RL R3 should be
   higher than CatIF on the fixture).

Catches every regression class behind external-review round 2 blocker B4:

- Wrapper / aggregator filename drift
- Method-name mapping drift
- score-dir glob mismatch
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "benchmark_smoke"


def test_build_master_score_dir_fixture(tmp_path):
    master_csv = tmp_path / "master.csv"

    # Run the CLI just like scripts/07_score_benchmark.sh does.
    result = subprocess.run(
        [
            sys.executable, "-m", "catif_rl.evaluation.build_master",
            "--score-dir", str(FIXTURE),
            "--output",    str(master_csv),
        ],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, (
        f"build_master CLI failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert master_csv.exists(), "master CSV was not written"

    df = pd.read_csv(master_csv)

    # The aggregator should have translated the directory names
    # 'catif' / 'catif_rl_r3' into the manuscript display names via the
    # _method_registry.
    expected_columns_subset = [
        "ProID",
        "CatIF__delta_lgKcat", "CatIF__recovery",
        "CatIF__Backbone_RMSD", "CatIF__Avg_pLDDT",
        "CatIF-RL R3__delta_lgKcat", "CatIF-RL R3__recovery",
        "CatIF-RL R3__Backbone_RMSD", "CatIF-RL R3__Avg_pLDDT",
    ]
    for col in expected_columns_subset:
        assert col in df.columns, (
            f"expected column {col!r} missing from master CSV; "
            f"actual columns = {sorted(df.columns)}"
        )

    # Sanity check the numerical aggregation: CatIF-RL R3 fixture is
    # constructed to dominate CatIF on every ProID, so the per-protein mean
    # delta should also dominate.
    assert df["CatIF-RL R3__delta_lgKcat"].mean() > df["CatIF__delta_lgKcat"].mean(), (
        "fixture: CatIF-RL R3 should beat CatIF on average delta_lgKcat"
    )

    # 5 distinct ProIDs in the fixture.
    assert df["ProID"].nunique() == 5
    assert len(df) == 5


def test_method_registry_round_trip():
    """Defensive: the registry exposes the methods statistics.py + build_master expect."""
    from catif_rl.evaluation import _method_registry as reg
    # Headline reference
    assert reg.HEADLINE in reg.METHODS
    # ORDER is a permutation of METHODS keys
    assert set(reg.ORDER) == set(reg.METHODS.keys())
    # Display names contain CatIF-RL R3 (the headline)
    display = reg.display_order()
    assert "CatIF-RL R3" in display
    assert display[reg.ORDER.index(reg.HEADLINE)] == reg.METHODS[reg.HEADLINE]
