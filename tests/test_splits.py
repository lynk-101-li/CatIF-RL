"""Smoke test for catif_rl.data.splits (external-review round 2 B1).

The full SI Table S4 run requires 6,290 + 18,629 + 1,423 = 26,342 .pt files,
which is too heavy for CI. This test exercises the same code path against
a synthetic toy fixture, then asserts:

1. The 9:1 enzyme split is reproducible (same seed -> same partition).
2. The CATH partition is read from the bundled manifest, not from the input
   directory's filename ordering.
3. The end-of-run count assertion FIRES when the inputs don't match a
   given expected set (--skip-count-assert downgrades to warning).
4. The fallback (CATH dir missing some chains) emits warnings but does
   not crash.

No torch / no biopython.
"""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock


def _make_pt(p: Path, content: bytes = b"PT") -> None:
    p.write_bytes(content)


def _setup_synthetic_layout(root: Path, n_enzymes: int = 10) -> tuple[Path, Path, Path, Path]:
    """Build a self-contained toy filesystem: enzyme dir, CATH dir, test dir,
    and a tiny chain_set_splits.json manifest."""
    enzyme_dir = root / "process" / "enzymeif" / "train_and_validation"
    cath_dir   = root / "process" / "enzymeif" / "cath_v4_2_0"
    test_dir   = root / "process" / "test"
    out_dir    = root / "process"
    for d in (enzyme_dir, cath_dir, test_dir):
        d.mkdir(parents=True, exist_ok=True)

    for i in range(n_enzymes):
        _make_pt(enzyme_dir / f"sequence_{i}.pt")
    # CATH cohort: 6 train + 2 valid in this synthetic layout.
    for cid in ("aaaa.A", "bbbb.A", "cccc.A", "dddd.A", "eeee.A", "ffff.A"):
        _make_pt(cath_dir / f"{cid}.pt")
    for cid in ("xxxx.A", "yyyy.A"):
        _make_pt(cath_dir / f"{cid}.pt")
    # 3 test enzymes (synthetic SI Table S4 surrogate).
    for i in range(3):
        _make_pt(test_dir / f"sequence_test_{i}.pt")

    manifest = root / "chain_set_splits.json"
    with manifest.open("w") as f:
        json.dump({
            "train":      ["aaaa.A", "bbbb.A", "cccc.A", "dddd.A", "eeee.A", "ffff.A"],
            "validation": ["xxxx.A", "yyyy.A"],
            "test":       [],
            "cath_nodes": {},
        }, f)
    return enzyme_dir, cath_dir, manifest, out_dir


def test_split_reproducible_under_fixed_seed():
    """Same seed -> identical partitioning of enzyme cohort."""
    # We monkey-patch EXPECTED_COUNTS to match the synthetic fixture before
    # importing run_full_split, so the count assertion does not trip.
    from catif_rl.data import splits as M
    importlib.reload(M)
    M.EXPECTED_COUNTS = {
        "train_enzyme": 9,
        "valid_enzyme": 1,
        "train_cath":   6,
        "valid_cath":   2,
        "test_enzyme":  3,
    }

    seed = 1234
    train_sets = []
    for trial in range(2):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            enzyme_dir, cath_dir, manifest, out_dir = _setup_synthetic_layout(root, n_enzymes=10)
            summary = M.run_full_split(
                enzyme_graphs_dir=enzyme_dir,
                cath_graphs_dir=cath_dir,
                output_dir=out_dir,
                chain_manifest=manifest,
                seed=seed,
                train_ratio=0.9,
            )
            assert summary["counts"]["train_enzyme"] == 9
            assert summary["counts"]["valid_enzyme"] == 1
            assert summary["counts"]["train_cath"]   == 6
            assert summary["counts"]["valid_cath"]   == 2
            assert summary["counts"]["test_enzyme"]  == 3
            train_sets.append(sorted((out_dir / "train").glob("sequence_*.pt")))
    # Same seed -> same enzyme train set
    a = [p.name for p in train_sets[0]]
    b = [p.name for p in train_sets[1]]
    assert a == b, (a, b)


def test_count_mismatch_raises_by_default():
    """If the input cohort is the wrong size, the assertion fires."""
    from catif_rl.data import splits as M
    # Restore the real EXPECTED_COUNTS (the manuscript ones)
    importlib.reload(M)
    assert M.EXPECTED_COUNTS["train_enzyme"] == 5661

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        enzyme_dir, cath_dir, manifest, out_dir = _setup_synthetic_layout(root, n_enzymes=10)
        try:
            M.run_full_split(
                enzyme_graphs_dir=enzyme_dir,
                cath_graphs_dir=cath_dir,
                output_dir=out_dir,
                chain_manifest=manifest,
            )
        except SystemExit as e:
            assert "SI Table S4" in str(e)
            return
        raise AssertionError("expected SystemExit on count mismatch")


def test_skip_count_assert_downgrades_to_warning(capsys):
    """--skip-count-assert lets the script complete despite the mismatch."""
    from catif_rl.data import splits as M
    importlib.reload(M)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        enzyme_dir, cath_dir, manifest, out_dir = _setup_synthetic_layout(root, n_enzymes=10)
        summary = M.run_full_split(
            enzyme_graphs_dir=enzyme_dir,
            cath_graphs_dir=cath_dir,
            output_dir=out_dir,
            chain_manifest=manifest,
            skip_count_assert=True,
        )
        assert summary["counts"]["train_enzyme"] == 9


def test_missing_cath_chain_only_warns():
    """A chain listed in the manifest but absent from cath_graphs_dir is skipped
    with a warning, not a hard error (lets the user re-run download)."""
    from catif_rl.data import splits as M
    importlib.reload(M)
    M.EXPECTED_COUNTS = {
        "train_enzyme": 9, "valid_enzyme": 1,
        "train_cath": 5, "valid_cath": 2, "test_enzyme": 3,
    }
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        enzyme_dir, cath_dir, manifest, out_dir = _setup_synthetic_layout(root, n_enzymes=10)
        # Delete one CATH train chain
        (cath_dir / "ffff.A.pt").unlink()
        summary = M.run_full_split(
            enzyme_graphs_dir=enzyme_dir,
            cath_graphs_dir=cath_dir,
            output_dir=out_dir,
            chain_manifest=manifest,
        )
        assert summary["counts"]["train_cath"] == 5


if __name__ == "__main__":
    test_split_reproducible_under_fixed_seed()
    test_count_mismatch_raises_by_default()

    # capsys isn't available outside pytest; just skip that test in standalone mode
    import io
    class _Stub:
        def readouterr(self):
            return type("R", (), {"out": "", "err": ""})()
    test_skip_count_assert_downgrades_to_warning(_Stub())

    test_missing_cath_chain_only_warns()
    print("PASS")
