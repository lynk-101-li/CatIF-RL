"""Train / validation / test splitting for the EnzymeIF + CATH dataset.

Implements the manuscript's split strategy (§2.1, SI Table S4) **directly**,
without delegating to the legacy four-script chain that used to live under
``catif_rl/data/_split_scripts/`` (those scripts had hardcoded private
paths and only ran on the author's workstation; they are kept under
``_split_scripts/`` for archival reference but no longer invoked).

Contract:

- The 1,423 DLKcat-test enzymes are kept out of the train+valid pool from
  the start; ``scripts/01_build_dataset.sh`` step 2 materialises them at
  ``data/process/test/`` via graph_construction. This module just verifies
  the count is right.
- The 6,290 non-test EnzymeIF cohort enzymes are split 9:1 into 5,661
  train and 629 valid using ``random.Random(seed)`` (default seed 1234).
- 18,021 CATH v4.2.0 chains are added as structural regularisers to train,
  608 to valid; the assignment comes from
  ``catif_rl/data/assets/chain_set_splits.json``.

End-state counts (asserted at the end of the run):

| Split | EnzymeIF cohort | CATH regularisers | total |
|-------|-----------------|--------------------|-------|
| train | 5,661           | 18,021             | 23,682 |
| valid |   629           |    608             |  1,237 |
| test  | 1,423           |     0              |  1,423 |

A non-match raises ``SystemExit`` -- the manuscript counts are a contract.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path


DEFAULT_SEED = 1234
TRAIN_RATIO = 0.9
DEFAULT_CHAIN_MANIFEST = (
    Path(__file__).resolve().parent / "assets" / "chain_set_splits.json"
)

# Expected counts per SI Table S4. Asserted at end of run.
EXPECTED_COUNTS = {
    "train_enzyme": 5661,
    "valid_enzyme": 629,
    "train_cath":   18021,
    "valid_cath":   608,
    "test_enzyme":  1423,
}


def _list_pt(directory: Path) -> list[str]:
    """Return sorted .pt filenames under ``directory`` (non-recursive)."""
    return sorted(f.name for f in Path(directory).iterdir()
                  if f.is_file() and f.suffix == ".pt")


def _split_indices(n: int, train_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_train = int(round(n * train_ratio))
    return idx[:n_train], idx[n_train:]


def _safe_copy(src: Path, dst: Path) -> bool:
    """Copy ``src`` to ``dst`` (creating dst.parent if needed). Returns
    ``True`` on success, ``False`` if ``src`` is missing."""
    if not src.exists():
        print(f"[splits][WARN] expected source file missing: {src}", file=sys.stderr)
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def run_full_split(
    enzyme_graphs_dir: Path,
    cath_graphs_dir: Path,
    output_dir: Path,
    chain_manifest: Path = DEFAULT_CHAIN_MANIFEST,
    seed: int = DEFAULT_SEED,
    train_ratio: float = TRAIN_RATIO,
    skip_count_assert: bool = False,
) -> dict:
    """End-to-end split orchestrator.

    Consumes the per-cohort .pt graph dirs produced by
    ``scripts/01_build_dataset.sh`` step 2 and emits

        <output_dir>/train/   (5,661 enzyme  +  18,021 CATH)
        <output_dir>/valid/   (  629 enzyme  +     608 CATH)

    ``<output_dir>/test/`` is assumed already populated by step 2 with the
    1,423 held-out enzyme graphs (its count is verified, but no copy is
    performed).

    Parameters
    ----------
    enzyme_graphs_dir
        Path to ``data/process/enzymeif/train_and_validation/`` containing
        the 6,290 EnzymeIF-cohort .pt graphs (pre-split).
    cath_graphs_dir
        Path to ``data/process/enzymeif/cath_v4_2_0/`` containing all CATH
        chain graphs.
    output_dir
        ``data/process/``; the train/ and valid/ subdirectories are
        created (or extended) under this root.
    chain_manifest
        Path to ``chain_set_splits.json`` (defaults to the bundled asset).
    seed, train_ratio
        Control the enzyme-cohort split.
    skip_count_assert
        If True, log the count mismatch but do not raise. Use only for
        debugging non-canonical inputs.

    Returns
    -------
    summary
        Dict containing per-split counts and the resolved input paths.
    """
    enzyme_graphs_dir = Path(enzyme_graphs_dir)
    cath_graphs_dir   = Path(cath_graphs_dir)
    output_dir        = Path(output_dir)
    chain_manifest    = Path(chain_manifest)

    if not enzyme_graphs_dir.is_dir():
        raise FileNotFoundError(f"enzyme_graphs_dir does not exist: {enzyme_graphs_dir}")
    if not cath_graphs_dir.is_dir():
        raise FileNotFoundError(f"cath_graphs_dir does not exist: {cath_graphs_dir}")
    if not chain_manifest.is_file():
        raise FileNotFoundError(f"chain_manifest does not exist: {chain_manifest}")

    # ---- 1. Enzyme split (9:1 with random.Random(seed)) ----------------------
    enzyme_files = _list_pt(enzyme_graphs_dir)
    print(f"[splits] enzyme cohort: {len(enzyme_files)} .pt graphs at {enzyme_graphs_dir}")
    train_idx, valid_idx = _split_indices(len(enzyme_files), train_ratio, seed)
    train_enzyme_names = [enzyme_files[i] for i in train_idx]
    valid_enzyme_names = [enzyme_files[i] for i in valid_idx]
    print(f"[splits]   train={len(train_enzyme_names)}, valid={len(valid_enzyme_names)}  "
          f"(seed={seed}, ratio={train_ratio})")

    # ---- 2. CATH split from manifest ----------------------------------------
    with chain_manifest.open() as f:
        manifest = json.load(f)
    cath_train_ids = manifest.get("train", [])
    cath_valid_ids = manifest.get("validation", [])
    print(f"[splits] CATH manifest: train={len(cath_train_ids)}, "
          f"validation={len(cath_valid_ids)}  ({chain_manifest})")

    available_cath = set(_list_pt(cath_graphs_dir))
    train_cath_names = [f"{cid}.pt" for cid in cath_train_ids if f"{cid}.pt" in available_cath]
    valid_cath_names = [f"{cid}.pt" for cid in cath_valid_ids if f"{cid}.pt" in available_cath]
    missing_train = [cid for cid in cath_train_ids if f"{cid}.pt" not in available_cath]
    missing_valid = [cid for cid in cath_valid_ids if f"{cid}.pt" not in available_cath]
    if missing_train or missing_valid:
        print(f"[splits][WARN] CATH dir is missing {len(missing_train)} train "
              f"and {len(missing_valid)} validation chains "
              f"(out of {len(cath_train_ids)} / {len(cath_valid_ids)} in the manifest). "
              f"Re-run scripts/01_build_dataset.sh step 1 to fetch them.", file=sys.stderr)
        if missing_train[:3]:
            print(f"[splits][WARN]   first missing train: {missing_train[:3]}", file=sys.stderr)

    # ---- 3. Copy into output/train/ and output/valid/ -----------------------
    train_out = output_dir / "train"
    valid_out = output_dir / "valid"
    train_out.mkdir(parents=True, exist_ok=True)
    valid_out.mkdir(parents=True, exist_ok=True)

    def _copy_batch(src_dir: Path, names: list[str], dst_dir: Path) -> int:
        n = 0
        for name in names:
            if _safe_copy(src_dir / name, dst_dir / name):
                n += 1
        return n

    n_train_enzyme = _copy_batch(enzyme_graphs_dir, train_enzyme_names, train_out)
    n_valid_enzyme = _copy_batch(enzyme_graphs_dir, valid_enzyme_names, valid_out)
    n_train_cath   = _copy_batch(cath_graphs_dir,   train_cath_names,   train_out)
    n_valid_cath   = _copy_batch(cath_graphs_dir,   valid_cath_names,   valid_out)

    # ---- 4. Verify test/ population (mirrored by step 2 from data/raw/test/)
    test_out = output_dir / "test"
    n_test_enzyme = len(_list_pt(test_out)) if test_out.is_dir() else 0

    # ---- 5. Assert manuscript counts ----------------------------------------
    counts = {
        "train_enzyme": n_train_enzyme,
        "valid_enzyme": n_valid_enzyme,
        "train_cath":   n_train_cath,
        "valid_cath":   n_valid_cath,
        "test_enzyme":  n_test_enzyme,
    }
    print(f"[splits] counts:    {counts}")
    print(f"[splits] expected:  {EXPECTED_COUNTS}  (SI Table S4)")
    diffs = [k for k in EXPECTED_COUNTS if counts[k] != EXPECTED_COUNTS[k]]
    if diffs:
        msg = "[ERROR] split counts do not match SI Table S4:\n"
        for k in diffs:
            msg += f"  {k}: got {counts[k]} != expected {EXPECTED_COUNTS[k]}\n"
        if skip_count_assert:
            print(msg, file=sys.stderr)
        else:
            raise SystemExit(msg.rstrip())
    else:
        print(f"[splits] all counts match SI Table S4 ✓")

    return {
        "enzyme_graphs_dir": str(enzyme_graphs_dir),
        "cath_graphs_dir":   str(cath_graphs_dir),
        "output_dir":        str(output_dir),
        "chain_manifest":    str(chain_manifest),
        "seed":              seed,
        "train_ratio":       train_ratio,
        "counts":            counts,
        "expected":          EXPECTED_COUNTS,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--enzyme-graphs-dir", type=Path, required=True,
                   help="data/process/enzymeif/train_and_validation/  (6,290 .pt)")
    p.add_argument("--cath-graphs-dir", type=Path, required=True,
                   help="data/process/enzymeif/cath_v4_2_0/  (CATH chains, "
                        "partitioned via the manifest at split time)")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="data/process/  (train/ and valid/ subdirs are created)")
    p.add_argument("--chain-manifest", type=Path, default=DEFAULT_CHAIN_MANIFEST,
                   help=f"chain split manifest (default: {DEFAULT_CHAIN_MANIFEST})")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    p.add_argument("--skip-count-assert", action="store_true",
                   help="downgrade SI Table S4 count mismatch from error to warning "
                        "(use only for debugging non-canonical inputs)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    run_full_split(
        enzyme_graphs_dir=args.enzyme_graphs_dir,
        cath_graphs_dir=args.cath_graphs_dir,
        output_dir=args.output_dir,
        chain_manifest=args.chain_manifest,
        seed=args.seed,
        train_ratio=args.train_ratio,
        skip_count_assert=args.skip_count_assert,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
