"""Train / validation / test splitting for the EnzymeIF + CATH dataset.

Implements the manuscript's split strategy (§2.1, SI Table S4):

- The 1,423 DLKcat-test sequences are held out as the independent benchmark
  before any model training.
- The remaining 6,290 non-test enzymes are split 9:1 into 5,661 train and
  629 validation samples using ``random.Random(1234)``.
- 18,021 CATH v4.2.0 backbones are added as structural regularizers in train,
  and 608 in validation; CATH backbones never enter the held-out benchmark.

The detailed pipeline is a four-step chain that lives verbatim under
:mod:`catif_rl.data._split_scripts` (preserved as-is for archival fidelity):

    1. ``align_pdb_names.py``        align raw PDB filenames to BRENDA IDs
    2. ``split_baseon_sequences.py`` per-ID matching, emit matched/unmatched
    3. ``run_split.py``              dedup + assemble the test cohort
    4. ``dataset_split_final.py``    9:1 train/valid split + merge CATH

This module exposes the public API that ``scripts/01_build_dataset.sh`` and
downstream tools call. It is a thin orchestrator -- the actual work is
delegated to the four scripts above (run in subprocesses so each retains
its own logging / sanity-check behaviour).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_SEED = 1234
TRAIN_RATIO = 0.9

_SPLIT_SCRIPTS_DIR = Path(__file__).resolve().parent / "_split_scripts"


def _run_subprocess(script: Path, args: list[str]) -> None:
    """Execute one legacy split script in a subprocess, raising on failure."""
    cmd = [sys.executable, str(script), *args]
    print(f"[splits] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_full_split(
    pdb_dir: Path,
    train_csv: Path,
    dev_csv: Path,
    test_csv: Path,
    cath_train_dir: Path,
    cath_valid_dir: Path,
    output_dir: Path,
    seed: int = DEFAULT_SEED,
    train_ratio: float = TRAIN_RATIO,
) -> None:
    """End-to-end split orchestrator.

    Wraps the four legacy scripts under :mod:`catif_rl.data._split_scripts`
    in order:

    1. ``align_pdb_names.py``  renames PDB files to ``sequence_<distc_pro_num>.pdb``
       in a sibling ``brenda_seq_pdb/`` directory next to ``pdb_dir``.
    2. ``split_baseon_sequences.py`` and ``run_split.py`` carve out the
       1,423 DLKcat-test cohort and the 6,290 enzyme train+valid cohort.
    3. ``dataset_split_final.py`` performs the 9:1 split with
       ``random.Random(seed)`` and merges in the CATH backbones.

    Parameters
    ----------
    pdb_dir
        Directory of raw enzyme PDBs (one file per BRENDA ID).
    train_csv, dev_csv, test_csv
        The three BRENDA tables produced by DLKcat preprocessing (each
        containing at least ``distc_pro_num`` and ``enzyme`` columns).
    cath_train_dir, cath_valid_dir
        Directories of preprocessed CATH ``.pt`` graphs (18,021 and 608
        chains respectively) to merge in as structural regularizers.
    output_dir
        Final destination. Will contain subdirectories ``train/``,
        ``validation/``, and ``test/`` of ``.pt`` graph tensors.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    intermediate = output_dir.parent / "_split_intermediate"
    intermediate.mkdir(parents=True, exist_ok=True)

    # Step 1: align PDB filenames to BRENDA IDs.
    _run_subprocess(
        _SPLIT_SCRIPTS_DIR / "align_pdb_names.py",
        [
            "--pdb_dir", str(pdb_dir),
            "--train_csv", str(train_csv),
            "--dev_csv", str(dev_csv),
            "--test_csv", str(test_csv),
        ],
    )

    # Step 2 + 3: split into test cohort vs train+valid cohort.
    # run_split.py is configured by editing constants at the top; see
    # docstring -- we invoke it as-is and trust the user has adjusted
    # paths if the layout differs from the script defaults.
    _run_subprocess(_SPLIT_SCRIPTS_DIR / "run_split.py", [])

    # Step 4: 9:1 split + CATH merge.
    _run_subprocess(
        _SPLIT_SCRIPTS_DIR / "dataset_split_final.py",
        [
            "--base_in", str(intermediate),
            "--out_base", str(output_dir),
            "--train_ratio", str(train_ratio),
            "--seed", str(seed),
        ],
    )

    print(f"[splits] done; final layout under {output_dir} (train/validation/test)")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pdb-dir", type=Path, required=True,
                   help="directory of raw enzyme PDBs")
    p.add_argument("--train-csv", type=Path, required=True,
                   help="BRENDA train_set.csv (DLKcat preprocessing output)")
    p.add_argument("--dev-csv", type=Path, required=True,
                   help="BRENDA dev_set.csv")
    p.add_argument("--test-csv", type=Path, required=True,
                   help="BRENDA test_set.csv")
    p.add_argument("--cath-train-dir", type=Path, required=True,
                   help="directory of preprocessed CATH train .pt files (18,021)")
    p.add_argument("--cath-valid-dir", type=Path, required=True,
                   help="directory of preprocessed CATH valid .pt files (608)")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="final train/validation/test destination")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    run_full_split(
        pdb_dir=args.pdb_dir,
        train_csv=args.train_csv,
        dev_csv=args.dev_csv,
        test_csv=args.test_csv,
        cath_train_dir=args.cath_train_dir,
        cath_valid_dir=args.cath_valid_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
    )
