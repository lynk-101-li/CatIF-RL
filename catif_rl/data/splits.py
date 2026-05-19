"""Train / validation / test splitting for the EnzymeIF + CATH dataset.

Preserves the original DLKcat split strategy (manuscript §2.1 and SI Table S4):

- The 1,423 DLKcat-test sequences are held out as the independent benchmark
  before any model training.
- The remaining 6,290 non-test enzymes are split 9:1 into 5,661 train and
  629 validation samples using ``random.Random(1234)``.
- 18,021 CATH v4.2.0 backbones are added as structural regularizers in train,
  and 608 in validation; CATH backbones never enter the held-out benchmark.

The detailed pipeline (a chain of three scripts -- ``split_baseon_sequences``
``run_split`` ``dataset_split_final`` plus the ``align_pdb_names`` helper) is
preserved verbatim under :mod:`catif_rl.data._split_scripts`. This module
exposes the public API that ``scripts/01_build_dataset.sh`` calls.
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_SEED = 1234
TRAIN_RATIO = 0.9


def run_full_split(
    distinct_enzymes_csv: Path,
    cath_backbones_dir: Path,
    output_dir: Path,
    seed: int = DEFAULT_SEED,
    train_ratio: float = TRAIN_RATIO,
) -> None:
    """End-to-end split orchestrator.

    Wraps the underlying scripts (see :mod:`catif_rl.data._split_scripts`)
    in a single call. Output directory contains:

    - ``enzyme_test_dataset/`` (1,423 PDB stems)
    - ``enzyme_train_and_valid_dataset/`` (5,661 + 629 split)
    - ``train_split_universal_graph/`` (18,021 CATH backbones)
    - ``valid_split_universal_graph/`` (608 CATH backbones)
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError(
        "Split orchestrator scaffold. Call sequence (see SI Table S4 walkthrough): "
        "split_baseon_sequences -> run_split -> dataset_split_final; "
        "align_pdb_names is used as a helper between steps. The underlying "
        "scripts live under catif_rl/data/_split_scripts/ and can be invoked "
        "directly until the wrapper is wired up."
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--distinct-enzymes", type=Path, required=True)
    p.add_argument("--cath-backbones-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    run_full_split(
        distinct_enzymes_csv=args.distinct_enzymes,
        cath_backbones_dir=args.cath_backbones_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
    )
