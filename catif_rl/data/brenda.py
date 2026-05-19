"""DLKcat-BRENDA enzyme-kinetic dataset loader and sequence-level filtering.

Implements the filtering pipeline summarised in SI Table S4(a):

1. Start from the DLKcat-BRENDA records (16,839 organism-substrate-sequence rows).
2. Remove sequences longer than 1,180 amino acids (-76 unique).
3. Remove sequences containing non-standard residues (-18 unique).
4. De-duplicate identical protein sequences (collapses 16,745 -> 7,713 distinct).

Step 4 emits one ESMFold-predicted backbone per distinct sequence.
"""

from __future__ import annotations

import argparse
from pathlib import Path


CANONICAL_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
MAX_LENGTH = 1180


def load_raw_records(path: Path):
    """Load the raw DLKcat-BRENDA records CSV (16,839 rows).

    Expected columns include ``ECNumber``, ``Organism``, ``Substrate``,
    ``SMILES``, ``Sequence``, ``Value``, ``Unit``. The format mirrors the
    ``case.csv`` shipped under ``case_study/<ec>_<organism>/``.
    """

    import pandas as pd

    return pd.read_csv(path)


def filter_pipeline(df, max_length: int = MAX_LENGTH):
    """Apply the SI Table S4(a) filtering pipeline.

    Returns
    -------
    pandas.DataFrame
        Frame with the 7,713 distinct enzyme sequences retained.
    """

    # length filter
    df = df[df["Sequence"].str.len() <= max_length]
    # non-standard residue filter
    def _canonical(seq: str) -> bool:
        return set(seq).issubset(CANONICAL_AA)

    df = df[df["Sequence"].map(_canonical)]
    # de-duplicate identical protein sequences
    df = df.drop_duplicates(subset=["Sequence"]).reset_index(drop=True)
    return df


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", type=Path, required=True, help="Raw BRENDA CSV")
    p.add_argument("--output", type=Path, required=True, help="Filtered CSV output path")
    p.add_argument("--max-length", type=int, default=MAX_LENGTH)
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    df = load_raw_records(args.records)
    out = filter_pipeline(df, max_length=args.max_length)
    out.to_csv(args.output, index=False)
    print("Retained " + str(len(out)) + " distinct enzyme sequences.")
