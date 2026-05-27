"""Convert a CatIF-RL candidate / scoring CSV into a multi-record FASTA.

The CSVs produced by ``catif_rl.sampling.batch`` and the
``catif_rl.reward.substrate_match`` template carry at least ``ProID`` and
``ProSeq'`` columns. ESMFold (``catif_rl.data.esmfold_backbones``) wants a
FASTA. This module bridges the gap.

Record ID convention is ``sequence_<ProID>_var<idx>``, where ``idx`` is a
per-``ProID`` counter over the deduplicated rows. The matching ref PDB
``data/raw/.../sequence_<ProID>.pdb`` is then recovered from the pred PDB
stem via the ``--ref-stem-regex`` flag of ``catif_rl.evaluation.structural``
(see that module's docstring for the matcher contract).
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import pandas as pd


DEFAULT_ID_COL = "ProID"
DEFAULT_SEQ_COL = "ProSeq'"


def csv_to_fasta(
    csv_path: Path,
    out_fasta: Path,
    id_col: str = DEFAULT_ID_COL,
    seq_col: str = DEFAULT_SEQ_COL,
) -> int:
    """Write deduplicated ``(id_col, seq_col)`` rows to ``out_fasta``.

    Returns the number of FASTA records emitted.
    """
    df = pd.read_csv(csv_path)
    missing = [c for c in (id_col, seq_col) if c not in df.columns]
    if missing:
        raise KeyError(
            f"{csv_path}: missing required columns {missing}; available = {sorted(df.columns)}"
        )
    df = df[[id_col, seq_col]].dropna().drop_duplicates(subset=[id_col, seq_col]).reset_index(drop=True)
    counter: dict[str, int] = collections.defaultdict(int)
    out_fasta = Path(out_fasta)
    out_fasta.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_fasta.open("w") as f:
        for _, row in df.iterrows():
            pid = str(row[id_col])
            seq = str(row[seq_col]).strip()
            if not seq:
                continue
            idx = counter[pid]
            counter[pid] += 1
            f.write(f">sequence_{pid}_var{idx}\n{seq}\n")
            n += 1
    return n


def fasta_dir_to_combined(
    fasta_dir: Path,
    out_fasta: Path,
) -> int:
    """Concatenate every ``.fa`` / ``.fasta`` under ``fasta_dir`` into one FASTA.

    Returns the number of files merged. Filenames are not encoded in the
    record IDs -- whatever header is already in each input FASTA is kept.
    For benchmark scoring (``07_score_benchmark.sh``), each per-test-enzyme
    FASTA already has the header that the ESMFold-produced PDB stem should
    inherit (``sequence_<ProID>``), so a simple concat is enough.
    """
    fasta_dir = Path(fasta_dir)
    out_fasta = Path(out_fasta)
    out_fasta.parent.mkdir(parents=True, exist_ok=True)
    paths = sorted(list(fasta_dir.glob("*.fa")) + list(fasta_dir.glob("*.fasta")))
    if not paths:
        raise FileNotFoundError(f"no .fa/.fasta files under {fasta_dir}")
    with out_fasta.open("w") as out:
        for p in paths:
            text = p.read_text()
            if not text.endswith("\n"):
                text += "\n"
            out.write(text)
    return len(paths)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert a (ProID, ProSeq') CSV into a multi-record FASTA "
                    "suitable for ESMFold refolding."
    )
    p.add_argument("--input",  type=Path, required=True,
                   help="input CSV path (must contain ProID and ProSeq' columns)")
    p.add_argument("--output", type=Path, required=True,
                   help="output FASTA path")
    p.add_argument("--id-col",  type=str, default=DEFAULT_ID_COL,
                   help=f"name of the ProID column (default: {DEFAULT_ID_COL})")
    p.add_argument("--seq-col", type=str, default=DEFAULT_SEQ_COL,
                   help=f"name of the sequence column (default: {DEFAULT_SEQ_COL})")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    n = csv_to_fasta(args.input, args.output,
                     id_col=args.id_col, seq_col=args.seq_col)
    print(f"[csv_to_fasta] wrote {n} records to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
