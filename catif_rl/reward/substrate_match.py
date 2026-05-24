#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replace the ``ProSeq'`` column in a (ProID, ProSeq, SMILES, ...) template
CSV with mutant sequences loaded from a directory of per-design FASTA
files, producing the (mutant, substrate) input expected by the *k*\\ :sub:`cat`
predictor wrappers.

FASTA files are expected to be named ``sequence_<ProID>.fasta`` or
``sequence_<ProID>.fa``, where ``<ProID>`` matches the ``ProID`` column of
the template CSV. When a matching FASTA is found, its sequence overwrites
that row's ``ProSeq'``; rows whose ``ProID`` cannot be matched are kept
as-is and a warning is printed.

Usage (positional args -- as invoked by ``scripts/07_score_benchmark.sh``)::

    python -m catif_rl.reward.substrate_match \\
      <template_csv> <fasta_dir> <output_csv>

Example, scoring the seed-1111 outputs of CatIF-RL R3::

    python -m catif_rl.reward.substrate_match \\
      data/brenda/test_mut_substrate_template.csv \\
      runs/benchmark/catif_rl_r3/seed_1111 \\
      runs/benchmark_scores/catif_rl_r3/test_mut_substrate_seed1111.csv

Arguments
---------
template_csv : Input CSV path; must contain columns ``ProID`` and ``ProSeq'``.
fasta_dir    : Directory of FASTA files named ``sequence_<ProID>.{fasta,fa}``.
output_csv   : Destination CSV path (the same columns as ``template_csv``).

Behaviour
---------
- The output CSV keeps all original columns; only ``ProSeq'`` is overwritten
  with the corresponding FASTA sequence when a match exists.
- If some ProIDs have no corresponding FASTA file, a warning is printed:
  ``Warning: no FASTA found for ProID(s): 101, 205, 309``.
"""

from __future__ import annotations
import argparse
import csv
from pathlib import Path

def load_fasta_sequences(fasta_dir: Path) -> dict[str, str]:
    sequences: dict[str, str] = {}

    fasta_paths = list(fasta_dir.glob("sequence_*.fasta")) + list(
        fasta_dir.glob("sequence_*.fa")
    )
    for fasta_path in sorted(fasta_paths):
        stem_parts = fasta_path.stem.split("_", 1)
        if len(stem_parts) != 2 or not stem_parts[1].isdigit():
            continue

        pro_id = stem_parts[1]
        with fasta_path.open("r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle if line.strip()]

        sequence = "".join(line for line in lines if not line.startswith(">"))
        if not sequence:
            continue

        sequences[pro_id] = sequence

    return sequences


def replace_sequences(
    csv_path: Path,
    fasta_dir: Path,
    output_path: Path,
) -> None:
    sequences = load_fasta_sequences(fasta_dir)

    missing_ids: set[str] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as src:
        reader = csv.DictReader(src)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError("Input CSV is empty or missing headers.")

        rows: list[dict[str, str]] = []
        for row in reader:
            pro_id = row.get("ProID", "").strip()
            if not pro_id:
                rows.append(row)
                continue

            replacement = sequences.get(pro_id)
            if replacement:
                row["ProSeq'"] = replacement
            else:
                missing_ids.add(pro_id)

            rows.append(row)

    with output_path.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if missing_ids:
        ids = ", ".join(sorted(missing_ids))
        print(f"Warning: no FASTA found for ProID(s): {ids}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to the source CSV file (e.g. output_gradeif_test_dataset_seed_12345.csv)",
    )
    parser.add_argument(
        "fasta_dir",
        type=Path,
        help="Directory containing sequence_*.fasta files",
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Path for the CSV file to write",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    replace_sequences(args.csv_path, args.fasta_dir, args.output_path)


if __name__ == "__main__":
    main()
