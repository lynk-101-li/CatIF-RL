#!/usr/bin/env python3
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
Replace the `ProSeq'` column with sequences loaded from FASTA files.
Program: replace_proseq_from_fasta.py
Description:
    本脚本用于将输入 CSV 文件中的 `ProSeq'` 列替换为来自指定 FASTA 文件夹中的序列。
    FASTA 文件命名格式为 `sequence_<ProID>.fasta` 或 `sequence_<ProID>.fa`，其中 <ProID> 必须与 CSV 表中 `ProID` 列的值一致。
    当存在匹配的 FASTA 文件时，将使用其序列内容替换对应行的 `ProSeq'` 值；
    若未找到匹配的 FASTA 文件，则该行保持原样，并在终端输出警告信息。

Usage:
    python replace_proseq_from_fasta.py <csv_path> <fasta_dir> <output_path>
Example:
    python replace_proseq_from_fasta.py test_mut_substrate_template.csv

Arguments:
    csv_path   : 输入 CSV 文件路径，需包含列 `ProID` 和 `ProSeq'`。
                  例如：output_gradeif_test_dataset_seed_12345.csv
    fasta_dir  : 存放 FASTA 文件的目录。每个 FASTA 文件名应为 sequence_<ProID>.fasta 或 sequence_<ProID>.fa。
                  例如：sequence_1024.fasta, sequence_2048.fa
    output_path: 替换完成后输出的新 CSV 文件路径。



Output:
    - 输出的新 CSV 文件保留原有列，仅 `ProSeq'` 列被替换为对应的 FASTA 序列。
    - 若某些 ProID 未在 fasta_dir 中找到对应的序列文件，将在终端输出警告：
        Warning: no FASTA found for ProID(s): 101, 205, 309

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
