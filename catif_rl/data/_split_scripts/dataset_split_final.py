#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dataset_split_final.py

Purpose:
1) Split enzyme_train_and_valid_dataset 9:1 (seed=1234) into train/valid (enzyme samples).
2) Copy enzyme_test_dataset verbatim into dataset/process/test.
3) Merge generic-protein .pt files from train_split_universal_graph into train.
4) Merge generic-protein .pt files from valid_split_universal_graph into validation.
5) Deduplicate and ensure no overlap across sets (filenames are the unique key);
   any duplicate name is skipped with a notice.
"""
"""
python dataset_src/data_split_for_gradeif_training/dataset_split_final.py \
  --base_in dataset_src/data_split_for_gradeif_training \
  --out_base dataset/process \
  --train_ratio 0.9 \
  --seed 1234
"""

import argparse
import os
import random
import shutil
from pathlib import Path
from typing import List, Set

SEED = 1234

def list_pt_files(folder: Path) -> List[Path]:
    if not folder.exists():
        print(f"[WARN] directory does not exist: {folder}")
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix == ".pt"])

def safe_copy(src: Path, dst_dir: Path, taken_names: Set[str]) -> bool:
    """Dedup by filename; skip duplicates. Returns True on successful copy."""
    name = src.name
    if name in taken_names:
        print(f"[SKIP] duplicate name (already present in another set): {name}")
        return False
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_dir / name)
    taken_names.add(name)
    return True

def split_indices(n: int, train_ratio: float, seed: int):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_train = int(round(n * train_ratio))
    return idx[:n_train], idx[n_train:]

def main():
    parser = argparse.ArgumentParser(description="Split enzyme datasets and merge universal graphs.")
    parser.add_argument("--base_in", default="dataset_src/data_split_for_gradeif_training", help="input base directory (contains enzyme_* and *_universal_graph)")
    parser.add_argument("--out_base", default="dataset/process", help="output base directory (train/validation/test)")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="train ratio for enzyme samples (default 0.9)")
    parser.add_argument("--seed", type=int, default=SEED, help="random seed (default 1234)")
    args = parser.parse_args()

    base_in = Path(args.base_in)
    out_base = Path(args.out_base)

    # Input directories
    d_test_enzyme = base_in / "enzyme_test_dataset"
    d_trainvalid_enzyme = base_in / "enzyme_train_and_valid_dataset"
    d_univ_train = base_in / "train_split_universal_graph"
    d_univ_valid = base_in / "valid_split_universal_graph"

    # Output directories
    d_out_train = out_base / "train"
    d_out_valid = out_base / "validation"
    d_out_test  = out_base / "test"

    print("=== Path check ===")
    print(f"input (test enzymes): {d_test_enzyme}")
    print(f"input (train+valid enzymes): {d_trainvalid_enzyme}")
    print(f"input (universal graphs - train): {d_univ_train}")
    print(f"input (universal graphs - valid): {d_univ_valid}")
    print(f"output (train): {d_out_train}")
    print(f"output (validation): {d_out_valid}")
    print(f"output (test): {d_out_test}")

    # Collect files
    test_files = list_pt_files(d_test_enzyme)
    tv_files = list_pt_files(d_trainvalid_enzyme)
    univ_train_files = list_pt_files(d_univ_train)
    univ_valid_files = list_pt_files(d_univ_valid)

    print("\n=== Input counts ===")
    print(f"enzyme_test_dataset: {len(test_files)} .pt")
    print(f"enzyme_train_and_valid_dataset: {len(tv_files)} .pt")
    print(f"train_split_universal_graph: {len(univ_train_files)} .pt")
    print(f"valid_split_universal_graph: {len(univ_valid_files)} .pt")

    # 1) Split train/valid (enzyme samples only)
    if len(tv_files) == 0:
        print("[ERROR] train+valid enzyme directory is empty.")
        return

    train_idx, valid_idx = split_indices(len(tv_files), args.train_ratio, args.seed)
    train_enzyme = [tv_files[i] for i in train_idx]
    valid_enzyme = [tv_files[i] for i in valid_idx]

    print("\n=== Split result (enzymes) ===")
    print(f"Train (enzyme): {len(train_enzyme)}")
    print(f"Valid (enzyme): {len(valid_enzyme)}")
    # Sanity: 6290 -> 5661 / 629; if counts differ, the ratio still applies.
    # Strict assertions could go here:
    # assert len(tv_files) == 6290 and len(train_enzyme) == 5661 and len(valid_enzyme) == 629

    # 2) Copy the test set (enzymes) to test/
    print("\n=== Copying test set (enzymes kept as-is) ===")
    taken_test = set()      # set of test filenames
    copied_test = 0
    for f in test_files:
        if safe_copy(f, d_out_test, taken_test):
            copied_test += 1
    print(f"[DONE] test: copied {copied_test} files.")

    # 3) Copy enzymes into train/valid (enzyme samples only for now)
    print("\n=== Copying train / valid (enzymes) ===")
    taken_train = set()     # set of train filenames
    taken_valid = set()     # set of valid filenames

    copied_train_enzyme = 0
    for f in train_enzyme:
        if safe_copy(f, d_out_train, taken_train):
            copied_train_enzyme += 1

    copied_valid_enzyme = 0
    for f in valid_enzyme:
        # Make sure no overlap with train/test
        if f.name in taken_train or f.name in taken_test:
            print(f"[SKIP] valid file collides with train/test: {f.name}")
            continue
        if safe_copy(f, d_out_valid, taken_valid):
            copied_valid_enzyme += 1

    print(f"[DONE] train (enzyme): {copied_train_enzyme}")
    print(f"[DONE] validation (enzyme): {copied_valid_enzyme}")

    # 4) Merge in universal graphs (generic protein .pt)
    print("\n=== Merge universal graphs: train_split_universal_graph -> train ===")
    copied_univ_train = 0
    for f in univ_train_files:
        # Prevent collisions with test/valid/train
        if f.name in taken_test or f.name in taken_valid or f.name in taken_train:
            print(f"[SKIP] universal-train collides with an existing set: {f.name}")
            continue
        if safe_copy(f, d_out_train, taken_train):
            copied_univ_train += 1
    print(f"[DONE] train (+universal): added {copied_univ_train} generic proteins")

    print("\n=== Merge universal graphs: valid_split_universal_graph -> validation ===")
    copied_univ_valid = 0
    for f in univ_valid_files:
        if f.name in taken_test or f.name in taken_train or f.name in taken_valid:
            print(f"[SKIP] universal-valid collides with an existing set: {f.name}")
            continue
        if safe_copy(f, d_out_valid, taken_valid):
            copied_univ_valid += 1
    print(f"[DONE] validation (+universal): added {copied_univ_valid} generic proteins")

    # 5) Summary
    n_train_total = len(list_pt_files(d_out_train))
    n_valid_total = len(list_pt_files(d_out_valid))
    n_test_total  = len(list_pt_files(d_out_test))

    print("\n=== Final summary ===")
    print(f"Train total: {n_train_total} (enzymes {copied_train_enzyme}, generic {copied_univ_train})")
    print(f"Validation total: {n_valid_total} (enzymes {copied_valid_enzyme}, generic {copied_univ_valid})")
    print(f"Test total: {n_test_total} (all enzymes)")
    print("\nDone.")

if __name__ == "__main__":
    main()
