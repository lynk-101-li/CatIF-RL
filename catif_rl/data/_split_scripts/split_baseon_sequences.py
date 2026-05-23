#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
import shutil

def split_sequences(csv_path: str, pdb_dir: str, matched_dir: str, unmatched_dir: str):
    import re
    # 1) Read the CSV and normalize IDs: strip whitespace; strip leading zeros if purely numeric. Keep IDs as strings.
    df = pd.read_csv(csv_path)
    if 'distc_pro_num' not in df.columns:
        raise KeyError(f'CSV is missing column distc_pro_num; columns found: {list(df.columns)}')

    def norm_id(x: str) -> str:
        s = str(x).strip()
        return str(int(s)) if re.fullmatch(r'\d+', s) else s  # pure digits: strip leading zeros
    ids_csv = set(df['distc_pro_num'].map(norm_id).astype(str))
    print(f'[CSV] unique IDs (after normalization): {len(ids_csv)}')

    # 2) Scan the directory (top level only); keep files starting with sequence/Sequence and ending in .pt or .pdb (case-insensitive)
    all_names = os.listdir(pdb_dir)
    cand = []
    for fname in all_names:
        if not fname.lower().startswith('sequence_'):
            continue
        lower = fname.lower()
        # Only accept .pt / .pdb (any case)
        if not (lower.endswith('.pt') or lower.endswith('.pdb')):
            continue
        cand.append(fname)

    print(f'[DIR] candidate files: {len(cand)} (start with sequence_, extension pt/pdb)')

    # 3) Pull every digit run from each filename; consider it a hit if any digit run equals one of the CSV IDs
    matched = []  # (src_path, dst_path)
    seen_hit_ids = set()
    sample_logs = []
    for i, fname in enumerate(cand[:10]):  # log first 10 samples for sanity
        digits = re.findall(r'\d+', fname)
        sample_logs.append((fname, digits))
    if sample_logs:
        print('[SAMPLE] filename -> digit runs:')
        for fn, ds in sample_logs:
            print('  -', fn, '->', ds)

    # Create output directories
    os.makedirs(matched_dir, exist_ok=True)
    os.makedirs(unmatched_dir, exist_ok=True)

    copied_match = 0
    copied_unmatch = 0

    for fname in cand:
        digits = re.findall(r'\d+', fname)
        hit = None
        # Prefer matches where the normalized digit run is in the CSV set
        for d in digits:
            if norm_id(d) in ids_csv:
                hit = norm_id(d)
                break

        src = os.path.join(pdb_dir, fname)
        if hit is not None:
            dst = os.path.join(matched_dir, fname)
            shutil.copy2(src, dst)
            copied_match += 1
            seen_hit_ids.add(hit)
        else:
            # Files that do not match any CSV ID are also copied into unmatched for inspection;
            # change this to "list only" if you do not want the copies.
            dst = os.path.join(unmatched_dir, fname)
            shutil.copy2(src, dst)
            copied_unmatch += 1

    # Write the list of CSV IDs that were not hit by any file
    unmatched_ids = sorted(ids_csv - seen_hit_ids, key=lambda x: (len(x), x))
    unmatched_csv_path = os.path.join(unmatched_dir, 'unmatched_ids.csv')
    pd.DataFrame({'distc_pro_num_norm': unmatched_ids}).to_csv(unmatched_csv_path, index=False)

    print(f'[RESULT] copied to matched: {copied_match} files; copied to unmatched: {copied_unmatch} files')
    print(f'[DONE] unmatched ID list written: {unmatched_csv_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Using the CSV list, split the sequence_n.pt files in pdb_output into matched / unmatched folders.'
    )
    parser.add_argument('--csv',      required=True, help='CSV file path; must contain a distc_pro_num column')
    parser.add_argument('--pdb_dir',  default='pdb_output', help='folder containing sequence_n.pt files')
    parser.add_argument('--matched',  default='matched',    help='output: matched files')
    parser.add_argument('--unmatched',default='unmatched',  help='output: unmatched files')
    args = parser.parse_args()

    split_sequences(args.csv, args.pdb_dir, args.matched, args.unmatched)
