#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
import shutil

def split_sequences(csv_path: str, pdb_dir: str, matched_dir: str, unmatched_dir: str):
    import re
    # 1) 读取 CSV，清洗 ID：去空格 → 若为纯数字则去前导0；保持为字符串集合
    df = pd.read_csv(csv_path)
    if 'distc_pro_num' not in df.columns:
        raise KeyError(f'CSV 缺少列 distc_pro_num，实际列有：{list(df.columns)}')

    def norm_id(x: str) -> str:
        s = str(x).strip()
        return str(int(s)) if re.fullmatch(r'\d+', s) else s  # 纯数字 → 去前导0
    ids_csv = set(df['distc_pro_num'].map(norm_id).astype(str))
    print(f'[CSV] 唯一ID数（清洗后）: {len(ids_csv)}')

    # 2) 扫描目录（仅当前层），过滤出以 sequence/Sequence 开头且扩展名为 pt/pdb（大小写不敏感）的文件
    all_names = os.listdir(pdb_dir)
    cand = []
    for fname in all_names:
        if not fname.lower().startswith('sequence_'):
            continue
        lower = fname.lower()
        # 只接受 .pt / .pdb 结尾；允许大小写
        if not (lower.endswith('.pt') or lower.endswith('.pdb')):
            continue
        cand.append(fname)

    print(f'[DIR] 候选文件数: {len(cand)}（以 sequence_ 开头，扩展名 pt/pdb）')

    # 3) 从文件名中提取所有数字段；只要有一个和 CSV ID 集合相同即可判为命中
    matched = []  # (src_path, dst_path)
    seen_hit_ids = set()
    sample_logs = []
    for i, fname in enumerate(cand[:10]):  # 打印前10条样例便于核对
        digits = re.findall(r'\d+', fname)
        sample_logs.append((fname, digits))
    if sample_logs:
        print('[SAMPLE] 文件名 → 数字段：')
        for fn, ds in sample_logs:
            print('  -', fn, '→', ds)

    # 建输出目录
    os.makedirs(matched_dir, exist_ok=True)
    os.makedirs(unmatched_dir, exist_ok=True)

    copied_match = 0
    copied_unmatch = 0

    for fname in cand:
        digits = re.findall(r'\d+', fname)
        hit = None
        # 优先匹配：与 CSV 集合完全相等的数字串（已去前导0）
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
            # 未命中 ID 的文件也可复制到 unmatched 方便排查；如果不想复制，改为“只写清单”
            dst = os.path.join(unmatched_dir, fname)
            shutil.copy2(src, dst)
            copied_unmatch += 1

    # 写未匹配 ID 清单（CSV 中没有被任何文件命中的那些 ID）
    unmatched_ids = sorted(ids_csv - seen_hit_ids, key=lambda x: (len(x), x))
    unmatched_csv_path = os.path.join(unmatched_dir, 'unmatched_ids.csv')
    pd.DataFrame({'distc_pro_num_norm': unmatched_ids}).to_csv(unmatched_csv_path, index=False)

    print(f'[RESULT] 复制到 matched: {copied_match} 个文件；复制到 unmatched: {copied_unmatch} 个文件')
    print(f'[DONE] 未匹配 ID 清单已写出: {unmatched_csv_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='根据 CSV 列表，将 pdb_output 中的 sequence_n.pt 分到 matched 和 unmatched 两个文件夹。'
    )
    parser.add_argument('--csv',      required=True, help='CSV 文件路径，包含 distc_pro_num 列')
    parser.add_argument('--pdb_dir',  default='pdb_output', help='存放 sequence_n.pt 的文件夹')
    parser.add_argument('--matched',  default='matched',    help='输出：匹配到的文件夹')
    parser.add_argument('--unmatched',default='unmatched',  help='输出：未匹配的文件夹')
    args = parser.parse_args()

    split_sequences(args.csv, args.pdb_dir, args.matched, args.unmatched)