#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dataset_split_final.py

功能：
1) 按 9:1（seed=1234）划分 enzyme_train_and_valid_dataset -> train/valid（酶样本）
2) 将 enzyme_test_dataset 原样复制到 dataset/process/test
3) 将 train_split_universal_graph 的普通蛋白 .pt 并入 train
4) 将 valid_split_universal_graph 的普通蛋白 .pt 并入 validation
5) 去重并保证集合间无重叠（以文件名为唯一键），重名将跳过并提示
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
        print(f"[WARN] 目录不存在：{folder}")
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix == ".pt"])

def safe_copy(src: Path, dst_dir: Path, taken_names: Set[str]) -> bool:
    """按文件名去重，若重复则跳过；成功复制返回 True。"""
    name = src.name
    if name in taken_names:
        print(f"[SKIP] 重名文件（已在其它集合出现）：{name}")
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
    parser.add_argument("--base_in", default="dataset_src/data_split_for_gradeif_training", help="输入基目录（包含 enzyme_* 与 *_universal_graph）")
    parser.add_argument("--out_base", default="dataset/process", help="输出基目录（train/validation/test）")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="酶样本训练集比例（默认 0.9）")
    parser.add_argument("--seed", type=int, default=SEED, help="随机种子（默认 1234）")
    args = parser.parse_args()

    base_in = Path(args.base_in)
    out_base = Path(args.out_base)

    # 输入目录
    d_test_enzyme = base_in / "enzyme_test_dataset"
    d_trainvalid_enzyme = base_in / "enzyme_train_and_valid_dataset"
    d_univ_train = base_in / "train_split_universal_graph"
    d_univ_valid = base_in / "valid_split_universal_graph"

    # 输出目录
    d_out_train = out_base / "train"
    d_out_valid = out_base / "validation"
    d_out_test  = out_base / "test"

    print("=== 路径检查 ===")
    print(f"输入(测试酶)：{d_test_enzyme}")
    print(f"输入(训练+验证酶)：{d_trainvalid_enzyme}")
    print(f"输入(通用图-训练)：{d_univ_train}")
    print(f"输入(通用图-验证)：{d_univ_valid}")
    print(f"输出(train)：{d_out_train}")
    print(f"输出(validation)：{d_out_valid}")
    print(f"输出(test)：{d_out_test}")

    # 收集文件
    test_files = list_pt_files(d_test_enzyme)
    tv_files = list_pt_files(d_trainvalid_enzyme)
    univ_train_files = list_pt_files(d_univ_train)
    univ_valid_files = list_pt_files(d_univ_valid)

    print("\n=== 统计输入 ===")
    print(f"enzyme_test_dataset: {len(test_files)} 个 .pt")
    print(f"enzyme_train_and_valid_dataset: {len(tv_files)} 个 .pt")
    print(f"train_split_universal_graph: {len(univ_train_files)} 个 .pt")
    print(f"valid_split_universal_graph: {len(univ_valid_files)} 个 .pt")

    # 1) 划分训练/验证（仅酶样本）
    if len(tv_files) == 0:
        print("[ERROR] 训练+验证酶样本目录为空。")
        return

    train_idx, valid_idx = split_indices(len(tv_files), args.train_ratio, args.seed)
    train_enzyme = [tv_files[i] for i in train_idx]
    valid_enzyme = [tv_files[i] for i in valid_idx]

    print("\n=== 划分结果（酶样本） ===")
    print(f"Train (enzyme): {len(train_enzyme)}")
    print(f"Valid (enzyme): {len(valid_enzyme)}")
    # 断言：6290 -> 5661 / 629
    # 但若实际数量不同，也允许按比例划分
    # 若需要严格校验，可在此添加断言：
    # assert len(tv_files) == 6290 and len(train_enzyme) == 5661 and len(valid_enzyme) == 629

    # 2) 拷贝测试集（酶样本）到 test/
    print("\n=== 复制测试集（酶样本保持原样） ===")
    taken_test = set()      # test 文件名集合
    copied_test = 0
    for f in test_files:
        if safe_copy(f, d_out_test, taken_test):
            copied_test += 1
    print(f"[DONE] test：复制 {copied_test} 个文件。")

    # 3) 复制训练/验证集合（先只放酶样本）
    print("\n=== 复制训练/验证（酶样本） ===")
    taken_train = set()     # train 文件名集合
    taken_valid = set()     # valid 文件名集合

    copied_train_enzyme = 0
    for f in train_enzyme:
        if safe_copy(f, d_out_train, taken_train):
            copied_train_enzyme += 1

    copied_valid_enzyme = 0
    for f in valid_enzyme:
        # 确保不与 train/test 重叠
        if f.name in taken_train or f.name in taken_test:
            print(f"[SKIP] 验证集与训练/测试重名：{f.name}")
            continue
        if safe_copy(f, d_out_valid, taken_valid):
            copied_valid_enzyme += 1

    print(f"[DONE] train（enzyme）：{copied_train_enzyme}")
    print(f"[DONE] validation（enzyme）：{copied_valid_enzyme}")

    # 4) 并入通用图（普通蛋白 .pt）
    print("\n=== 并入通用图：train_split_universal_graph -> train ===")
    copied_univ_train = 0
    for f in univ_train_files:
        # 防止与 test/valid/train 重名
        if f.name in taken_test or f.name in taken_valid or f.name in taken_train:
            print(f"[SKIP] 通用图(train)与现有集合重名：{f.name}")
            continue
        if safe_copy(f, d_out_train, taken_train):
            copied_univ_train += 1
    print(f"[DONE] train（+universal）：新增 {copied_univ_train} 个普通蛋白")

    print("\n=== 并入通用图：valid_split_universal_graph -> validation ===")
    copied_univ_valid = 0
    for f in univ_valid_files:
        if f.name in taken_test or f.name in taken_train or f.name in taken_valid:
            print(f"[SKIP] 通用图(valid)与现有集合重名：{f.name}")
            continue
        if safe_copy(f, d_out_valid, taken_valid):
            copied_univ_valid += 1
    print(f"[DONE] validation（+universal）：新增 {copied_univ_valid} 个普通蛋白")

    # 5) 汇总
    n_train_total = len(list_pt_files(d_out_train))
    n_valid_total = len(list_pt_files(d_out_valid))
    n_test_total  = len(list_pt_files(d_out_test))

    print("\n=== 最终汇总 ===")
    print(f"Train 总数：{n_train_total}（其中酶样本 {copied_train_enzyme}，普通蛋白 {copied_univ_train}）")
    print(f"Validation 总数：{n_valid_total}（其中酶样本 {copied_valid_enzyme}，普通蛋白 {copied_univ_valid}）")
    print(f"Test 总数：{n_test_total}（全部为酶样本）")
    print("\n完成。")

if __name__ == "__main__":
    main()
