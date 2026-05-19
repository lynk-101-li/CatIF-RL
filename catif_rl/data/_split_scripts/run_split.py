"""
run_split.py

1. 去重 distc_pro_num
2. 调用 split_baseon_sequences.py中的函数
   将 process/brenda_seq_pdb 中的 sequence_n.pt 按编号拆分
使用前终端进入目录autodl-tmp/gradeifprotein/GraDe_IF-main
输入
python dataset_src/data_split_for_gradeif_training/run_split.py
"""

import os
import pandas as pd
from split_baseon_sequences import split_sequences

# —— 用户配置区 —— #
# CSV 列表文件（保证都有 distc_pro_num 这一列）
csv_list = ['dataset_src/data_split_for_gradeif_training/brenda_dataset_split/test_set.csv']

# 存放 sequence_n.pt 的文件夹
pdb_dir  = 'dataset/process/brenda_seq_pdb'

# 输出文件夹
matched_dir   = 'dataset_src/data_split_for_gradeif_training/enzyme_test_dataset'
unmatched_dir = 'dataset_src/data_split_for_gradeif_training/enzyme_train_and_valid_dataset'
# —— 配置区结束 —— #

def main():
    # 1. 读取并合并 CSV
    dfs = []
    for p in csv_list:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"找不到 CSV 文件：{p}")
        dfs.append(pd.read_csv(p))
    df_merged = pd.concat(dfs, ignore_index=True)
    # 去重
    df_merged = df_merged.drop_duplicates(subset='distc_pro_num')
    # 写入临时合并文件
    merged_csv = 'merged.csv'
    df_merged.to_csv(merged_csv, index=False)
    print(f'已生成合并文件：{merged_csv} （共 {len(df_merged)} 条唯一记录）')

    # 2. 确保输出目录存在
    os.makedirs(matched_dir, exist_ok=True)
    os.makedirs(unmatched_dir, exist_ok=True)

    # 3. 调用拆分函数
    split_sequences(
        csv_path=merged_csv,
        pdb_dir=pdb_dir,
        matched_dir=matched_dir,
        unmatched_dir=unmatched_dir
    )

    # 4. 去重 unmatched_dir 中的文件，只保留唯一文件名
    seen = set()
    for fname in os.listdir(unmatched_dir):
        if fname in seen:
            os.remove(os.path.join(unmatched_dir, fname))
        else:
            seen.add(fname)

    print('全部完成！')

if __name__ == '__main__':
    main()
