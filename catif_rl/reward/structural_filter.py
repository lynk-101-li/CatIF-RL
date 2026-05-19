'''
本脚本用作筛选计算rmsd与plddt两个指标后的新生成序列（pdb）的脚本
过滤标准为(df["CA_RMSD"] > 4) & (df["Backbone_RMSD"] > 4) | (df["Avg_pLDDT"] < 90)
'''


import pandas as pd
import os

# ==== 1. 路径 ====
INPUT_CSV = "evaluation/pred_output_rmsd_plddt_table/output_enzymeif_train_valid_mut_merged.csv"        # 原始 4 列 csv
GOOD_CSV  = "KcatPred/ideal_rmsf_plddt_enzymeif_merged.csv"      # 输出：达标
BAD_CSV   = "KcatPred/unqualified_rmsf_plddt_enzymeif_merged.csv"       # 输出：不达标

# ==== 2. 读取 ====
df = pd.read_csv(INPUT_CSV)

# 确保列名与你文件一致。如果不是这几个名字，请改成实际列名
# 例如：df.columns = ["Filename", "CA_RMSD", "Backbone_RMSD", "Avg_pLDDT"]

# ==== 3. 构造布尔条件 ====
bad_mask = (df["CA_RMSD"] > 4) & (df["Backbone_RMSD"] > 4) | (df["Avg_pLDDT"] < 90)
good_mask = ~bad_mask

# ==== 4. 切分并保存 ====
df_good = df[good_mask].reset_index(drop=True)
df_bad  = df[bad_mask].reset_index(drop=True)

df_good.to_csv(GOOD_CSV, index=False)
df_bad.to_csv(BAD_CSV,  index=False)

print(f"✅ Done!  达标样本: {len(df_good)}  |  不达标样本: {len(df_bad)}")
print(f"• {GOOD_CSV}  |  {BAD_CSV}")
