"""
Filter newly generated sequences (PDB) using the RMSD and pLDDT metrics
computed by the structural evaluator.

Rejection rule:
    (CA_RMSD > 4) & (Backbone_RMSD > 4) | (Avg_pLDDT < 90)
"""


import pandas as pd
import os

# ==== 1. Paths ====
INPUT_CSV = "evaluation/pred_output_rmsd_plddt_table/output_enzymeif_train_valid_mut_merged.csv"        # raw 4-column CSV
GOOD_CSV  = "KcatPred/ideal_rmsf_plddt_enzymeif_merged.csv"      # output: kept
BAD_CSV   = "KcatPred/unqualified_rmsf_plddt_enzymeif_merged.csv"       # output: rejected

# ==== 2. Read ====
df = pd.read_csv(INPUT_CSV)

# Make sure column names match your file. If they differ, rename here:
# e.g. df.columns = ["Filename", "CA_RMSD", "Backbone_RMSD", "Avg_pLDDT"]

# ==== 3. Build boolean masks ====
bad_mask = (df["CA_RMSD"] > 4) & (df["Backbone_RMSD"] > 4) | (df["Avg_pLDDT"] < 90)
good_mask = ~bad_mask

# ==== 4. Split and persist ====
df_good = df[good_mask].reset_index(drop=True)
df_bad  = df[bad_mask].reset_index(drop=True)

df_good.to_csv(GOOD_CSV, index=False)
df_bad.to_csv(BAD_CSV,  index=False)

print(f"[OK] Done!  kept: {len(df_good)}  |  rejected: {len(df_bad)}")
print(f"  {GOOD_CSV}  |  {BAD_CSV}")
