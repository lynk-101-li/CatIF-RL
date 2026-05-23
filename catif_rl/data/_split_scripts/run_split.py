"""
run_split.py

1. Deduplicate distc_pro_num.
2. Call the helpers in split_baseon_sequences.py to split the
   sequence_n.pt files in process/brenda_seq_pdb by their ID number.

Before running, cd to autodl-tmp/gradeifprotein/GraDe_IF-main, then:
   python dataset_src/data_split_for_gradeif_training/run_split.py
"""

import os
import pandas as pd
from split_baseon_sequences import split_sequences

# ---- User configuration ---- #
# CSV file list (each file must contain a distc_pro_num column)
csv_list = ['dataset_src/data_split_for_gradeif_training/brenda_dataset_split/test_set.csv']

# Folder holding the sequence_n.pt files
pdb_dir  = 'dataset/process/brenda_seq_pdb'

# Output folders
matched_dir   = 'dataset_src/data_split_for_gradeif_training/enzyme_test_dataset'
unmatched_dir = 'dataset_src/data_split_for_gradeif_training/enzyme_train_and_valid_dataset'
# ---- End of configuration ---- #

def main():
    # 1. Read and concatenate the CSV(s)
    dfs = []
    for p in csv_list:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"CSV file not found: {p}")
        dfs.append(pd.read_csv(p))
    df_merged = pd.concat(dfs, ignore_index=True)
    # Deduplicate
    df_merged = df_merged.drop_duplicates(subset='distc_pro_num')
    # Write the temporary merged file
    merged_csv = 'merged.csv'
    df_merged.to_csv(merged_csv, index=False)
    print(f'merged file written: {merged_csv} ({len(df_merged)} unique records)')

    # 2. Make sure the output directories exist
    os.makedirs(matched_dir, exist_ok=True)
    os.makedirs(unmatched_dir, exist_ok=True)

    # 3. Call the splitter
    split_sequences(
        csv_path=merged_csv,
        pdb_dir=pdb_dir,
        matched_dir=matched_dir,
        unmatched_dir=unmatched_dir
    )

    # 4. Deduplicate the unmatched_dir, keeping only unique filenames
    seen = set()
    for fname in os.listdir(unmatched_dir):
        if fname in seen:
            os.remove(os.path.join(unmatched_dir, fname))
        else:
            seen.add(fname)

    print('All done.')

if __name__ == '__main__':
    main()
