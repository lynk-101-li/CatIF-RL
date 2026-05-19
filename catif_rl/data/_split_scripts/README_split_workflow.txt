# 1. test_dataset.csv
     |seq of test dataset from DLKcat。

# 2. gradeif的测试集需要全部来自test_set.csv里的seq；而训练和验证集则无所谓，可以来自dev_set.csv与train_set.csv；

## 
# 3. 通过ID比对，从brenda_seq_pdb(graph)提取出DLKcat的test_set中酶对应的graph文件：test_graph_dataset_for_GDIF

## test_split_enzyme_graph_final
# 4. 从test_graph_dataset_for_GDIF取部分拿到最终用于gradeif训练中测试集的部分

# 5. 剩余的蛋白与train_and_valid_graph_dataset_for_GDIF中的蛋白混合，
# 按比例分为训练gradeif用的train set与valid set，再分别与比例分配的通用蛋白train_split_universal_graph与valid_split_universal_graph混合，得到train_split_enzyme_graph_final和valid_split_enzyme_graph_final。

# 6. train_split_enzyme_graph_final, valid_split_enzyme_graph_final, test_split_enzyme_graph_final分别被输入到dataset/process/，train，valid，test

####    OVERVIEW    ####
split_baseon_sequences.py --> run_split.py(split_baseon_sequences.py) --> dataset_split_final.py
          align_pdb_names.py _^