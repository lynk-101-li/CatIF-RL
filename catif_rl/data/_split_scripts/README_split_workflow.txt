# 1. test_dataset.csv
#    Sequences for the test dataset taken from DLKcat.

# 2. The GraDe-IF test set must come entirely from sequences in test_set.csv;
#    the train and validation sets can draw from dev_set.csv and train_set.csv
#    without restriction.

#
# 3. Using ID matching, extract from brenda_seq_pdb (graphs) the .pt graph
#    files that correspond to enzymes in DLKcat's test_set:
#    test_graph_dataset_for_GDIF

## test_split_enzyme_graph_final
# 4. Carve out a portion of test_graph_dataset_for_GDIF to use as the final
#    test split for GraDe-IF training.

# 5. Mix the remaining proteins with those in train_and_valid_graph_dataset_for_GDIF,
#    then split by ratio into the GraDe-IF train / valid sets. Each is then
#    blended with the proportionally allocated generic-protein pools
#    train_split_universal_graph / valid_split_universal_graph, producing
#    train_split_enzyme_graph_final and valid_split_enzyme_graph_final.

# 6. train_split_enzyme_graph_final, valid_split_enzyme_graph_final,
#    test_split_enzyme_graph_final are written into dataset/process/{train,valid,test}.

####    OVERVIEW    ####
split_baseon_sequences.py --> run_split.py(split_baseon_sequences.py) --> dataset_split_final.py
          align_pdb_names.py _^
