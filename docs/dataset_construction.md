# Dataset construction

This document expands manuscript §2.1 (Dataset Construction and Leakage
Control) and SI Table S4 with concrete pipeline detail.

## Source-level filtering (SI Table S4(a))

1. Start from the DLKcat-BRENDA records (16,839 organism-substrate-sequence rows).
2. Remove sequences longer than 1,180 amino acids -- removes 76 unique entries.
3. Remove sequences containing non-standard residues -- removes a further 18.
4. De-duplicate identical protein sequences -- collapses to 7,713 distinct entries.

`catif_rl.data.brenda.filter_pipeline` implements steps 2-4 directly.

## ESMFold backbone prediction

The 7,713 distinct sequences are folded with ESMFold (Lin et al., 2023). Each
fold uses a fixed random seed of 1234 and produces one `.pdb` per sequence,
which becomes the conditioning graph for inverse folding.

`catif_rl.data.esmfold_backbones.fold_sequences` wraps the `esm` package.

## Train / validation / test split (SI Table S4(b))

- The 1,423 DLKcat-test sequences are locked aside *before* any model training.
- The remaining 6,290 enzymes are split 9:1 (`random.Random(1234)`) into
  5,661 train and 629 validation samples.
- 18,021 CATH v4.2.0 backbones are added to train, and 608 to validation, as
  general-protein structural regularizers.
- CATH backbones never enter the test split.

The orchestrator `catif_rl.data.splits.run_full_split` chains the underlying
scripts shipped under `catif_rl/data/_split_scripts/`.

## Graph construction

Each PDB is converted to a PyTorch-Geometric `Data` object by
`catif_rl.data.graph_construction`. Node features carry the one-hot residue
identity together with the physicochemical and geometric attributes used by
the upstream backbone (`mu_r_norm`, secondary-structure encoding, SASA).

## Leakage-control mechanics (SI Table S4(c))

- The DLKcat protein-cluster identifier (`distc_pro_num`) is propagated end
  to end, so distinct-sequence variants of any test cluster cannot enter
  train or valid.
- File-name uniqueness is enforced across the three split directories.
- CATH backbones are added only to train and valid -- never to test.
