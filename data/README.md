# `data/`

The CatIF-RL training and evaluation data live outside the repository --
per-protein graph tensors and ESMFold-predicted backbones are too large to
ship in git. They are archived separately and downloaded into this folder.

## Expected layout

```
data/
├── process/
│   ├── train/             # 23,682 *.pt graph tensors (5,661 enzymes + 18,021 CATH regularizers)
│   ├── valid/             # 1,237 *.pt (629 enzymes + 608 CATH)
│   └── test/              # 1,423 *.pt (held-out DLKcat benchmark)
├── raw/                   # ESMFold-predicted *.pdb backbones (one per distinct sequence;
│                          # only needed if you plan to regenerate the .pt graphs from
│                          # scratch -- see catif_rl/data/graph_construction.py)
├── brenda/                # DLKcat-BRENDA exports (records CSV + train/dev/test sets)
├── enzyme_train_and_valid_dataset/   # Per-enzyme conditioning graphs used during GRPO
└── reward/                            # Offline reward CSVs per round
    ├── round1_reward_data.csv
    ├── round2_reward_data.csv
    └── round3_reward_data.csv
```

## How to obtain it

A Zenodo archive will be linked here once the deposit is finalised. Until
then, the dataset can be reconstructed from public sources:

- **DLKcat-BRENDA records** — Li et al., 2022 [12]; obtainable from the
  DLKcat repository (`SysBioChalmers/DLKcat`).
- **CATH v4.2.0 backbones** — Sillitoe et al., *Nucleic Acids Res.* 49 (2021).
  Available at <http://www.cathdb.info/>.
- **ESMFold predicted structures** — generated locally with the `esm`
  Python package (see `external/README.md`).

Once the raw inputs are downloaded, recreate the processed `.pt` tensors with:

```bash
bash scripts/01_build_dataset.sh
```

## Underlying public sources cited in the manuscript

- UniProt Knowledgebase -- enzyme sequences and EC annotations.
- BRENDA -- per-enzyme catalytic constant (*k*<sub>cat</sub>) measurements.
- PDB / CATH v4.2.0 -- general protein backbones used as structural regularizers.

Detailed dataset construction is described in the manuscript's Supporting
Information Section S1 (extended methods) and Table S4 (leakage control).
