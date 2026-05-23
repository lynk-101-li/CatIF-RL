# `data/`

The CatIF-RL training and evaluation data live outside the repository --
per-protein graph tensors and ESMFold-predicted backbones are too large to
ship in git. They are archived separately and downloaded into this folder.

## Expected layout

Raw PDB structures only. Processed `.pt` graph tensors are not shipped —
the full processed set is ~53 GB and exceeds Zenodo's 50 GB per-record
limit, and the `.pt` format is also tightly coupled to specific PyTorch /
PyG versions, which makes it brittle for long-term archival. Regenerate
the `.pt` graphs in ~1-2 hours with:

```bash
bash scripts/01_build_dataset.sh
```

This also doubles as an end-to-end verification of the graph-construction
pipeline.

```
data/
├── raw/
│   ├── enzymeif/
│   │   ├── train_and_validation/  # 6,290 sequence_<N>.pdb
│   │   │                          # Native enzyme structures for EnzymeIF training.
│   │   │                          # Train+valid are NOT pre-split here — the
│   │   │                          # 5,661 / 629 split happens during graph
│   │   │                          # construction with random.Random(1234)
│   │   │                          # (manuscript §2.1, SI Table S4(b)).
│   │   └── cath_v4_2_0/           # 18,629 <pdbid>.<chain>.pdb
│   │                              # CATH v4.2.0 structural regularizers
│   │                              # (manuscript §2.2). 18,021 enter train,
│   │                              # 608 enter valid. Sourced from cathdb.info
│   │                              # and the RCSB PDB.
│   ├── catif/
│   │   ├── train/                 # 5,430 sequence_<N>_group<M>.pdb
│   │   └── valid/                 # 604 sequence_<N>_group<M>.pdb
│   │                              # 5,430 + 604 = 6,034 activity-positive GDC
│   │                              # variants (manuscript §2.3); _group<M>
│   │                              # identifies the GDC sample group. Pre-split
│   │                              # train / valid by the GDC pipeline.
│   └── test/                      # 1,423 sequence_<N>.pdb
│                                  # Shared held-out DLKcat benchmark
│                                  # (manuscript §2.7).
│
├── gdc/
│   └── gdc_variants_6034.csv      # 6,034 rows; columns: Group, ProID, ProSeq', mean3.
│                                  # Each row corresponds to one PDB under data/raw/catif/.
│
└── reward/                        # Offline RL inner-loop reward signals (manuscript §2.5)
    ├── round1_reward_data.csv     # CatIF       -> CatIF-RL Round-1
    ├── round2_reward_data.csv     # CatIF-RL R1 -> CatIF-RL Round-2
    └── round3_reward_data.csv     # CatIF-RL R2 -> CatIF-RL Round-3 (final)
```

**Three separate training cohorts.** EnzymeIF and CatIF do **not** share
their training data:

- EnzymeIF trains against native enzyme structures (`data/raw/enzymeif/train_and_validation/`) plus CATH structural regularizers (`data/raw/enzymeif/cath_v4_2_0/`).
- CatIF trains from scratch against ESMFold-refolded structures of the 6,034 GDC-curated mutants (`data/raw/catif/`).
- CatIF-RL initialises from CatIF and uses the same `data/raw/catif/` (or its derived `.pt` graphs) as conditioning during GRPO outer-loop sampling.

Only the held-out test set (`data/raw/test/`) is shared across all three stages.

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

## License and attribution

The training data are derived from BRENDA
(<https://www.brenda-enzymes.org/>; Schomburg et al., *Nucleic Acids Res.*
49, D498-D508, 2021), released under **CC BY 4.0 Non-Commercial**.

This repository does **not** redistribute BRENDA records or the derived
DLKcat splits. Users must obtain BRENDA records directly from the BRENDA
website in accordance with BRENDA's terms of use. Processed graph datasets
reconstructed by `scripts/01_build_dataset.sh` inherit the CC BY-NC 4.0
restriction and may be used only for non-commercial research purposes.

CATH v4.2.0 (Sillitoe et al., *Nucleic Acids Res.* 49, D266-D273, 2021)
is distributed under CC BY 4.0 and is freely usable for both commercial and
non-commercial purposes; this repository likewise does not redistribute the
CATH archive, only references it.

See [`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) for the full
per-component license summary.
