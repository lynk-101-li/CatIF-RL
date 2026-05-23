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
│
├── raw/                   # ESMFold-predicted *.pdb backbones
│   ├── brenda_seq_pdb/    # 7,713 PDB files (one per distinct enzyme sequence)
│   └── test/              # 1,423 PDB files (held-out subset, also in brenda_seq_pdb/)
│
├── gdc/                   # GDC sequence selections (manuscript §2.3)
│   └── gdc_variants_6034.csv     # 6,034 activity-positive variants; columns:
│                                 #   Group, ProID, ProSeq' (GDC-selected sequence), mean3
│                                 # CatIF training reads the EnzymeIF backbones from
│                                 # data/process/train/ and overrides the native
│                                 # sequence labels with ProSeq' for each ProID found
│                                 # in this CSV.
│
└── reward/                # Offline RL inner-loop reward signals (manuscript §2.5)
    ├── round1_reward_data.csv    # CatIF -> Round-1 policy update
    ├── round2_reward_data.csv    # Round 1 -> Round-2 update
    └── round3_reward_data.csv    # Round 2 -> Round-3 update (final CatIF-RL)
```

The `process/` graph tensors are the canonical structural backbones used by
**all three** training stages: EnzymeIF trains against the native sequence
label baked into each `.pt`; CatIF trains against the GDC-selected label
substituted in from `gdc/gdc_variants_6034.csv` at load time; CatIF-RL
samples from `process/train/` and `process/valid/` as conditioning during
GRPO outer-loop generation.

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
