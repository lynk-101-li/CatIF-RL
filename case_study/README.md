# `case_study/`

Worked examples used in the paper (Section 3.5). Three are global
redesigns and one is a motif-preserving inpainting case; together they
illustrate the unconstrained protocol and the discrete subgraph inpainting
sampler (Algorithm S1).

| Folder | EC | Enzyme | Organism | Substrate | Native *k*<sub>cat</sub> (s⁻¹) | Type |
|---|---|---|---|---|---|---|
| `EC1.4.1.20_Lsphaericus/` | 1.4.1.20 | leucine dehydrogenase variant | *Lysinibacillus sphaericus* | 4-Methylthio-2-oxobutanoate | 34.0 | global redesign |
| `EC2.4.2.1_Hsapiens/` | 2.4.2.1 | purine nucleoside phosphorylase | *Homo sapiens* | Inosine | 8.8 | global redesign |
| `EC5.3.1.1_Tbrucei/` | 5.3.1.1 | triose-phosphate isomerase | *Trypanosoma brucei* | Dihydroxyacetone phosphate | 63.0 | global redesign |
| `EC1.1.1.248_SalR/` | 1.1.1.248 | salutaridine reductase (SalR) | *Papaver bracteatum* | NADPH cofactor | 2.1 | motif-preserving inpaint |

Each folder contains:

- `native.pdb` — the ESMFold-predicted backbone used as the conditioning structure.
- `case.json` — substrate / SMILES / sequence / native *k*<sub>cat</sub> metadata.
- `motif_mask.json` — *(SalR only)* the four fixed catalytic positions.

## Reproduction

```bash
bash scripts/08_run_case_studies.sh
```

By default the script uses the final CatIF-RL Round-3 checkpoint and seed
12345. Outputs land under `runs/case_studies/<case_name>/`.

The SalR motif positions (Asn152, Ser180, Tyr236, Lys240) follow the
proton-transfer assignments of Geissler et al., *Plant Physiology* 143 (2007).
