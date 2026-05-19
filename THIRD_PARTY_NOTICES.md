# Third-Party Notices

The source code authored by the CatIF-RL contributors and shipped in this
repository is released under the MIT License (see [LICENSE](LICENSE)).

Several components imported, depended on, referenced, or invoked by
CatIF-RL are governed by other terms. This document enumerates them and
the obligations they imply for downstream users of this repository.

## 1. Upstream code repositories (cloned at install time)

### 1.1 GraDe-IF

- **Repository**: <https://github.com/ykiiiiii/GraDe_IF>
- **Reference**: Yi, K.; Zhou, B.; Shen, Y.; Lio, P.; Wang, Y. G.
  *Graph Denoising Diffusion for Inverse Protein Folding*. NeurIPS 2023.
- **Upstream license**: **NONE PROVIDED**.
- **How CatIF-RL uses it**: The graph-diffusion backbone
  (`diffusion.utils`, `diffusion.model.egnn_pytorch.*`) is cloned at
  install time into `external/GraDe_IF/` and loaded onto `sys.path` by
  `catif_rl/models/gradeif_adapter.py`. No upstream source is mirrored
  in this repository.
- **Implications**: The upstream repository does not ship an explicit
  license. This repository cites GraDe-IF in the accompanying manuscript
  and clones it at install time under academic citation conventions for
  non-commercial research use. Users wishing to use GraDe-IF commercially
  should contact the GraDe-IF authors directly.

### 1.2 UniKP

- **Repository**: <https://github.com/Luo-SynBioLab/UniKP>
- **Reference**: Yu, H.; Deng, H.; He, J.; Keasling, J. D.; Luo, X.
  *UniKP: A Unified Framework for the Prediction of Enzyme Kinetic
  Parameters*. *Nature Communications* 14, 8211 (2023).
- **Upstream license**: **NONE PROVIDED**.
- **How CatIF-RL uses it**: UniKP is cloned at install time into
  `external/UniKP/` and invoked only via subprocess by
  `catif_rl/reward/predictors/unikp.py`. No UniKP source is mirrored.
- **Implications**: Same posture as GraDe-IF -- academic citation
  conventions for non-commercial research use; users wishing to use UniKP
  commercially should contact the UniKP authors directly.

### 1.3 DLKcat

- **Repository**: <https://github.com/SysBioChalmers/DLKcat>
- **Reference**: Li, F.; Yuan, L.; Lu, H.; *et al.* *Deep Learning-Based
  Kcat Prediction Enables Improved Enzyme-Constrained Model
  Reconstruction*. *Nature Catalysis* 5, 662-672 (2022).
- **Upstream license**: **GNU General Public License v3**.
- **How CatIF-RL uses it**: DLKcat is cloned at install time into
  `external/DLKcat5/` and invoked only via subprocess by
  `catif_rl/reward/predictors/dlkcat.py`. The wrapper does not
  statically link to DLKcat source code, so the GPL "viral" provision does
  not propagate to this repository.
- **Implications**: Redistribution of the cloned `external/DLKcat5/`
  checkout retains DLKcat's GPL v3 terms. Modifying DLKcat source itself
  imposes GPL v3 obligations on the modified work.

### 1.4 CataPro

- **Repository**: <https://github.com/zchwang/CataPro>
- **Reference**: Wang, Z.; Xie, D.; Wu, D.; *et al.* *Robust Enzyme
  Discovery and Engineering with Deep Learning Using CataPro*.
  *Nature Communications* 16, 2736 (2025).
- **Upstream license**: **MIT** (compatible with CatIF-RL's MIT license).
- **How CatIF-RL uses it**: CataPro is cloned at install time into
  `external/CataPro-master/` and invoked only via subprocess by
  `catif_rl/reward/predictors/catapro.py`.
- **Implications**: No additional restrictions.

### 1.5 ESMFold

- **Source**: The `esm` package on PyPI
  (`pip install "fair-esm[esmfold]"`); upstream at
  <https://github.com/facebookresearch/esm>.
- **Reference**: Lin, Z.; *et al.* *Evolutionary-Scale Prediction of
  Atomic-Level Protein Structure with a Language Model*. *Science* 379,
  1123-1130 (2023).
- **Upstream license**: **MIT**.
- **How CatIF-RL uses it**: Subprocess invocation for backbone prediction
  during dataset construction and for refolding generated sequences during
  structural evaluation (manuscript Section 2.7).

## 2. Data sources

### 2.1 BRENDA / DLKcat-BRENDA enzyme kinetic dataset

- **Source**: BRENDA database, <https://www.brenda-enzymes.org/>;
  DLKcat-BRENDA derived release accompanying Li et al., 2022.
- **Reference**: Schomburg, A.; *et al.* *BRENDA, the ELIXIR Core Data
  Resource in 2021*. *Nucleic Acids Res.* 49, D498-D508 (2021).
- **License**: **CC BY 4.0 Non-Commercial**.
- **How CatIF-RL uses it**: BRENDA-derived sequences, substrate SMILES,
  and *k*<sub>cat</sub> values are the primary training and evaluation
  data (manuscript Section 2.1).
- **Implications**: This repository does **not** redistribute BRENDA
  records or the derived DLKcat splits. Users must obtain BRENDA data
  directly from the BRENDA website in accordance with BRENDA's terms.
  Processed graph datasets reconstructed by `scripts/01_build_dataset.sh`
  from BRENDA-sourced sequences inherit the **CC BY-NC 4.0 restriction**
  and may be used only for non-commercial research purposes.

### 2.2 CATH v4.2.0

- **Source**: <http://www.cathdb.info/>
- **Reference**: Sillitoe, I.; *et al.* *CATH: Increased Structural
  Coverage of Functional Space*. *Nucleic Acids Res.* 49, D266-D273
  (2021).
- **License**: **CC BY 4.0**.
- **How CatIF-RL uses it**: General protein backbones used as
  structural regularizers during EnzymeIF training (manuscript
  Section 2.2).
- **Implications**: Not redistributed in this repository; freely
  available from the CATH website.

## 3. Summary

| Component | Kind | Upstream license | Effect on CatIF-RL repository |
|---|---|---|---|
| GraDe-IF | code | none | Cloned at install time; academic citation conventions for non-commercial research use |
| UniKP | code | none | Subprocess-only; no source mirrored |
| DLKcat | code | GPL v3 | Subprocess wrapper exempt from static-link clause |
| CataPro | code | MIT | Fully compatible |
| ESMFold | code | MIT | Fully compatible |
| BRENDA | data | CC BY-NC 4.0 | Not redistributed; derived datasets inherit the non-commercial restriction |
| CATH | data | CC BY 4.0 | Not redistributed |

## 4. Citation

Please cite the upstream references above when using the corresponding
components, in addition to the CatIF-RL manuscript itself.
