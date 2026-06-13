# Reproducing the paper

The pipeline is implemented as numbered shell entry points under `scripts/`.
Run them in order; intermediate artefacts land under `runs/` so a partial
re-run only repeats the affected stage.

## End-to-end sequence

| # | Script | Manuscript ref | Outputs | Hardware footprint |
|---|--------|----------------|---------|--------------------|
| 1 | `scripts/01_build_dataset.sh` | §2.1 / SI Table S4 | `data/process/{train,valid,test}/*.pt` | CPU + ESMFold GPU (24 GB) |
| 2 | `scripts/02_train_enzymeif.sh` | §2.2 / SI Table S1 | `runs/enzymeif/best_epoch467.pt` | 1 x RTX 4090 (24 GB), days |
| 3 | `scripts/03_run_gdc.sh` | §2.3 / SI Table S5 | `runs/gdc/activity_positive.csv` (~6,034 variants) | Sampling + predictor wall-clock |
| 4 | `scripts/04_train_catif.sh` | §2.4 / SI Table S2 | `runs/catif/best_epoch228.pt` | 1 x RTX 4060 Ti (16 GB), days |
| 5 | `scripts/05a_rl_round1.sh` -> `05b_rl_round2.sh` -> `05c_rl_round3.sh` | §2.5 / SI Table S3 + Alg S2 | `runs/grpo_round{1,2,3}/policy_epoch02.pt` | 1 x RTX 4060 Ti (16 GB), hours per round |
| 6 | `scripts/06_sample_benchmark.sh` | §2.7 / SI Table S6 | `runs/benchmark/<method>/seed_<s>/*.fasta` (11 methods x 5 seeds) | Mostly inference |
| 7 | `scripts/07_score_benchmark.sh` | §2.7 / SI Tables S7-S10 | `runs/benchmark_scores/{master_per_protein.csv, tables/*}` | Subprocess to 3 predictor envs + ESMFold refold of one seed |
| 8 | `scripts/08_run_case_studies.sh` | §3.5 / SI Tables S11-S12 | `runs/case_studies/<EC>_<organism>/` | Lightweight (4 single-seed runs) |

## Stage-by-stage notes

### 1. Build the dataset

The author-original enzyme PDBs (`data/raw/enzymeif/...`,
`data/raw/catif/...`, `data/raw/test/...`) are downloaded by the user
from the Zenodo deposit beforehand (see `data/README.md`). With those
in place, `01_build_dataset.sh` runs three steps:

```
(1) catif_rl.data.download_pdb     -> CATH v4.2.0 chains pulled from RCSB
                                      into data/raw/enzymeif/cath_v4_2_0/
                                      (18,021 train + 608 valid = 18,629)
(2) catif_rl.data.graph_construction (per cohort)
                                   -> .pdb -> PyG .pt for each of:
                                      data/process/enzymeif/train_and_validation/   (6,290)
                                      data/process/enzymeif/cath_v4_2_0/            (18,629)
                                      data/process/catif/{train,valid}/             (5,430 + 604)
                                      data/process/test/                            (1,423)
(3) catif_rl.data.splits           -> 9:1 enzyme split + CATH merge:
                                      data/process/train/  5,661 + 18,021 = 23,682
                                      data/process/valid/    629 +    608 =  1,237
                                      (asserts SI Table S4 counts; non-match is fatal)
```

The DLKcat-test sequences are held out from training before any model is
fit; CATH regularizers never enter the test split.

### 2-4. Supervised stages

Both `02_` and `04_` invoke `catif_rl.training.train_supervised` -- the same
underlying code path -- with different YAML configs. EnzymeIF (`enzymeif.yaml`)
trains on the enzyme + CATH mixture; CatIF (`catif.yaml`) trains from scratch
on the 6,034 activity-positive variants emitted by `03_run_gdc.sh`. Each
stage uses Adam (lr 5e-4, wd 1e-5, dropout 0.1, batch 32, EMA decay 0.995)
with cross-entropy on the predicted clean amino-acid distribution.

### 5. CatIF-RL GRPO refinement (3 rounds)

Each round runs the full sample -> score -> train cycle:

1. Sample `G = 5` candidates per (enzyme, substrate) using the previous
   round's checkpoint, with DDIM step = 100 and diverse decoding.
2. Score each candidate with DLKcat / UniKP / CataPro, normalize via the
   frozen 10th/90th quantile range from GDC, and mean to obtain
   `S_ensemble`.
3. Run the inner GRPO update for 2 operational epochs (training-run length
   flag = 50 for Round 1, 25 for Rounds 2 and 3 per SI Table S3).

The reference policy is the frozen CatIF checkpoint throughout all three
rounds; the KL target stays at 0.01 with adaptive beta in
`[5e-4, 0.5]`.

### 6-7. Benchmark on the 1,423-enzyme held-out test set

`06_sample_benchmark.sh` covers all eleven methods (GraDe-IF, EnzymeIF, CatIF,
CatIF-RL R1 / R2 / R3, ProteinMPNN, ESM-IF, LigandMPNN, PiFold, ABACUS-T) with
five seeds each: `{1111, 2222, 3333, 4444, 5555}`.

`07_score_benchmark.sh` then computes the four primary metrics:

- Δlog10 *k*<sub>cat</sub> (paired mutant-native DLKcat predictions averaged
  per protein)
- Recovery rate
- pLDDT (single-seed structural-evaluation subset)
- Backbone RMSD (same subset)

and the joint success rate at six δ thresholds (SI Table S9).

### 8. Case studies

Four representative test enzymes from §3.5:

- Three global redesigns (EC 1.4.1.20, EC 2.4.2.1, EC 5.3.1.1).
- One motif-preserving SalR redesign (EC 1.1.1.248) using
  `catif_rl.sampling.inpaint` with the four catalytic residues fixed
  (Asn152, Ser180, Tyr236, Lys240).

All four cases use a fixed seed of 12345 for direct reproducibility of the
numbers reported in SI Table S11.

## Wall-clock estimates

The reported pipeline took approximately:

- Dataset build: ~12 hours wall (mostly ESMFold).
- EnzymeIF pretraining: ~3-5 days on an RTX 4090 24 GB.
- GDC sampling + ensemble scoring: ~12 hours on the local 16 GB card.
- CatIF supervised training: ~2 days.
- Three RL rounds: ~6-10 hours each.
- Full benchmark sampling + scoring: ~1 day.
- Case studies: ~15 minutes.

Numbers will vary with GPU model and disk throughput; an A100 cuts the
supervised stages substantially.
