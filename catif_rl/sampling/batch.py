#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample raw RL candidates (no scoring) and export to CSV: epochX_raw.csv.

- Supports multiple condition-dirs (train / validation).
- Reuses the EMA + DDIM sampling flow from infrs_preds.py (sequences only).
- Groups by {ProID, SMILES} as group_id (option A).
- Draws K samples per condition (--group_size).
- Output columns: data_index, cond_name, group, ProID, SMILES, wt_seq, seq,
  sample_idx, seed, step, ckpt_path

Example:
python -m sampling.sample_rl \
  --condition_dirs dataset_src/data_split_for_gradeif_training/enzyme_train_and_valid_dataset \
  --pairs_csv KcatPred/brenda_train_and_dev_set.csv \
  --ckpt_path diffusion/results/weight_rl/Nov19_epoch1/policy_epoch03.pt \
  --epoch 1 \
  --group_size 5 \
  --step 100 \
  --out_csv sampling/epoch2_raw_Nov19.csv \
  --device cuda:0 \
  --seed 11 \
  --diverse

Notes:
- pairs_csv must contain at least ``ProID`` and ``SMILES``. If a ``cond_name``
  column is present, it is used directly; otherwise the script falls back to
  matching ``{ProID}.pt`` condition files.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import argparse
from pathlib import Path
import random
import numpy as np
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from ema_pytorch import EMA
from catif_rl.models.gradeif_app import EGNN_NET, GraDe_IF
from catif_rl.data.large_dataset import Cath

AMINO_CODES = ['A','R','N','D','C','Q','E','G','H','I',
               'L','K','M','F','P','S','T','W','Y','V']
AA_TO_IDX = {aa:i for i,aa in enumerate(AMINO_CODES)}

def set_seed(seed: int):
    if seed is None or seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def data_to_sequence(data: Data) -> str:
    """Recover the WT sequence from the first 20 one-hot dims of node features
    (node order == sequence order)."""
    idx = data.x[:, :20].argmax(dim=-1).tolist()
    return ''.join(AMINO_CODES[i] for i in idx)

def sequence_recovery(wt_seq: str, mut_seq: str) -> float:
    """
    Sequence recovery: position-wise identity between the generated sequence
    and the WT on the overlapping prefix.
    If lengths differ, only the min(len_wt, len_mut) prefix is compared.
    """
    L = min(len(wt_seq), len(mut_seq))
    if L == 0:
        return 0.0
    match = sum(1 for a, b in zip(wt_seq[:L], mut_seq[:L]) if a == b)
    return match / L

def sequence_perplexity(seq: str) -> float:
    """
    Entropy-based perplexity from empirical amino-acid frequencies:
      H = -sum_a p(a) log p(a)
      ppl = exp(H)
    Note: this measures amino-acid diversity, not language-model log-prob
    perplexity.
    """
    if not seq:
        return float('nan')
    counts = {}
    for aa in seq:
        counts[aa] = counts.get(aa, 0) + 1
    total = len(seq)
    ps = np.array([c / total for c in counts.values()], dtype=np.float64)
    H = -(ps * np.log(ps)).sum()
    return float(np.exp(H))

def build_model_from_config(config: dict, sample: Data):
    base_model = EGNN_NET(
        input_feat_dim = config['input_feat_dim'],
        hidden_channels= config['hidden_dim'],
        edge_attr_dim  = config['edge_attr_dim'],
        dropout        = config['drop_out'],
        n_layers       = config['depth'],
        update_edge    = config.get('update_edge'),
        embedding      = config.get('embedding'),
        embedding_dim  = config.get('embedding_dim'),
        norm_feat      = config.get('norm_feat'),
        embed_ss       = config.get('embed_ss'),
    )
    diffusion = GraDe_IF(
        model     = base_model,
        timesteps = config['timesteps'],
        objective = config.get('objective', 'pred_x0'),
        config    = config
    )
    return diffusion

def load_pairs_csv(pairs_csv: Path):
    """
    Load the {ProID, SMILES[, cond_name]} mapping.

    Returns:
      - by_proid: {ProID -> {'SMILES': ..., 'cond_name': optional}}
      - rows:     raw rows (kept for future extensions)
    """
    by_proid = {}
    rows = []
    with pairs_csv.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        needed = {'ProID','SMILES'}
        if not needed.issubset(set(reader.fieldnames or [])):
            raise RuntimeError(f"{pairs_csv} must contain columns: {needed}")
        for r in reader:
            pro = r['ProID'].strip()
            smi = r['SMILES'].strip()
            cn  = r.get('cond_name', '').strip() if 'cond_name' in r else ''
            if pro:
                by_proid[pro] = {'SMILES': smi, 'cond_name': cn}
                rows.append(r)
    return by_proid, rows

def guess_cond_name_for_proid(proid: str, cond_files: set[str]) -> str | None:
    """
    Default rule: cond_name = sequence_{ProID}.pt or {ProID}.PT.
    Filenames with extra prefixes/suffixes can be supported with a
    more aggressive matcher if needed.
    """
    cand1 = f"sequence_{proid}.pt"
    cand2 = f"{proid}.PT"
    if cand1 in cond_files: return cand1
    if cand2 in cond_files: return cand2
    # Looser fallback: stem equal to proid and .pt extension
    for name in cond_files:
        stem = name[:-3] if name.lower().endswith('.pt') else name
        if stem == proid:
            return name
    return None

def main():
    p = argparse.ArgumentParser(description="Sample RL raw candidates from GraDe_IF and export CSV.")
    p.add_argument('--condition_dirs', nargs='+', required=True,
                   help='one or more dirs containing .pt condition graphs')
    p.add_argument('--pairs_csv', required=True,
                   help='CSV with at least columns: ProID, SMILES[, cond_name]')
    p.add_argument('--ckpt_path', required=True,
                   help='checkpoint (.pt) with EMA for sampling (catif starting point)')
    p.add_argument('--out_csv', default='epoch_raw.csv',
                   help='output CSV path')
    p.add_argument('--epoch', type=int, default=1,
                   help='epoch tag written into the CSV (recordkeeping)')
    p.add_argument('--group_size', type=int, default=4,
                   help='K: number of samples per condition')
    p.add_argument('--batch_size', type=int, default=8,
                   help='graphs per forward DDIM call (loading only; does not change the K-per-condition logic)')
    p.add_argument('--step', type=int, default=100,
                   help='DDIM sampling step')
    p.add_argument('--diverse', action='store_true',
                   help='stochastic sampling (True) or greedy (False)')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--log_every', type=int, default=50,
                   help='number of samples between rolling recovery/perplexity log lines')
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 1) Gather all .pt condition files
    cond_paths: list[Path] = []
    for d in args.condition_dirs:
        dpth = Path(d)
        if not dpth.exists():
            raise RuntimeError(f"condition dir not found: {d}")
        cond_paths += sorted([p for p in dpth.glob('*.pt')])
    if not cond_paths:
        raise RuntimeError("no .pt graphs found in given condition_dirs")

    cond_names = [p.name for p in cond_paths]
    cond_name_to_path = {p.name: p for p in cond_paths}
    cond_name_set = set(cond_names)

    # 2) Load the pairs mapping (ProID, SMILES[, cond_name])
    by_proid, _ = load_pairs_csv(Path(args.pairs_csv))

    # 3) Build the CATH dataset (cond_names as id list)
    dataset = Cath(cond_names, str(cond_paths[0].parent))  # Cath will look for matching .pt files under the given root
    if len(dataset) == 0:
        raise RuntimeError('Cath dataset is empty - check your condition files.')

    # 4) Load the model (supports both catif's original ckpt and RL policy_epochXX.ckpt)
    ckpt = torch.load(args.ckpt_path, map_location='cpu')

    # 4.1 Resolve config first:
    #   - original catif:  ckpt['config']
    #   - post-RL:         ckpt['meta']['config']
    if 'config' in ckpt:
        config = ckpt['config']
    elif 'meta' in ckpt and isinstance(ckpt['meta'], dict) and 'config' in ckpt['meta']:
        config = ckpt['meta']['config']
    else:
        raise RuntimeError(
            f"checkpoint {args.ckpt_path} contains neither 'config' nor meta['config']; "
            "cannot build the GraDe_IF model."
        )

    diffusion = build_model_from_config(config, dataset[0]).to(device)

    # 4.2 Then load the weights:
    #   - 'ema' present: use the original EMA inference path (catif-era ckpt)
    #   - no 'ema' but has 'model': RL-saved policy weights, load directly into diffusion
    if 'ema' in ckpt:
        ema = EMA(diffusion)
        ema.load_state_dict(ckpt['ema'])
        model = ema.ema_model.to(device).eval()
    elif 'model' in ckpt:
        diffusion.load_state_dict(ckpt['model'], strict=False)
        model = diffusion.to(device).eval()
    else:
        raise RuntimeError(
            f"checkpoint {args.ckpt_path} contains neither 'ema' nor 'model'; "
            "cannot load weights."
        )

    # 5) Walk through every condition graph, assign group by {ProID, SMILES}, draw K samples
    rows_to_write = []
    name_to_index = {name: i for i, name in enumerate(cond_names)}

    # Global recovery / perplexity tracking
    total_samples = 0
    sum_recovery = 0.0
    sum_perplexity = 0.0

    for cond_idx, cond_name in enumerate(cond_names):
        data_index = name_to_index[cond_name]
        data = dataset[data_index]

        # Resolve the WT ProID and SMILES for this condition
        proid = None
        smiles = None
        # 1) If pairs.csv gave an explicit cond_name mapping, reverse-look up ProID
        for pid, info in by_proid.items():
            cn = info.get('cond_name', '')
            if cn and cn == cond_name:
                proid = pid
                smiles = info['SMILES']
                break
        # 2) Otherwise guess using the {ProID}.pt convention
        if proid is None:
            hit = [pid for pid in by_proid.keys()
                   if guess_cond_name_for_proid(pid, {cond_name}) is not None]
            if len(hit) == 1:
                proid = hit[0]
                smiles = by_proid[proid]['SMILES']
            else:
                # No ProID->cond mapping found, skip this condition
                continue

        wt_seq = data_to_sequence(data)

        # Sample K times from this condition graph
        for k in range(args.group_size):
            per_sample_seed = (args.seed if args.seed is not None else 0) + k
            set_seed(per_sample_seed)

            single = Batch.from_data_list([data.clone()]).to(device)
            with torch.no_grad():
                _, pred_onehot = model.ddim_sample(
                    single,
                    diverse=True,       # force stochasticity to get within-group diversity
                    step=args.step
                )
            oh = pred_onehot[:, :20]
            idx = oh.argmax(dim=-1).tolist()
            seq = ''.join(AMINO_CODES[i] for i in idx)

            # --------- per-sample recovery & perplexity ----------
            rec = sequence_recovery(wt_seq, seq)
            ppl = sequence_perplexity(seq)

            total_samples += 1
            sum_recovery += rec
            # ppl may be NaN; skip those in the running sum
            if not (isinstance(ppl, float) and np.isnan(ppl)):
                sum_perplexity += ppl
            # --------------------------------------------------

            rows_to_write.append({
                'epoch': args.epoch,
                'data_index': data_index,
                'cond_name': cond_name,
                'group': f"{proid}||{smiles}",
                'ProID': proid,
                'SMILES': smiles,
                'ProSeq': wt_seq,
                "ProSeq'": seq,
                'sample_idx': k+1,
                'seed': per_sample_seed,     # actual seed used for this sample
                'step': args.step,
                'ckpt_path': args.ckpt_path,
                'recovery': rec,
                'perplexity': ppl,
            })

            # Live progress logging
            if args.log_every > 0 and (total_samples % args.log_every == 0):
                mean_rec = sum_recovery / total_samples if total_samples > 0 else 0.0
                mean_ppl = sum_perplexity / total_samples if total_samples > 0 else float('nan')
                print(
                    f"[{total_samples} samples] "
                    f"last_rec={rec*100:.2f}% last_ppl={ppl:.3f} | "
                    f"mean_rec={mean_rec*100:.2f}% mean_ppl={mean_ppl:.3f}"
                )

    # 6) Write the CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'epoch','data_index','cond_name','group',
            'ProID','SMILES','ProSeq',"ProSeq'",
            'sample_idx','seed','step','ckpt_path',
            'recovery','perplexity',
        ])
        writer.writeheader()
        for r in rows_to_write:
            writer.writerow(r)

    # Final means
    if total_samples > 0:
        mean_rec = sum_recovery / total_samples
        mean_ppl = sum_perplexity / total_samples
        print(f"[OK] Exported {len(rows_to_write)} rows to {out_csv}")
        print(
            f"[stats] Overall stats: samples={total_samples}, "
            f"mean_recovery={mean_rec*100:.2f}%, mean_perplexity={mean_ppl:.3f}"
        )
    else:
        print(f"[WARN] No samples exported. Please check your inputs. (out_csv={out_csv})")

if __name__ == "__main__":
    main()
