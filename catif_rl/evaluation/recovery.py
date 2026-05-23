#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import sys
import numpy as np
import pandas as pd
import torch.nn.functional as F
from ema_pytorch import EMA
from torch_geometric.loader import DataLoader
from catif_rl.data.large_dataset import Cath
from catif_rl.models.gradeif_app import EGNN_NET, GraDe_IF

# Neutralize torch.manual_seed so that the sampler stays diverse across runs
# regardless of any seeding done by external scripts.
torch.manual_seed = lambda *args, **kwargs: None


# 1. Configuration: checkpoint path, test set directory, batch size,
#    ensemble count, DDIM sampling step, and device.
CKPT = 'diffusion/results/weight_rl/Nov26_epoch1/policy_epoch01.pt'
TEST_DIR = 'dataset/process/test/'
BATCH_SIZE = 300
ENSEMBLE_NUM = 50    # number of sampling repetitions to ensemble over
DDIM_STEP = 250      # DDIM sampling step count
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --------- Unified ckpt loading (compatible with original CatIF + RL policy_epochXX) ---------
ckpt = torch.load(CKPT, map_location=DEVICE)

# 1) Resolve the model config: prefer ckpt['config'], otherwise look under meta.
if 'config' in ckpt:
    config = ckpt['config']
elif 'meta' in ckpt and isinstance(ckpt['meta'], dict) and 'config' in ckpt['meta']:
    config = ckpt['meta']['config']
else:
    raise KeyError(
        f"checkpoint {CKPT} does not contain a config; "
        f"expected either ckpt['config'] or ckpt['meta']['config']."
    )

# 2) Rebuild the underlying EGNN_NET with the same hyperparameters used in training.
input_feat_dim = config['input_feat_dim']
edge_attr_dim  = config['edge_attr_dim']
base_model = EGNN_NET(
    input_feat_dim=input_feat_dim,
    hidden_channels=config['hidden_dim'],
    edge_attr_dim=edge_attr_dim,
    dropout=config['drop_out'],
    n_layers=config['depth'],
    update_edge=config.get('update_edge', True),
    embedding=config.get('embedding', False),
    embedding_dim=config.get('embedding_dim', 16),
    norm_feat=config.get('norm_feat', False),
    embed_ss=config.get('embed_ss', -1),
)

# Wrap the base model in the GraDe_IF discrete-diffusion framework.
diffusion = GraDe_IF(
    model=base_model,
    timesteps=config['timesteps'],
    objective=config.get('objective', 'pred_x0'),
    config=config
)

# 3) Support both checkpoint formats:
#    - Original CatIF supervised: ckpt['ema'] holds the EMA state -> wrap and restore via EMA(diffusion)
#    - RL policy_epochXX.pt: only ckpt['model'] is present -> load directly into diffusion
if 'ema' in ckpt:
    # Original CatIF / EMA-tracked checkpoint
    ema = EMA(diffusion)
    ema.load_state_dict(ckpt['ema'])
    model = ema.ema_model.to(DEVICE).eval()
else:
    # RL trainer output: only 'model', no EMA
    if 'model' not in ckpt:
        raise KeyError(
            f"checkpoint {CKPT} contains neither 'ema' nor 'model'; cannot load weights."
        )
    diffusion.load_state_dict(ckpt['model'], strict=False)
    model = diffusion.to(DEVICE).eval()

# From here on the rest of the script uses `model.ddim_sample` regardless of provenance.


# 3. Build the test-set DataLoader.
test_ids = sorted(os.listdir(TEST_DIR))       # all .pt filenames in the test directory
test_ds  = Cath(test_ids, TEST_DIR)           # load graphs through the Cath dataset class
test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
    num_workers=6
)

# 4. Evaluate on the test set with multiple random samples; track per-run and ensemble metrics.
records = []         # one record per run: {'run', 'recovery', 'perplexity'}
ensemble_accum = []  # accumulated per-run probability tensors for the running ensemble

for run in range(ENSEMBLE_NUM):
    all_prob   = []  # this run's per-batch probability distributions
    all_seq    = []  # this run's per-batch ground-truth labels
    ind_accum  = []  # this run's per-node correctness indicators

    with torch.no_grad():
        # DDIM-sample every batch
        for data in test_loader:
            data = data.to(DEVICE)
            prob, sample = model.ddim_sample(
                data,
                diverse=True,
                step=DDIM_STEP
            )
            # Per-batch recovery (fraction of correctly predicted residues)
            seq_true = data.x.argmax(dim=1)
            seq_pred = sample.argmax(dim=1)
            rr = (seq_true == seq_pred).float().mean().item()
            ind_accum.append((seq_true == seq_pred).cpu())

            # Stash probabilities and labels for the global aggregate
            all_prob.append(prob.cpu())
            all_seq.append(seq_true.cpu())

    # Concatenate batches to obtain the per-run test-set totals
    all_prob = torch.cat(all_prob, dim=0)   # [N_total, 20]
    all_seq  = torch.cat(all_seq, dim=0)    # [N_total]
    ind_all  = torch.cat(ind_accum, dim=0)  # [N_total]
    # Per-run recovery rate and perplexity = exp(mean cross-entropy)
    rr = ind_all.float().mean().item()
    ppl = np.exp(F.cross_entropy(all_prob, all_seq, reduction='mean').item())

    # Record this run's single-sample metrics
    records.append({'run': run, 'recovery': rr, 'perplexity': ppl})
    ensemble_accum.append(all_prob)

    # Starting from the second run, also record the running ensemble metric
    if run > 0:
        ens_prob = torch.stack(ensemble_accum, dim=0).mean(dim=0)  # mean per-node probability
        ens_pred = ens_prob.argmax(dim=1)
        ens_rr   = (ens_pred == all_seq).float().mean().item()
        ens_ppl  = np.exp(F.cross_entropy(ens_prob, all_seq, reduction='mean').item())
        records.append({
            'run': f'ensemble_{run}',
            'recovery': ens_rr,
            'perplexity': ens_ppl
        })

# 5. Persist the records to CSV and echo to stdout.
df = pd.DataFrame(records)
df.to_csv('evaluation/Nov26_epoch1_evaluation_results.csv', index=False)
print(df)
