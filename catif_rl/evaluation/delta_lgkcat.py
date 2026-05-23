"""
Legacy evaluation entry point retained for reference. The global RNG seed is
not unsealed here, so this is not the canonical scoring script -- see
``catif_rl.evaluation.recovery`` for the maintained evaluator.
"""

import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from ema_pytorch import EMA
from torch_geometric.loader import DataLoader
from catif_rl.data.large_dataset import Cath
from catif_rl.models.gradeif_app import EGNN_NET, GraDe_IF

# 1. Configuration
CKPT = 'diffusion/results/weight/Jul01_result_lr=0.0005_dp=0.1_clip=1.0_timestep=500_depth=6_hidden=128_embedding=True_embed_dim=128_ss=-1_noise=blosum_467.pt'  # path to the .pt checkpoint
TEST_DIR = 'dataset/process/test/'  # path to the held-out test set
BATCH_SIZE = 300
ENSEMBLE_NUM = 50
DDIM_STEP = 250  # sampling step
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Load the checkpoint and rebuild the EMA model
ckpt = torch.load(CKPT, map_location=DEVICE)
config = ckpt['config']

# Build the underlying network from the embedded config
input_feat_dim = config['input_feat_dim']
edge_attr_dim = config['edge_attr_dim']
base_model = EGNN_NET(
    input_feat_dim=input_feat_dim,
    hidden_channels=config['hidden_dim'],
    edge_attr_dim=edge_attr_dim,
    dropout=config['drop_out'],
    n_layers=config['depth'],
    update_edge=True,
    embedding=config['embedding'],
    embedding_dim=config['embedding_dim'],
    norm_feat=config['norm_feat'],
    embed_ss=config['embed_ss']
)

diffusion = GraDe_IF(model=base_model,
                     timesteps=config['timesteps'],
                     objective=config['objective'],
                     config=config)
# Wrap with EMA and restore the weights
ema = EMA(diffusion)
ema.load_state_dict(ckpt['ema'])
ema_model = ema.ema_model.to(DEVICE)
ema_model.eval()

# 3. Build the test-set DataLoader
test_ids = sorted(os.listdir(TEST_DIR))
test_ds = Cath(test_ids, TEST_DIR)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         pin_memory=True, num_workers=6)

# 4. Evaluation loop
records = []          # per-run recovery rate and perplexity
ensemble_accum = []   # accumulated per-run probability tensors

for run in range(ENSEMBLE_NUM):
    all_prob = []
    all_seq = []
    ind_accum = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(DEVICE)
            prob, sample = ema_model.ddim_sample(data,
                                                diverse=True,
                                                step=DDIM_STEP)
            # Single-run recovery
            seq_true = data.x.argmax(dim=1)
            seq_pred = sample.argmax(dim=1)
            rr = (seq_true == seq_pred).float().mean().item()
            ind_accum.append((seq_true == seq_pred).cpu())

            all_prob.append(prob.cpu())
            all_seq.append(seq_true.cpu())

    all_prob = torch.cat(all_prob, dim=0)
    all_seq = torch.cat(all_seq, dim=0)
    ind_all = torch.cat(ind_accum, dim=0)
    ppl = np.exp(F.cross_entropy(all_prob, all_seq, reduction='mean').item())

    records.append({'run': run, 'recovery': rr, 'perplexity': ppl})
    ensemble_accum.append(all_prob)

    # Running ensemble metric (from the second run onwards)
    if run > 0:
        ens_prob = torch.stack(ensemble_accum).mean(dim=0)
        ens_pred = ens_prob.argmax(dim=1)
        ens_rr = (ens_pred == all_seq).float().mean().item()
        ens_ppl = np.exp(F.cross_entropy(ens_prob, all_seq, reduction='mean').item())
        records.append({'run': f'ensemble_{run}', 'recovery': ens_rr, 'perplexity': ens_ppl})

# 5. Persist results
df = pd.DataFrame(records)
df.to_csv('gradeifevaluation_results.csv', index=False)
print(df)
