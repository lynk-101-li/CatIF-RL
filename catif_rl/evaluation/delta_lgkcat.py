'''
此为老版本脚本，没有解封随机种子，不是用评测
'''

import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from ema_pytorch import EMA
from torch_geometric.loader import DataLoader
from catif_rl.data.large_dataset import Cath
from catif_rl.models.gradeif_app import EGNN_NET, GraDe_IF

# 1. 配置
CKPT = 'diffusion/results/weight/Jul01_result_lr=0.0005_dp=0.1_clip=1.0_timestep=500_depth=6_hidden=128_embedding=True_embed_dim=128_ss=-1_noise=blosum_467.pt'  #### 路径到 .pt 文件
TEST_DIR = 'dataset/process/test/' ###路径到测试集
BATCH_SIZE = 300
ENSEMBLE_NUM = 50
DDIM_STEP = 250  # sampling step
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. 加载 checkpoint 并还原 EMA 模型
ckpt = torch.load(CKPT, map_location=DEVICE)
config = ckpt['config']

# 构建底层模型
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
# 包装 EMA 并加载
ema = EMA(diffusion)
ema.load_state_dict(ckpt['ema'])
ema_model = ema.ema_model.to(DEVICE)
ema_model.eval()

# 3. 测试集 DataLoader
test_ids = sorted(os.listdir(TEST_DIR))
test_ds = Cath(test_ids, TEST_DIR)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         pin_memory=True, num_workers=6)

# 4. 评估
records = []  # 存单次 run 的 rr, ppl
ensemble_accum = []  # 存每次循环的 all_prob

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
            # recovery
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

    # 每次 ensemble
    if run > 0:
        ens_prob = torch.stack(ensemble_accum).mean(dim=0)
        ens_pred = ens_prob.argmax(dim=1)
        ens_rr = (ens_pred == all_seq).float().mean().item()
        ens_ppl = np.exp(F.cross_entropy(ens_prob, all_seq, reduction='mean').item())
        records.append({'run': f'ensemble_{run}', 'recovery': ens_rr, 'perplexity': ens_ppl})

# 5. 保存结果
df = pd.DataFrame(records)
df.to_csv('gradeifevaluation_results.csv', index=False)
print(df)

