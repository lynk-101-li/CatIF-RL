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

# 防止外部脚本改 seed，这里保留原来的行
torch.manual_seed = lambda *args, **kwargs: None


# 1. 配置部分：指定 checkpoint 路径、测试集目录、批大小、集成次数、采样步数和设备
CKPT = 'diffusion/results/weight_rl/Nov26_epoch1/policy_epoch01.pt'
TEST_DIR = 'dataset/process/test/'
BATCH_SIZE = 300
ENSEMBLE_NUM = 50    # 要重复采样几次并做 ensemble
DDIM_STEP = 250      # DDIM 采样时的 step 参数
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --------- 统一加载 ckpt（兼容原始 catif + 强化学习 policy_epochXX） ---------
ckpt = torch.load(CKPT, map_location=DEVICE)

# 1) 取 config：优先 ckpt['config']，否则看 meta 里有没有 config
if 'config' in ckpt:
    config = ckpt['config']
elif 'meta' in ckpt and isinstance(ckpt['meta'], dict) and 'config' in ckpt['meta']:
    config = ckpt['meta']['config']
else:
    raise KeyError(
        f"checkpoint {CKPT} 中没有找到 config，"
        f"请确认是否包含 'config' 或 meta['config']"
    )

# 2) 重新构建底层的 EGNN_NET，参数要与训练时一致
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

# 将底层模型包装进 GraDe_IF 扩散框架
diffusion = GraDe_IF(
    model=base_model,
    timesteps=config['timesteps'],
    objective=config.get('objective', 'pred_x0'),
    config=config
)

# 3) 兼容两种权重格式：
#    - 训练阶段原始 catif: ckpt['ema'] 存 EMA 状态 -> 用 EMA(diffusion) 还原 ema_model
#    - 强化学习阶段 policy_epochXX.pt: 只存 ckpt['model'] -> 直接加载到 diffusion
if 'ema' in ckpt:
    # 原始 catif / 带 EMA 的 checkpoint
    ema = EMA(diffusion)
    ema.load_state_dict(ckpt['ema'])
    model = ema.ema_model.to(DEVICE).eval()
else:
    # RL trainer 输出：只有 'model'，没有 EMA
    if 'model' not in ckpt:
        raise KeyError(
            f"checkpoint {CKPT} 既没有 'ema' 也没有 'model'，无法加载权重"
        )
    diffusion.load_state_dict(ckpt['model'], strict=False)
    model = diffusion.to(DEVICE).eval()

# 到这里，后面统一用 `model` 做 ddim_sample，不再区分 ema / 非 ema


# 3. 构建测试集 DataLoader
test_ids = sorted(os.listdir(TEST_DIR))       # 得到所有 .pt 文件名
test_ds  = Cath(test_ids, TEST_DIR)           # 用 Cath 类加载图数据
test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
    num_workers=6
)

# 4. 在测试集上评估：多次随机采样、计算单次和 ensemble 的指标
records = []         # 存每次 run 的 {'run', 'recovery', 'perplexity'}
ensemble_accum = []  # 存每次 run 全量的 all_prob，用于后续 ensemble

for run in range(ENSEMBLE_NUM):
    all_prob   = []  # 本次 run 下，每个 batch 的概率分布
    all_seq    = []  # 本次 run 下，对应的真实标签
    ind_accum  = []  # 本次 run 下，每个节点的正确/错误布尔向量

    with torch.no_grad():
        # 遍历所有 batch，做 DDIM 采样
        for data in test_loader:
            data = data.to(DEVICE)
            prob, sample = model.ddim_sample(
                data,
                diverse=True,
                step=DDIM_STEP
            )
            # 计算本 batch 的恢复率（预测正确比例）
            seq_true = data.x.argmax(dim=1)
            seq_pred = sample.argmax(dim=1)
            rr = (seq_true == seq_pred).float().mean().item()
            ind_accum.append((seq_true == seq_pred).cpu())

            # 收集概率分布和真实标签
            all_prob.append(prob.cpu())
            all_seq.append(seq_true.cpu())

    # 拼接所有 batch 得到测试集全量结果
    all_prob = torch.cat(all_prob, dim=0)   # [N_total, 20]
    all_seq  = torch.cat(all_seq, dim=0)    # [N_total]
    ind_all  = torch.cat(ind_accum, dim=0)  # [N_total]
    # 计算全量rr与 perplexity = exp(平均交叉熵)
    rr = ind_all.float().mean().item()
    ppl = np.exp(F.cross_entropy(all_prob, all_seq, reduction='mean').item())

    # 记录本次 run 的单次指标
    records.append({'run': run, 'recovery': rr, 'perplexity': ppl})
    ensemble_accum.append(all_prob)

    # 从第二次 run 开始，每次都做 ensemble 平均并记录
    if run > 0:
        ens_prob = torch.stack(ensemble_accum, dim=0).mean(dim=0)  # 平均概率
        ens_pred = ens_prob.argmax(dim=1)
        ens_rr   = (ens_pred == all_seq).float().mean().item()
        ens_ppl  = np.exp(F.cross_entropy(ens_prob, all_seq, reduction='mean').item())
        records.append({
            'run': f'ensemble_{run}',
            'recovery': ens_rr,
            'perplexity': ens_ppl
        })

# 5. 将记录写入 CSV，并打印
df = pd.DataFrame(records)
df.to_csv('evaluation/Nov26_epoch1_evaluation_results.csv', index=False)
print(df)