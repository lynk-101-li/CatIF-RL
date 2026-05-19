#!/usr/bin/env python
# diffusion/inference.py

"""
示例用法：
for i in 
python -u -m sampling.infrs_pred \
  --test_dir   dataset/process/test \
  --ckpt_path  diffusion/results/weight_rl/Nov19_epoch1/policy_epoch03.pt \
  --output_dir sampling/output_test_dataset_mut_epoch1_Nov19 \
  --seed 1

for s in {1..5}; do
  python -u -m sampling.infrs_pred \
    --test_dir dataset/process/test \
    --ckpt_path diffusion/results/weight_rl/Nov27.5_round1/policy_epoch01.pt \
    --output_dir sampling/output_test_dataset_mut_round1_Nov27.5_seed${s} \
    --seed $s
done
"""

import os
import argparse
import torch
import random
import numpy as np
from ema_pytorch import EMA
from torch_geometric.loader import DataLoader
from catif_rl.models.gradeif_app import EGNN_NET, GraDe_IF
from catif_rl.data.large_dataset import Cath
#torch.manual_seed = lambda *args, **kwargs: None #### 覆盖 manual_seed，让它不做任何事

# 20 种氨基酸单字母码，用于将 one-hot 向量转换为序列
AMINO_CODES = ['A','R','N','D','C','Q','E','G','H','I',
               'L','K','M','F','P','S','T','W','Y','V']

def onehot_to_seq(onehot):
    """
    将模型输出的 one-hot 张量转换为氨基酸序列字符串。
    onehot: [total_nodes, 20]
    返回: 长度等于 total_nodes 的字符串
    """
    idx = onehot.argmax(dim=-1).cpu().tolist()
    return ''.join(AMINO_CODES[i] for i in idx)

def set_seed(seed: int):
    if seed is None or seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 如需更强确定性，可开启（通常不必）：
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    # 1. 参数解析
    p = argparse.ArgumentParser(description="GraDe_IF inference with EMA weights")
    p.add_argument('--test_dir',   type=str, required=True, help=".pt 测试文件目录")
    p.add_argument('--ckpt_path',  type=str, required=True, help="包含 EMA 权重的 .pt 文件路径")
    p.add_argument('--batch_size', type=int, default=32, help="DataLoader batch_size")
    p.add_argument('--device',     type=str, default='cuda:0', help="device, e.g. cuda:0 or cpu")
    p.add_argument('--output_dir', type=str, required=True, help="输出 fasta 文件目录")
    p.add_argument('--seed', type=int, default=-1, help='Random seed; set >=0 for reproducible DDIM sampling')
    args = p.parse_args()

    # 2. 设备 & 输出目录
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # 3. 测试集 DataLoader
    test_ids = sorted([f for f in os.listdir(args.test_dir) if f.endswith('.pt')])
    test_ds  = Cath(test_ids, args.test_dir)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4
    )

    # 4. 从 checkpoint 恢复网络结构并加载 EMA 权重
    ckpt    = torch.load(args.ckpt_path, map_location=device)
    # 兼容两种 ckpt 格式：
    # 1) 监督训练: 直接有 ckpt['config']
    # 2) RL 微调: 只有 ckpt['meta']['config']
    if 'config' in ckpt:
        config = ckpt['config'] # config 中已经存了所有超参和输入维度
    elif 'meta' in ckpt and isinstance(ckpt['meta'], dict) and 'config' in ckpt['meta']:
        config = ckpt['meta']['config']
    else:
        raise KeyError("config not found in ckpt; expected 'config' or 'meta[\"config\"]'")
    
    # 4.1 动态构建 EGNN_NET，参数与训练时完全一致
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
    
    # 4.2 包装成 GraDe_IF
    diffusion = GraDe_IF(
        model      = base_model,
        timesteps  = config['timesteps'],
        objective  = config.get('objective', 'pred_x0'),
        config     = config
    ).to(device)
    
    # 4.3 根据 ckpt 结构选择加载方式
    if 'ema' in ckpt:
        # 旧的监督训练 ckpt：用 EMA 平滑权重
        from ema_pytorch import EMA
        ema = EMA(diffusion)
        ema.load_state_dict(ckpt['ema'])
        model = ema.ema_model.to(device).eval()
    else:
        # RL ckpt：只有 'model'，直接用微调后的 diffusion
        diffusion.load_state_dict(ckpt['model'], strict=False)
        model = diffusion.to(device).eval()

    # 6. 推理 & 拆分写文件
    global_idx = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            # 用 EMA 模型做 DDIM 采样
            zt, pred_onehot = model.ddim_sample(
                batch,
                diverse=True,
                step=100      # 可根据需求调整
            )
            # 每个子图的节点数
            node_counts = [g.x.size(0) for g in batch.to_data_list()]

            start = 0
            for count in node_counts:
                oh = pred_onehot[start:start+count]
                seq = onehot_to_seq(oh)
                pt_name = test_ids[global_idx]
                fasta_fn = os.path.splitext(pt_name)[0] + '.fasta'
                out_path = os.path.join(args.output_dir, fasta_fn)
                with open(out_path, 'w') as f:
                    f.write(f'>{pt_name}\n{seq}\n')
                start += count
                global_idx += 1

    print(f"✅ 共处理 {global_idx} 个样本，结果保存在 {args.output_dir}")
