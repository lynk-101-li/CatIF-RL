#!/usr/bin/env python
# diffusion/inference.py

"""
示例用法：1
# basic
python -u -m sampling.infrs_pred \
  --test_dir dataset/process/test \
  --ckpt_path ... \
  --output_dir output/normal
  --seed 1

# Repaint sampling example:
python -u -m sampling.infrs_pred \
  --test_dir dataset/process/test \
  --ckpt_path diffusion/results/ \
  --output_dir sampling/ \
  --no_diverse
  --use_repaint \
  --mask_indices "5,10-20" \
  --repaint_jump_n 1 \
  --seed 1
"""

#!/usr/bin/env python
# diffusion/infrs_pred.py

#!/usr/bin/env python
# diffusion/infrs_pred.py

import os
import argparse
import torch
import random
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from catif_rl.models.gradeif_app import EGNN_NET, GraDe_IF
from catif_rl.data.large_dataset import Cath
from ema_pytorch import EMA


# ==========================================
# 1. 基础函数
# ==========================================
AMINO_CODES = ['A','R','N','D','C','Q','E','G','H','I',
               'L','K','M','F','P','S','T','W','Y','V']

def onehot_to_seq(onehot):
    idx = onehot.argmax(dim=-1).cpu().tolist()
    return ''.join(AMINO_CODES[i] for i in idx)

def set_seed(seed):
    if seed < 0: return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ==========================================
# 2. 辅助工具：Mask解析 & 高级采样逻辑
# ==========================================

def parse_mask_indices(indices_str):
    """
    解析索引字符串，支持 "1,2,5-8" 格式。
    返回 0-based 的索引列表。
    """
    if not indices_str:
        return []
    
    indices = set()
    parts = indices_str.split(',')
    for part in parts:
        part = part.strip()
        if '-' in part:
            start, end = map(int, part.split('-'))
            indices.update(range(start, end + 1))
        else:
            if part:
                indices.add(int(part))
    return sorted(list(indices))

def create_batch_mask(batch, user_indices, device):
    """
    构建全局 Mask：1=Keep(GT), 0=Inpaint(Gen)
    """
    total_nodes = batch.x.shape[0]
    mask = torch.ones((total_nodes, 1), device=device) # 默认全保留
    
    if not hasattr(batch, 'ptr'):
        ptr = [0, total_nodes]
    else:
        ptr = batch.ptr.cpu().tolist()
    
    for i in range(len(ptr) - 1):
        start_node = ptr[i]
        end_node = ptr[i+1]
        length = end_node - start_node
        
        for local_idx in user_indices:
            if local_idx < length:
                global_idx = start_node + local_idx
                mask[global_idx] = 0.0
                
    return mask

def repaint_ddim_sample(diffusion_model, data, mask, gt_x, step=100, jump_n=10, diverse=True):
    """
    支持 Resampling (Jump) 的 RePaint 采样函数。
    """
    timesteps = diffusion_model.timesteps
    device = data.x.device
    
    # 初始化噪声
    limit_dist = torch.ones(20) / 20
    zt = diffusion_model.sample_discrete_feature_noise(limit_dist=limit_dist, num_node=data.x.shape[0])
    zt = zt.to(device)
    
    # 生成时间步列表 (倒序)
    times_list = list(reversed(range(0, timesteps, step)))
    
    # 进度条描述
    pbar_desc = f"Repaint(jump={jump_n}, div={diverse})"
    
    for i, s_int in enumerate(tqdm(times_list, desc=pbar_desc)):
        # s_int 是当前要去的目标时刻 (next step)
        # t_int 是上一步的时刻 (current step)
        t_int = s_int + step
        
        # 计算归一化时间
        s_array = s_int * torch.ones((data.batch[-1]+1, 1)).to(device)
        t_array = t_int * torch.ones((data.batch[-1]+1, 1)).to(device)
        s_norm = s_array / timesteps
        t_norm = t_array / timesteps
        
        # === Resampling Loop (倒退-前进 循环) ===
        # 如果是最后一步(s=0)，通常不需要再 jump 回去，除非为了最后refine
        # RePaint 论文逻辑：在每一步 t->s 后，跳回 t，重复 U 次
        
        # 确定循环次数：如果是最后一步，只做 1 次（不跳回）；否则做 jump_n 次
        current_jump_n = 1 if s_int == 0 else jump_n
        
        for u in range(current_jump_n):
            
            # 1. Denoise (t -> s)
            # 调用 GraDe-IF 的 p(z_s | z_t)
            is_last_step = (s_int == 0) and (u == current_jump_n - 1)
            
            # 注意：sample_p_zs_given_zt 需要的是 t_norm (当前) 和 s_norm (目标)
            zt_pred, final_predicted_X = diffusion_model.sample_p_zs_given_zt(
                t_norm, s_norm, zt, data, cond=False, diverse=diverse, step=step, last_step=is_last_step
            )
            
            # 2. Replacement (注入 GT)
            # 获取 s 时刻的 GT 噪声分布
            # 使用 diffusion_model.apply_noise (它利用 Qt_bar 从 x0 -> xs)
            s_int_batch = torch.full((data.batch[-1]+1, 1), s_int, device=device).float()
            temp_data = data.clone()
            temp_data.x = gt_x
            zt_known = diffusion_model.apply_noise(temp_data, s_int_batch).x
            
            # 融合
            zt = mask * zt_known + (1 - mask) * zt_pred
            
            # 3. Jump Back (s -> t) [加噪]
            # 如果不是最后一次循环，且不是最后一步，我们需要把 zt (即 xs) 加噪回 xt
            if u < current_jump_n - 1 and s_int > 0:
                # 我们需要从 s 跳回 t (s < t)
                # 对于 Uniform 噪声，转移矩阵 Q_{t|s} 可以通过相对 alpha_bar 计算
                
                # 获取 alpha_bar
                alpha_t_bar = diffusion_model.noise_schedule.get_alpha_bar(t_normalized=t_norm)
                alpha_s_bar = diffusion_model.noise_schedule.get_alpha_bar(t_normalized=s_norm)
                
                # 计算相对 alpha: alpha_{t|s} = alpha_t / alpha_s
                # (注意：alpha_bar 是单调递减的，t > s implies alpha_t < alpha_s, 所以 ratio < 1)
                # 这里为了防止数值不稳，加个 clamp
                alpha_rel = (alpha_t_bar / alpha_s_bar).clamp(0, 1)
                
                # 获取转移矩阵 Q_{t|s}
                # 注意：GraDe_IF 里的 get_Qt_bar 是生成 "Keep" 概率矩阵
                if diffusion_model.config['noise_type'] == 'uniform':
                    Qt_jump = diffusion_model.transition_model.get_Qt_bar(alpha_rel, device=device)
                    
                    # 采样: z_t ~ z_s @ Q_{t|s}
                    prob_jump = (Qt_jump[data.batch] @ zt.unsqueeze(2)).squeeze()
                    zt_idx = prob_jump.multinomial(1).squeeze()
                    zt = torch.nn.functional.one_hot(zt_idx, num_classes=20).float()
                else:
                    # 如果是 BLOSUM 等其他噪声，相对 alpha 计算可能不准确
                    # 这里做一个简单的回退：只做单纯的 Replacement，不进行加噪 Jump
                    # 或者你可以调用 step 次单步 apply_noise，但比较慢。
                    # 暂时保持 zt 不变 (相当于 jump_n 无效，退化为普通 repaint)
                    pass

    return zt, final_predicted_X

# ==========================================
# 3. 主程序
# ==========================================


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="GraDe_IF Inference with Repaint & Resampling")
    
    # 基础参数
    p.add_argument('--test_dir',   type=str, required=True)
    p.add_argument('--ckpt_path',  type=str, required=True)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--device',     type=str, default='cuda:0')
    p.add_argument('--output_dir', type=str, required=True)
    p.add_argument('--seed',       type=int, default=-1)
    
    # 采样参数
    p.add_argument('--step',       type=int, default=50, help="DDIM 采样步长")
    p.add_argument('--no_diverse', action='store_true', help="如果设置，将关闭 diverse 采样 (使用 argmax)")
    
    # Repaint 参数
    p.add_argument('--use_repaint', action='store_true', help="开启 Repaint 模式")
    p.add_argument('--mask_indices', type=str, default="", help="需要重绘的索引 (0-based), e.g. '10-20,35'")
    p.add_argument('--repaint_jump_n', type=int, default=10, help="Repaint 重采样次数 (默认 10). 设为 1 则不进行重采样回跳.")
    
    args = p.parse_args()

    # 处理参数
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理 diverse (默认为 True，除非用户加了 --no_diverse)
    is_diverse = not args.no_diverse

    # 加载数据
    test_ids = sorted([f for f in os.listdir(args.test_dir) if f.endswith('.pt')])
    test_ds  = Cath(test_ids, args.test_dir)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

    # 加载 Checkpoint
    print(f"Loading checkpoint from {args.ckpt_path} ...")
    ckpt = torch.load(args.ckpt_path, map_location=device)
    
    # 兼容性处理：不同训练阶段的 ckpt 结构可能不同
    if 'config' in ckpt:
        config = ckpt['config']
    elif 'meta' in ckpt:
        config = ckpt['meta']['config']
    else:
        # 尝试直接从 ckpt 根目录找 (有些简单的 save 逻辑)
        config = ckpt.get('args', {}) 
        # 如果还是找不到，可能需要手动硬编码 defaults
        if not config: print("Warning: Config not found in ckpt, using defaults might fail.")

    # 动态构建模型
    base_model = EGNN_NET(
        input_feat_dim = config.get('input_feat_dim', 0), # 需确保 config 里有这个，或者从 dataset 推断
        hidden_channels= config['hidden_dim'],
        edge_attr_dim  = config.get('edge_attr_dim', 0),
        dropout        = config['drop_out'],
        n_layers       = config['depth'],
        update_edge    = config.get('update_edge', True),
        embedding      = config.get('embedding', False), 
        embedding_dim  = config.get('embedding_dim', 64),
        norm_feat      = config.get('norm_feat', False),
        embed_ss       = config.get('embed_ss', -1),
    )
    
    diffusion = GraDe_IF(
        model      = base_model,
        timesteps  = config['timesteps'],
        objective  = config.get('objective', 'pred_x0'),
        config     = config
    ).to(device)
    
    # 加载权重
    if 'ema' in ckpt:
        ema = EMA(diffusion)
        ema.load_state_dict(ckpt['ema'])
        model = ema.ema_model.to(device).eval()
    elif 'model' in ckpt:
        diffusion.load_state_dict(ckpt['model'], strict=False)
        model = diffusion.to(device).eval()
    else:
        # 最后的尝试：ckpt 本身就是 state_dict
        diffusion.load_state_dict(ckpt, strict=False)
        model = diffusion.to(device).eval()

    # 打印模式信息
    if args.use_repaint:
        indices = parse_mask_indices(args.mask_indices)
        print(f"🎨 Mode: Repaint | Jump N: {args.repaint_jump_n} | Diverse: {is_diverse}")
        print(f"📍 Masked Indices (Inpainting): {indices}")
    else:
        print(f"🚀 Mode: Standard Generation | Diverse: {is_diverse}")

    # 推理循环
    global_idx = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            # 确保 input_feat_dim 匹配 (防止 config 里的 dim 只有占位符)
            # 这里的逻辑是为了应对 config 中没有保存 dim 的情况
            if base_model.lin.in_features != batch.x.shape[1] + batch.extra_x.shape[1]:
               # 如果发现维度对不上，可能需要在这里做一些特殊的 hack，或者信任 ckpt 是对的
               pass 

            pred_onehot = None
            
            if args.use_repaint:
                # 1. 准备 GT
                gt_x = batch.x.clone()
                # 2. 构建 Mask
                user_mask_indices = parse_mask_indices(args.mask_indices)
                mask = create_batch_mask(batch, user_mask_indices, device)
                
                # 3. Repaint 采样
                zt, pred_onehot = repaint_ddim_sample(
                    model, 
                    batch, 
                    mask, 
                    gt_x, 
                    step=args.step, 
                    jump_n=args.repaint_jump_n,
                    diverse=is_diverse
                )
            else:
                # 标准采样
                zt, pred_onehot = model.ddim_sample(
                    batch, 
                    diverse=is_diverse, 
                    step=args.step
                )

            # 保存
            node_counts = [g.x.size(0) for g in batch.to_data_list()]
            start = 0
            for count in node_counts:
                if global_idx >= len(test_ids): break
                
                oh = pred_onehot[start:start+count]
                seq = onehot_to_seq(oh)
                pt_name = test_ids[global_idx]
                fasta_fn = os.path.splitext(pt_name)[0] + '.fasta'
                out_path = os.path.join(args.output_dir, fasta_fn)
                
                with open(out_path, 'w') as f:
                    f.write(f'>{pt_name}\n{seq}\n')
                    
                start += count
                global_idx += 1

    print(f"✅ All Done. Output saved to {args.output_dir}")