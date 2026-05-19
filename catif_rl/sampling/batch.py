#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
采样 RL 原始样本（不做打分），导出 CSV：epochX_raw.csv

- 兼容多个 condition-dir（train / validation）
- 参考 infrs_preds.py 的 EMA + DDIM 采样流程（仅用于生成序列）
- 按 {ProID, SMILES} 定义 group_id（方案 A）
- 每个条件采样 K 条（--group_size）
- 输出列：data_index, cond_name, group, ProID, SMILES, wt_seq, seq, sample_idx, seed, step, ckpt_path

用法示例：
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

说明：
- pairs_csv 至少包含列：ProID, SMILES。若还有 cond_name 列可直接映射；否则默认用 {ProID}.pt 去匹配条件图文件。
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
    """从节点特征的前20维 one-hot 反推 WT 序列（节点顺序=序列顺序）"""
    idx = data.x[:, :20].argmax(dim=-1).tolist()
    return ''.join(AMINO_CODES[i] for i in idx)

def sequence_recovery(wt_seq: str, mut_seq: str) -> float:
    """
    序列回复率：生成序列与 WT 在重叠长度上的按位一致比例。
    如果长度不同，只对 min(len_wt, len_mut) 的前缀做比较。
    """
    L = min(len(wt_seq), len(mut_seq))
    if L == 0:
        return 0.0
    match = sum(1 for a, b in zip(wt_seq[:L], mut_seq[:L]) if a == b)
    return match / L

def sequence_perplexity(seq: str) -> float:
    """
    基于经验氨基酸频率的熵定义的 perplexity：
      H = -Σ p(a) log p(a)
      ppl = exp(H)
    注意：这是“序列氨基酸多样性”的度量，不是语言模型意义上的 log-prob perplexity。
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
    读取 {ProID, SMILES[, cond_name]} 映射
    返回：
      - by_proid: {ProID -> {'SMILES':..., 'cond_name':optional}}
      - rows:     原始行（保留方便扩展）
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
    缺省规则：cond_name = sequence_{ProID}.pt 或 {ProID}.PT
    也允许文件名前后附加前缀/后缀（可按需改进）
    """
    cand1 = f"sequence_{proid}.pt"
    cand2 = f"{proid}.PT"
    if cand1 in cond_files: return cand1
    if cand2 in cond_files: return cand2
    # 允许更宽松匹配：包含 proid 且以 .pt 结尾
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
                   help='checkpoint (.pt) with EMA for sampling (catif 起点)')
    p.add_argument('--out_csv', default='epoch_raw.csv',
                   help='output CSV path')
    p.add_argument('--epoch', type=int, default=1,
                   help='epoch tag written into CSV (for记录)')
    p.add_argument('--group_size', type=int, default=4,
                   help='K: number of samples per condition')
    p.add_argument('--batch_size', type=int, default=8,
                   help='graphs per forward DDIM call (只是载入，不影响每组K条的逻辑)')
    p.add_argument('--step', type=int, default=100,
                   help='DDIM sampling step')
    p.add_argument('--diverse', action='store_true',
                   help='stochastic sampling (True) or greedy (False)')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--log_every', type=int, default=50,
                   help='每多少条样本在终端打印一次当前/平均 recovery 与 perplexity')
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 1) 收集所有 .pt 条件文件
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

    # 2) 读 pairs 映射（ProID, SMILES[, cond_name]）
    by_proid, _ = load_pairs_csv(Path(args.pairs_csv))

    # 3) 构建 CATH 数据集（使用 cond_names 作为 id 列表）
    dataset = Cath(cond_names, str(cond_paths[0].parent))  # CatH 实现会在给定根目录下找名字匹配的 .pt
    if len(dataset) == 0:
        raise RuntimeError('Cath dataset is empty — check your condition files.')

    # 4) 加载模型（兼容 catif 原始 ckpt 和 RL 后 policy_epochXX.ckpt）
    ckpt = torch.load(args.ckpt_path, map_location='cpu')

    # 4.1 先拿 config：
    #   - 原始 catif:  ckpt['config']
    #   - RL 之后:     ckpt['meta']['config']
    if 'config' in ckpt:
        config = ckpt['config']
    elif 'meta' in ckpt and isinstance(ckpt['meta'], dict) and 'config' in ckpt['meta']:
        config = ckpt['meta']['config']
    else:
        raise RuntimeError(
            f"Checkpoint {args.ckpt_path} 中既没有 'config' 也没有 meta['config']，"
            "无法构建 GraDe_IF 模型结构。"
        )

    diffusion = build_model_from_config(config, dataset[0]).to(device)

    # 4.2 再加载权重：
    #   - 如果有 'ema'，走原来 EMA 推理路径（catif 训练期的 ckpt）
    #   - 如果没有 'ema'，但有 'model'，说明是 RL 后保存的策略权重，直接加载到 diffusion
    if 'ema' in ckpt:
        ema = EMA(diffusion)
        ema.load_state_dict(ckpt['ema'])
        model = ema.ema_model.to(device).eval()
    elif 'model' in ckpt:
        diffusion.load_state_dict(ckpt['model'], strict=False)
        model = diffusion.to(device).eval()
    else:
        raise RuntimeError(
            f"Checkpoint {args.ckpt_path} 既不包含 'ema' 也不包含 'model'，"
            "无法加载参数。"
        )

    # 5) 遍历每个条件图，按 {ProID, SMILES} 确定 group，并采样 K 条
    rows_to_write = []
    name_to_index = {name: i for i, name in enumerate(cond_names)}

    # 统计全局 recovery / perplexity
    total_samples = 0
    sum_recovery = 0.0
    sum_perplexity = 0.0

    for cond_idx, cond_name in enumerate(cond_names):
        data_index = name_to_index[cond_name]
        data = dataset[data_index]

        # 先推断 WT 的 ProID 和 SMILES
        proid = None
        smiles = None
        # 1) 如果 pairs.csv 给了 cond_name 精确映射，就反查 ProID
        for pid, info in by_proid.items():
            cn = info.get('cond_name', '')
            if cn and cn == cond_name:
                proid = pid
                smiles = info['SMILES']
                break
        # 2) 否则用 {ProID}.pt 规则猜测
        if proid is None:
            hit = [pid for pid in by_proid.keys()
                   if guess_cond_name_for_proid(pid, {cond_name}) is not None]
            if len(hit) == 1:
                proid = hit[0]
                smiles = by_proid[proid]['SMILES']
            else:
                # 找不到 ProID→cond 的映射，跳过这个条件
                continue

        wt_seq = data_to_sequence(data)

        # 对该条件图采样 K 条
        for k in range(args.group_size):
            per_sample_seed = (args.seed if args.seed is not None else 0) + k
            set_seed(per_sample_seed)

            single = Batch.from_data_list([data.clone()]).to(device)
            with torch.no_grad():
                _, pred_onehot = model.ddim_sample(
                    single,
                    diverse=True,       # 组内想要差异化，就强制启用随机
                    step=args.step
                )
            oh = pred_onehot[:, :20]
            idx = oh.argmax(dim=-1).tolist()
            seq = ''.join(AMINO_CODES[i] for i in idx)

            # --------- 计算本条的 recovery & perplexity ----------
            rec = sequence_recovery(wt_seq, seq)
            ppl = sequence_perplexity(seq)

            total_samples += 1
            sum_recovery += rec
            # ppl 可能为 NaN，这里简单跳过 NaN
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
                'seed': per_sample_seed,     # 记录每条的实际种子
                'step': args.step,
                'ckpt_path': args.ckpt_path,
                'recovery': rec,
                'perplexity': ppl,
            })

            # 实时打印进度
            if args.log_every > 0 and (total_samples % args.log_every == 0):
                mean_rec = sum_recovery / total_samples if total_samples > 0 else 0.0
                mean_ppl = sum_perplexity / total_samples if total_samples > 0 else float('nan')
                print(
                    f"[{total_samples} samples] "
                    f"last_rec={rec*100:.2f}% last_ppl={ppl:.3f} | "
                    f"mean_rec={mean_rec*100:.2f}% mean_ppl={mean_ppl:.3f}"
                )

    # 6) 写出 CSV
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

    # 最终平均值输出
    if total_samples > 0:
        mean_rec = sum_recovery / total_samples
        mean_ppl = sum_perplexity / total_samples
        print(f"✅ Exported {len(rows_to_write)} rows to {out_csv}")
        print(
            f"📊 Overall stats: samples={total_samples}, "
            f"mean_recovery={mean_rec*100:.2f}%, mean_perplexity={mean_ppl:.3f}"
        )
    else:
        print(f"⚠ No samples exported. Please check your inputs. (out_csv={out_csv})")

if __name__ == "__main__":
    main()
