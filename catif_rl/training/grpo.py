#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Editdate : Jan 30
# Author   : Yanheng Li
# Device   : NVIDIA RTX 5060 Ti (16 GB)

# =============================================================================
# grpo_trainer6.py -- main changes vs. grpo_trainer4.py
# -----------------------------------------------------------------------------
# 1) Offline GRPO training: rewards / predictors are no longer called during
#    training; the policy is fine-tuned from a pre-scored CSV that already
#    contains delta / reward.
# 2) Mutant-sequence injection and tracking: mutant sequences from the CSV
#    are turned directly into onehot inputs, and mutation_fraction / recovery
#    are computed for the penalty term and training diagnostics.
# 3) Token-averaged policy gradient: the PG term uses mean_logp (averaged per
#    token) instead of the length-summed logp, preventing long sequences from
#    dominating the gradient.
# 4) Stable KL surrogate: KL is a clipped MSE surrogate on
#    (mean_logp - ref_mean_logp), combined with an adaptive beta (with an
#    upper bound) to keep the policy drift in check.
# 5) Diagnostics for "are we actually learning?": added per-group
#    icorr(reward, mean_logp) and top-bottom gap, plus loss / pg_term /
#    mean_logp / ref_mean_logp in the logs, summaries, and plots.
# 6) Plot redesign: epochXX_steps.png uses scatter plots only (groups appear
#    in random order, which avoids spurious trend lines), and now focuses on
#    loss / pg_term / dlogP / icorr / gap.
# 7) Learnability filter and dedup: within each group we dedup by ProSeq',
#    and we only keep groups with >= 3 distinct reward values, so that there
#    is a real within-group ranking signal to learn from.
# 8) Stronger weight / config compatibility: model config is resolved
#    uniformly from either the checkpoint root or meta.config, supporting
#    both the original CatIF weights and the RL-saved policy_epochXX.pt.
# =============================================================================

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import math
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.cuda.amp import GradScaler, autocast
from torch_geometric.data import Batch, Data
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from catif_rl.models.gradeif_app import EGNN_NET, GraDe_IF
from catif_rl.data.large_dataset import Cath


# -------------------------
# Constants
# -------------------------

AMINO_CODES = ['A','R','N','D','C','Q','E','G','H','I',
               'L','K','M','F','P','S','T','W','Y','V']
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_CODES)}


@dataclass
class SampleBatch:
    batch: Batch
    onehot: Tensor
    seqs: List[str]
    mut_frac: Tensor


@dataclass
class OfflineSample:
    cond_idx: int
    group_key: str      # e.g., "ProID||SMILES"
    seq: str
    mean_delta: float
    reward: Optional[float] = None
    mut_frac: Optional[float] = None


# -------------------------
# Utilities
# -------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model_from_config(config: Dict[str, object], sample: Data) -> Tuple[EGNN_NET, GraDe_IF]:
    extra_dim_t = getattr(sample, 'extra_x', None)
    extra_dim = int(extra_dim_t.shape[1]) if isinstance(extra_dim_t, torch.Tensor) else 0
    input_dim = int(config.get('input_feat_dim', sample.x.shape[1] + extra_dim))
    edge_dim = int(config.get('edge_attr_dim', sample.edge_attr.shape[1]))
    hidden = int(config.get('hidden_dim', 256))
    dropout = float(config.get('drop_out', 0.0))
    depth = int(config.get('depth', 1))
    embedding = bool(config.get('embedding', False))
    embed_dim = int(config.get('embedding_dim', 16))
    embed_ss = int(config.get('embed_ss', -1))
    update_edge = bool(config.get('update_edge', True))
    norm_feat = bool(config.get('norm_feat', False))
    noise_type = str(config.get('noise_type', 'blosum'))
    timesteps = int(config.get('timesteps', 500))
    objective = str(config.get('objective', 'pred_x0'))
    base = EGNN_NET(
        input_feat_dim=input_dim,
        hidden_channels=hidden,
        edge_attr_dim=edge_dim,
        dropout=dropout,
        n_layers=depth,
        update_edge=update_edge,
        embedding=embedding,
        embedding_dim=embed_dim,
        norm_feat=norm_feat,
        embed_ss=embed_ss
    )
    diffusion = GraDe_IF(base, timesteps=timesteps, objective=objective,
                         config={'noise_type': noise_type, **config})
    return base, diffusion

def extract_config(ckpt: Dict[str, object]) -> Dict[str, object]:
    """
    Support two checkpoint formats:
    1) Original catif: { 'model': ..., 'config': {...}, 'ema': ... }
    2) Post-RL:        { 'model': ..., 'meta': { 'config': {...}, ... } }
    """
    # Case 1: top-level config (original catif weights)
    cfg = ckpt.get('config')
    if isinstance(cfg, dict):
        return cfg

    # Case 2: RL trainer's policy_epochXX.pt
    meta = ckpt.get('meta')
    if isinstance(meta, dict):
        cfg2 = meta.get('config')
        if isinstance(cfg2, dict):
            return cfg2

    raise RuntimeError(
        "Could not find config in checkpoint; "
        "expected either an original catif weight or an RL-saved policy_epochXX.pt."
    )


def data_to_sequence(data: Data) -> str:
    idx = data.x[:, :20].argmax(dim=-1).tolist()
    return ''.join(AMINO_CODES[i] for i in idx)


def mutation_fraction(batch: Batch, onehot: Tensor) -> Tensor:
    wt_idx = batch.x[:, :20].argmax(dim=-1)
    sample_idx = onehot[:, :20].argmax(dim=-1)
    diff = (wt_idx != sample_idx).float()
    counts = torch.zeros(batch.num_graphs, device=onehot.device)
    lengths = torch.zeros(batch.num_graphs, device=onehot.device)
    counts.scatter_add_(0, batch.batch, diff)
    lengths.scatter_add_(0, batch.batch, torch.ones_like(diff))
    return counts / lengths.clamp_min(1.0)


def batch_mutant_onehot_from_seqs(batch: Batch, seqs: List[str]) -> Tensor:
    device = batch.x.device
    x_mut = batch.x.clone()
    ptr = batch.ptr.tolist()
    if len(ptr) - 1 != len(seqs):
        raise ValueError(f"number of graphs ({len(ptr)-1}) != number of seqs ({len(seqs)})")
    for g_idx, seq in enumerate(seqs):
        start, end = ptr[g_idx], ptr[g_idx + 1]
        if (end - start) != len(seq):
            raise ValueError(
                f"seq length mismatch: graph {g_idx} has {end-start} nodes, "
                f"but seq length = {len(seq)}"
            )
        x_mut[start:end, :20] = 0.0
        aa_idx = [AA_TO_IDX.get(a, None) for a in seq]
        if any(i is None for i in aa_idx):
            bad = {a for a, i in zip(seq, aa_idx) if i is None}
            raise ValueError(f"invalid amino acids in seq: {bad}")
        rows = torch.arange(start, end, device=device)
        cols = torch.tensor(aa_idx, device=device)
        x_mut[rows, cols] = 1.0
    return x_mut


# >>> NEW: compute_log_prob now also returns "sequence logP and mean token logP"; used later for perplexity
def compute_log_prob(diffusion: GraDe_IF, sample: SampleBatch, step: int,
                     requires_grad: bool) -> Tuple[Tensor, Tensor]:
    """
    Returns:
      seq_logp:  per-sequence log p(seq) = sum_token log p(a_t)
      mean_logp: per-sequence mean token log p = seq_logp / L
    """
    data = sample.batch.clone()
    data.x = sample.onehot
    num_graphs = data.num_graphs
    t_val = torch.full((num_graphs, 1), fill_value=step,
                       device=data.x.device, dtype=data.x.dtype)
    ctx = torch.enable_grad() if requires_grad else torch.no_grad()
    with ctx:
        logits = diffusion.model(data, t_val)
        log_probs = F.log_softmax(logits, dim=-1)
        idx = sample.onehot[:, :20].argmax(dim=-1, keepdim=True)
        token_logp = log_probs.gather(1, idx).squeeze(1)  # [num_nodes]

        # Aggregate per graph
        seq_logp = torch.zeros(num_graphs, device=data.x.device)
        lengths = torch.zeros(num_graphs, device=data.x.device)
        seq_logp.scatter_add_(0, data.batch, token_logp)
        lengths.scatter_add_(0, data.batch, torch.ones_like(token_logp))
        mean_logp = seq_logp / lengths.clamp_min(1.0)
    return seq_logp, mean_logp
# <<< NEW


def reward_from_delta(mean_delta: Tensor,
                      mode: str,
                      threshold: float,
                      tau: float,
                      mut_frac: Optional[Tensor],
                      free_frac: float,
                      penalty: float) -> Tensor:
    if mode == 'bin':
        reward = (mean_delta >= threshold).float()
    elif mode == 'lin':
        reward = torch.clamp((mean_delta - threshold) / tau, min=0.0, max=1.0)
    elif mode == 'lin_sym':
        reward = torch.clamp(mean_delta / tau, min=-1.0, max=1.0)
    elif mode == 'bin3':
        reward = torch.zeros_like(mean_delta)
        reward = torch.where(mean_delta >= threshold, torch.ones_like(reward), reward)
        reward = torch.where(mean_delta <= -threshold, -torch.ones_like(reward), reward)
    else:
        raise ValueError(f'unknown reward mode {mode}')

    if mut_frac is not None and penalty > 0:
        reward = reward - penalty * torch.relu(mut_frac - free_frac)
    return reward


def compute_group_advantages(reward: Tensor, group_ids: Tensor) -> Tensor:
    unique = group_ids.unique()
    baseline = torch.zeros_like(reward)
    for g in unique:
        idx = (group_ids == g)
        if idx.any():
            baseline[idx] = reward[idx].mean()
    adv = reward - baseline
    std = adv.std(unbiased=False)
    if torch.isnan(std) or std < 1e-6:
        return adv
    return (adv - adv.mean()) / (std + 1e-6)


def _safe_corr_pearson(x: Tensor, y: Tensor) -> float:
    """
    Pearson corr between 1D tensors; returns NaN if ill-defined.
    """
    x = x.detach().float().reshape(-1)
    y = y.detach().float().reshape(-1)
    if x.numel() < 2 or y.numel() < 2:
        return float('nan')
    if torch.isnan(x).any() or torch.isnan(y).any():
        return float('nan')
    x = x - x.mean()
    y = y - y.mean()
    denom = (x.std(unbiased=False) * y.std(unbiased=False)).item()
    if (not math.isfinite(denom)) or denom < 1e-8:
        return float('nan')
    return float((x * y).mean().item() / denom)


def _top_bottom_gap(reward: Tensor, score: Tensor, frac: float = 0.2) -> float:
    """
    gap = mean(score[top reward]) - mean(score[bottom reward])
    """
    r = reward.detach().float().reshape(-1)
    s = score.detach().float().reshape(-1)
    n = r.numel()
    if n < 2:
        return float('nan')
    k = max(1, int(round(n * frac)))
    # top-k by reward
    top_idx = torch.topk(r, k=k, largest=True).indices
    bot_idx = torch.topk(r, k=k, largest=False).indices
    return float(s[top_idx].mean().item() - s[bot_idx].mean().item())
def dedup_and_filter_groups(
    groups: Dict[str, List[OfflineSample]],
    reward_distinct_min: int = 3
) -> Dict[str, List[OfflineSample]]:
    """
    1) within each group, deduplicate by seq (ProSeq')
    2) keep group only if distinct reward-count > 2
       - if sample.reward exists, use it; else fall back to mean_delta
    """
    out: Dict[str, List[OfflineSample]] = {}
    for gk, recs in groups.items():
        if not recs:
            continue
        # dedup by seq, keep first occurrence
        seen = set()
        uniq: List[OfflineSample] = []
        for r in recs:
            s = r.seq
            if s in seen:
                continue
            seen.add(s)
            uniq.append(r)
        if len(uniq) < 2:
            continue
        vals = []
        for r in uniq:
            v = r.reward if (r.reward is not None) else r.mean_delta
            vals.append(float(v))
        # distinct count
        distinct = len(set(vals))
        if distinct >= reward_distinct_min:
            out[gk] = uniq
    return out

def save_policy(diffusion: GraDe_IF, path: Path, meta: Dict[str, object]) -> None:
    payload = {'model': diffusion.state_dict(), 'meta': meta}
    torch.save(payload, path)


# -------------------------
# CSV loading & grouping
# -------------------------

def load_offline_samples(csv_path: Path,
                         cond_name_to_idx: Dict[str, int]) -> List[OfflineSample]:
    samples: List[OfflineSample] = []
    with csv_path.open('r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        required_cols = {'group', "ProSeq'"}
        if not required_cols.issubset(reader.fieldnames or []):
            raise RuntimeError(f"CSV must contain columns: {required_cols}")
        for row in reader:
            if 'data_index' in row and row['data_index'] != '':
                cond_idx = int(float(row['data_index']))
            elif 'cond_name' in row and row['cond_name'] != '':
                name = row['cond_name']
                if name not in cond_name_to_idx:
                    raise KeyError(f"cond_name {name} not found in condition_dir")
                cond_idx = cond_name_to_idx[name]
            else:
                raise RuntimeError("Each row must have either data_index or cond_name")

            group_key = row['group'].strip()
            seq = row["ProSeq'"].strip()

            if 'mean_delta' in row and row['mean_delta'] != '':
                mean_delta = float(row['mean_delta'])
            elif 'delta' in row and row['delta'] != '':
                mean_delta = float(row['delta'])
            else:
                raise RuntimeError("Each row must have mean_delta or delta column")

            reward = float(row['reward']) if 'reward' in row and row['reward'] != '' else None
            mut_frac = float(row['mut_frac']) if 'mut_frac' in row and row['mut_frac'] != '' else None

            samples.append(
                OfflineSample(
                    cond_idx=cond_idx,
                    group_key=group_key,
                    seq=seq,
                    mean_delta=mean_delta,
                    reward=reward,
                    mut_frac=mut_frac,
                )
            )
    return samples


def group_samples_by_key(samples: List[OfflineSample]) -> Dict[str, List[OfflineSample]]:
    groups: Dict[str, List[OfflineSample]] = {}
    for s in samples:
        groups.setdefault(s.group_key, []).append(s)
    return groups


# -------------------------
# Plotting utilities
# -------------------------

def _save_epoch_step_plots(out_dir: Path,
                           epoch_idx: int,
                           step_stats: Dict[str, List[float]]) -> None:
    # Step-level plots: keep only the most diagnostic signals (avoid low-signal ref_mean_logp)
    fig, axes = plt.subplots(4, 3, figsize=(14, 12))
    axes = axes.ravel()
    # We plot dlogp = mean_logp - ref_mean_logp instead of ref_mean_logp alone.
    keys = [
        'reward', 'hit', 'kl', 'beta', 'delta', 'mut',
        'loss', 'pg_term', 'dlogp', 'icorr', 'gap'
    ]
    titles = [
        'Reward(mean)', 'Hit(>0)', 'KL(proxy)', 'Beta', 'Mean delta', 'Mutation frac',
        'Loss', 'PG term', 'dlogP (mean_logp - ref_mean_logp)',
        'icorr(reward, mean_logp)', 'Top-Bottom gap'
    ]

    for ax, k, t in zip(axes, keys, titles):
        if k in step_stats and len(step_stats[k]) > 0:
            ys = np.array(step_stats[k], dtype=float)
            xs = np.arange(1, len(ys) + 1)
            # Scatter is more faithful here because "steps" are shuffled groups, not a smooth time series.
            ax.scatter(xs, ys, s=10)
            ax.set_xlabel('Group step')
            ax.set_ylabel(k)
            ax.set_title(t)
    # hide unused axes
    for ax in axes[len(keys):]:
        ax.axis('off')
    plt.tight_layout()
    fig.savefig(out_dir / f'epoch{epoch_idx:02d}_steps.png', dpi=600)
    plt.close(fig)


def _save_epoch_agg_plots(out_dir: Path,
                          hist_epochs: Dict[str, List[float]]) -> None:
    # Build "panels" explicitly to avoid zip-truncation bugs when metrics grow.
    panels: List[Tuple[str, str]] = [
        ('reward', 'Reward(mean/Epoch)'),
        ('hit', 'Hit(Epoch)'),
        ('kl', 'KL(proxy/Epoch)'),
        ('beta', 'Beta(Epoch end)'),
        ('delta', 'Mean delta(Epoch)'),
        ('mut', 'Mutation frac(Epoch)'),
        ('rec', 'Recovery(Epoch)'),
        ('loss', 'Loss(Epoch)'),
        # Special panels below: logP pair + dlogp
        ('logp_pair', 'Mean/Ref mean token logP(Epoch)'),
        ('dlogp', 'dlogP (mean_logp - ref_mean_logp) (Epoch)'),
        ('icorr', 'mean_logp_icorr(Epoch)'),
        ('gap', 'Top-Bottom gap(Epoch)'),
    ]
    n = len(panels)
    ncol = 2
    nrow = int(math.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 2.8 * nrow))
    axes = np.array(axes).ravel()

    for i, (key, title) in enumerate(panels):
        ax = axes[i]
        ax.set_title(title)
        ax.set_xlabel('Epoch')

        if key == 'logp_pair':
            # plot mean_logp & ref_mean_logp on same axis
            if ('mean_logp' in hist_epochs and len(hist_epochs['mean_logp']) > 0) or \
               ('ref_mean_logp' in hist_epochs and len(hist_epochs['ref_mean_logp']) > 0):
                if 'mean_logp' in hist_epochs and len(hist_epochs['mean_logp']) > 0:
                    ys = np.array(hist_epochs['mean_logp'], dtype=float)
                    xs = np.arange(1, len(ys) + 1)
                    ax.plot(xs, ys, marker='o', label='mean_logp')
                if 'ref_mean_logp' in hist_epochs and len(hist_epochs['ref_mean_logp']) > 0:
                    ys = np.array(hist_epochs['ref_mean_logp'], dtype=float)
                    xs = np.arange(1, len(ys) + 1)
                    ax.plot(xs, ys, marker='o', label='ref_mean_logp')
                ax.legend()
            ax.set_ylabel('logp')
            continue

        # Normal 1-series panels
        if key in hist_epochs and len(hist_epochs[key]) > 0:
            ys = np.array(hist_epochs[key], dtype=float)
            xs = np.arange(1, len(ys) + 1)
            ax.plot(xs, ys, marker='o')
        ax.set_ylabel(key)

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    fig.savefig(out_dir / 'training_summary.png', dpi=600)
    plt.close(fig)


# -------------------------
# Training entry point
# -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline GRPO fine-tuning for GraDe_IF using pre-scored CSV."
    )
    p.add_argument('--policy-ckpt', required=True)
    p.add_argument('--ref-ckpt', required=True)
    p.add_argument('--condition-dir', required=True)
    p.add_argument('--scored-csv', required=True)
    p.add_argument('--output-dir', default='runs/grpo_offline',
                   help='where to write policy_epochXX.pt, train_log.jsonl, and per-epoch plots')

    p.add_argument('--device', default='cuda:0')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--max-groups', type=int, default=None)

    p.add_argument('--sample-step', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--weight-decay', type=float, default=1e-2)
    p.add_argument('--beta', type=float, default=0.01)
    p.add_argument('--kl-target', type=float, default=0.05)

    p.add_argument('--reward-mode', choices=['bin', 'lin', 'lin_sym', 'bin3'], default='bin3')
    p.add_argument('--warmup-epochs', type=int, default=0)
    p.add_argument('--reward-threshold', type=float, default=0.01)
    p.add_argument('--reward-tau', type=float, default=0.05)
    p.add_argument('--mutation-penalty', type=float, default=0.2)
    p.add_argument('--mutation-free-frac', type=float, default=0.05)

    p.add_argument('--accum-steps', type=int, default=1)
    p.add_argument('--grad-clip', type=float, default=2.0)
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save-every', type=int, default=1)

    # Numerical stability knobs
    p.add_argument('--kl-clip', type=float, default=5.0,
                   help='clip |logp - ref_logp| before squaring for KL proxy')
    p.add_argument('--max-beta', type=float, default=0.5,
                   help='maximum value for adaptive KL coefficient beta')

    # Plotting
    p.add_argument('--no-plots', action='store_true', help='disable plotting')
    p.add_argument('--plot-every', type=int, default=1, help='plot every N epochs')

    # Group filtering (>=3 distinct reward values are kept for training)
    p.add_argument('--min-reward-distinct', type=int, default=3,
                   help='only train on groups with distinct reward count >= this (after seq dedup)')
    return p.parse_args()


def train_offline() -> None:
    args = parse_args()
    set_seed(args.seed)

    command_line = ' '.join(sys.argv)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'train_log.jsonl'
    log_file = log_path.open('w')
    summary_path = out_dir / 'training_summary.jsonl'
    summary_file = summary_path.open('a')
    summary_header = {
        'event': 'run_start',
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'command': command_line,
        'output_dir': str(out_dir),
        'scored_csv': str(Path(args.scored_csv).resolve()),
    }
    summary_file.write(json.dumps(summary_header, ensure_ascii=False) + '\n')

    # ---------- Condition graphs ----------
    cond_paths = sorted(f for f in Path(args.condition_dir).glob('*.pt'))
    if not cond_paths:
        raise RuntimeError('no conditioning graphs found in condition_dir')
    cond_names = [p.name for p in cond_paths]
    cond_name_to_idx = {name: i for i, name in enumerate(cond_names)}
    dataset = Cath(cond_names, args.condition_dir)
    sample_data = dataset[0]

    # ---------- Policy ----------
    policy_ckpt = torch.load(args.policy_ckpt, map_location='cpu')
    # Resolve config from ckpt uniformly (supports catif & policy_epochXX.pt)
    config = extract_config(policy_ckpt)

    # Build the model using the policy ckpt's config (the ref uses the same shape)
    _, policy_diffusion = build_model_from_config(config, sample_data)
    policy_diffusion.load_state_dict(policy_ckpt['model'], strict=False)
    policy_diffusion.to(device)
    policy_diffusion.train()

    ref_ckpt = torch.load(args.ref_ckpt, map_location='cpu')
    _, ref_diffusion = build_model_from_config(config, sample_data)
    ref_diffusion.load_state_dict(ref_ckpt['model'], strict=False)
    ref_diffusion.to(device)
    ref_diffusion.eval()
    for p in ref_diffusion.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        policy_diffusion.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98)
    )
    scaler = GradScaler(enabled=(not args.no_amp) and device.type == 'cuda')

    # ---------- Data ----------
    scored_csv = Path(args.scored_csv)
    samples = load_offline_samples(scored_csv, cond_name_to_idx)
    if not samples:
        raise RuntimeError('no samples found in scored-csv')

    group2id: Dict[str, int] = {gk: i for i, gk in enumerate(sorted({s.group_key for s in samples}))}
    groups = group_samples_by_key(samples)
    n_groups_raw = len(groups)
    # dedup by seq within group + filter by distinct reward count
    groups = dedup_and_filter_groups(groups, reward_distinct_min=args.min_reward_distinct)
    n_groups_kept = len(groups)
    group_keys_all = list(groups.keys())
    print(f'[filter] groups: raw={n_groups_raw} -> kept={n_groups_kept} '
          f'(min_reward_distinct={args.min_reward_distinct}, seq_dedup=on)')

    beta = args.beta
    beta_min = 5e-4  # lower bound
    global_step = 0

    hist_epochs = {'reward': [], 'hit': [], 'kl': [], 'beta': [], 'delta': [], 'mut': [], 'rec': [],
                   'loss': [], 'mean_logp': [], 'ref_mean_logp': [], 'dlogp': [],
                   'icorr': [], 'gap': []}

    # >>> NEW: global recovery stats
    global_rec_sum = 0.0
    global_n_samples = 0
    # <<< NEW

    for epoch in range(args.epochs):
        reward_mode = 'lin_sym' if epoch < args.warmup_epochs else args.reward_mode
        random.shuffle(group_keys_all)
        group_iter = group_keys_all[:min(args.max_groups, len(group_keys_all))] if args.max_groups else group_keys_all

        step_stats = {'reward': [], 'hit': [], 'kl': [], 'beta': [], 'delta': [], 'mut': [],
                      'loss': [], 'pg_term': [], 'mean_logp': [], 'ref_mean_logp': [], 'icorr': [], 'gap': []}
        epoch_rewards: List[float] = []
        pos_hits: List[float] = []
        kl_values: List[float] = []
        delta_values: List[float] = []
        mut_values: List[float] = []
        loss_values: List[float] = []
        mean_logp_values: List[float] = []
        ref_mean_logp_values: List[float] = []
        icorr_values: List[float] = []
        gap_values: List[float] = []

        # >>> NEW: per-epoch recovery
        rec_epoch_sum = 0.0
        epoch_n_samples = 0
        # <<< NEW

        optimizer.zero_grad(set_to_none=True)
        accum = 0

        pbar = tqdm(group_iter, desc=f"Epoch {epoch+1:02d}", leave=True)
        for g_key in pbar:
            recs = groups[g_key]
            if not recs:
                continue

            cond_idx = recs[0].cond_idx
            data = dataset[cond_idx]

            clones = [data.clone() for _ in recs]
            batch_wt = Batch.from_data_list(clones).to(device)

            seqs = [r.seq for r in recs]
            try:
                x_mut = batch_mutant_onehot_from_seqs(batch_wt, seqs)
            except ValueError as e:
                pbar.write(f"[warn] skip group {g_key} due to: {e}")
                continue

            mut_frac_tensor = mutation_fraction(batch_wt, x_mut)
            batch_mut = batch_wt.clone()
            batch_mut.x = x_mut
            sample_batch = SampleBatch(batch=batch_mut, onehot=x_mut, seqs=seqs, mut_frac=mut_frac_tensor)

            mean_delta = torch.tensor([r.mean_delta for r in recs], dtype=torch.float32, device=device)
            if all(r.reward is not None for r in recs):
                reward = torch.tensor([r.reward for r in recs], dtype=torch.float32, device=device)
            else:
                reward = reward_from_delta(mean_delta, reward_mode, args.reward_threshold, args.reward_tau,
                                           mut_frac=mut_frac_tensor.to(device),
                                           free_frac=args.mutation_free_frac,
                                           penalty=args.mutation_penalty)

            group_ids_tensor = torch.full((len(recs),), fill_value=group2id[g_key],
                                          dtype=torch.long, device=device)
            advantages = compute_group_advantages(reward, group_ids_tensor)

            # Numerical guard: bail out early if reward / adv has NaN
            if torch.isnan(reward).any() or torch.isnan(advantages).any():
                pbar.write(f"[warn] skip group {g_key} due to NaN reward/adv.")
                continue

            loss_value = None  # will fill after forward

            with autocast(enabled=scaler.is_enabled()):
                # >>> NEW: get both seq_logp and mean_logp
                logp, mean_logp = compute_log_prob(
                    policy_diffusion, sample_batch, step=args.sample_step, requires_grad=True
                )
                with torch.no_grad():
                    ref_logp, ref_mean_logp = compute_log_prob(
                        ref_diffusion, sample_batch, step=args.sample_step, requires_grad=False
                    )
                # <<< NEW

                # Sanity check on logp / ref_logp
                if torch.isnan(logp).any() or torch.isnan(ref_logp).any() \
                   or torch.isinf(logp).any() or torch.isinf(ref_logp).any():
                    pbar.write(f"[warn] skip group {g_key} due to NaN/Inf in logp.")
                    continue

                # KL surrogate: clip |logp - ref_logp| first, then mean-of-squares
                kl_sample = (mean_logp - ref_mean_logp)  # per-sequence mean token logp difference
                kl_sample = torch.clamp(kl_sample, -args.kl_clip, args.kl_clip)
                kl = (kl_sample ** 2).mean()

                if torch.isnan(kl) or torch.isinf(kl):
                    pbar.write(f"[warn] skip group {g_key} due to NaN/Inf KL.")
                    continue

                # policy-gradient term (length-normalized): use mean_logp instead of seq_logp
                pg_term = -(advantages.detach() * mean_logp).mean()
                loss = pg_term + beta * kl
                loss = loss / args.accum_steps
                # keep a fp32 scalar for logs
                loss_value = float(loss.detach().float().item())

            scaler.scale(loss).backward()
            accum += 1
            if accum % args.accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(policy_diffusion.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            reward_mean = reward.mean().item()
            hit_rate = (reward > 0).float().mean().item()
            kl_value = float(kl.detach().item())
            delta_mean = mean_delta.mean().item()
            mut_mean = mut_frac_tensor.mean().item()

            # group-level learning diagnostics
            mean_logp_mean = float(mean_logp.detach().float().mean().item())
            ref_mean_logp_mean = float(ref_mean_logp.detach().float().mean().item())
            pg_value = float(pg_term.detach().float().item())
            # per-sample correlation and gap (within this group)
            icorr = _safe_corr_pearson(reward, mean_logp)
            gap = _top_bottom_gap(reward, mean_logp, frac=0.2)

            # logp stats (these DO change as policy moves)
            mean_logp_mean = float(mean_logp.detach().float().mean().item())
            ref_mean_logp_mean = float(ref_mean_logp.detach().float().mean().item())
            pg_value = float(pg_term.detach().float().item())
            if loss_value is None:
                loss_value = float((pg_term + beta * kl).detach().float().item())

            epoch_rewards.append(reward_mean)
            pos_hits.append(hit_rate)
            kl_values.append(kl_value)
            delta_values.append(delta_mean)
            mut_values.append(mut_mean)
            loss_values.append(loss_value)
            mean_logp_values.append(mean_logp_mean)
            ref_mean_logp_values.append(ref_mean_logp_mean)
            if math.isfinite(icorr):
                icorr_values.append(icorr)
            if math.isfinite(gap):
                gap_values.append(gap)

            # >>> NEW: recovery from mut_frac
            # recovery = 1 - mutation_fraction
            rec_vec = (1.0 - mut_frac_tensor).detach()               # [B]
            rec_mean = rec_vec.mean().item()

            # Accumulate per-epoch + global
            batch_size = len(recs)
            rec_epoch_sum += float(rec_vec.sum().item())
            epoch_n_samples += batch_size

            global_rec_sum += float(rec_vec.sum().item())
            global_n_samples += batch_size
            # <<< NEW

            # beta adapts within bounds
            if kl_value > args.kl_target * 1.5:
                beta *= 1.5
            elif kl_value < args.kl_target * 0.5:
                beta *= 0.7
            beta = float(min(max(beta, beta_min), args.max_beta))

            pbar.set_postfix({
                'reward': f"{reward_mean:+.3f}",
                'hit': f"{hit_rate:.3f}",
                'kl': f"{kl_value:.4f}",
                'beta': f"{beta:.3e}",
                # >>> NEW: live recovery in tqdm
                'rec': f"{rec_mean:.3f}",
                # <<< NEW
                # policy movement signals
                'loss': f"{loss_value:.3e}",
                'mlogp': f"{mean_logp_mean:.3f}",
                'icorr': f"{icorr:.3f}" if math.isfinite(icorr) else "nan",
                'gap': f"{gap:.3f}" if math.isfinite(gap) else "nan",
            })

            step_stats['reward'].append(reward_mean)
            step_stats['hit'].append(hit_rate)
            step_stats['kl'].append(kl_value)
            step_stats['beta'].append(beta)
            step_stats['delta'].append(delta_mean)
            step_stats['mut'].append(mut_mean)
            step_stats['loss'].append(loss_value)
            step_stats['pg_term'].append(pg_value)
            step_stats['mean_logp'].append(mean_logp_mean)
            step_stats['ref_mean_logp'].append(ref_mean_logp_mean)
            # dlogp for step plots: policy vs ref drift at token-mean scale
            try:
                dlogp = float(mean_logp_mean - ref_mean_logp_mean)
            except Exception:
                dlogp = float('nan')
            if 'dlogp' not in step_stats:
                step_stats['dlogp'] = []
            if math.isfinite(dlogp):
                step_stats['dlogp'].append(dlogp)
            if math.isfinite(icorr): step_stats['icorr'].append(icorr)
            if math.isfinite(gap): step_stats['gap'].append(gap)

            with torch.no_grad():
                q10, q50, q90 = torch.quantile(
                    reward, torch.tensor([0.1, 0.5, 0.9], device=device)
                ).tolist()
            log_record = {
                'epoch': epoch + 1,
                'group_key': str(g_key),
                'cond_index': int(cond_idx),
                'reward_mean': reward_mean,
                'reward_hit_rate': hit_rate,
                'reward_q10': float(q10),
                'reward_q50': float(q50),
                'reward_q90': float(q90),
                'kl': kl_value,
                'beta': beta,
                'delta_mean': delta_mean,
                'mutation_frac_mean': mut_mean,
                # >>> NEW: record per-batch rec in the log too (optional)
                'recovery_mean': rec_mean,
                # <<< NEW
                # >>> NEW: signals that reflect policy movement under offline fixed rewards
                'loss': loss_value,
                'pg_term': pg_value,
                'mean_logp_mean': mean_logp_mean,
                'ref_mean_logp_mean': ref_mean_logp_mean,
                'corr_r_logp': icorr,
                'top_bottom_gap': gap,
                # <<< NEW
                'step': global_step,
                'reward_mode': reward_mode,
            }
            log_file.write(json.dumps(log_record, ensure_ascii=False) + '\n')
            log_file.flush()

        def avg(xs: List[float]) -> float:
            return float(sum(xs) / max(len(xs), 1))

        reward_epoch = avg(epoch_rewards)
        hit_epoch = avg(pos_hits)
        kl_epoch = avg(kl_values)
        delta_epoch = avg(delta_values)
        mut_epoch = avg(mut_values)
        loss_epoch = avg(loss_values)
        mean_logp_epoch = avg(mean_logp_values)
        ref_mean_logp_epoch = avg(ref_mean_logp_values)
        dlogp_epoch = float(mean_logp_epoch - ref_mean_logp_epoch)
        # for icorr/gap we averaged only finite ones
        icorr_epoch = avg(icorr_values) if len(icorr_values) > 0 else float('nan')
        gap_epoch = avg(gap_values) if len(gap_values) > 0 else float('nan')
        # >>> NEW: per-epoch mean recovery
        rec_epoch = rec_epoch_sum / max(epoch_n_samples, 1)
        # <<< NEW

        print(
            f'Epoch {epoch+1:02d} | reward={reward_epoch:.4f} | '
            f'hit={hit_epoch:.3f} | kl={kl_epoch:.4f} | '
            f'delta={delta_epoch:.4f} | mut={mut_epoch:.4f} | '
            f'rec={rec_epoch:.4f} | '  # >>> NEW
            f'loss={loss_epoch:.4f} | mlogp={mean_logp_epoch:.4f} | rmlogp={ref_mean_logp_epoch:.4f} | '
            f'icorr={icorr_epoch:.4f} | gap={gap_epoch:.4f} | '
            f'beta={beta:.4f} | mode={reward_mode}'
        )

        # Append to training_summary.jsonl
        summary_epoch = {
            'event': 'epoch_end',
            'epoch': epoch + 1,
            'reward': reward_epoch,
            'hit': hit_epoch,
            'kl': kl_epoch,
            'delta': delta_epoch,
            'mut': mut_epoch,
            'rec': rec_epoch,
            'loss': loss_epoch,
            'mean_logp': mean_logp_epoch,
            'ref_mean_logp': ref_mean_logp_epoch,
            'dlogp': float(mean_logp_epoch - ref_mean_logp_epoch),
            'mean_logp_icorr': icorr_epoch,
            'top_bottom_gap': gap_epoch,
            'beta': beta,
            'mode': reward_mode,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
        }
        summary_file.write(json.dumps(summary_epoch, ensure_ascii=False) + '\n')
        summary_file.flush()

        hist_epochs['reward'].append(reward_epoch)
        hist_epochs['hit'].append(hit_epoch)
        hist_epochs['kl'].append(kl_epoch)
        hist_epochs['delta'].append(delta_epoch)
        hist_epochs['mut'].append(mut_epoch)
        hist_epochs['beta'].append(beta)
        hist_epochs['rec'].append(rec_epoch)
        hist_epochs['loss'].append(loss_epoch)
        hist_epochs['mean_logp'].append(mean_logp_epoch)
        hist_epochs['ref_mean_logp'].append(ref_mean_logp_epoch)
        hist_epochs['dlogp'].append(dlogp_epoch)
        hist_epochs['icorr'].append(icorr_epoch)
        hist_epochs['gap'].append(gap_epoch)

        if (not args.no_plots) and ((epoch + 1) % args.plot_every == 0):
            _save_epoch_step_plots(out_dir, epoch + 1, step_stats)
            _save_epoch_agg_plots(out_dir, hist_epochs)

        if (epoch + 1) % args.save_every == 0:
            save_path = out_dir / f'policy_epoch{epoch+1:02d}.pt'
            meta = {
                'epoch': epoch + 1,
                'beta': beta,
                'reward_mode': reward_mode,
                'kl_target': args.kl_target,
                'config': config,
                'scored_csv': str(scored_csv),
            }
            save_policy(policy_diffusion, save_path, meta)

    log_file.close()
    summary_file.close()

    # >>> NEW: global mean recovery after all training is done
    if global_n_samples > 0:
        global_rec = global_rec_sum / global_n_samples
        print(
            f'Overall | samples={global_n_samples} | '
            f'mean_recovery={global_rec:.4f}'
        )
    # <<< NEW


if __name__ == '__main__':
    train_offline()
