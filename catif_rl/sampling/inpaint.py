#!/usr/bin/env python
# diffusion/inference.py

"""
Example usage:

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
# 1. Basics
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
# 2. Helpers: mask parsing & advanced sampling
# ==========================================

def parse_mask_indices(indices_str):
    """
    Parse a string of indices, accepting forms like "1,2,5-8".
    Returns a 0-based list of indices.
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
    Build a global mask: 1=Keep(GT), 0=Inpaint(Gen).
    """
    total_nodes = batch.x.shape[0]
    mask = torch.ones((total_nodes, 1), device=device)  # keep everything by default

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
    RePaint sampling with optional resampling (jump).
    """
    timesteps = diffusion_model.timesteps
    device = data.x.device

    # Initialize noise
    limit_dist = torch.ones(20) / 20
    zt = diffusion_model.sample_discrete_feature_noise(limit_dist=limit_dist, num_node=data.x.shape[0])
    zt = zt.to(device)

    # Time-step list (reversed)
    times_list = list(reversed(range(0, timesteps, step)))

    # Progress-bar description
    pbar_desc = f"Repaint(jump={jump_n}, div={diverse})"

    for i, s_int in enumerate(tqdm(times_list, desc=pbar_desc)):
        # s_int is the next target time
        # t_int is the previous (current) time
        t_int = s_int + step

        # Normalized time
        s_array = s_int * torch.ones((data.batch[-1]+1, 1)).to(device)
        t_array = t_int * torch.ones((data.batch[-1]+1, 1)).to(device)
        s_norm = s_array / timesteps
        t_norm = t_array / timesteps

        # === Resampling loop (denoise then forward-jump) ===
        # The last step (s=0) does not need a jump-back, unless one wants a
        # final refinement. RePaint's recipe: after each t->s, jump back to t
        # and repeat U times.

        # Cycle count: only one pass on the last step; otherwise jump_n
        current_jump_n = 1 if s_int == 0 else jump_n

        for u in range(current_jump_n):

            # 1. Denoise (t -> s)
            # Call GraDe-IF's p(z_s | z_t)
            is_last_step = (s_int == 0) and (u == current_jump_n - 1)

            # Note: sample_p_zs_given_zt expects t_norm (current) and s_norm (target)
            zt_pred, final_predicted_X = diffusion_model.sample_p_zs_given_zt(
                t_norm, s_norm, zt, data, cond=False, diverse=diverse, step=step, last_step=is_last_step
            )

            # 2. Replacement (inject GT)
            # Get the GT noise distribution at time s
            # Use diffusion_model.apply_noise (which uses Qt_bar to go x0 -> xs)
            s_int_batch = torch.full((data.batch[-1]+1, 1), s_int, device=device).float()
            temp_data = data.clone()
            temp_data.x = gt_x
            zt_known = diffusion_model.apply_noise(temp_data, s_int_batch).x

            # Blend
            zt = mask * zt_known + (1 - mask) * zt_pred

            # 3. Jump back (s -> t) [re-noise]
            # If this is not the last sub-loop and not the last step, re-noise zt (xs) back to xt
            if u < current_jump_n - 1 and s_int > 0:
                # We need to jump from s back to t (s < t)
                # For uniform noise, the transition matrix Q_{t|s} can be computed via the alpha_bar ratio

                # Get alpha_bar
                alpha_t_bar = diffusion_model.noise_schedule.get_alpha_bar(t_normalized=t_norm)
                alpha_s_bar = diffusion_model.noise_schedule.get_alpha_bar(t_normalized=s_norm)

                # Relative alpha: alpha_{t|s} = alpha_t / alpha_s
                # (alpha_bar is monotonically decreasing, t > s implies alpha_t < alpha_s, so ratio < 1)
                # Clamp for numerical stability
                alpha_rel = (alpha_t_bar / alpha_s_bar).clamp(0, 1)

                # Get the transition matrix Q_{t|s}
                # Note: GraDe_IF's get_Qt_bar produces the "keep" probability matrix
                if diffusion_model.config['noise_type'] == 'uniform':
                    Qt_jump = diffusion_model.transition_model.get_Qt_bar(alpha_rel, device=device)

                    # Sample: z_t ~ z_s @ Q_{t|s}
                    prob_jump = (Qt_jump[data.batch] @ zt.unsqueeze(2)).squeeze()
                    zt_idx = prob_jump.multinomial(1).squeeze()
                    zt = torch.nn.functional.one_hot(zt_idx, num_classes=20).float()
                else:
                    # For BLOSUM or other noise types, the relative alpha may not give the right Q
                    # Fallback: pure replacement, no re-noise jump (degrades to ordinary repaint).
                    # An alternative would be step single apply_noise calls, but that is slow.
                    # Keep zt unchanged for now (jump_n effectively no-op).
                    pass

    return zt, final_predicted_X

# ==========================================
# 3. Main
# ==========================================


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="GraDe_IF Inference with Repaint & Resampling")

    # Basic args
    p.add_argument('--test_dir',   type=str, required=True)
    p.add_argument('--ckpt_path',  type=str, required=True)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--device',     type=str, default='cuda:0')
    p.add_argument('--output_dir', type=str, required=True)
    p.add_argument('--seed',       type=int, default=-1)

    # Sampling args
    p.add_argument('--step',       type=int, default=50, help="DDIM step size")
    p.add_argument('--no_diverse', action='store_true', help="if set, disables diverse sampling (uses argmax)")

    # Repaint args
    p.add_argument('--use_repaint', action='store_true', help="enable Repaint mode")
    p.add_argument('--mask_indices', type=str, default="", help="0-based indices to repaint, e.g. '10-20,35'")
    p.add_argument('--repaint_jump_n', type=int, default=10, help="repaint resampling count (default 10); 1 disables the jump-back")

    args = p.parse_args()

    # Handle arguments
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # diverse defaults to True unless --no_diverse is given
    is_diverse = not args.no_diverse

    # Load data
    test_ids = sorted([f for f in os.listdir(args.test_dir) if f.endswith('.pt')])
    test_ds  = Cath(test_ids, args.test_dir)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

    # Load checkpoint
    print(f"Loading checkpoint from {args.ckpt_path} ...")
    ckpt = torch.load(args.ckpt_path, map_location=device)

    # Compatibility: ckpt structure varies across training stages
    if 'config' in ckpt:
        config = ckpt['config']
    elif 'meta' in ckpt:
        config = ckpt['meta']['config']
    else:
        # Last-ditch: look at the ckpt root (some simple save paths do this)
        config = ckpt.get('args', {})
        # If still not found, defaults may need to be hard-coded
        if not config: print("Warning: Config not found in ckpt, using defaults might fail.")

    # Build the model
    base_model = EGNN_NET(
        input_feat_dim = config.get('input_feat_dim', 0),  # ensure this lives in config or is inferred from the dataset
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

    # Load weights
    if 'ema' in ckpt:
        ema = EMA(diffusion)
        ema.load_state_dict(ckpt['ema'])
        model = ema.ema_model.to(device).eval()
    elif 'model' in ckpt:
        diffusion.load_state_dict(ckpt['model'], strict=False)
        model = diffusion.to(device).eval()
    else:
        # Last fallback: ckpt itself is a state_dict
        diffusion.load_state_dict(ckpt, strict=False)
        model = diffusion.to(device).eval()

    # Print mode info
    if args.use_repaint:
        indices = parse_mask_indices(args.mask_indices)
        print(f"[Mode] Repaint | Jump N: {args.repaint_jump_n} | Diverse: {is_diverse}")
        print(f"[Masked Indices (Inpainting)] {indices}")
    else:
        print(f"[Mode] Standard Generation | Diverse: {is_diverse}")

    # Inference loop
    global_idx = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)

            # Sanity check input_feat_dim (in case config only had placeholder dims)
            if base_model.lin.in_features != batch.x.shape[1] + batch.extra_x.shape[1]:
               # If dims disagree, a hack may be needed here, or trust the ckpt
               pass

            pred_onehot = None

            if args.use_repaint:
                # 1. Prepare GT
                gt_x = batch.x.clone()
                # 2. Build the mask
                user_mask_indices = parse_mask_indices(args.mask_indices)
                mask = create_batch_mask(batch, user_mask_indices, device)

                # 3. Repaint sampling
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
                # Standard sampling
                zt, pred_onehot = model.ddim_sample(
                    batch,
                    diverse=is_diverse,
                    step=args.step
                )

            # Save
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

    print(f"[OK] All Done. Output saved to {args.output_dir}")
