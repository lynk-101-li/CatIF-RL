#!/usr/bin/env python
"""
DDIM inference for CatIF / CatIF-RL policies.

Two input modes are supported:

1. **Batch mode** (``--test_dir <dir>``): read every ``.pt`` graph under the
   directory, sample one sequence per graph, and write one FASTA per graph
   into ``--output_dir``. This is what ``scripts/06_sample_benchmark.sh``
   uses for the held-out benchmark.
2. **Single-PDB mode** (``--input-pdb <file.pdb>``): build a graph on the
   fly with :func:`catif_rl.data.graph_construction.pdb_to_sample_data`,
   sample one sequence, and write the resulting FASTA into ``--output_dir``
   under the PDB's basename. This is what ``scripts/08_run_case_studies.sh``
   uses for the per-case redesigns.

Batch mode, single-seed run on the held-out benchmark::

    python -u -m catif_rl.sampling.infer \\
      --test_dir   data/process/test \\
      --ckpt_path  checkpoints/catif_rl_R3_epoch02.pt \\
      --output_dir runs/benchmark/catif_rl_r3/seed_1 \\
      --seed 1

Batch mode, five-seed sweep (matches ``scripts/06_sample_benchmark.sh``)::

    for s in 1111 2222 3333 4444 5555; do
      python -u -m catif_rl.sampling.infer \\
        --test_dir   data/process/test \\
        --ckpt_path  checkpoints/catif_rl_R3_epoch02.pt \\
        --output_dir runs/benchmark/catif_rl_r3/seed_${s} \\
        --seed $s
    done

Single-PDB mode (matches ``scripts/08_run_case_studies.sh``)::

    python -u -m catif_rl.sampling.infer \\
      --input-pdb  case_study/EC1.4.1.20_Lsphaericus/native.pdb \\
      --ckpt_path  checkpoints/catif_rl_R3_epoch02.pt \\
      --output_dir runs/case_studies/EC1.4.1.20_Lsphaericus \\
      --seed 12345

The checkpoint loader autodetects both the supervised EMA-tracked format
(``ckpt['ema']``) and the RL policy format (``ckpt['model']`` only), and
rebuilds the EGNN_NET / GraDe_IF from ``ckpt['config']`` or
``ckpt['meta']['config']``.
"""

import os
import argparse
import torch
import random
import numpy as np
from ema_pytorch import EMA
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from catif_rl.models.gradeif_app import EGNN_NET, GraDe_IF
from catif_rl.data.large_dataset import Cath
# torch.manual_seed = lambda *args, **kwargs: None  # override manual_seed to be a no-op (kept for reference)

# 20 amino-acid one-letter codes, used to map one-hot back to a sequence
AMINO_CODES = ['A','R','N','D','C','Q','E','G','H','I',
               'L','K','M','F','P','S','T','W','Y','V']

def onehot_to_seq(onehot):
    """
    Convert a one-hot tensor produced by the model to an amino-acid string.
    onehot: [total_nodes, 20]
    Returns a string of length total_nodes.
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
    # For stricter determinism (usually not needed):
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    # 1. Argument parsing
    p = argparse.ArgumentParser(description="GraDe_IF inference with EMA weights")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--test_dir',  type=str, default=None,
                     help="directory containing .pt test graphs (batch mode)")
    src.add_argument('--input-pdb', dest='input_pdb', type=str, default=None,
                     help="single PDB file to redesign (single-PDB mode); built on the fly via catif_rl.data.graph_construction")
    p.add_argument('--ckpt_path',  type=str, required=True, help="path to .pt checkpoint with EMA weights")
    p.add_argument('--batch_size', type=int, default=32, help="DataLoader batch_size (ignored in single-PDB mode)")
    p.add_argument('--device',     type=str, default='cuda:0', help="device, e.g. cuda:0 or cpu")
    p.add_argument('--output_dir', type=str, required=True, help="directory to write FASTA files")
    p.add_argument('--seed', type=int, default=-1, help='Random seed; set >=0 for reproducible DDIM sampling')
    args = p.parse_args()
    set_seed(args.seed)

    # 2. Device & output directory
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # 3. Input loader -- batch mode reads a directory of .pt files via Cath;
    #    single-PDB mode builds one Data on the fly and wraps it in a single-element batch.
    if args.input_pdb is not None:
        from catif_rl.data.graph_construction import pdb_to_sample_data
        single_data = pdb_to_sample_data(args.input_pdb)
        if single_data is None:
            raise RuntimeError(
                f"failed to build a graph from {args.input_pdb}; "
                "check that the PDB has at least the minimum number of residues "
                "and that mkdssp is on PATH."
            )
        test_ids = [os.path.splitext(os.path.basename(args.input_pdb))[0] + '.pt']
        test_loader = [Batch.from_data_list([single_data])]  # iterable of length 1
    else:
        test_ids = sorted([f for f in os.listdir(args.test_dir) if f.endswith('.pt')])
        test_ds  = Cath(test_ids, args.test_dir)
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=4
        )

    # 4. Restore the network from the checkpoint and load EMA weights
    ckpt    = torch.load(args.ckpt_path, map_location=device)
    # Two ckpt formats are supported:
    # 1) supervised training: ckpt['config'] directly
    # 2) RL fine-tuning:      ckpt['meta']['config']
    if 'config' in ckpt:
        config = ckpt['config']  # config holds all hyperparameters and input dims
    elif 'meta' in ckpt and isinstance(ckpt['meta'], dict) and 'config' in ckpt['meta']:
        config = ckpt['meta']['config']
    else:
        raise KeyError("config not found in ckpt; expected 'config' or 'meta[\"config\"]'")

    # 4.1 Dynamically build EGNN_NET with the same hyperparameters as training
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

    # 4.2 Wrap in GraDe_IF
    diffusion = GraDe_IF(
        model      = base_model,
        timesteps  = config['timesteps'],
        objective  = config.get('objective', 'pred_x0'),
        config     = config
    ).to(device)

    # 4.3 Load weights using the ckpt structure
    if 'ema' in ckpt:
        # Original supervised ckpt: use EMA-smoothed weights
        from ema_pytorch import EMA
        ema = EMA(diffusion)
        ema.load_state_dict(ckpt['ema'])
        model = ema.ema_model.to(device).eval()
    else:
        # RL ckpt: only 'model'; use the fine-tuned diffusion directly
        diffusion.load_state_dict(ckpt['model'], strict=False)
        model = diffusion.to(device).eval()

    # 6. Inference and per-graph file write
    global_idx = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            # DDIM sample with the EMA model
            zt, pred_onehot = model.ddim_sample(
                batch,
                diverse=True,
                step=100      # adjust as needed
            )
            # Node counts per sub-graph
            node_counts = [g.x.size(0) for g in batch.to_data_list()]

            start = 0
            for count in node_counts:
                oh = pred_onehot[start:start+count]
                seq = onehot_to_seq(oh)
                pt_name = test_ids[global_idx]
                # Strip the graph-tensor extension from the FASTA record id.
                # ESMFold (and several other downstream consumers) name the
                # predicted PDB after the record id; leaving '.pt' in the
                # header would produce sequence_<n>.pt.pdb, which then fails
                # to match data/raw/test/sequence_<n>.pdb under
                # catif_rl.evaluation.structural's default basename contract.
                stem = os.path.splitext(pt_name)[0]
                fasta_fn = stem + '.fasta'
                out_path = os.path.join(args.output_dir, fasta_fn)
                with open(out_path, 'w') as f:
                    f.write(f'>{stem}\n{seq}\n')
                start += count
                global_idx += 1

    print(f"[OK] processed {global_idx} samples, output saved under {args.output_dir}")
