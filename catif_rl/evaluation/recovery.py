#!/usr/bin/env python3
"""Sequence-recovery evaluation on the held-out test set.

For each of ``--ensemble-num`` independent DDIM samples, this script:

1. Runs ``model.ddim_sample`` over every test backbone (in batches).
2. Reports per-run recovery (fraction of correctly predicted residues) and
   perplexity ``exp(mean CE)``.
3. Starting at the second run, also reports the running ensemble metric
   (mean of per-node probabilities across the runs so far).

A single CSV with one row per run (and one extra row per running ensemble
size from run 2 onwards) is written to ``--output-csv``.

CLI
---

    python -m catif_rl.evaluation.recovery \\
      --ckpt        checkpoints/catif_rl_R3_epoch02.pt \\
      --test-dir    data/process/test \\
      --output-csv  runs/eval/recovery_results.csv

The defaults match the manuscript (CatIF-RL Round-3 checkpoint, batch 300,
ensemble 50, DDIM step 250).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# These heavy imports are kept at module top level so that the smoke CI's
# `python -m … --help` accepts a clean ModuleNotFoundError; on a system
# with the catif env activated they import fine. argparse construction does
# NOT touch torch / torch_geometric.
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from ema_pytorch import EMA
from torch_geometric.loader import DataLoader

from catif_rl.data.large_dataset import Cath
from catif_rl.models.gradeif_app import EGNN_NET, GraDe_IF


# Neutralize torch.manual_seed so that the sampler stays diverse across runs
# regardless of any seeding done by external scripts.
torch.manual_seed = lambda *args, **kwargs: None


DEFAULT_CKPT = "checkpoints/catif_rl_R3_epoch02.pt"
DEFAULT_TEST_DIR = "dataset/process/test/"
DEFAULT_OUTPUT = "evaluation/recovery_results.csv"
DEFAULT_BATCH = 300
DEFAULT_ENSEMBLE = 50
DEFAULT_DDIM_STEP = 250


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m catif_rl.evaluation.recovery",
        description=(
            "Sequence-recovery + perplexity evaluation on the held-out test set "
            "with N-sample DDIM ensembling. Reproduces the manuscript's Table 2 "
            "/ SI Table S7 recovery numbers."
        ),
    )
    p.add_argument("--ckpt", type=Path, default=Path(DEFAULT_CKPT),
                   help=("Path to a CatIF (supervised, with EMA) or CatIF-RL "
                         "(policy_epochXX.pt) checkpoint. Default: %(default)s."))
    p.add_argument("--test-dir", type=Path, default=Path(DEFAULT_TEST_DIR),
                   help="Directory of per-protein test graphs (*.pt). Default: %(default)s.")
    p.add_argument("--output-csv", type=Path, default=Path(DEFAULT_OUTPUT),
                   help="Destination CSV (run, recovery, perplexity). Default: %(default)s.")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH,
                   help="DataLoader batch size (default: %(default)s).")
    p.add_argument("--ensemble-num", type=int, default=DEFAULT_ENSEMBLE,
                   help="Number of independent DDIM samples to ensemble (default: %(default)s).")
    p.add_argument("--ddim-step", type=int, default=DEFAULT_DDIM_STEP,
                   help="DDIM sampling step count (default: %(default)s).")
    p.add_argument("--num-workers", type=int, default=6,
                   help="DataLoader worker count (default: %(default)s).")
    p.add_argument("--device", default=None,
                   help=("Torch device string (cuda / cuda:0 / cpu). "
                         "Default: cuda if available, else cpu."))
    return p


def _load_model(ckpt_path: Path, device: torch.device):
    """Unified checkpoint loader for supervised CatIF (EMA) and RL policy_epochXX."""
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=device)
    # Resolve the model config: prefer ckpt['config'], else look under meta.
    if "config" in ckpt:
        config = ckpt["config"]
    elif "meta" in ckpt and isinstance(ckpt["meta"], dict) and "config" in ckpt["meta"]:
        config = ckpt["meta"]["config"]
    else:
        raise KeyError(
            f"checkpoint {ckpt_path} does not contain a config; "
            f"expected either ckpt['config'] or ckpt['meta']['config']."
        )

    base_model = EGNN_NET(
        input_feat_dim=config["input_feat_dim"],
        hidden_channels=config["hidden_dim"],
        edge_attr_dim=config["edge_attr_dim"],
        dropout=config["drop_out"],
        n_layers=config["depth"],
        update_edge=config.get("update_edge", True),
        embedding=config.get("embedding", False),
        embedding_dim=config.get("embedding_dim", 16),
        norm_feat=config.get("norm_feat", False),
        embed_ss=config.get("embed_ss", -1),
    )
    diffusion = GraDe_IF(
        model=base_model,
        timesteps=config["timesteps"],
        objective=config.get("objective", "pred_x0"),
        config=config,
    )
    # Two supported checkpoint formats:
    #   - supervised CatIF: ckpt['ema']  -> wrap in EMA(diffusion) and restore.
    #   - RL policy_epochXX: ckpt['model'] -> load directly.
    if "ema" in ckpt:
        ema = EMA(diffusion)
        ema.load_state_dict(ckpt["ema"])
        return ema.ema_model.to(device).eval()
    if "model" in ckpt:
        diffusion.load_state_dict(ckpt["model"], strict=False)
        return diffusion.to(device).eval()
    raise KeyError(
        f"checkpoint {ckpt_path} contains neither 'ema' nor 'model'; cannot load weights."
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if not args.test_dir.is_dir():
        print(f"[recovery][ERROR] test dir missing: {args.test_dir}", file=sys.stderr)
        return 2

    model = _load_model(args.ckpt, device)

    test_ids = sorted(os.listdir(args.test_dir))
    test_ds = Cath(test_ids, str(args.test_dir))
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    records = []
    ensemble_accum = []

    for run in range(args.ensemble_num):
        all_prob = []
        all_seq = []
        ind_accum = []

        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                prob, sample = model.ddim_sample(
                    data, diverse=True, step=args.ddim_step,
                )
                seq_true = data.x.argmax(dim=1)
                seq_pred = sample.argmax(dim=1)
                ind_accum.append((seq_true == seq_pred).cpu())
                all_prob.append(prob.cpu())
                all_seq.append(seq_true.cpu())

        all_prob = torch.cat(all_prob, dim=0)
        all_seq = torch.cat(all_seq, dim=0)
        ind_all = torch.cat(ind_accum, dim=0)
        rr = ind_all.float().mean().item()
        ppl = float(np.exp(F.cross_entropy(all_prob, all_seq, reduction="mean").item()))
        records.append({"run": run, "recovery": rr, "perplexity": ppl})
        ensemble_accum.append(all_prob)

        if run > 0:
            ens_prob = torch.stack(ensemble_accum, dim=0).mean(dim=0)
            ens_pred = ens_prob.argmax(dim=1)
            ens_rr = (ens_pred == all_seq).float().mean().item()
            ens_ppl = float(np.exp(
                F.cross_entropy(ens_prob, all_seq, reduction="mean").item()
            ))
            records.append({
                "run": f"ensemble_{run}",
                "recovery": ens_rr,
                "perplexity": ens_ppl,
            })

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(args.output_csv, index=False)
    print(df.to_string(index=False))
    print(f"[save] {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
