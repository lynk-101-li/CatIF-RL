"""Unified supervised-training entry point for EnzymeIF and CatIF.

Both supervised stages (manuscript §2.2 and §2.4) use the same EGNN-based
discrete-diffusion backbone, the same optimizer (Adam, lr 5e-4, weight
decay 1e-5), the same training objective (residue-level cross-entropy on the
predicted clean amino-acid distribution), and the same EMA / checkpoint
selection policy. They differ only in:

- Training distribution (enzyme + CATH regularizer vs. GDC activity-positive
  variants).
- Run identifier / date tag (Jul01 vs. Sep24).
- Selected epoch (467 vs. 228).

This module is a *thin shim* over the existing trainer
``catif_rl.models.gradeif_app`` (the GraDe-IF-derived ``__main__`` block):
the YAML config is translated into the argv vector that ``gradeif_app``
expects, then we re-invoke ``gradeif_app`` as a module so its training loop
runs against the same code path the authors used to produce the released
EnzymeIF (epoch 467) and CatIF (epoch 228) checkpoints.

Usage::

    python -m catif_rl.training.train_supervised \\
        --config catif_rl/config/enzymeif.yaml

Mapping from YAML fields to gradeif_app flags follows SI Tables S1 / S2:

    YAML field                         -> gradeif_app flag
    ---------------------------------------------------------
    run.name + run.date_tag            -> --Date '<name>_<date>'
    run.output_dir                     -> --output_dir
    dataset.train_split                -> --train_dir
    dataset.valid_split                -> --val_dir
    architecture.depth                 -> --depth
    architecture.hidden_channels       -> --hidden_dim
    architecture.embedding             -> --embedding (flag)
    architecture.embedding_dim         -> --embedding_dim
    architecture.norm_feat             -> --norm_feat (flag)
    architecture.embed_ss              -> --embed_ss
    architecture.noise_type            -> --noise_type
    architecture.timesteps             -> --timesteps
    architecture.objective             -> --objective (mapped: 'pred_x0')
    optimizer.lr                       -> --lr
    optimizer.weight_decay             -> --wd
    regularization.dropout             -> --drop_out
    regularization.grad_clip_norm      -> --clip_grad_norm
    regularization.ema_decay           -> --ema_decay
    training.batch_size                -> --batch_size
    training.seed                      -> --seed
    training.max_epochs                -> --epochs
    training.save_every_n_epochs       -> --save_every_n_epochs

CLI overrides
-------------

``--epochs`` and ``--output-dir`` can be passed alongside ``--config``
to override the YAML values without editing the file (useful for the
shell pipeline and for quick smoke runs). Anything passed on the CLI
wins over the YAML; anything missing from both falls back to
``gradeif_app``'s built-in defaults.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import yaml


# YAML's friendlier 'cross_entropy_clean_x0' label maps to gradeif_app's
# canonical 'pred_x0' objective (cross-entropy on the predicted clean x_0
# distribution, per the GraDe-IF paper).
_OBJECTIVE_MAP = {
    "cross_entropy_clean_x0": "pred_x0",
    "pred_x0": "pred_x0",
    "smooth_x0": "smooth_x0",
}


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _config_to_argv(config: dict) -> list[str]:
    """Translate a YAML config dict into argv for ``gradeif_app.__main__``.

    Unknown / missing fields fall back to the defaults declared inside
    ``gradeif_app`` itself.
    """
    run         = config.get("run", {})            or {}
    dataset     = config.get("dataset", {})        or {}
    arch        = config.get("architecture", {})   or {}
    optim       = config.get("optimizer", {})      or {}
    reg         = config.get("regularization", {}) or {}
    training    = config.get("training", {})       or {}

    argv: list[str] = []

    # Run identification + output directory
    name = run.get("name", "supervised")
    date_tag = run.get("date_tag", "")
    argv += ["--Date", f"{name}_{date_tag}" if date_tag else name]
    if "output_dir" in run:
        argv += ["--output_dir", str(run["output_dir"])]

    # Data paths (gradeif_app appends a trailing slash internally, so
    # normalise here for consistency).
    def _slashed(p: object) -> str:
        s = str(p)
        return s if s.endswith("/") else s + "/"

    if "train_split" in dataset:
        argv += ["--train_dir", _slashed(dataset["train_split"])]
    if "valid_split" in dataset:
        argv += ["--val_dir",   _slashed(dataset["valid_split"])]
    test_split = dataset.get("test_split", dataset.get("valid_split"))
    if test_split:
        argv += ["--test_dir",  _slashed(test_split)]

    # Always train on the CATH-style PyG Cath wrapper.
    argv += ["--dataset", "CATH"]

    # Architecture
    if "depth" in arch:
        argv += ["--depth", str(arch["depth"])]
    if "hidden_channels" in arch:
        argv += ["--hidden_dim", str(arch["hidden_channels"])]
    if "embedding_dim" in arch:
        argv += ["--embedding_dim", str(arch["embedding_dim"])]
    if arch.get("embedding", False):
        argv += ["--embedding"]
    if arch.get("norm_feat", False):
        argv += ["--norm_feat"]
    if "embed_ss" in arch:
        argv += ["--embed_ss", str(arch["embed_ss"])]
    if "noise_type" in arch:
        argv += ["--noise_type", str(arch["noise_type"])]
    if "timesteps" in arch:
        argv += ["--timesteps", str(arch["timesteps"])]
    obj = arch.get("objective", "pred_x0")
    argv += ["--objective", _OBJECTIVE_MAP.get(obj, obj)]

    # Optimizer
    if "lr" in optim:
        argv += ["--lr", str(optim["lr"])]
    if "weight_decay" in optim:
        argv += ["--wd", str(optim["weight_decay"])]

    # Regularization
    if "dropout" in reg:
        argv += ["--drop_out", str(reg["dropout"])]
    if "grad_clip_norm" in reg:
        argv += ["--clip_grad_norm", str(reg["grad_clip_norm"])]
    if "ema_decay" in reg:
        argv += ["--ema_decay", str(reg["ema_decay"])]

    # Training loop
    if "batch_size" in training:
        argv += ["--batch_size", str(training["batch_size"])]
    if "seed" in training:
        argv += ["--seed", str(training["seed"])]
    if "max_epochs" in training:
        argv += ["--epochs", str(training["max_epochs"])]
    if "save_every_n_epochs" in training:
        argv += ["--save_every_n_epochs", str(training["save_every_n_epochs"])]

    return argv


def train(config: dict) -> None:
    """Translate ``config`` to gradeif_app argv and invoke the trainer.

    Replaces argv in-process and runs ``catif_rl.models.gradeif_app`` as
    its own module so its ``if __name__ == "__main__":`` block executes.
    """
    argv = _config_to_argv(config)
    print(f"[train_supervised] invoking gradeif_app with argv: {argv}")

    old_argv = sys.argv[:]
    try:
        sys.argv = ["catif_rl.models.gradeif_app", *argv]
        runpy.run_module("catif_rl.models.gradeif_app", run_name="__main__")
    finally:
        sys.argv = old_argv


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True,
                   help="Path to YAML config (catif_rl/config/{enzymeif,catif}.yaml).")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override training.max_epochs from YAML.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Override run.output_dir from YAML.")
    p.add_argument("--save-every-n-epochs", type=int, default=None,
                   help="Override training.save_every_n_epochs from YAML.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    config = load_config(args.config)
    # CLI overrides win over the YAML for the M4 knobs.
    if args.epochs is not None:
        config.setdefault("training", {})["max_epochs"] = args.epochs
    if args.output_dir is not None:
        config.setdefault("run", {})["output_dir"] = str(args.output_dir)
    if args.save_every_n_epochs is not None:
        config.setdefault("training", {})["save_every_n_epochs"] = args.save_every_n_epochs
    train(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
