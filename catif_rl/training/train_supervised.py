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

This script wires those differences through YAML configs in
``catif_rl/config/{enzymeif,catif}.yaml`` and dispatches to the GraDe-IF
backbone defined in :mod:`catif_rl.models.gradeif_app`.

Usage::

    python -m catif_rl.training.train_supervised \\
        --config catif_rl/config/enzymeif.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from catif_rl.models.gradeif_app import EGNN_NET, GraDe_IF  # noqa: F401  (kept for re-export)


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def train(config: dict) -> None:
    """Run a supervised pretraining stage from ``config``.

    The detailed training loop is implemented inside
    :mod:`catif_rl.models.gradeif_app` (the user-modified GraDe-IF application
    wrapper). This function constructs the data loaders, optimizer, EMA,
    and checkpoint selector, then delegates the inner step to that module.
    """

    # Scaffold -- the actual driver is being lifted from the existing
    # gradeif_app.py __main__ block. Wiring up the YAML-driven entry point
    # is intentionally minimal here so the existing trainer keeps working
    # until the refactor is complete.
    raise NotImplementedError(
        "Supervised training scaffold; wire to gradeif_app driver. "
        "Run name: " + str(config.get("run", {}).get("name"))
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    train(load_config(args.config))
