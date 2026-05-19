"""Training callbacks shared by EnzymeIF / CatIF supervised training and GRPO RL.

Includes:

- Exponential Moving Average (EMA) tracking of policy weights, with decay 0.995
  used during EnzymeIF and CatIF supervised training (SI Tables S1, S2).
- Lowest-validation-loss checkpoint selection used to pick the round-end model
  in all three supervised stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass
class CheckpointSelector:
    """Tracks the lowest validation loss seen so far and saves on improvement."""

    output_dir: Path
    metric_name: str = "val_loss"
    best_metric: float = float("inf")
    best_epoch: Optional[int] = None
    best_path: Optional[Path] = None

    def update(self, epoch: int, metric_value: float, state_dict: dict) -> bool:
        """Save a checkpoint if ``metric_value`` improves on the current best.

        Returns
        -------
        bool
            True when a new best was recorded and a checkpoint written.
        """

        self.output_dir.mkdir(parents=True, exist_ok=True)
        improved = metric_value < self.best_metric
        if improved:
            self.best_metric = metric_value
            self.best_epoch = epoch
            self.best_path = self.output_dir / ("best_epoch" + str(epoch) + ".pt")
            torch.save(state_dict, self.best_path)
        return improved


def make_ema(model: torch.nn.Module, decay: float = 0.995):
    """Construct an EMA wrapper compatible with the upstream ``ema_pytorch`` API.

    The supervised training stages use decay 0.995 across all parameters.
    """

    from ema_pytorch import EMA

    return EMA(
        model,
        beta=decay,
        update_after_step=0,
        update_every=1,
        include_online_model=False,
    )
