"""Smoke test: verify every top-level module can be imported.

This test does NOT require any pretrained weights or datasets. It only
exercises Python-level wiring, so it can run on a CPU-only machine.
"""

from __future__ import annotations

import importlib

import pytest


MODULES = [
    "catif_rl",
    "catif_rl.config",
    "catif_rl.data",
    "catif_rl.data.brenda",
    "catif_rl.data.dataset",
    "catif_rl.data.splits",
    "catif_rl.evaluation",
    "catif_rl.evaluation.success_rate",
    "catif_rl.evaluation.baselines",
    "catif_rl.reward",
    "catif_rl.reward.gdc",
    "catif_rl.reward.predictors",
    "catif_rl.reward.predictors.dlkcat",
    "catif_rl.reward.predictors.unikp",
    "catif_rl.reward.predictors.catapro",
    "catif_rl.training",
    "catif_rl.training.callbacks",
    "catif_rl.training.train_supervised",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_import(module_name: str) -> None:
    importlib.import_module(module_name)


def test_version_is_string() -> None:
    import catif_rl

    assert isinstance(catif_rl.__version__, str)
