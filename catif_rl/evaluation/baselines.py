"""Unified interface for the eleven baselines reported in manuscript §3.3.

The benchmark compares CatIF-RL (round 3) against ten reference models:

- Four pipeline variants: GraDe-IF, EnzymeIF, CatIF, CatIF-RL R1 / R2 / R3
- Five external inverse-folding baselines: ProteinMPNN, ESM-IF, LigandMPNN,
  PiFold, ABACUS-T

The per-baseline sampling commands (5 seeds, 1 design per seed) live in
``materials/07_code_snippets/baseline_sampling_configs/`` in the SI workspace;
this module exposes a Python dispatch table that maps baseline name to the
external command used to sample it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict


BENCHMARK_SEEDS = (1111, 2222, 3333, 4444, 5555)  # SI Table S6 (5 seeds x 1 design per seed)


@dataclass
class BaselineConfig:
    name: str
    conda_env: str
    repo_relative_to_external: Path
    sample_script: str          # e.g. "sample.sh" relative to repo_relative_to_external
    notes: str = ""


# Default repo layouts assume scripts/00_setup_external.sh placed each baseline
# under external/<name>/ with the same internal structure used in the SI's
# materials/07_code_snippets/baseline_sampling_configs/ archive.
BASELINES: Dict[str, BaselineConfig] = {
    "proteinmpnn": BaselineConfig(
        name="ProteinMPNN",
        conda_env="pxdesign",
        repo_relative_to_external=Path("ProteinMPNN_and_ESMIF1"),
        sample_script="sample.sh",
    ),
    "esmif": BaselineConfig(
        name="ESM-IF",
        conda_env="pxdesign",
        repo_relative_to_external=Path("ProteinMPNN_and_ESMIF1"),
        sample_script="sample.sh",
    ),
    "ligandmpnn": BaselineConfig(
        name="LigandMPNN",
        conda_env="ligandmpnn",
        repo_relative_to_external=Path("LigandMPNN"),
        sample_script="sample.sh",
    ),
    "pifold": BaselineConfig(
        name="PiFold",
        conda_env="pifold",
        repo_relative_to_external=Path("PiFold"),
        sample_script="sample.sh",
    ),
    "abacust": BaselineConfig(
        name="ABACUS-T",
        conda_env="abacust",
        repo_relative_to_external=Path("ABACUST-v2-pub-main"),
        sample_script="sample.sh",
    ),
    "gradeif": BaselineConfig(
        name="GraDe-IF",
        conda_env="catif",
        repo_relative_to_external=Path("GraDe_IF"),
        sample_script="infrs_gradeif.sh",
    ),
    "enzymeif": BaselineConfig(
        name="EnzymeIF",
        conda_env="catif",
        repo_relative_to_external=Path("..").joinpath("..").joinpath("..").joinpath("scripts"),  # in-tree
        sample_script="infrs_enzymeif.sh",
        notes="Local pipeline variant; called via scripts/, not external/.",
    ),
    "catif": BaselineConfig(
        name="CatIF",
        conda_env="catif",
        repo_relative_to_external=Path("..").joinpath("..").joinpath("..").joinpath("scripts"),
        sample_script="infrs_catif.sh",
        notes="Local pipeline variant; called via scripts/, not external/.",
    ),
    "catif_rl_r1": BaselineConfig(
        name="CatIF-RL R1",
        conda_env="catif",
        repo_relative_to_external=Path("..").joinpath("..").joinpath("..").joinpath("scripts"),
        sample_script="infrs_catif_rl.sh",
        notes="Pipeline variant -- round-1 checkpoint.",
    ),
    "catif_rl_r2": BaselineConfig(
        name="CatIF-RL R2",
        conda_env="catif",
        repo_relative_to_external=Path("..").joinpath("..").joinpath("..").joinpath("scripts"),
        sample_script="infrs_catif_rl.sh",
        notes="Pipeline variant -- round-2 checkpoint.",
    ),
    "catif_rl_r3": BaselineConfig(
        name="CatIF-RL R3",
        conda_env="catif",
        repo_relative_to_external=Path("..").joinpath("..").joinpath("..").joinpath("scripts"),
        sample_script="infrs_catif_rl.sh",
        notes="Pipeline variant -- round-3 checkpoint (the main result).",
    ),
}


def list_baselines() -> Dict[str, BaselineConfig]:
    """Return the dispatch table of known baselines."""
    return BASELINES
