"""Canonical method-id <-> display-name registry.

One source of truth used by:

- ``catif_rl.evaluation.build_master`` to translate per-method subdirectory
  names under ``runs/benchmark_scores/`` into the column-prefix labels that
  the master CSV (and downstream Tables S7-S10) expect.
- ``catif_rl.evaluation.statistics`` for ``BASELINE_ORDER`` and
  ``CATIF_RL`` references.
- ``scripts/06_sample_benchmark.sh`` for the list of method IDs it iterates.
- The metrics-comparison notebook for cross-referencing the legacy
  ``MODEL_CONFIGS`` keys.

The ID is the lowercase, filesystem-safe slug used as the per-method
subdirectory name (matches ``06_sample_benchmark.sh``'s ``METHODS`` list).
The display name is what appears in published tables / figures.
"""

from __future__ import annotations

from typing import Final


# Method id (subdirectory + glob prefix)   ->   display name (tables, plots, paper)
METHODS: Final[dict[str, str]] = {
    "gradeif":      "GraDe-IF",
    "enzymeif":     "EnzymeIF",
    "catif":        "CatIF",
    "catif_rl_r1":  "CatIF-RL R1",
    "catif_rl_r2":  "CatIF-RL R2",
    "catif_rl_r3":  "CatIF-RL R3",
    "proteinmpnn":  "ProteinMPNN",
    "esmif":        "ESM-IF",
    "ligandmpnn":   "LigandMPNN",
    "pifold":       "PiFold",
    "abacust":      "ABACUS-T",
}

# Canonical ordering used by statistics.BASELINE_ORDER and the Tables S7-S10
# row order. Roughly: external baselines first, then the in-repo lineage
# (GraDe-IF -> EnzymeIF -> CatIF -> RL rounds).
ORDER: Final[list[str]] = [
    "proteinmpnn", "esmif", "ligandmpnn", "pifold", "abacust",
    "gradeif", "enzymeif", "catif",
    "catif_rl_r1", "catif_rl_r2", "catif_rl_r3",
]

# The reference / headline method used in paired comparisons.
HEADLINE: Final[str] = "catif_rl_r3"


def display_name(method_id: str) -> str:
    """Map a method id to its published display name. Unknown ids pass through."""
    return METHODS.get(method_id, method_id)


def display_order() -> list[str]:
    """Return ORDER as display names (in paper order)."""
    return [METHODS[i] for i in ORDER]


def headline_display() -> str:
    return METHODS[HEADLINE]
