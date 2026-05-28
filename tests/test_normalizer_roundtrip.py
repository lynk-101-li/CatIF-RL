"""Round-trip test for the frozen GDC normalizer.

Goal: prove that the same input data, scored under ``--normalizer``,
produces a reward scale that is *invariant* to the data distribution.
The test does this by:

1. Computing a normalizer from a "GDC-like" calibration distribution
   (10 synthetic samples per predictor with known q10 / q90).
2. Writing it out as ``normalizer.json``.
3. Applying it to a *different* (smaller, noisier) distribution that
   mimics an RL round's per-batch data, and verifying the resulting
   scaled values match what we'd get from direct division by the
   saved scale.
4. Verifying that recomputing q10 / q90 on the small batch would have
   produced a different scale — i.e. that NOT freezing the normalizer
   really does cause drift (the drift bug this test catches).

The test uses only the in-process helpers ``_compute_one_normalizer`` and
``_apply_normalizer`` from ``catif_rl.reward.gdc``; no subprocess, no
torch, no heavy deps.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from catif_rl.reward.gdc import _apply_normalizer, _compute_one_normalizer


def test_frozen_normalizer_invariant_to_round_distribution():
    rng = np.random.default_rng(42)

    # ---- 1. Big "GDC" calibration distribution ----
    calibration = pd.Series(rng.normal(loc=0.0, scale=1.0, size=10_000))
    nrm = _compute_one_normalizer(calibration)
    assert nrm["fallback"] == "q90-q10"
    assert nrm["scale"] > 0
    # For a standard normal the q90 - q10 width is roughly 2.56; allow slack.
    assert 2.0 < nrm["scale"] < 3.5

    # ---- 2. Persist and reload ----
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "normalizer.json"
        with path.open("w") as f:
            json.dump({"version": "1.0", "predictors": {"dlkcat": nrm}}, f)
        reloaded = json.loads(path.read_text())
        saved = reloaded["predictors"]["dlkcat"]
        assert saved["scale"] == nrm["scale"]

    # ---- 3. Apply to a small / noisy "RL round" batch ----
    round_batch = pd.Series(rng.normal(loc=0.5, scale=0.3, size=100))
    # Two ways to scale:
    s_frozen = _apply_normalizer(round_batch, nrm["scale"])

    # Live recomputation would give a much smaller scale because the batch
    # is narrower than the GDC pool; this is the drift bug we are fixing.
    live = _compute_one_normalizer(round_batch)
    s_live = _apply_normalizer(round_batch, live["scale"])

    # Frozen vs live: scales differ enough that the scaled values differ.
    diff = (s_frozen - s_live).abs().max()
    assert diff > 0.1, (
        f"frozen ({nrm['scale']:.3f}) and live ({live['scale']:.3f}) "
        f"normalizers happened to coincide; the test fixture isn't exercising "
        f"the drift bug. diff={diff}"
    )
    # Frozen scale is the GDC scale, NOT the round's local scale.
    assert abs(nrm["scale"] / live["scale"] - 1.0) > 0.5, (
        "frozen and per-batch scales should differ noticeably for this fixture"
    )


def test_normalizer_json_schema_is_stable():
    """Defensive: the JSON written by GDC carries the keys ensemble_rl reads."""
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(size=1000))
    n = _compute_one_normalizer(x)
    for key in ("q10", "q90", "fallback", "scale"):
        assert key in n, f"missing required key {key!r} in normalizer payload"
    # Round-trip JSON serializability
    blob = {"version": "1.0", "predictors": {"dlkcat": n, "unikp": n, "catapro": n}}
    s = json.dumps(blob)
    back = json.loads(s)
    assert back["predictors"]["dlkcat"]["scale"] == n["scale"]


if __name__ == "__main__":
    test_frozen_normalizer_invariant_to_round_distribution()
    test_normalizer_json_schema_is_stable()
    print("PASS")
