# `catif_rl.models`

This package contains the diffusion-policy backbone used throughout the pipeline.

## Why this is a thin adapter, not a vendored copy

The graph-denoising-diffusion architecture (EGNN-based discrete diffusion over
amino-acid node types with a BLOSUM substitution kernel) was first introduced
by GraDe-IF (Yi et al., NeurIPS 2023). The upstream repository at
<https://github.com/ykiiiiii/GraDe_IF> contains the reference implementation
that this project builds on.

Rather than vendoring the upstream code, `catif_rl.models` follows a
*depend-don't-redistribute* pattern:

- `scripts/00_setup_external.sh` clones the upstream into
  `external/GraDe_IF/`.
- `gradeif_adapter.py` inserts that directory onto `sys.path`, so all of the
  `diffusion.utils.*`, `diffusion.model.egnn_pytorch.*`, and related symbols
  remain importable using their original module paths.
- The application wrapper `gradeif_app.py` shipped here contains the local
  modifications used to train EnzymeIF, CatIF, and CatIF-RL. It imports from
  `diffusion.*` (the cloned upstream) but exposes its public symbols
  (`EGNN_NET`, `GraDe_IF`) under `catif_rl.models.gradeif_app`.

## Public API

```python
from catif_rl.models.gradeif_app import EGNN_NET, GraDe_IF
```

That import implicitly runs the path injection.

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | Re-exports ``gradeif_adapter`` so that just importing the package triggers `sys.path` setup. |
| `gradeif_adapter.py` | Inserts `external/GraDe_IF/` onto `sys.path`. |
| `gradeif_app.py` | Application wrapper layered on top of the GraDe-IF backbone. Defines `EGNN_NET` and `GraDe_IF` with the project's training/sampling extensions. |
