# `checkpoints/`

Trained policy weights for CatIF-RL are too large to ship in git. Download
them separately into this folder before running training or sampling.

## Expected layout

```
checkpoints/
├── enzymeif_Jul01_epoch467.pt        # Supervised EnzymeIF (Supporting Information Table S1)
├── catif_Sep24_epoch228.pt           # Supervised CatIF (Supporting Information Table S2)
├── catif_rl_R1_epoch02.pt            # CatIF-RL Round 1
├── catif_rl_R2_epoch02.pt            # CatIF-RL Round 2
└── catif_rl_R3_epoch02.pt            # CatIF-RL Round 3 (the main result)
```

The filename ``_epoch02`` suffix on the RL checkpoints reflects the inner-
epoch budget E_k = 2 selected for downstream use in every RL round
(Supporting Information Table S3).

## How to obtain them

A Zenodo / HuggingFace archive will be linked here once the deposit is
finalised. Each checkpoint embeds its full model configuration under
``ckpt['config']``, so inference scripts auto-construct the matching
``EGNN_NET`` / ``GraDe_IF`` instance -- there are no extra hyperparameter
flags to remember:

```python
import torch
from catif_rl.models.gradeif_app import EGNN_NET, GraDe_IF

ckpt = torch.load("checkpoints/catif_rl_R3_epoch02.pt", map_location="cpu")
cfg = ckpt["config"]
# ... see catif_rl/sampling/infer.py for the canonical loading sequence
```

For inference convenience, ``catif_rl/sampling/infer.py`` also restores EMA
weights when they are present in the checkpoint.

## License

The trained weights are released by the same authors as the code under the
same MIT licence. Upstream backbone code (GraDe-IF) is cloned separately and
remains under its own terms; see ``external/README.md``.
