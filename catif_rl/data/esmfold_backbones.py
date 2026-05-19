"""ESMFold backbone prediction wrapper.

After sequence-level filtering (see :mod:`catif_rl.data.brenda`), each of the
7,713 distinct enzyme sequences is folded with ESMFold (Lin et al., 2023) to
obtain a predicted backbone structure. The predicted PDB then feeds into the
graph-construction pipeline.

ESMFold runs in its own conda environment (``esmfold``) -- see
``environment.yml`` cross-references and ``model_environments_yml_database/
esmfold_environment.yml``. This module therefore delegates folding to a
subprocess.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Optional


def fold_sequences(
    fasta_input: Path,
    output_dir: Path,
    conda_env: str = "esmfold",
    max_tokens_per_batch: int = 1024,
    chunk_size: int = 64,
) -> None:
    """Run ESMFold over a FASTA file, emitting one PDB per record.

    Parameters
    ----------
    fasta_input
        Input FASTA with one record per distinct sequence.
    output_dir
        Where to write ``<record_id>.pdb`` files.
    conda_env
        Name of the conda environment containing the ``esm`` package.
    max_tokens_per_batch / chunk_size
        ESMFold throughput knobs; safe defaults for an RTX 4090 24 GB or a
        16 GB card with chunking.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = (
        "source $(conda info --base)/etc/profile.d/conda.sh && "
        "conda activate " + conda_env + " && "
        "python -m esm.esmfold.v1.pretrained --fasta " + str(fasta_input) + " "
        "--pdb " + str(output_dir) + " "
        "--max-tokens-per-batch " + str(max_tokens_per_batch) + " "
        "--chunk-size " + str(chunk_size)
    )
    subprocess.run(["bash", "-c", cmd], check=True)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fasta", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--conda-env", default="esmfold")
    p.add_argument("--max-tokens-per-batch", type=int, default=1024)
    p.add_argument("--chunk-size", type=int, default=64)
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    fold_sequences(
        fasta_input=args.fasta,
        output_dir=args.output_dir,
        conda_env=args.conda_env,
        max_tokens_per_batch=args.max_tokens_per_batch,
        chunk_size=args.chunk_size,
    )
