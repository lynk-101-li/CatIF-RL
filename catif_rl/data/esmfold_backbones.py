"""ESMFold backbone prediction wrapper.

Refolds a set of input protein sequences into PDB structures using ESMFold
(Lin et al., 2023). One PDB is written per FASTA record into the
``--output-dir`` (record ID becomes the PDB stem).

The ``--fasta`` argument is flexible and accepts any of:

- A single multi-record ``.fasta`` / ``.fa`` file. Passed to ESMFold directly.
- A directory containing per-record ``.fasta`` / ``.fa`` files (the layout
  produced by ``catif_rl.sampling.{infer,inpaint}`` and by the archived
  baseline outputs). Auto-concatenated into a temporary combined FASTA
  before invoking ESMFold.
- A ``.csv`` file with at least the columns ``ProID`` and ``ProSeq'``
  (the schema produced by ``catif_rl.sampling.batch`` and consumed by the
  GDC funnel). Auto-converted via ``catif_rl.data.csv_to_fasta`` --
  records are deduplicated on ``(ProID, ProSeq')`` and IDed as
  ``sequence_<ProID>_var<idx>``.

ESMFold runs in its own conda environment (``esmfold``) -- see
``environment.yml`` cross-references and ``model_environments_yml_database/
esmfold_environment.yml``. This module therefore delegates folding to a
subprocess.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


def _resolve_input_to_fasta(fasta_arg: Path) -> tuple[Path, Optional[tempfile.TemporaryDirectory]]:
    """Translate the polymorphic ``--fasta`` argument into one FASTA path.

    Returns ``(fasta_path, owner)``. If a temporary file/directory was
    created to hold the materialised FASTA, ``owner`` is the
    ``TemporaryDirectory`` that owns it (caller must keep it alive until
    the subprocess has consumed the file); otherwise ``owner`` is ``None``.
    """
    p = Path(fasta_arg)
    if not p.exists():
        raise FileNotFoundError(f"--fasta argument does not exist: {p}")

    # Case 1: directory of per-record FASTAs -- concat into one tmp file.
    if p.is_dir():
        from catif_rl.data.csv_to_fasta import fasta_dir_to_combined
        tmp = tempfile.TemporaryDirectory(prefix="esmfold_combined_")
        out = Path(tmp.name) / "combined.fasta"
        n = fasta_dir_to_combined(p, out)
        print(f"[esmfold] merged {n} .fa/.fasta file(s) from {p} -> {out}")
        return out, tmp

    # Case 2: CSV -- transcode via csv_to_fasta.
    if p.suffix.lower() == ".csv":
        from catif_rl.data.csv_to_fasta import csv_to_fasta
        tmp = tempfile.TemporaryDirectory(prefix="esmfold_from_csv_")
        out = Path(tmp.name) / (p.stem + ".fasta")
        n = csv_to_fasta(p, out)
        print(f"[esmfold] converted {p.name} ({n} unique (ProID, ProSeq') records) -> {out}")
        return out, tmp

    # Case 3: single FASTA -- use as-is.
    if p.suffix.lower() in (".fa", ".fasta"):
        return p, None

    raise ValueError(
        f"unsupported --fasta target {p!s}; expected a .fasta/.fa file, "
        "a directory of such files, or a CSV with ProID/ProSeq' columns"
    )


def fold_sequences(
    fasta_input: Path,
    output_dir: Path,
    conda_env: str = "esmfold",
    max_tokens_per_batch: int = 1024,
    chunk_size: int = 64,
) -> None:
    """Run ESMFold over a FASTA file (or CSV/dir; auto-detected), emitting PDBs.

    Parameters
    ----------
    fasta_input
        Input source. May be a single ``.fasta`` / ``.fa`` file, a
        directory of such files (auto-concatenated), or a CSV with
        ``ProID`` and ``ProSeq'`` columns (auto-converted; see module
        docstring).
    output_dir
        Where to write ``<record_id>.pdb`` files.
    conda_env
        Name of the conda environment containing the ``esm`` package.
    max_tokens_per_batch / chunk_size
        ESMFold throughput knobs; safe defaults for an RTX 4090 24 GB or
        a 16 GB card with chunking.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fasta_path, _tmp_owner = _resolve_input_to_fasta(fasta_input)

    cmd = (
        "source $(conda info --base)/etc/profile.d/conda.sh && "
        "conda activate " + conda_env + " && "
        "python -m esm.esmfold.v1.pretrained --fasta " + str(fasta_path) + " "
        "--pdb " + str(output_dir) + " "
        "--max-tokens-per-batch " + str(max_tokens_per_batch) + " "
        "--chunk-size " + str(chunk_size)
    )
    try:
        subprocess.run(["bash", "-c", cmd], check=True)
    finally:
        if _tmp_owner is not None:
            _tmp_owner.cleanup()


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fasta", type=Path, required=True,
                   help="input FASTA file, OR directory of FASTAs, OR CSV with "
                        "ProID/ProSeq' columns (auto-detected by extension/type)")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="directory to write <record_id>.pdb files")
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
