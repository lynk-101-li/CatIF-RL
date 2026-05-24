#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute structural / sequence metrics for a directory of predicted PDB
files against a reference PDB directory.

Supported metrics (selectable via ``--metrics``):
    - CA-RMSD
    - Backbone-RMSD
    - Mean pLDDT (from B-factor)
    - Recovery rate (residue-level sequence identity vs. reference)

The script writes one CSV row per prediction with the columns
``["ProID", "ProSeq", "ProSeq'", "Recovery_rate", "CA_RMSD",
"Backbone_RMSD", "Avg_pLDDT"]`` and appends a final ``"Mean"`` row
averaging the requested metric columns.

Examples (as invoked by ``scripts/03_run_gdc.sh`` and ``07_score_benchmark.sh``)
-------------------------------------------------------------------------------

Default RMSD + pLDDT pass on the held-out benchmark refolds::

    python -m catif_rl.evaluation.structural \\
      --ref-dir  data/raw/test \\
      --pred-dir runs/benchmark_scores/catif_rl_r3/refold_seed1111 \\
      --csv-out  runs/benchmark_scores/catif_rl_r3/rmsd_plddt.csv

All three metrics, used during the GDC structural gate::

    python -m catif_rl.evaluation.structural \\
      --ref-dir  data/raw/enzymeif/train_and_validation \\
      --pred-dir runs/gdc/refold \\
      --csv-out  runs/gdc/rmsd_plddt_recovery.csv \\
      --metrics  rmsd,plddt,recovery
"""

import os, re, csv, argparse
import numpy as np
from Bio.PDB import PDBParser, Superimposer

AA3_TO_1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C",
    "GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I",
    "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
    "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
}

OUT_COLS = ["ProID","ProSeq","ProSeq'","Recovery_rate","CA_RMSD","Backbone_RMSD","Avg_pLDDT"]

def get_sequence(pdb_file: str) -> str:
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("tmp", pdb_file)
    seq = []
    for res in struct.get_residues():
        if res.get_id()[0] != " ":
            continue
        aa3 = res.get_resname().upper()
        seq.append(AA3_TO_1.get(aa3, "X"))
    return "".join(seq)

def align_proteins_and_calculate_rmsd(ref_pdb: str, pred_pdb: str):
    parser = PDBParser(QUIET=True)
    s1 = parser.get_structure("ref", ref_pdb)
    s2 = parser.get_structure("pred", pred_pdb)

    ca1 = [a for a in s1.get_atoms() if a.get_name() == "CA"]
    ca2 = [a for a in s2.get_atoms() if a.get_name() == "CA"]
    bb1 = [a for a in s1.get_atoms() if a.get_name() in ("CA","C","N","O")]
    bb2 = [a for a in s2.get_atoms() if a.get_name() in ("CA","C","N","O")]

    if len(ca1) != len(ca2) or len(bb1) != len(bb2):
        raise ValueError(f"atom count mismatch: CA({len(ca1)} vs {len(ca2)}), BB({len(bb1)} vs {len(bb2)})")

    sup = Superimposer()
    sup.set_atoms(ca1, ca2)
    ca_rmsd = sup.rms

    sup.set_atoms(bb1, bb2)
    bb_rmsd = sup.rms
    return ca_rmsd, bb_rmsd

def average_plddt(pdb_file: str) -> float:
    parser = PDBParser(QUIET=True)
    model = parser.get_structure("pred", pdb_file)
    bf = [a.get_bfactor() for a in model.get_atoms()]
    return float(np.mean(bf)) if bf else float("nan")

def recovery_rate(ref_seq: str, pred_seq: str) -> float:
    L = min(len(ref_seq), len(pred_seq))
    if L == 0:
        return float("nan")
    if len(ref_seq) != len(pred_seq):
        print(f"[WARN] sequence length mismatch: ref={len(ref_seq)} pred={len(pred_seq)}, computing recovery on min={L}")
    match = sum(1 for i in range(L) if ref_seq[i] == pred_seq[i])
    return match / L

def safe_proid_from_fname(fname: str) -> str:
    nums = re.findall(r"\d+", fname)
    return nums[0] if nums else os.path.splitext(fname)[0]

def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

def build_ref_path(ref_dir: str, pred_fname: str, ref_pattern: str) -> str:
    stem = os.path.splitext(pred_fname)[0]
    ref_name = ref_pattern.format(fname=pred_fname, stem=stem)
    return os.path.join(ref_dir, ref_name)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir",  required=True, help="reference PDB directory (e.g. data/raw/test)")
    ap.add_argument("--pred-dir", required=True, help="predicted PDB directory (e.g. runs/benchmark_scores/<method>/refold_seedXXXX)")
    ap.add_argument("--csv-out",  required=True, help="output CSV path (e.g. runs/benchmark_scores/<method>/rmsd_plddt.csv)")
    ap.add_argument("--ref-pattern", default="{fname}", help='reference filename template; default is matching basename, others such as "{stem}.pdb" supported')
    ap.add_argument(
        "--metrics",
        default="rmsd,plddt",
        help="metrics to compute, comma-separated: rmsd,plddt,recovery (default: rmsd,plddt)"
    )
    return ap.parse_args()

def main():
    args = parse_args()
    ref_dir, pred_dir, csv_out, ref_pattern = args.ref_dir, args.pred_dir, args.csv_out, args.ref_pattern
    metrics = {m.strip().lower() for m in args.metrics.split(",") if m.strip()}

    do_rmsd = "rmsd" in metrics
    do_plddt = "plddt" in metrics
    do_recovery = "recovery" in metrics

    if not os.path.isdir(pred_dir):
        raise SystemExit(f"pred_dir does not exist or is not a directory: {pred_dir}")
    if not os.path.isdir(ref_dir):
        print(f"[WARN] ref_dir does not exist or is not a directory: {ref_dir}")

    rows = []
    for fname in sorted(os.listdir(pred_dir)):
        if not fname.lower().endswith(".pdb"):
            continue

        pred_p = os.path.join(pred_dir, fname)
        ref_p  = build_ref_path(ref_dir, fname, ref_pattern)

        if not os.path.exists(ref_p):
            print(f"[MISS] reference structure missing: {os.path.basename(ref_p)}  (pred={fname})")
            continue

        try:
            pro_id   = safe_proid_from_fname(fname)
            ref_seq  = get_sequence(ref_p)
            pred_seq = get_sequence(pred_p)

            rec = ca_r = bb_r = plddt = float("nan")

            if do_recovery:
                rec = recovery_rate(ref_seq, pred_seq)
            if do_rmsd:
                ca_r, bb_r = align_proteins_and_calculate_rmsd(ref_p, pred_p)
            if do_plddt:
                plddt = average_plddt(pred_p)

            msg = [fname]
            if do_recovery: msg.append(f"Rec={rec:.3f}")
            if do_rmsd:     msg.append(f"CA={ca_r:.3f} BB={bb_r:.3f}")
            if do_plddt:    msg.append(f"pLDDT={plddt:.2f}")
            print("  ".join(msg))

            rows.append([pro_id, ref_seq, pred_seq, rec, ca_r, bb_r, plddt])

        except Exception as e:
            print(f"[ERROR] {fname} failed: {e}")

    if not rows:
        print("[WARN] no valid results; CSV not written")
        return

    # Compute Mean row: only over metrics that were actually requested.
    def mean_or_blank(vals, enabled: bool):
        if not enabled:
            return ""
        arr = np.array(vals, dtype=float)
        arr = arr[~np.isnan(arr)]
        return float(np.mean(arr)) if arr.size else ""

    rec_vals = [r[3] for r in rows]
    ca_vals  = [r[4] for r in rows]
    bb_vals  = [r[5] for r in rows]
    pld_vals = [r[6] for r in rows]

    mean_row = [
        "Mean", "", "",
        mean_or_blank(rec_vals, do_recovery),
        mean_or_blank(ca_vals,  do_rmsd),
        mean_or_blank(bb_vals,  do_rmsd),
        mean_or_blank(pld_vals, do_plddt),
    ]

    ensure_parent_dir(csv_out)
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(OUT_COLS)
        w.writerows(rows)
        w.writerow(mean_row)

    print(f"\n[OK] wrote {csv_out}  ({len(rows)} records)")

if __name__ == "__main__":
    main()
