#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
## 可选计算：CA-RMSD / Backbone-RMSD / 平均 pLDDT / Recovery rate
## 并把 ProID、参考序列、预测序列 一并写入 CSV
## 最后一行求平均

# 输出列固定：
["ProID","ProSeq","ProSeq'","Recovery_rate","CA_RMSD","Backbone_RMSD","Avg_pLDDT"]

# 默认 metrics = rmsd,plddt
python eval_metrics.py \
  --ref-dir  dataset/raw/test \
  --pred-dir sampling/mut_seq2pdb/pdb_Catif_RL_Nov26_test_mut_1 \
  --csv-out  evaluation/pred_output_rmsd_plddt_table/metrics_default.csv

# rmsd, plddt,rr都计算
python evaluation/rmsd_plddt_eval_new.py \
  --ref-dir  dataset/raw/test \
  --pred-dir sampling/mut_seq2pdb/test_output_ProteinMPNN_pdb_1 \
  --csv-out  evaluation/pred_output_rmsd_plddt_table/rmsd_plddt_proteinmpnn_test_mut_1.csv \
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
        print(f"⚠️ 序列长度不一致：ref={len(ref_seq)} pred={len(pred_seq)}，按 min={L} 计算 recovery")
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
    ap.add_argument("--ref-dir",  default="dataset/raw/test", help="参考 PDB 目录")
    ap.add_argument("--pred-dir", default="sampling/mut_seq2pdb/pdb_Catif_RL_Nov26_test_mut_1", help="预测 PDB 目录")
    ap.add_argument("--csv-out",  default="evaluation/pred_output_rmsd_plddt_table/metrics.csv", help="输出 CSV 路径")
    ap.add_argument("--ref-pattern", default="{fname}", help='参考文件名模板：默认同名，可用 "{stem}.pdb" 等')
    ap.add_argument(
        "--metrics",
        default="rmsd,plddt",
        help="要计算的指标，逗号分隔：rmsd,plddt,recovery（默认 rmsd,plddt）"
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
        raise SystemExit(f"pred_dir 不存在或不是目录：{pred_dir}")
    if not os.path.isdir(ref_dir):
        print(f"⚠️ ref_dir 不存在或不是目录：{ref_dir}")

    rows = []
    for fname in sorted(os.listdir(pred_dir)):
        if not fname.lower().endswith(".pdb"):
            continue

        pred_p = os.path.join(pred_dir, fname)
        ref_p  = build_ref_path(ref_dir, fname, ref_pattern)

        if not os.path.exists(ref_p):
            print(f"❗ 缺失参考结构: {os.path.basename(ref_p)}  (pred={fname})")
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
            print(f"❌ {fname} 处理失败: {e}")

    if not rows:
        print("⚠️ 无有效结果，未生成 CSV")
        return

    # 计算 Mean：仅对“实际计算”的列求均值；没算的列留空
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

    print(f"\n✅ 已写入 {csv_out}  (共 {len(rows)} 条记录)")

if __name__ == "__main__":
    main()