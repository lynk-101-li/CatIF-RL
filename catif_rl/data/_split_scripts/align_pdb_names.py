'''
python dataset_src/data_split_for_gradeif_training/align_pdb_names.py \
  --pdb_dir dataset/raw/pdb_output \
  --train_csv dataset_src/data_split_for_gradeif_training/Brenda_dataset_split/train_set.csv \
  --dev_csv dataset_src/data_split_for_gradeif_training/Brenda_dataset_split/dev_set.csv \
  --test_csv dataset_src/data_split_for_gradeif_training/Brenda_dataset_split/test_set.csv \
  --no-strict
'''

# align_pdb_names.py
import argparse
import os
from pathlib import Path
import pandas as pd
from collections import defaultdict, Counter

# Biopython
from Bio.PDB import PDBParser, PPBuilder

VALID20 = set(list("ACDEFGHIKLMNPQRSTVWY"))

def normalize_seq(seq: str, strict: bool = True) -> str:
    """Uppercase, remove spaces, and (optionally) map non-20AA to X."""
    s = "".join(ch for ch in seq.upper() if ch.isalpha())
    if strict:
        # 严格模式：保留 20AA 和常见 'X'（若有，其实CSV一般不会有X）
        return "".join(ch if ch in VALID20 else "X" for ch in s)
    else:
        # 宽松模式：全部非 20AA -> X
        return "".join(ch if ch in VALID20 else "X" for ch in s)

def pdb_chains_to_seqs(pdb_path: Path, prefer_longest: bool, strict: bool):
    """从 PDB 提取每条多肽链序列（list of (chain_id, seq)）。"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("model", str(pdb_path))
    ppb = PPBuilder()

    chain_seqs = []
    # 遍历每条链，构建多肽并拼接序列
    for model in structure:
        for chain in model:
            pps = ppb.build_peptides(chain)
            if not pps:
                continue
            seq = "".join([str(pp.get_sequence()) for pp in pps])
            if seq:
                chain_seqs.append((chain.id, normalize_seq(seq, strict=strict)))

    if not chain_seqs:
        return []

    if prefer_longest:
        # 把最长的放到最前，后续优先尝试
        chain_seqs.sort(key=lambda x: len(x[1]), reverse=True)
    return chain_seqs

def load_sequence_id_map(train_csv, dev_csv, test_csv, strict: bool):
    """读取三张表，构建 {normalized_enzyme_seq -> set(distc_pro_num)} 的映射。"""
    dfs = []
    for p in [train_csv, dev_csv, test_csv]:
        df = pd.read_csv(p)
        # 只取需要的列
        required = {"distc_pro_num", "enzyme"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{p} 缺少列：{missing}")
        dfs.append(df[["distc_pro_num", "enzyme"]].copy())

    all_df = pd.concat(dfs, ignore_index=True)
    all_df["norm_enzyme"] = all_df["enzyme"].astype(str).map(lambda s: normalize_seq(s, strict=strict))

    seq2ids = defaultdict(set)
    for _, row in all_df.iterrows():
        seq2ids[row["norm_enzyme"]].add(int(row["distc_pro_num"]))
    return seq2ids

def main():
    ap = argparse.ArgumentParser(description="对齐 pdb_output/ 中 PDB 的序列到 BRENDA 三表的 enzyme，并重命名为 sequence__{distc_pro_num}.pdb")
    ap.add_argument("--pdb_dir", required=True, help="PDB 文件夹（包含 sequence_xxx.pdb）")
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--dev_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--strict", action="store_true", default=True, help="严格模式：仅 20AA，非标准残基映射为 X（默认开启）")
    ap.add_argument("--no-strict", dest="strict", action="store_true", help="关闭严格模式：放宽匹配，把非 20AA 都视作 X")
    ap.add_argument("--prefer-longest-chain", action="store_true", default=True, help="多链优先最长链（默认开启）")
    ap.add_argument("--no-prefer-longest-chain", dest="prefer_longest_chain", action="store_false")
    args = ap.parse_args()

    pdb_dir = Path(args.pdb_dir).resolve()
    if not pdb_dir.exists() or not pdb_dir.is_dir():
        raise SystemExit(f"PDB 目录不存在：{pdb_dir}")

    # 输出目录：与 pdb_dir 同级的 brenda_seq_pdb
    out_dir = pdb_dir.parent / "brenda_seq_pdb"
    out_dir.mkdir(exist_ok=True, parents=True)

    # 读取三表，构建序列->ID 映射
    seq2ids = load_sequence_id_map(args.train_csv, args.dev_csv, args.test_csv, strict=args.strict)

    # 遍历 PDB
    records = []
    processed = 0
    matched = 0
    multi_match = 0
    no_match = 0

    pdb_files = sorted([p for p in pdb_dir.iterdir() if p.suffix.lower() == ".pdb"])
    for pdb_path in pdb_files:
        processed += 1
        chain_seqs = pdb_chains_to_seqs(pdb_path, prefer_longest=args.prefer_longest_chain, strict=args.strict)
        status = "no_match"
        picked_id = None
        picked_chain = None
        matched_ids = set()

        # 逐链尝试（若启用 prefer_longest，则最长链先尝试）
        for chain_id, norm_seq in chain_seqs:
            if norm_seq in seq2ids:
                matched_ids = seq2ids[norm_seq]
                if len(matched_ids) == 1:
                    picked_id = list(matched_ids)[0]
                    picked_chain = chain_id
                    status = "ok"
                    break
                else:
                    # 多重匹配（同一条序列关联了多个 distc_pro_num）
                    picked_id = sorted(list(matched_ids))[0]  # 选一个稳定的（最小）ID
                    picked_chain = chain_id
                    status = "multi_match"
                    break

        if status == "ok":
            matched += 1
            new_name = f"sequence_{picked_id}.pdb"
            dst = out_dir / new_name
            # 复制*或*硬链接/软链接？为简单与通用性，选择复制
            data = pdb_path.read_bytes()
            dst.write_bytes(data)
        elif status == "multi_match":
            multi_match += 1
            new_name = f"sequence_{picked_id}.pdb"
            dst = out_dir / new_name
            data = pdb_path.read_bytes()
            dst.write_bytes(data)
        else:
            no_match += 1

        records.append({
            "pdb_file": str(pdb_path.name),
            "chains_parsed": ",".join(f"{cid}:{len(seq)}" for cid, seq in chain_seqs) if chain_seqs else "",
            "status": status,
            "picked_chain": picked_chain if picked_chain is not None else "",
            "picked_distc_pro_num": picked_id if picked_id is not None else "",
            "all_matched_ids_if_any": ";".join(map(str, sorted(matched_ids))) if matched_ids else ""
        })

    # 保存对齐报告
    report_df = pd.DataFrame(records)
    report_path = out_dir / "align_report.csv"
    report_df.to_csv(report_path, index=False)

    print(f"[DONE] 处理文件数: {processed}")
    print(f"  - 成功匹配: {matched}")
    print(f"  - 多重匹配（已选择最小ID）: {multi_match}")
    print(f"  - 未匹配: {no_match}")
    print(f"输出目录: {out_dir}")
    print(f"对齐报告: {report_path}")

if __name__ == "__main__":
    main()
