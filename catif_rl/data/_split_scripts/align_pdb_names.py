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
        # Strict: keep only 20AA and the conventional 'X' (CSV typically has no X)
        return "".join(ch if ch in VALID20 else "X" for ch in s)
    else:
        # Loose: map every non-20AA to X
        return "".join(ch if ch in VALID20 else "X" for ch in s)

def pdb_chains_to_seqs(pdb_path: Path, prefer_longest: bool, strict: bool):
    """Extract per-chain peptide sequences from a PDB (list of (chain_id, seq))."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("model", str(pdb_path))
    ppb = PPBuilder()

    chain_seqs = []
    # Iterate chains, assemble peptides, concatenate sequence
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
        # Put the longest chain first so it is tried first
        chain_seqs.sort(key=lambda x: len(x[1]), reverse=True)
    return chain_seqs

def load_sequence_id_map(train_csv, dev_csv, test_csv, strict: bool):
    """Read the three tables and build {normalized_enzyme_seq -> set(distc_pro_num)}."""
    dfs = []
    for p in [train_csv, dev_csv, test_csv]:
        df = pd.read_csv(p)
        # Keep only the needed columns
        required = {"distc_pro_num", "enzyme"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{p} is missing columns: {missing}")
        dfs.append(df[["distc_pro_num", "enzyme"]].copy())

    all_df = pd.concat(dfs, ignore_index=True)
    all_df["norm_enzyme"] = all_df["enzyme"].astype(str).map(lambda s: normalize_seq(s, strict=strict))

    seq2ids = defaultdict(set)
    for _, row in all_df.iterrows():
        seq2ids[row["norm_enzyme"]].add(int(row["distc_pro_num"]))
    return seq2ids

def main():
    ap = argparse.ArgumentParser(description="Align PDB sequences in pdb_output/ to BRENDA enzymes from the three tables and rename to sequence__{distc_pro_num}.pdb")
    ap.add_argument("--pdb_dir", required=True, help="PDB folder (containing sequence_xxx.pdb)")
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--dev_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--strict", action="store_true", default=True, help="strict mode: 20AA only, non-standard residues mapped to X (default on)")
    ap.add_argument("--no-strict", dest="strict", action="store_true", help="disable strict mode: loosen matching, treat every non-20AA as X")
    ap.add_argument("--prefer-longest-chain", action="store_true", default=True, help="for multi-chain entries, prefer the longest chain (default on)")
    ap.add_argument("--no-prefer-longest-chain", dest="prefer_longest_chain", action="store_false")
    args = ap.parse_args()

    pdb_dir = Path(args.pdb_dir).resolve()
    if not pdb_dir.exists() or not pdb_dir.is_dir():
        raise SystemExit(f"PDB directory does not exist: {pdb_dir}")

    # Output directory: brenda_seq_pdb, sibling of pdb_dir
    out_dir = pdb_dir.parent / "brenda_seq_pdb"
    out_dir.mkdir(exist_ok=True, parents=True)

    # Read the three tables and build sequence -> ID map
    seq2ids = load_sequence_id_map(args.train_csv, args.dev_csv, args.test_csv, strict=args.strict)

    # Walk the PDB files
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

        # Try chains one at a time (longest first if prefer_longest is on)
        for chain_id, norm_seq in chain_seqs:
            if norm_seq in seq2ids:
                matched_ids = seq2ids[norm_seq]
                if len(matched_ids) == 1:
                    picked_id = list(matched_ids)[0]
                    picked_chain = chain_id
                    status = "ok"
                    break
                else:
                    # Multi-match (one sequence linked to multiple distc_pro_num)
                    picked_id = sorted(list(matched_ids))[0]  # deterministic pick (smallest ID)
                    picked_chain = chain_id
                    status = "multi_match"
                    break

        if status == "ok":
            matched += 1
            new_name = f"sequence_{picked_id}.pdb"
            dst = out_dir / new_name
            # Copy *or* hardlink/symlink? Prefer copy for portability.
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

    # Save alignment report
    report_df = pd.DataFrame(records)
    report_path = out_dir / "align_report.csv"
    report_df.to_csv(report_path, index=False)

    print(f"[DONE] files processed: {processed}")
    print(f"  - matched: {matched}")
    print(f"  - multi-match (smallest ID picked): {multi_match}")
    print(f"  - not matched: {no_match}")
    print(f"output directory: {out_dir}")
    print(f"alignment report: {report_path}")

if __name__ == "__main__":
    main()
