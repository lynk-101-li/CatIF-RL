"""external-review round 2 B2: CSV->FASTA + directory-merge contract for ESMFold input.

Covers:

- ``csv_to_fasta``: dedup on (ProID, ProSeq'), record IDs
  ``sequence_<ProID>_var<idx>``, idx counters reset per ProID.
- ``fasta_dir_to_combined``: concatenates every .fa/.fasta under a directory
  into a single multi-record FASTA, in lexicographic filename order, with
  headers preserved verbatim.
- ``structural.build_ref_path`` with ``--ref-pattern "{ref_stem}.pdb"`` and
  ``--ref-stem-regex "^(sequence_\\d+)"``: pred stems with sample/seed
  suffixes resolve back to the ProID-keyed WT reference.

No heavy deps (only pandas).
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import pandas as pd

from catif_rl.data.csv_to_fasta import csv_to_fasta, fasta_dir_to_combined
from catif_rl.evaluation.structural import build_ref_path


def test_csv_to_fasta_dedup_and_id_counter():
    """Per-row dedup, record IDs are sequence_<ProID>_var<n> with per-ProID counters."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        csv = tmp / "cand.csv"
        pd.DataFrame({
            "ProID":   [1,   1,   1,    2,   2,   3],
            "ProSeq":  ["A", "A", "A",  "B", "B", "C"],
            "ProSeq'": ["X", "Y", "X",  "P", "Q", "R"],   # row 0 and row 2 are exact duplicates
            "extra":   [0,   1,   2,    3,   4,   5],
        }).to_csv(csv, index=False)

        out = tmp / "out.fasta"
        n = csv_to_fasta(csv, out)
        assert n == 5, f"expected 5 unique (ProID, ProSeq') rows, got {n}"

        text = out.read_text().splitlines()
        headers = [ln[1:] for ln in text if ln.startswith(">")]
        # Expected (in input order, dedup keeps first occurrence per group):
        # (1, X), (1, Y), (2, P), (2, Q), (3, R)
        assert headers == [
            "sequence_1_var0",
            "sequence_1_var1",
            "sequence_2_var0",
            "sequence_2_var1",
            "sequence_3_var0",
        ], headers


def test_csv_to_fasta_missing_columns():
    """A CSV without ProSeq' should fail with a clear error."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        csv = tmp / "bad.csv"
        pd.DataFrame({"ProID": [1, 2], "wrong_col": ["X", "Y"]}).to_csv(csv, index=False)
        out = tmp / "x.fasta"
        try:
            csv_to_fasta(csv, out)
        except KeyError as e:
            # The repr of the apostrophe in ProSeq' varies between escaping
            # styles; check for the unambiguous "ProSeq" prefix instead.
            assert "ProSeq" in str(e), str(e)
            return
        raise AssertionError("expected KeyError on missing ProSeq' column")


def test_fasta_dir_to_combined_concat_in_order():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "a.fa").write_text(">sequence_1\nAA\n")
        (tmp / "b.fasta").write_text(">sequence_2\nBB\n")
        (tmp / "c.fa").write_text(">sequence_10\nCC\n")
        combined = tmp / "combined.fasta"
        n = fasta_dir_to_combined(tmp, combined)
        assert n == 3
        text = combined.read_text()
        # Lexicographic file order: a, b, c -> sequence_1, sequence_2, sequence_10
        headers_in_order = re.findall(r"^>(\S+)", text, flags=re.MULTILINE)
        assert headers_in_order == ["sequence_1", "sequence_2", "sequence_10"], headers_in_order
        # Sequences carried through
        assert "AA" in text and "BB" in text and "CC" in text


def test_build_ref_path_resolves_via_regex():
    """For a GDC-style pred stem, the regex pulls the ProID-keyed ref name."""
    regex = re.compile(r"^(sequence_\d+)")
    out = build_ref_path("data/raw/enzymeif/train_and_validation",
                         "sequence_42_var3.pdb",
                         "{ref_stem}.pdb",
                         regex)
    assert out.endswith("data/raw/enzymeif/train_and_validation/sequence_42.pdb"), out

    # And for the benchmark case (default --ref-pattern), the legacy
    # filename-equality still works.
    out2 = build_ref_path("data/raw/test", "sequence_7.pdb", "{fname}", None)
    assert out2.endswith("data/raw/test/sequence_7.pdb"), out2


if __name__ == "__main__":
    test_csv_to_fasta_dedup_and_id_counter()
    test_csv_to_fasta_missing_columns()
    test_fasta_dir_to_combined_concat_in_order()
    test_build_ref_path_resolves_via_regex()
    print("PASS")
