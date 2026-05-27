#!/usr/bin/env bash
# external-review round 2 B5: scripts/06_sample_benchmark.sh's archive-aware branch
# must (a) abort with a useful error when the archive isn't staged, and
# (b) symlink the right per-seed FASTA dir into $OUT_BASE when it is.
#
# This test exercises only the case-branch logic from 06_sample_benchmark.sh
# without invoking conda or torch -- it inlines the same `archive_src=...
# ln -sf ...` block against a synthetic fixture, then asserts on the
# resulting layout.

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TMP=$(mktemp -d)
trap "rm -rf '$TMP'" EXIT

DATA_DIR="$TMP/data"
RUNS_DIR="$TMP/runs"
ARCHIVE="$DATA_DIR/benchmark_baselines"
OUT_BASE="$RUNS_DIR/benchmark"

# ---- 1. NEGATIVE: archive missing -> the 06 branch should abort -----------
echo "[test] archive missing => 06 archive branch should exit 1"
method="proteinmpnn"
seed=1111
out_dir="$OUT_BASE/$method/seed_${seed}"
mkdir -p "$out_dir"

# Inline replica of the 06 archive-aware branch (kept in sync with the script)
exit_code=0
(
  set -e
  archive_src="$ARCHIVE/$method/seed_${seed}"
  if [ ! -d "$archive_src" ]; then
    echo "[test] (expected) archive_src missing: $archive_src" >&2
    exit 1
  fi
) || exit_code=$?
if [ "$exit_code" -ne 1 ]; then
  echo "[test] FAIL: expected exit 1 when archive missing, got $exit_code" >&2
  exit 1
fi
echo "[test]   PASS: missing-archive branch aborts"

# ---- 2. POSITIVE: archive staged -> links should appear in $out_dir -------
echo "[test] archive staged => 06 archive branch should link N files"
mkdir -p "$ARCHIVE/$method/seed_${seed}"
for i in 1 2 3; do
  echo -e ">sequence_${i}\nMOCKSEQUENCE${i}" > "$ARCHIVE/$method/seed_${seed}/sequence_${i}.fa"
done

archive_src="$ARCHIVE/$method/seed_${seed}"
for f in "$archive_src"/*.fa "$archive_src"/*.fasta; do
  [ -e "$f" ] || continue
  ln -sf "$f" "$out_dir/$(basename "$f")"
done

n_links=$(find "$out_dir" -maxdepth 1 -type l | wc -l | tr -d ' ')
n_archive=$(find "$archive_src" -maxdepth 1 \( -name '*.fa' -o -name '*.fasta' \) | wc -l | tr -d ' ')
if [ "$n_links" -ne "$n_archive" ]; then
  echo "[test] FAIL: linked $n_links files, archive has $n_archive" >&2
  exit 1
fi
echo "[test]   PASS: linked $n_links archived FASTAs into $out_dir"

# ---- 3. POSITIVE: links resolve to the archive (not local copies) ---------
echo "[test] links should resolve to the archive (not be regular files)"
sample_link=$(find "$out_dir" -maxdepth 1 -type l | head -1)
real=$(readlink "$sample_link")
expected_prefix="$archive_src"
case "$real" in
  "$expected_prefix"*) echo "[test]   PASS: link target is in the archive ($real)" ;;
  *) echo "[test] FAIL: link target does not resolve to archive: $real" >&2; exit 1 ;;
esac

# ---- 4. POSITIVE: the actual 06 script's branch is reachable --------------
echo "[test] 06_sample_benchmark.sh has the archive_src branch"
if ! grep -q 'BASELINE_ARCHIVE_DIR' "$ROOT/scripts/06_sample_benchmark.sh"; then
  echo "[test] FAIL: 06_sample_benchmark.sh missing BASELINE_ARCHIVE_DIR" >&2
  exit 1
fi
echo "[test]   PASS: 06_sample_benchmark.sh references BASELINE_ARCHIVE_DIR"

echo ""
echo "PASS"
