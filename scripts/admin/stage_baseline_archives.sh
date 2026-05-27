#!/usr/bin/env bash
# Stage the five external-baseline FASTA archives (ProteinMPNN / ESM-IF /
# LigandMPNN / PiFold / ABACUS-T) into a single tar.gz suitable for upload
# to Zenodo as part of the catif_rl_baselines_v<X>.tar.gz package.
#
# This is a *maintenance* / *release-engineering* script -- not part of the
# numbered pipeline. The end-user pipeline (scripts/01..08) never invokes
# it. Run it once on a workstation that has the upstream-sampling output
# directories mounted, hand the resulting tar.gz to Zenodo, and update
# data/README.md with the resulting DOI + sha256.
#
# After the release, users reconstruct the layout by:
#
#     wget https://zenodo.org/.../catif_rl_baselines_v<X>.tar.gz
#     mkdir -p data/benchmark_baselines
#     tar -xzf catif_rl_baselines_v<X>.tar.gz -C data/benchmark_baselines/
#
# Resulting layout (consumed by scripts/06_sample_benchmark.sh):
#
#     data/benchmark_baselines/<method>/seed_<NNNN>/sequence_<id>.{fa,fasta}
#
# Usage:
#   bash scripts/admin/stage_baseline_archives.sh \
#     --src <upstream-output-root> \
#     --out <staging-output-root> \
#     [--version v0.1.0] [--dry-run]
#
# Example (for the author's workstation):
#   bash scripts/admin/stage_baseline_archives.sh \
#     --src "/Volumes/IXUNICS/Cat-IF-Rl-${DUMMY}/CatIF_RL-main/sampling" \
#     --out "/tmp/catif_rl_stage" \
#     --version v0.1.0

set -euo pipefail

# ---- arg parsing --------------------------------------------------------
SRC=""
OUT=""
VERSION="v0.1.0"
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --src)      SRC="$2"; shift 2;;
    --out)      OUT="$2"; shift 2;;
    --version)  VERSION="$2"; shift 2;;
    --dry-run)  DRY_RUN=1; shift;;
    -h|--help)
      head -30 "$0" | sed -n '/^# Usage:/,/--version/p'
      exit 0
      ;;
    *) echo "[stage][ERROR] unknown arg: $1" >&2; exit 1;;
  esac
done
[[ -z "$SRC" ]] && { echo "[stage][ERROR] --src is required" >&2; exit 1; }
[[ -z "$OUT" ]] && { echo "[stage][ERROR] --out is required" >&2; exit 1; }
[[ ! -d "$SRC" ]] && { echo "[stage][ERROR] --src does not exist: $SRC" >&2; exit 1; }

# Per-method source-path lookup. macOS default bash is 3.2 which does not
# support associative arrays, so we use a case statement that maps
# (method, seed-index) -> source path relative to --src. Seed indices
# 1..5 in the upstream layout map to CatIF-RL canonical seeds 1111..5555.
src_subpath_for() {
  local method="$1" idx="$2"
  case "$method" in
    proteinmpnn) printf "ProteinMPNN_on_Catif_test_dataset_outputs/ProteinMPNN_on_Catif_test_dataset_outputs_%s/seqs" "$idx" ;;
    esmif)       printf "ESMIF_on_Catif_test_dataset_outputs/output/ESMIF_on_Catif_test_dataset_outputs_%s" "$idx" ;;
    ligandmpnn)  printf "LigandMPNN_on_Catif_test_dataset_outputs/LigandMPNN_on_Catif_test_dataset_outputs%s" "$idx" ;;
    pifold)      printf "PiFold_on_Catif_test_dataset_outputs/PiFold_on_Catif_test_dataset_outputs%s" "$idx" ;;
    abacust)     printf "ABACUS-T_on_Catif_test_dataset_outputs/ABACUS-T_on_Catif_test_dataset_outputs%s" "$idx" ;;
    *)           echo "[stage][ERROR] unknown method: $method" >&2; return 1 ;;
  esac
}

# Canonical CatIF-RL benchmark seeds (matches scripts/lib_env.sh).
SEEDS=(1111 2222 3333 4444 5555)
METHODS=(proteinmpnn esmif ligandmpnn pifold abacust)

STAGE_DIR="$OUT/stage_$VERSION"
mkdir -p "$STAGE_DIR"
echo "[stage] staging to: $STAGE_DIR"
echo "[stage] dry-run:    $DRY_RUN"
echo ""

# ---- per-method staging -----------------------------------------------
total_files=0
for method in "${METHODS[@]}"; do
  echo "--- $method ---"
  for i in 1 2 3 4 5; do
    seed="${SEEDS[$((i-1))]}"
    subpath="$(src_subpath_for "$method" "$i")"
    src="$SRC/$subpath"
    dst="$STAGE_DIR/$method/seed_$seed"
    if [[ ! -d "$src" ]]; then
      echo "  seed_${seed}: MISSING source $src" >&2; exit 1
    fi
    n=$(find "$src" -maxdepth 1 \( -name '*.fa' -o -name '*.fasta' \) 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "  [dry-run] $method seed_$seed: would stage $n .fa/.fasta from ${src#$SRC/}"
    else
      mkdir -p "$dst"
      # Use rsync to preserve mtimes and skip already-staged files on retry.
      rsync -a --include='*.fa' --include='*.fasta' --exclude='*' \
        "$src/" "$dst/" >/dev/null
      n_dst=$(find "$dst" -maxdepth 1 \( -name '*.fa' -o -name '*.fasta' \) | wc -l | tr -d ' ')
      if [[ "$n_dst" -ne "$n" ]]; then
        echo "[stage][ERROR] $method seed_$seed: staged $n_dst but source has $n" >&2; exit 1
      fi
      sz=$(du -sh "$dst" | awk '{print $1}')
      echo "  $method seed_$seed: $sz, $n_dst files"
    fi
    total_files=$((total_files + n))
  done
done
echo ""
echo "[stage] total files: $total_files (expect 5 methods x 5 seeds x 1423 = 35,575)"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[stage] dry-run complete. Re-run without --dry-run to materialise the staging tree."
  exit 0
fi

# ---- tar + sha256 -----------------------------------------------------
TARBALL="$OUT/catif_rl_baselines_${VERSION}.tar.gz"
SHAFILE="$TARBALL.sha256"
echo ""
echo "[stage] creating tarball: $TARBALL"
# Pipe `tar` -> `gzip -n` so the gzip header carries no original filename
# or compress-time mtime; the published sha256 then only depends on the
# staged tree's contents and per-file mtimes (which rsync -a preserved
# from --src). This form is portable across GNU tar (Linux) and bsdtar
# (macOS); the previous `tar --options 'gzip:!filename'` was GNU-only and
# broke macOS staging.
( cd "$STAGE_DIR" && tar -cf - . | gzip -n > "$TARBALL" )
shasum -a 256 "$TARBALL" > "$SHAFILE"
echo ""
echo "[stage] tarball ready:"
ls -lh "$TARBALL" | awk '{print "  size:   "$5}'
echo "  sha256: $(awk '{print $1}' "$SHAFILE")"
echo ""
echo "Next steps:"
echo "  1. Upload $TARBALL to Zenodo (concept DOI 10.5281/zenodo.20357062;"
echo "     add as a new file to the existing v0.1.0 record, or create a v0.2 release)."
echo "  2. Note the resulting Zenodo URL + the sha256 above."
echo "  3. Paste both into the placeholder block at data/README.md (search for"
echo "     'CATIF_RL_BASELINES_TARBALL_URL' and 'CATIF_RL_BASELINES_TARBALL_SHA256')."
