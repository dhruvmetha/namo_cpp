#!/bin/bash
# Per-template parallel tar+zstd → parallel rsync to Amarel.
# Designed to run in tmux. Logs to /tmp/v2_transfer_<phase>.log.
#
# Usage:
#   tmux new -s v2xfer
#   bash scripts/transfer_v2_to_amarel.sh tar         # phase 1: parallel tar
#   bash scripts/transfer_v2_to_amarel.sh push        # phase 2: parallel rsync
#   bash scripts/transfer_v2_to_amarel.sh extract     # phase 3: print extract cmd for Amarel
#   bash scripts/transfer_v2_to_amarel.sh all         # tar then push (no extract — that runs on Amarel)

set -euo pipefail

SRC_DIR=/common/users/dm1487/corl2026/namo/envs/v2
ARCHIVE_DIR=/common/users/dm1487/v2_archives
AMAREL_USER=dm1487
AMAREL_HOST=amarel.rutgers.edu
AMAREL_DEST=/scratch/dm1487/datasets/car_envs
PARALLEL=20
ZSTD_THREADS=1
ZSTD_LEVEL=3

TEMPLATE_LIST=/tmp/v2_templates.txt
TAR_LOG=/tmp/v2_transfer_tar.log
PUSH_LOG=/tmp/v2_transfer_push.log

phase=${1:-all}

phase_tar() {
  mkdir -p "$ARCHIVE_DIR"
  cd "$SRC_DIR"
  echo "[$(date)] listing template subdirs..." | tee -a "$TAR_LOG"
  find . -mindepth 3 -maxdepth 3 -type d | sed 's|^\./||' > "$TEMPLATE_LIST"
  n=$(wc -l < "$TEMPLATE_LIST")
  echo "[$(date)] found $n template subdirs; tarring with -P $PARALLEL" | tee -a "$TAR_LOG"

  cat "$TEMPLATE_LIST" | xargs -P "$PARALLEL" -I {} bash -c '
    template="$1"
    name=$(echo "$template" | tr "/" "_")
    out="'"$ARCHIVE_DIR"'/${name}.tar.zst"
    if [ -s "$out" ]; then
      echo "[skip] $name (already exists)"
      exit 0
    fi
    tar -cf - -C "'"$SRC_DIR"'" "$template" \
      | zstd -T'"$ZSTD_THREADS"' -'"$ZSTD_LEVEL"' -o "$out" -q
    echo "[done] $name $(ls -lh "$out" | awk "{print \$5}")"
  ' _ {} 2>&1 | tee -a "$TAR_LOG"

  echo "[$(date)] tar phase complete" | tee -a "$TAR_LOG"
  du -sh "$ARCHIVE_DIR" | tee -a "$TAR_LOG"
  echo "tarball count: $(ls "$ARCHIVE_DIR"/*.tar.zst | wc -l)" | tee -a "$TAR_LOG"
}

phase_push() {
  echo "[$(date)] ensuring dest dirs on Amarel..." | tee -a "$PUSH_LOG"
  ssh "${AMAREL_USER}@${AMAREL_HOST}" \
    "mkdir -p ${AMAREL_DEST}/_archives ${AMAREL_DEST}/v2"

  echo "[$(date)] rsyncing $(ls "$ARCHIVE_DIR"/*.tar.zst | wc -l) archives with -P $PARALLEL" | tee -a "$PUSH_LOG"
  ls "$ARCHIVE_DIR"/*.tar.zst \
    | xargs -P "$PARALLEL" -I {} \
        rsync -a --partial {} \
          "${AMAREL_USER}@${AMAREL_HOST}:${AMAREL_DEST}/_archives/" \
        2>&1 | tee -a "$PUSH_LOG"

  echo "[$(date)] push complete" | tee -a "$PUSH_LOG"
}

phase_extract_hint() {
  cat <<EOF
# Run this on Amarel inside a compute alloc:
#   ssh ${AMAREL_USER}@${AMAREL_HOST}
#   srun --partition=main-redhat --cpus-per-task=32 --mem=32G --time=2:00:00 --pty bash
cd ${AMAREL_DEST}
ls _archives/*.tar.zst \\
  | xargs -P 20 -I {} sh -c '
      zstd -d -T1 -c "{}" | tar -xf - -C v2/
    '
# verify count, then:
# rm -rf _archives/
EOF
}

case "$phase" in
  tar)      phase_tar ;;
  push)     phase_push ;;
  extract)  phase_extract_hint ;;
  all)      phase_tar; phase_push; phase_extract_hint ;;
  *)        echo "usage: $0 [tar|push|extract|all]"; exit 2 ;;
esac
