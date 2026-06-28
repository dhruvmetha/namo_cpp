#!/bin/bash
# Pack a shard of POST-PUSH npz -> scorer H5 (v2 OOD data). Sharded by npz-list (one list per task).
# Required env: LIST_DIR (dir of part_XX.txt lists), OUT_DIR. Each task packs $LIST_DIR/part_${TASK}.txt
#   -> $OUT_DIR/shard_${TASK}.h5 . Join these into the v2 training mix data_dir (';'-separated).
#SBATCH --job-name=pp_pack
#SBATCH --partition=main,main-redhat
#SBATCH --array=0-3
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=logs/pppack-%A_%a.out
#SBATCH --error=logs/pppack-%A_%a.err
set -euo pipefail
cd "$NAMO_REPO"
export PYTHONPATH="$PWD/build_python:$PWD/python:$PWD/scripts:$SAGE_REPO"
: "${LIST_DIR:?}"; : "${OUT_DIR:?}"
mkdir -p "$OUT_DIR"
LIST="$LIST_DIR/part_$(printf '%02d' ${SLURM_ARRAY_TASK_ID}).txt"
echo "task ${SLURM_ARRAY_TASK_ID}: list=$LIST ($(wc -l <"$LIST") npz) -> $OUT_DIR/shard_${SLURM_ARRAY_TASK_ID}.h5 on $(hostname)"
"$NAMO_PYTHON" -u scripts/pipeline/build_postpush_h5.py \
    --npz-list "$LIST" --out-h5 "$OUT_DIR/shard_${SLURM_ARRAY_TASK_ID}.h5"
echo "task ${SLURM_ARRAY_TASK_ID} DONE"
