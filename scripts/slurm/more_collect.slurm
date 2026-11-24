#!/bin/bash
# MORE's self-supervised collection: one array task per shard of the training manifest.
#
# MORE's own plain-UCT search with no network (see scripts/rl_loop/more_collect.py), so there
# is no checkpoint argument and nothing to warm up. It imports MORE's tree code from the pinned
# clone at $MORE_REF (default $NAMO_SCRATCH/reference/arc-l_more, revision 70402c0), which must
# exist on the box. Each shard writes its own JSONL and is independent.
#
# Usage (from the repo root on Amarel):
#   ROOT=$NAMO_SCRATCH/more_collect_20260921 NSHARDS=40 SCENES=2000 \
#     sbatch --array=0-39 scripts/slurm/more_collect.slurm
#
# SCENES caps how much of the manifest the campaign covers, so a pilot and the full
# run use the same script with one number changed.
#
# SIZING. The numbers in this header predate the 2026-11-24 rewrite to MORE's own search
# (300 iterations per search, re-search after every executed push) and must be re-measured
# before a campaign. Calibrate on the box you will run on. Partial shards are not lost: the
# collector flushes one JSON line per episode.
#SBATCH --job-name=more-collect
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail

: "${ROOT:?set ROOT to the campaign dir holding manifest.txt}"
: "${NSHARDS:?set NSHARDS to the array size}"
SCENES=${SCENES:-0}                       # 0 = the whole manifest
NAMO_REPO=${NAMO_REPO:-${SLURM_SUBMIT_DIR:?no NAMO_REPO and no SLURM_SUBMIT_DIR}}
cd "$NAMO_REPO"
source scripts/amarel/activate.sh
source env.amarel.sh

# CS boxes count THREADS against ulimit -u and OpenBLAS spawns one per core, so every
# pool is pinned to 1. One shard is one core; the parallelism is the array, not threads.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1
export CUDA_VISIBLE_DEVICES=""            # collection is CPU-only and loads no model

TOTAL=$(wc -l < "$ROOT/manifest.txt")
[ "$SCENES" -gt 0 ] && [ "$SCENES" -lt "$TOTAL" ] && TOTAL=$SCENES
I=${SLURM_ARRAY_TASK_ID:?}
PER=$(( (TOTAL + NSHARDS - 1) / NSHARDS ))
S=$(( I * PER ))
E=$(( S + PER ))
[ "$E" -gt "$TOTAL" ] && E=$TOTAL

echo "MORE-COLLECT start $(date) host=$(hostname) shard=$I range=[$S:$E) of $TOTAL"
if [ "$S" -ge "$E" ]; then echo "empty shard, nothing to do"; exit 0; fi

"$NAMO_PYTHON" scripts/rl_loop/more_collect.py \
  --manifest "$ROOT/manifest.txt" --start "$S" --end "$E" \
  --iterations "${ITERATIONS:-300}" --out "$ROOT/raw/shard_$(printf '%04d' "$I").jsonl"

echo "MORE-COLLECT done $(date) shard=$I"
