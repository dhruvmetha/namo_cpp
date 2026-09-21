#!/bin/bash
# MORE's self-supervised collection: one array task per shard of the training manifest.
#
# Plain UCT with no network (see scripts/rl_loop/more_collect.py), so there is no
# checkpoint argument and nothing to warm up. Each shard writes its own JSONL and is
# independent, so a failed task is just a resubmit of that index.
#
# Usage (from the repo root on Amarel):
#   ROOT=$NAMO_SCRATCH/more_collect_20260921 NSHARDS=40 SCENES=2000 \
#     sbatch --array=0-39 scripts/slurm/more_collect.slurm
#
# SCENES caps how much of the manifest the campaign covers, so a pilot and the full
# run use the same script with one number changed.
#
# SIZING. An episode costs roughly 400 s on an Amarel core at budget 600 with
# stop_on_open off, against 102 s measured on arrakis: the cores are slower and dense
# scenes with many reachable contacts are far slower than the mean. A scene yields about
# 1.35 episodes. So 20 scenes per shard is ~3 h and overruns a 3 h wall, which is exactly
# what happened on 51 of 100 tasks in job 61727373. Calibrate on the box you will run on,
# not the one you are typing on. Keep shards at 10 scenes or fewer and the wall at 8 h.
# Partial shards are not lost: the collector flushes one JSON line per episode.
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
  --budget "${BUDGET:-600}" --out "$ROOT/raw/shard_$(printf '%04d' "$I").jsonl"

echo "MORE-COLLECT done $(date) shard=$I"
