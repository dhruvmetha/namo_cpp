#!/bin/bash
# ============================================================================================
# FAMILY corpus collection template (Amarel main-redhat). One array task = one manifest slice.
# All three campaign passes use THIS file, differing only in CONFIG + MANIFEST + slice size:
#   BASE:       CONFIG=region_opening_family_car.yaml  (capped-12, censored-honest)
#   AUDIT:      CONFIG=region_opening_tree_car.yaml    MANIFEST=<random 2% slice>
#   ESCALATION: CONFIG=region_opening_tree_car.yaml    MANIFEST=<suspect episodes>
# Sizing = the Colossus-measured shape: waves <=470 tasks, XMLS_PER_TASK=350, 14 CPU / 12 workers
# (see EXP-2026-07-21 §Compute; profiling: 83% of worker time is push execution — width is the
# only lever). Physics gate: run scripts/check_box_sync.sh + BUILD_INFO post-5daaed5 BEFORE waves.
#
# Usage (Amarel):
#   CONFIG=python/namo/data_collection/region_opening_family_car.yaml \
#   MANIFEST=<scratch>/family_manifest.txt OUT=<scratch>/family0/base \
#   sbatch --array=0-469 scripts/slurm/family_collect.slurm
# Tunables: XMLS_PER_TASK(350) WORKERS(12)
# ============================================================================================
#SBATCH --job-name=family_collect
#SBATCH --partition=main-redhat
#SBATCH --cpus-per-task=14
#SBATCH --mem=24G
#SBATCH --time=3:00:00
#SBATCH --output=%x_%A_%a.out
set -uo pipefail

: "${CONFIG:?}"; : "${MANIFEST:?}"; : "${OUT:?}"
REPO=${NAMO_REPO:?set NAMO_REPO=<amarel namo checkout>}
cd "$REPO" && source env.amarel.sh >/dev/null 2>&1
PY="${NAMO_PYTHON:?env.amarel.sh must set NAMO_PYTHON}"
export PYTHONPATH="$REPO/build_python:$REPO/python:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

XPT=${XMLS_PER_TASK:-350}
TID=${SLURM_ARRAY_TASK_ID:-0}
START=$((TID * XPT)); END=$(((TID + 1) * XPT))
mkdir -p "$OUT"
echo "FAMILY task=$TID slice=[$START,$END) config=$(basename "$CONFIG") $(date) host=$(hostname)"
"$PY" python/namo/data_collection/modular_parallel_collection.py \
  --config-yaml "$CONFIG" --manifest "$MANIFEST" \
  --start-idx "$START" --end-idx "$END" \
  --workers "${WORKERS:-12}" --output-dir "$OUT/shard_${TID}"
echo "FAMILY task=$TID DONE $(date)"
