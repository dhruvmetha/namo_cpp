#!/bin/bash
# One arm of the chained-2-hop Full-NAMO evaluation, run directly on a CS box (no SLURM).
#
# The population manifest is an INPUT, so this re-runs unchanged when the keyhole-2 chain
# grows the table. Everything else matches the Amarel showcase protocol exactly: path-length 2,
# hmax=2 per keyhole, a fresh SIM_BUDGET per keyhole, car 1x config, 1x_car_d5_ primitives.
#
# Usage: run_chained_2hop_eval.sh <manifest> <out_root> <prior> <seed> <workers> [ckpt]
set -euo pipefail
MANIFEST=${1:?manifest}
OUT_ROOT=${2:?out root}
PRIOR=${3:?model|uniform}
SEED=${4:?seed}
WORKERS=${5:?workers}
CKPT=${6:-/common/users/dm1487/scratch_namo/aquaman/round0/models/HY5U_s2/checkpoints/epoch011-val_loss0.3256.ckpt}
SIM_BUDGET=${SIM_BUDGET:-900}
PATH_LENGTH=${PATH_LENGTH:-2}

REPO=${NAMO_REPO:-/common/home/dm1487/robotics_research/ktamp/namo}
cd "$REPO"
source env.ilab.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# OpenCV keeps its OWN thread pool, sized to the machine's core count and untouched by the four
# vars above. On a 256-core CS box every worker process spawned 256 cv2 threads, so a few dozen
# workers blew straight through `ulimit -u` (2000) and the run died on "Can't spawn new thread:
# res = 11". This is the one that actually bites when moving off SLURM onto a big shared box.
export OPENCV_FOR_THREADS_NUM=${OPENCV_FOR_THREADS_NUM:-1}

OUT="$OUT_ROOT/${PRIOR}_s${SEED}"
mkdir -p "$OUT"
echo "CHAINED start $(date) host=$(hostname) prior=$PRIOR seed=$SEED workers=$WORKERS budget=$SIM_BUDGET n=$(wc -l < "$MANIFEST")"
"$NAMO_PYTHON" -m namo.solvability_runner \
  --manifest "$MANIFEST" \
  --path-length "$PATH_LENGTH" \
  --config-file config/namo_config_complete_skill15_car_1x.yaml \
  --goal-strategy scorer \
  --region-max-chain-depth 2 \
  --primitive-data-dir data \
  --primitive-prefix 1x_car_d5_ \
  --region-success-min-reachable 20 \
  --goals-per-region 100 \
  --seed "$SEED" \
  --simulation-budget "$SIM_BUDGET" \
  --simulation-budget-scope keyhole \
  --local-search best_first \
  --best-first-prior "$PRIOR" \
  --scorer-ckpt "$CKPT" \
  --ml-device cpu \
  --workers "$WORKERS" \
  --output-dir "$OUT"
echo "CHAINED done $(date) prior=$PRIOR seed=$SEED"
