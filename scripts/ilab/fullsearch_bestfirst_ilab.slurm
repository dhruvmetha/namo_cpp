#!/bin/bash
# iLab twin of scripts/amarel/time_bestfirst_shard.slurm — instrumented best-first SEARCH shard on CPU.
# Used for the 1push best-first campaign (hmax=1) that never ran on Amarel. Shared-FS mjxrl python; model on CPU.
# Env in: OUT_DIR, MODELS (required); opt NOHZ_CKPT, RNG_SEED, KEY, HMAX, BUDGET, SHARD.
# iLab specifics: partition=unlimited, NO --cpus-per-task (partition rejects it), NO --gres (CPU inference).
#SBATCH --job-name=bf1push
#SBATCH --partition=unlimited
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=logs/bf1push-%A_%a.out
#SBATCH --error=logs/bf1push-%A_%a.err
set -euo pipefail
REPO=/common/home/dm1487/robotics_research/ktamp/namo
cd "$REPO"
export MJ_PATH=/common/users/dm1487/ktamp/mujoco
export PYTHONPATH="$REPO/build_python:$REPO/python:$REPO/scripts:$REPO/scripts/sandbox:$REPO/scripts/pipeline:/common/home/dm1487/robotics_research/ktamp/sage_learning"
export LD_LIBRARY_PATH="$MJ_PATH/build/lib:$MJ_PATH/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
# thread-spawn hardening: iLab login nproc ulimit propagates + OpenCV spawns worker threads ->
# "Can't spawn new thread res=11" (EAGAIN) killed shards after ~1 scene. Raise soft nproc to hard,
# pin OpenCV/BLAS to 1 thread so the model-scoring render doesn't exhaust the process thread budget.
ulimit -u "$(ulimit -Hu)" 2>/dev/null || true
export OPENCV_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/common/users/dm1487/envs/mjxrl/bin/python
: "${OUT_DIR:?}"; : "${MODELS:?}"; : "${KEY:?}"; SHARD=${SHARD:-1400}
TID=${SLURM_ARRAY_TASK_ID:-0}
START=$(( TID * SHARD )); END=$(( START + SHARD )); mkdir -p "$OUT_DIR" logs
echo "shard ${TID}: [$START,$END) models=$MODELS budget=${BUDGET:-900} hmax=${HMAX:-1} key=$(basename "$KEY") on $(hostname)"
"$PY" -u scripts/sandbox/time_bestfirst.py --start "$START" --end "$END" \
  --budget "${BUDGET:-900}" --hmax "${HMAX:-1}" --key "$KEY" \
  ${NOHZ_CKPT:+--nohz-ckpt "$NOHZ_CKPT"} ${HZ_CKPT:+--hz-ckpt "$HZ_CKPT"} --rng-seed "${RNG_SEED:-7}" --models "$MODELS" \
  --out "$OUT_DIR/shard_${TID}.jsonl"
echo "shard ${TID} DONE"
