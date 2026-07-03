#!/bin/bash
# iLab twin of scripts/amarel/reactive_argmax.slurm — PURE reactive (argmax setup@H -> argmax finish@H1),
# sharded by xml-index. Runs model-free RANDOM (PRIOR=uniform) or model (PRIOR=q) on CPU.
# Env in: CKPT, OUT_DIR (required); PRIOR(q), SEED(7000), H(2), KEY(default pure2push), SHARD(80), ARRAY(0-12).
# iLab specifics: partition=unlimited, NO --cpus-per-task (partition rejects it), shared-FS paths.
#SBATCH --job-name=reactarg
#SBATCH --partition=unlimited
#SBATCH --array=0-12
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/reactarg_ilab-%A_%a.out
#SBATCH --error=logs/reactarg_ilab-%A_%a.err
set -euo pipefail
REPO=/common/home/dm1487/robotics_research/ktamp/namo
cd "$REPO"
export MJ_PATH=/common/users/dm1487/ktamp/mujoco
export PYTHONPATH="$REPO/build_python:$REPO/python:$REPO/scripts:$REPO/scripts/sandbox:$REPO/scripts/pipeline:/common/home/dm1487/robotics_research/ktamp/sage_learning"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=/common/users/dm1487/envs/mjxrl/bin/python
: "${CKPT:?}"; : "${OUT_DIR:?}"; SHARD=${SHARD:-80}
START=$(( SLURM_ARRAY_TASK_ID * SHARD )); END=$(( START + SHARD )); mkdir -p "$OUT_DIR" logs
"$PY" -u scripts/sandbox/eval_reactive_argmax.py \
    --ckpt "$CKPT" --prior "${PRIOR:-q}" --seed "${SEED:-7000}" --h "${H:-2}" ${KEY:+--key "$KEY"} \
    --start "$START" --end "$END" \
    --out "$OUT_DIR/shard_${SLURM_ARRAY_TASK_ID}.json" --leaf-out "$OUT_DIR/shard_${SLURM_ARRAY_TASK_ID}.jsonl"
