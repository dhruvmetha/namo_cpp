#!/bin/bash
# Phase-0 GATE (EXP-2026-07-06-rl-only-self-imitation). iLab `unlimited`, sharded by xml-index. CPU-only.
# Runs the WORKTREE phase0_oracle_decomp.py against MAIN's compiled bindings on the shared FS.
# Env in: CKPT, OUT_DIR (required); H(2), MAXSETUPS(12), SHARD(50). Submit once per ckpt-seed.
#SBATCH --job-name=p0gate
#SBATCH --partition=unlimited
#SBATCH --array=0-19
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/common/users/dm1487/scratch_namo/eval/phase0_gate/logs/p0-%A_%a.out
#SBATCH --error=/common/users/dm1487/scratch_namo/eval/phase0_gate/logs/p0-%A_%a.err
set -euo pipefail
MAIN=/common/home/dm1487/robotics_research/ktamp/namo
WT=/common/home/dm1487/robotics_research/ktamp/namo/.claude/worktrees/agent-aa6c1e592dbdacbf9
cd "$MAIN"
export MJ_PATH=/common/users/dm1487/ktamp/mujoco
export SAGE_REPO=/common/home/dm1487/robotics_research/ktamp/sage_learning
export PYTHONPATH="$MAIN/build_python:$MAIN/python:$MAIN/scripts:$MAIN/scripts/sandbox:$MAIN/scripts/pipeline:$SAGE_REPO"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=/common/users/dm1487/envs/mjxrl/bin/python
: "${CKPT:?}"; : "${OUT_DIR:?}"; SHARD=${SHARD:-50}
START=$(( SLURM_ARRAY_TASK_ID * SHARD )); END=$(( START + SHARD ))
mkdir -p "$OUT_DIR" /common/users/dm1487/scratch_namo/eval/phase0_gate/logs
"$PY" -u "$WT/scripts/rl_loop/phase0_oracle_decomp.py" \
    --ckpt "$CKPT" --h "${H:-2}" --max-setups "${MAXSETUPS:-12}" \
    --start "$START" --end "$END" \
    --out "$OUT_DIR/shard_${SLURM_ARRAY_TASK_ID}.json" \
    --leaf-out "$OUT_DIR/shard_${SLURM_ARRAY_TASK_ID}.jsonl"
