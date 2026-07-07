#!/bin/bash
# Per-generation 2push testset eval: reactive greedy open@1/@2 on namo_testset_v1/pure2push,
# stratified by division via agg_testset_reactive.py. This is the 2push row + kill-signal-2 the
# h1 pool's dev eval can't give. iLab `unlimited`, CPU-only, sharded by xml-index.
#   Env in: CKPT, OUT_DIR (required); KEY, SHARD(62), MAXP(2). One array per pi ckpt.
#   Submit: CKPT=<pi.ckpt> OUT_DIR=<dir> sbatch --array=0-15 scripts/rl_loop/testset_reactive.slurm
#   Then:   python scripts/rl_loop/agg_testset_reactive.py --leaf-glob "<dir>/shard_*.jsonl" \
#             --divisions <labels>/pure2push_divisions.json --out <dir>/agg.json
#SBATCH --job-name=ts_react
#SBATCH --partition=unlimited
#SBATCH --array=0-15
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/common/users/dm1487/scratch_namo/rl_runs/logs/tsreact-%A_%a.out
#SBATCH --error=/common/users/dm1487/scratch_namo/rl_runs/logs/tsreact-%A_%a.err
set -euo pipefail
MAIN=/common/home/dm1487/robotics_research/ktamp/namo
cd "$MAIN"
source env.ilab.sh >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=/common/users/dm1487/envs/mjxrl/bin/python
: "${CKPT:?set CKPT}"; : "${OUT_DIR:?set OUT_DIR}"
KEY=${KEY:-/common/users/dm1487/scratch_namo/datasets/namo_testset_v1/labels/pure2push.json}
SHARD=${SHARD:-62}; MAXP=${MAXP:-2}
START=$(( SLURM_ARRAY_TASK_ID * SHARD )); END=$(( START + SHARD ))
mkdir -p "$OUT_DIR" /common/users/dm1487/scratch_namo/rl_runs/logs
"$PY" -u scripts/sandbox/eval_reactive_argmax.py \
  --ckpt "$CKPT" --key "$KEY" --start "$START" --end "$END" \
  --max-pushes "$MAXP" --prior q \
  --out "$OUT_DIR/shard_${SLURM_ARRAY_TASK_ID}.json" \
  --leaf-out "$OUT_DIR/shard_${SLURM_ARRAY_TASK_ID}.jsonl"
