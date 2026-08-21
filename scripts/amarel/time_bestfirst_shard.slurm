#!/bin/bash
# FULL-SET best-first WALL-CLOCK timing, sharded. Every shard is --exclusive AND pinned to ONE CPU type
# (sbatch --constraint=icelake) so all shards run on identical hardware => pooling times is same-machine-valid.
# Within a shard, the SELECTED models (default Hz,NoHz,random; set MODELS=NoHz or MODELS=random for a subset)
# run interleaved per episode. Env: OUT_DIR, SHARD (eps/shard); opt BUDGET,HMAX,KEY,MODELS,RNG_SEED.
#SBATCH --job-name=bftime_shard
#SBATCH --partition=main,main
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=logs/bftime_shard-%A_%a.out
set -euo pipefail
cd "${NAMO_REPO:-/cache/home/dm1487/projects/namo/namo_cpp}"
source env.amarel.sh >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${OUT_DIR:?}"; : "${SHARD:?}"
mkdir -p "$OUT_DIR"
START=$(( SLURM_ARRAY_TASK_ID * SHARD )); END=$(( START + SHARD ))
KEY=${KEY:-$("$NAMO_PYTHON" -m namo.eval_sets pure2push_manifest)}
echo "shard ${SLURM_ARRAY_TASK_ID}: [$START,$END) budget=${BUDGET:-900} hmax=${HMAX:-2} on $(hostname) [$(scontrol show node "$(hostname)" 2>/dev/null | grep -oE 'icelake|sapphirerapids' | head -1)]"
"$NAMO_PYTHON" -u scripts/sandbox/time_bestfirst.py --start "$START" --end "$END" \
  --budget "${BUDGET:-900}" --hmax "${HMAX:-2}" --key "$KEY" \
  ${STRAT:+--strat} --n-per-tier "${NPER:-50}" \
  ${HZ_CKPT:+--hz-ckpt "$HZ_CKPT"} ${NOHZ_CKPT:+--nohz-ckpt "$NOHZ_CKPT"} --rng-seed "${RNG_SEED:-7}" \
  ${MODELS:+--models "$MODELS"} \
  --out "$OUT_DIR/shard_${SLURM_ARRAY_TASK_ID}.jsonl"
echo "shard ${SLURM_ARRAY_TASK_ID} DONE"
