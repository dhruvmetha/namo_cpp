#!/bin/bash
#SBATCH --job-name=c0collect
#SBATCH --account=general
#SBATCH --partition=main-redhat
#SBATCH --exclude=halk[0001-0159]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=32G
#SBATCH --time=18:00:00
#SBATCH --output=/scratch/dm1487/curriculum2_amarel/colossus0_1m/logs/collect_%A_%a.out
#SBATCH --error=/scratch/dm1487/curriculum2_amarel/colossus0_1m/logs/collect_%A_%a.err
set -euo pipefail

: "${NAMO_REPO:?}"
: "${ROOT:?}"
: "${BATCH_START:?}"
: "${BATCH_END:?}"
: "${WAVE:?}"
cd "$NAMO_REPO"
source env.amarel.sh >/dev/null 2>&1
export PYTHONPATH="$NAMO_REPO/build_python:$NAMO_REPO/python:$NAMO_REPO/scripts:$NAMO_REPO/scripts/sandbox:$NAMO_REPO/scripts/pipeline:${SAGE_REPO:-}:${PYTHONPATH:-}"
export NAMO_PIN_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENCV_NUM_THREADS=1

shard=350
start=$((BATCH_START + SLURM_ARRAY_TASK_ID * shard))
end=$((start + shard))
if ((end > BATCH_END)); then end=$BATCH_END; fi
echo "wave=$WAVE task=$SLURM_ARRAY_TASK_ID scenes=[$start,$end) host=$(hostname) start=$(date -Is)"
"$NAMO_PYTHON" python/namo/data_collection/modular_parallel_collection.py \
  --config-yaml python/namo/data_collection/region_opening_beast_round2_probe_car.yaml \
  --manifest "$ROOT/manifests/colossus0_1m.txt" --start-idx "$start" --end-idx "$end" --workers 12 \
  --scorer-ckpt "$ROOT/d20_finish_ranker.ckpt" --ml-device cpu --target-goal-region \
  --output-dir "$ROOT/collect" --unique-run-dir --run-name "c0_w${WAVE}_t${SLURM_ARRAY_TASK_ID}"
echo "done=$(date -Is)"
