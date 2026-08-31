#!/bin/bash
# Target-box smoke or full three-seed HY5U ablation fleet on an Amarel SLURM GPU allocation.
# Submit smoke: sbatch --array=0-2 --export=ALL,MODE=smoke scripts/amarel/hy5u_ablations_train.slurm
# Submit full after calibrating the smoke: sbatch --array=0-8 --export=ALL,MODE=full ...
#SBATCH --job-name=hy5u_ab
#SBATCH --account=general
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail

REPO=${NAMO_REPO:-/cache/home/dm1487/projects/namo/namo_cpp}
cd "$REPO"
source env.amarel.sh >/dev/null 2>&1

MODE=${MODE:-smoke}
R0=${HY5U_R0:-$NAMO_SCRATCH/aquaman/round0}
H5=${HY5U_H5:-$R0/hybrid_train_v1.h5}
TASK=${SLURM_ARRAY_TASK_ID:-0}

arms=(HY5U_no_family HY5U_regression HY5U_independent)
if [ "$MODE" = "smoke" ]; then
  [ "$TASK" -lt 3 ] || { echo "smoke array must be 0-2" >&2; exit 2; }
  ARM=${arms[$TASK]}
  SEED=1
  EPOCHS=1
  OUT=$R0/ablations_20260831/smoke_amarel/${ARM}_s1
else
  [ "$MODE" = "full" ] || { echo "unknown MODE=$MODE" >&2; exit 2; }
  [ "$TASK" -lt 9 ] || { echo "full array must be 0-8" >&2; exit 2; }
  ARM=${arms[$((TASK / 3))]}
  SEED=$((TASK % 3 + 1))
  EPOCHS=12
  OUT=$R0/ablations_20260831/models/${ARM}_s${SEED}
fi

case "$ARM" in
  HY5U_no_family)  EGMM=0;   RANK=0.1; LOWER=0.05; EDGE=1 ;;
  HY5U_regression) EGMM=0;   RANK=0;   LOWER=0;    EDGE=1 ;;
  HY5U_independent) EGMM=0.1; RANK=0.1; LOWER=0.05; EDGE=0 ;;
esac

[ -f "$H5" ] || { echo "missing H5: $H5" >&2; exit 1; }
[ ! -e "$OUT/train.log" ] || { echo "refusing to overwrite: $OUT" >&2; exit 1; }
mkdir -p "$OUT"

echo "ABLATION mode=$MODE arm=$ARM seed=$SEED task=$TASK job=$SLURM_JOB_ID host=$(hostname)"
env NAMO_REPO="$REPO" NAMO_ENV_SCRIPT=env.amarel.sh NAMO_USE_SRUN=1 \
  H5="$H5" OUT="$OUT" TRAIN_SCRIPT=scripts/rl_loop/train_q2_round2.py \
  EPOCHS="$EPOCHS" BATCH=256 WORKERS=6 SEED="$SEED" POSTCHECK_LIMIT=64 \
  NAMO_GAMMA=0.5 NAMO_UNREACH_W=0.1 NAMO_GROUP_EPISODES=1 \
  EGMM_LAMBDA="$EGMM" RANK_LAMBDA="$RANK" LOWER_RANK_LAMBDA="$LOWER" \
  NAMO_EDGE_SELF_ATTN="$EDGE" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  bash scripts/slurm/train.slurm
