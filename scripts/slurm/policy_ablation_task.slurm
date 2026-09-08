#!/bin/bash
#SBATCH --job-name=pol_ablate
#SBATCH --partition=main
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
set -euo pipefail
: "${CAMPAIGN_ROOT:?}"
: "${NAMO_REPO:?}"
: "${CAMPAIGN_STAGE:?}"
cd "$NAMO_REPO"
TASK_SAGE=${SAGE_REPO:?}
TASK_BIND=${NAMO_BINDINGS:?}
source env.amarel.sh
export SAGE_REPO="$TASK_SAGE"
export NAMO_CFG="$PWD/config/margin_5mm/namo_config_complete_skill15_car_1x.yaml"
export PYTHONPATH="$TASK_BIND:$PWD/python:$PWD/scripts:$PWD/scripts/sandbox:$PWD/scripts/rl_loop:$PWD/scripts/pipeline:$SAGE_REPO"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENCV_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
"$NAMO_PYTHON" scripts/rl_loop/policy_ablation_campaign.py "$CAMPAIGN_STAGE" --root "$CAMPAIGN_ROOT" --index "${SLURM_ARRAY_TASK_ID:-0}"
