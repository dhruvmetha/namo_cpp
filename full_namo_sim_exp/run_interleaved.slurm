#!/bin/bash
# One SLURM array task evaluates the same frozen shard sequentially with the
# model and all five Random seeds. Rotation by shard balances arm order while
# keeping every wall-clock comparison on the same physical node.
#SBATCH --job-name=full_namo_final
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail

: "${EXPERIMENT:?set EXPERIMENT to the frozen experiment JSON}"
: "${SLURM_ARRAY_TASK_ID:?submit this script as an array}"

NAMO_REPO=${NAMO_REPO:-${SLURM_SUBMIT_DIR:?no NAMO_REPO and no SLURM_SUBMIT_DIR}}
cd "$NAMO_REPO"
if [ -n "${NAMO_ENV_FILE:-}" ]; then
    source "$NAMO_ENV_FILE"
elif [[ "$NAMO_REPO" == /common/* ]]; then
    source env.ilab.sh
elif [ -f env.amarel.sh ] && [[ "$NAMO_REPO" == /cache/* || "$NAMO_REPO" == /scratch/* ]]; then
    source env.amarel.sh
elif [ -f env.robotlearning.sh ]; then
    source env.robotlearning.sh
else
    echo "Set NAMO_ENV_FILE to this machine's NAMO environment script." >&2
    exit 1
fi

# Machine env files derive SAGE_REPO from the current checkout's parent. A Git
# worktree has no sibling Sage checkout, so fall back to the main worktree's
# sibling without weakening the provenance check.
if [ ! -d "${SAGE_REPO:-}" ]; then
    git_common_dir=$(git rev-parse --git-common-dir)
    if [[ "$git_common_dir" != /* ]]; then
        git_common_dir="$NAMO_REPO/$git_common_dir"
    fi
    export SAGE_REPO="$(dirname "$(dirname "$git_common_dir")")/sage_learning"
fi
if [ ! -d "$SAGE_REPO" ]; then
    echo "Could not resolve the Sage scorer repository: $SAGE_REPO" >&2
    exit 1
fi

export PYTHONPATH="$NAMO_REPO/build_python:$NAMO_REPO/python:$NAMO_REPO"
export PYTHONPATH="$PYTHONPATH:$SAGE_REPO"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1

"${NAMO_PYTHON:?environment did not define NAMO_PYTHON}" -m full_namo_sim_exp.run_shard \
    --experiment "$EXPERIMENT" \
    --shard-index "$SLURM_ARRAY_TASK_ID"
