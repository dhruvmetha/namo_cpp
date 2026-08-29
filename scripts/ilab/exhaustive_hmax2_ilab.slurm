#!/bin/bash
# iLab array for scripts/pipeline/exhaustive_hmax2.py — exhaustive depth-2 labelling of a scene pool.
# Works for single-movable pools and for --n-movables 2 pools, where both the first push and the
# finish range over every movable in the doorway.
# Env in: SCENES, OUT_DIR, POOL_ROOT (required); DEPTH1(unset -> full depth 2).
# iLab specifics: partition=unlimited, NO --cpus-per-task (the partition rejects it), shared-FS paths.
#
# Sizing, measured not guessed: one two-movable scene at depth 2 ran 1254 sims in 269s on arrakis,
# and a single-movable scene ran 169 sims in 21s, so both land near 5-8 sims/sec single-threaded.
#
# Cost is uneven across scene flavors, so size for the worst one. Measured, not reasoned: the first
# 130 scenes averaged 1884-2140 sims each whichever flavor they came from. 100 tasks over 300 scenes
# is 3 scenes each, and --time is set far above the median because the tail is long.
#
# ⛔ A "domino" scene, where no movable opens the goal when deleted on its own, is NOT push-proof at
# depth 1, and predicting otherwise was wrong. The argument was that deleting an object is strictly
# more favourable than pushing it, so if deleting leaves the goal shut, pushing must too. It ignores
# the whole point of these scenes: a push on one block MOVES THE OTHER. Deleting object 0 leaves
# object 1 where it was, while pushing object 0 drives object 1 out of the way as well. Measured, 19
# of the first 22 domino scenes had a single-push opener, and 90.4% of those openers reported
# block-on-block contact, the highest rate of any flavor.
#SBATCH --job-name=exh2
#SBATCH --partition=unlimited
#SBATCH --array=0-99
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/exh2_ilab-%A_%a.out
#SBATCH --error=logs/exh2_ilab-%A_%a.err
set -euo pipefail
REPO=/common/home/dm1487/robotics_research/ktamp/namo
cd "$REPO"
export MJ_PATH=/common/users/dm1487/ktamp/mujoco
export PYTHONPATH="$REPO/build_python:$REPO/python"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/common/users/dm1487/envs/mjxrl/bin/python
: "${SCENES:?}"; : "${OUT_DIR:?}"; : "${POOL_ROOT:?}"
mkdir -p "$OUT_DIR" logs
"$PY" -u scripts/pipeline/exhaustive_hmax2.py \
    --scenes "$SCENES" --out "$OUT_DIR" --pool-root "$POOL_ROOT" \
    --shard "$SLURM_ARRAY_TASK_ID" --nshards "$SLURM_ARRAY_TASK_COUNT" \
    ${DEPTH1:+--depth1-only}
