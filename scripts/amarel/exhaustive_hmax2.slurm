#!/bin/bash
# Amarel array for scripts/pipeline/exhaustive_hmax2.py — exhaustive depth-2 labelling of a scene
# pool. Amarel twin of scripts/ilab/exhaustive_hmax2_ilab.slurm; the iLab one carries the same
# sizing note. Handles single-movable pools and --n-movables 2 pools, where the first push and the
# finish both range over every movable in the doorway.
#
# Env in: SCENES, OUT_DIR, POOL_ROOT (required); DEPTH1 (set to any value -> depth-1 only).
#
# ⛔ The pool XMLs and the scenes list must live on Amarel's own filesystem. This box does not
# share /common/users with the CS estate, and `--scenes` holds absolute paths, so a list written on
# arrakis points at nothing here. Copy the pool over and rebuild the list with
# manifest_to_scenes.py on THIS box.
#
# ⛔ Run scripts/check_box_sync.sh here first. The bindings are per-machine and Amarel's .so was
# three weeks stale when this file was written, which is a silent physics mismatch, not an error.
#
# One task per core, single-threaded, because the work is one MuJoCo env per scene and the
# throughput measured on arrakis was 5-8 sims/sec whatever the core count.
#SBATCH --job-name=exh2
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --output=logs/exh2_amarel-%A_%a.out
#SBATCH --error=logs/exh2_amarel-%A_%a.err
set -euo pipefail
REPO=/cache/home/dm1487/projects/namo/namo_cpp
cd "$REPO"
source "$REPO/scripts/amarel/activate.sh"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${SCENES:?}"; : "${OUT_DIR:?}"; : "${POOL_ROOT:?}"
mkdir -p "$OUT_DIR" logs
python -u scripts/pipeline/exhaustive_hmax2.py \
    --scenes "$SCENES" --out "$OUT_DIR" --pool-root "$POOL_ROOT" \
    --shard "$SLURM_ARRAY_TASK_ID" --nshards "$SLURM_ARRAY_TASK_COUNT" \
    ${DEPTH1:+--depth1-only}
