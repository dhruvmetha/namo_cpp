#!/bin/bash
# Replay each arm's own solved plans from the 2026-09-02 two-movable gallery eval, one file per
# (run, arm, card), via build_scene_replay.py --from-eval. Cross-box replay is the point: the
# set_full_state restore path differs from CS by about 1 mm, so plans reproduce differently here
# than on the box that ran the eval (see reference_crossboard_physics_identical, project_focus
# memories -- the sim itself is bit-identical, only the mid-solve restore drifts).
#SBATCH --job-name=evalreplay
#SBATCH --partition=main
#SBATCH --array=0-5
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/cache/home/dm1487/projects/namo/namo_cpp/logs/evalreplay-%A_%a.out
#SBATCH --error=/cache/home/dm1487/projects/namo/namo_cpp/logs/evalreplay-%A_%a.err
set -euo pipefail
cd /cache/home/dm1487/projects/namo/namo_cpp
source env.amarel.sh

# Same reasoning as hy5u_2mov_gallery_1hop.slurm: pin every math library to one thread, or a
# worker's forward passes spawn their own pool and 8 allocated cores get fought over by 50+ threads.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PY=/scratch/dm1487/envs/namo/bin/python
GAL=/scratch/dm1487/viz/real_2mov
RUN=/scratch/dm1487/eval/two_movable_1hop_gallery_20260902

case $SLURM_ARRAY_TASK_ID in
  0) ARM=HY5U_s1 ;;
  1) ARM=HY5U_s2 ;;
  2) ARM=HY5U_s3 ;;
  3) ARM=rand_s7000 ;;
  4) ARM=rand_s8000 ;;
  5) ARM=rand_s9000 ;;
esac

echo "arm=$ARM commit=$(git rev-parse --short HEAD)"
PYTHONPATH=build_python:python $PY scripts/viz/build_scene_replay.py \
  --out "$GAL" \
  --from-eval "$RUN" \
  --arms "$ARM" \
  --nshards 1 \
  --remap /common/users/dm1487/scratch_namo/=/scratch/dm1487/
echo "ARM $ARM DONE"
