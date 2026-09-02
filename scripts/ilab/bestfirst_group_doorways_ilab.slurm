#!/bin/bash
# Does best-first solve a doorway that needs BOTH blocks, when there is no way around it?
#
# The hardware side ran full_namo on a real captured two-block scene and found the arm this paper is
# built on failing where the older opener succeeds: best_first burned all 900 sims committing zero
# pushes, uniform and model alike, while region_bfs solved it in 281 attempts. Their scene had a
# reroute available. This asks the same question on OUR pool, on the scenes that have none.
#
# Two populations, both one-hop, both from the 2220-card two-movable gallery:
#   NOREROUTE  240 rooms / 329 cards. Robot and goal share a boundary that no single block opens,
#              and no path to the goal avoids a group edge. If best_first exhausts here it is not
#              that these scenes are expensive, they are outside its reach.
#   CONTROL    136 rooms. Same kind of two-block boundary, but a solo-object route exists. This is
#              the arm that says whether any exhaustion is about the doorway or about the pool.
#
# Read `simulation_budget_exhausted` with zero pushes. That is the hardware failure signature.
#
# ⛔ Same two overrides as scripts/amarel/hy5u_2mov_1hop.slurm and for the same reason: the runner
# defaults min-reachable to 1, which calls a door open on a single reachable point, while every
# label in this pool means twenty of a hundred.
#
# ⛔ Not comparable to any row in horizon_q_model_registry.md. Different harness, different
# population. Needs its own entry.
#SBATCH --job-name=bfgroup
#SBATCH --partition=unlimited
#SBATCH --array=0-3
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/bfgroup-%A_%a.out
#SBATCH --error=logs/bfgroup-%A_%a.err
set -euo pipefail
REPO=/common/home/dm1487/robotics_research/ktamp/namo
cd "$REPO"
source env.ilab.sh
# One thread per math library. Without this each worker spawns its own pool for the scorer's
# forward passes and the model arm crawls with no error: measured 974% CPU across three workers.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
PY=/common/users/dm1487/envs/mjxrl/bin/python
G=$NAMO_SCRATCH/viz/real_2mov
CK=$NAMO_SCRATCH/aquaman/round0/models/HY5U_s1/checkpoints/epoch011-val_loss0.3213.ckpt
OUT=$NAMO_SCRATCH/eval/group_doorway_bestfirst

case $SLURM_ARRAY_TASK_ID in
  0) POP=noreroute;  MAN=$G/_onehop_noreroute.txt;  PRIOR=model;   CKPT="--scorer-ckpt $CK"; SEED=1 ;;
  1) POP=noreroute;  MAN=$G/_onehop_noreroute.txt;  PRIOR=uniform; CKPT="";                  SEED=7000 ;;
  2) POP=control;    MAN=$G/_onehop_hasreroute.txt; PRIOR=model;   CKPT="--scorer-ckpt $CK"; SEED=1 ;;
  3) POP=control;    MAN=$G/_onehop_hasreroute.txt; PRIOR=uniform; CKPT="";                  SEED=7000 ;;
esac

echo "pop=$POP prior=$PRIOR seed=$SEED commit=$(git rev-parse --short HEAD) scenes=$(wc -l < $MAN)"
PYTHONPATH=build_python:python $PY -m namo.solvability_runner \
  --manifest "$MAN" \
  --path-length 1 \
  --local-search best_first \
  --best-first-prior "$PRIOR" $CKPT \
  --region-selection-strategy ml_first \
  --region-max-chain-depth 2 \
  --goals-per-region 100 \
  --region-success-min-reachable 20 \
  --simulation-budget 900 \
  --simulation-budget-scope keyhole \
  --use-cpp-snapshot \
  --seed "$SEED" \
  --workers 16 \
  --output-dir "$OUT/${POP}_${PRIOR}"
echo "DONE $POP $PRIOR"
