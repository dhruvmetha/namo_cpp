#!/bin/bash
# HY5U best-first on EVERY room in the real_2mov gallery (1436 rooms), against uniform random.
#
# Second pass over this pool. The 2026-08-30 runs covered 572 rooms across three manifests (227 v1,
# 240 no-reroute, 136 control) with different seed counts. This is ONE protocol over the whole
# gallery so every card can carry a model-vs-random solve and speed-up. Split by door type
# afterwards with scripts/pipeline/report_two_movable_1hop.py --meta gallery_1hop_meta.json.
# MAN and OUT are overridable so the same file runs the smoke (5 rooms) and the full array.
# Canonical HY5U protocol. hmax=2, no-op dedupe AND jam pruning both ON (hardcoded at
# best_first_region_opening.py:509-510 and 578-579), combine=q, discount off.
#
# ⛔ Two deliberate overrides of the runner's own defaults, both so this matches the labelling in
# exhaustive_hmax2.py: --goals-per-region 100 and --region-success-min-reachable 20. The runner
# defaults min-reachable to 1, which calls a door open on ONE reachable point, while every label in
# this pool means twenty of a hundred. Left alone, nothing would line up with the tiers.
#
# ⛔ NOT comparable to any row in horizon_q_model_registry.md. Those run eval_bestfirst on
# testset_v3; this runs solvability_runner -> FullNAMOPlanner -> BestFirstRegionOpeningPlanner. A
# different harness and a different population. It needs its own registry entry.
#
# Requires the region-graph fix (commit 41ba153 or later) AND the Amarel bindings rebuilt on top of
# it. Without that, _boundary_objects returns nothing on a two-block door and every such scene fails
# instantly with no_blocking_objects.
#SBATCH --job-name=hy5ugal
#SBATCH --partition=main
#SBATCH --array=0-5
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=/cache/home/dm1487/projects/namo/namo_cpp/logs/hy5ugal-%A_%a.out
#SBATCH --error=/cache/home/dm1487/projects/namo/namo_cpp/logs/hy5ugal-%A_%a.err
set -euo pipefail
cd /cache/home/dm1487/projects/namo/namo_cpp
source env.amarel.sh

# ⛔ Pin every math library to one thread. Without this each worker spawns its own pool for the
# scorer's forward passes: measured 974% CPU across THREE workers, one of them alone at 680%. With
# --workers 16 on 16 allocated cores that is 100+ threads fighting over 16, and the arm crawls with
# no error and no crash. The random arm is unaffected because uniform prior never calls the model,
# which is exactly why the first attempt looked like the model arm had hung.
# Every other eval script in scripts/slurm/ already does this; this one did not, which was my miss.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PY=/scratch/dm1487/envs/namo/bin/python
MAN=${MAN:-/scratch/dm1487/real_buildable_2mov/gallery_1hop_manifest_amarel.txt}
OUT=${OUT:-/scratch/dm1487/eval/two_movable_1hop_gallery_20260902}
CK=/cache/home/dm1487/aquaman0/ckpts_bfix

case $SLURM_ARRAY_TASK_ID in
  0) ARM=HY5U_s1;    PRIOR=model;   CKPT="--scorer-ckpt $CK/HY5U_s1.ckpt"; SEED=1 ;;
  1) ARM=HY5U_s2;    PRIOR=model;   CKPT="--scorer-ckpt $CK/HY5U_s2.ckpt"; SEED=2 ;;
  2) ARM=HY5U_s3;    PRIOR=model;   CKPT="--scorer-ckpt $CK/HY5U_s3.ckpt"; SEED=3 ;;
  3) ARM=rand_s7000; PRIOR=uniform; CKPT="";                               SEED=7000 ;;
  4) ARM=rand_s8000; PRIOR=uniform; CKPT="";                               SEED=8000 ;;
  5) ARM=rand_s9000; PRIOR=uniform; CKPT="";                               SEED=9000 ;;
esac

echo "arm=$ARM prior=$PRIOR seed=$SEED commit=$(git rev-parse --short HEAD)"
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
  --output-dir "$OUT/$ARM"
echo "ARM $ARM DONE"
