#!/bin/bash
# best-first@2 ACROSS SEEDS with combine=q (the [USER] "don't multiply state value for the dive" fix).
# Pairs with the reactive@2 multi-seed table (38.5±2.1 Hz / 38.2±3.0 NoHz) to put error bars on the DIVE TAX.
# combine=q => priority = raw Q(s,a) (NO 0.5Q+0.5V blend). Budget=900 => one run yields BOTH s@2 (= solve within
# 2 sims, the reactive-equivalent / dive-tax point) AND s@900 (full-search) — greedy best-first's first 2 pops are
# budget-independent, so s@2 read off a budget-900 run == a budget-2 run. Ckpts = registry best-val (NOT globbed).
#   bash scripts/amarel/bestfirst_multiseed_q.sh
set -euo pipefail
R=/scratch/dm1487/sage_outputs/scorer
MANIFEST=/scratch/dm1487/manifests/test_pure2_fromkey.txt
COMBINE=q SIM_BUDGET=900 HMAX=2 PRIOR=model AGG=mean5
# label -> registry best-val ckpt (Horizon-v2 / NoHorizon-v2, 3 seeds each)
declare -A CK=(
  [hz_v2_s1]=qfull_v2_v4hq_s1/namo-classifier/10whb62b/checkpoints/epoch008-val_loss0.6728.ckpt
  [hz_v2_s2]=qfull_v2_v4hq_s2/namo-classifier/whv2sdf3/checkpoints/epoch008-val_loss0.6771.ckpt
  [hz_v2_s3]=qfull_v2_v4hq_s3/namo-classifier/a81jq5ob/checkpoints/epoch008-val_loss0.6689.ckpt
  [nohz_v2_s1]=qfull_nohz_v2_v4hq_s1/namo-classifier/4w1hovo4/checkpoints/epoch007-val_loss0.7041.ckpt
  [nohz_v2_s2]=qfull_nohz_v2_v4hq_s2/namo-classifier/rbbqq0ya/checkpoints/epoch009-val_loss0.7004.ckpt
  [nohz_v2_s3]=qfull_nohz_v2_v4hq_s3/namo-classifier/c82jwuw5/checkpoints/epoch010-val_loss0.6968.ckpt
)
cd /cache/home/dm1487/projects/namo/namo_cpp
for lbl in hz_v2_s1 hz_v2_s2 hz_v2_s3 nohz_v2_s1 nohz_v2_s2 nohz_v2_s3; do
  CKPT="$R/${CK[$lbl]}"; [ -f "$CKPT" ] || { echo "MISS $lbl -> $CKPT"; exit 1; }
  OUT_DIR=/scratch/dm1487/eval/bfq_${lbl}        # bfq_ = best-first combine=Q (distinct from blend bf900_*)
  JID=$(sbatch --parsable --array=0-39 \
    --export="ALL,CKPT=$CKPT,MANIFEST=$MANIFEST,OUT_DIR=$OUT_DIR,COMBINE=$COMBINE,SIM_BUDGET=$SIM_BUDGET,HMAX=$HMAX,PRIOR=$PRIOR,AGG=$AGG,SHARD=26" \
    scripts/amarel/bestfirst_eval.slurm)
  echo "  $lbl -> job $JID  ($OUT_DIR)"
done
echo "BFQ_LAUNCHED combine=q budget=900 (s@2 + s@900) for 6 v2 cells"
