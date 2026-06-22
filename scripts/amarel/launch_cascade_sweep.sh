#!/bin/bash
# CASCADE (dive-bonus) sweep — fix #1 for the reactive deficit (cross-head scale mismatch).
# Re-runs the 900-cap best-first SOLVE eval with --dive-bonus > 0 so a simulated setup's child outranks a
# fresh first-push (the search DIVES instead of restarting). dive=0 baselines already exist and are REUSED:
#   Hz-v2   -> /scratch/dm1487/eval/bf900_qfull_v2_v4hq_s1        (@2 24.2, @900 94.9)
#   NoHz-v2 -> /scratch/dm1487/eval/bf900_qfull_nohz_v2_v4hq_s1   (@2 32.6, @900 91.6)
# Expectation (journal FORCED-DIVE CEILING, n=150): Hz-v2 reactive @2 24.2 -> ~39 at forced dive; NoHz ~flat
# (already dives 78%). C (strategic re-eval) reads these curves. Everything else matches the 2x2 baseline
# (region criterion, sigmoid/blend/mean5, hmax2, key=pure2push, 40 shards x SHARD=13).
set -euo pipefail
cd /cache/home/dm1487/projects/namo/namo_cpp
SLURM=scripts/amarel/bestfirst_eval.slurm
MAN=/scratch/dm1487/manifests/test_pure2_fromkey.txt
HZ=/scratch/dm1487/sage_outputs/scorer/qfull_v2_v4hq_s1/namo-classifier/10whb62b/checkpoints/epoch008-val_loss0.6728.ckpt
NOHZ=/scratch/dm1487/sage_outputs/scorer/qfull_nohz_v2_v4hq_s1/namo-classifier/4w1hovo4/checkpoints/epoch007-val_loss0.7041.ckpt
[ -f "$HZ" ] || { echo "MISSING Hz ckpt $HZ"; exit 1; }
[ -f "$NOHZ" ] || { echo "MISSING NoHz ckpt $NOHZ"; exit 1; }

submit () {  # $1=tag(hz|nohz) $2=ckpt $3=dive_bonus
  local tag=$1 ck=$2 db=$3
  local base; [ "$tag" = hz ] && base=bf900_qfull_v2_v4hq_s1 || base=bf900_qfull_nohz_v2_v4hq_s1
  local out=/scratch/dm1487/eval/${base}_dive${db}
  echo ">> $tag dive=$db -> $out"
  # --array=0-75: the slurm DEFAULTS to 0-39 (=520 scenes) which UNDER-COVERS the 983-scene manifest
  # (SHARD=13 -> ceil(983/13)=76 shards). Must override to 0-75 for the full 1018 episodes, else the
  # curve is on a ~539 subset and NOT comparable to the full baseline.
  CKPT="$ck" MANIFEST="$MAN" OUT_DIR="$out" SIM_BUDGET=900 DIVE_BONUS="$db" \
    sbatch --array=0-75 --job-name="casc_${tag}_${db}" "$SLURM" | tee -a /tmp/cascade_jobids.txt
}

: > /tmp/cascade_jobids.txt
for db in 0.05 0.1 0.3; do submit hz   "$HZ"   "$db"; done
for db in 0.05 0.3;      do submit nohz "$NOHZ" "$db"; done
echo "=== submitted ==="; cat /tmp/cascade_jobids.txt
