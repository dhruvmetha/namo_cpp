#!/bin/bash
# afterany eval-chain: fires when a training seed finishes (COMPLETE *or* TIMEOUT — these converge by hitting the 14h
# wall, so afterok would never fire). Resolves the best-val ckpt from RUN_DIR at RUNTIME (path unknown at submit time),
# guards on converged-enough (best epoch >= MINEP), then submits reactive@2 + best-first@2 (combine=q) for that ckpt.
# Pure SLURM => survives session restarts. Submit one per pending seed via launch_seed_evals.sh.
#   env: RUN_DIR (e.g. qfull_v3_v4hq_s2), LABEL (e.g. Hz_v3_s2); opt MINEP(8)
#SBATCH --job-name=evalchain
#SBATCH --partition=main,main-redhat
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=logs/evalchain-%j.out
set -uo pipefail
cd "$NAMO_REPO" || exit 1
: "${RUN_DIR:?}"; : "${LABEL:?}"; MINEP=${MINEP:-8}
R="$NAMO_SCRATCH/sage_outputs/scorer"
MANIFEST="$NAMO_MANIFESTS/test_pure2_fromkey.txt"
# best-val ckpt = min val_loss across all wandb subdirs (epoch*.ckpt only, not last.ckpt)
CKPT=$(ls $R/$RUN_DIR/namo-classifier/*/checkpoints/epoch*.ckpt 2>/dev/null \
  | sed -E 's/.*val_loss([0-9.]+)\.ckpt$/\1 &/' | sort -n | head -1 | awk '{print $2}')
EP=$(echo "$CKPT" | grep -oE 'epoch[0-9]+' | grep -oE '[0-9]+' | sed 's/^0*//'); EP=${EP:-0}
if [ -z "$CKPT" ] || [ "$EP" -lt "$MINEP" ]; then
  echo "GUARD: $RUN_DIR not converged (ckpt='$CKPT' ep=$EP < $MINEP) — skip eval, training likely died early"; exit 0
fi
echo "resolved $LABEL -> $CKPT (ep$EP); submitting reactive@2 + best-first@2(q)"
sbatch --export="ALL,CKPT=$CKPT,OUT_DIR=$NAMO_SCRATCH/eval/reactarg_${LABEL},SHARD=80,PRIOR=q,SEED=7000" \
  scripts/amarel/reactive_argmax.slurm
sbatch --array=0-39 \
  --export="ALL,CKPT=$CKPT,MANIFEST=$MANIFEST,OUT_DIR=$NAMO_SCRATCH/eval/bfq_${LABEL},COMBINE=q,SIM_BUDGET=900,HMAX=2,PRIOR=model,AGG=mean5,SHARD=26" \
  scripts/amarel/bestfirst_eval.slurm
echo "EVAL_SUBMITTED $LABEL (reactarg_${LABEL} + bfq_${LABEL})"
