#!/bin/bash
# Full eval for ONE budget-Q model = its 2x2-matrix cell: ranking (eval_scorer hit@k, H=1 & H=2 on the onepush
# key) + SOLVE (best-first @900, object-constrained pure2push). Random baseline is model-agnostic (shared, already
# computed in bf900_uniform_s0..4) so we only run the MODEL solve here.
#   usage: eval_one_model.sh <run_name e.g. qfull_v2_v4hq_s1>
# Picks the lowest-val_loss epoch ckpt in the run dir. Idempotent-ish: re-submitting overwrites the named outputs.
set -euo pipefail
REPO=/cache/home/dm1487/projects/namo/namo_cpp
RUN="${1:?usage: eval_one_model.sh <run_name>}"
RUNDIR=/scratch/dm1487/sage_outputs/scorer/$RUN
# best (lowest val_loss) epoch ckpt
CKPT=$(ls "$RUNDIR"/namo-classifier/*/checkpoints/epoch*.ckpt 2>/dev/null \
       | sed -E 's/.*val_loss([0-9.]+)\.ckpt/\1 &/' | sort -n | head -1 | awk '{print $2}')
[ -n "$CKPT" ] && [ -f "$CKPT" ] || { echo "NO ckpt for $RUN — abort"; exit 1; }
echo "=== eval $RUN  ckpt=$(basename "$CKPT") ==="
cd "$REPO"

# RANKING H=1 and H=2 (eval_scorer feeler; onepush key default)
for H in 1 2; do
  sbatch --parsable --export=ALL,CKPTS=$CKPT,OUT_DIR=/scratch/dm1487/eval/${RUN}_rank,EVAL_H=$H \
    scripts/amarel/eval_scorer_feeler.slurm | sed "s/^/  rank H=$H job /"
done

# SOLVE best-first @900 (model, object-constrained pure2push)
sbatch --parsable --array=0-75%30 \
  --export=ALL,CKPT=$CKPT,MANIFEST=/scratch/dm1487/manifests/test_pure2_fromkey.txt,OUT_DIR=/scratch/dm1487/eval/bf900_${RUN},SIM_BUDGET=900,PRIOR=model,SHARD=13 \
  scripts/amarel/bestfirst_eval.slurm | sed "s/^/  solve bf900 job /"
echo "EVAL_LAUNCHED $RUN"
