#!/bin/bash
# Run on Amarel after the selected architecture-ablation checkpoints have been transferred.
set -euo pipefail

: "${CKPT_ROOT:?set CKPT_ROOT to the Amarel checkpoint directory}"
: "${OUT_ROOT:?set OUT_ROOT to a new Amarel evaluation root}"
: "${SAGE_REPO:?set SAGE_REPO to the matching Sage checkout}"
: "${NAMO_BINDINGS:?set NAMO_BINDINGS to the canonical Amarel build_python directory}"

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SAGE=$SAGE_REPO
BIND=$NAMO_BINDINGS
cd "$REPO"
source env.amarel.sh >/dev/null 2>&1
export SAGE_REPO="$SAGE"
export NAMO_BINDINGS="$BIND"
mkdir -p logs

if [ -e "$OUT_ROOT/jobs.tsv" ]; then
  echo "refusing to reuse launched evaluation root: $OUT_ROOT" >&2
  exit 1
fi

arms=(${HY5U_ARCH_ARMS:-HY5U_global HY5U_no_local})
for arm in "${arms[@]}"; do
  for seed in 1 2 3; do
    test -f "$CKPT_ROOT/${arm}_s${seed}.ckpt"
  done
done

mkdir -p "$OUT_ROOT/smoke" "$OUT_ROOT/full"
printf 'kind\tarm\tseed\tjob_id\tdependency\n' > "$OUT_ROOT/jobs.tsv"

for arm in "${arms[@]}"; do
  case "$arm" in
    HY5U_global) short=gl ;;
    HY5U_no_local) short=nl ;;
    HY5U_no_edge) short=ne ;;
    *) echo "unknown architecture arm: $arm" >&2; exit 2 ;;
  esac
  smoke_out="$OUT_ROOT/smoke/${arm}_s1"
  smoke_job=$(sbatch --parsable \
    --job-name="archsm_${short}" --partition=main --array=0-0 \
    --cpus-per-task=14 --mem=64G --time=00:30:00 \
    --export="ALL,NAMO_REPO=$REPO,SAGE_REPO=$SAGE,NAMO_BINDINGS=$BIND,CKPT_ROOT=$CKPT_ROOT,CKPT=$CKPT_ROOT/${arm}_s1.ckpt,OUT=$smoke_out,N1SH=7,N2SH=7,WORKERS_PER_TASK=14,ONEPUSH_LIMIT=7,TWOPUSH_LIMIT=7,REFUSE_OVERWRITE=1,HMAX=2,SIM_BUDGET=900,PRIOR=model,SEED_BASE=7000" \
    scripts/slurm/aquaman_eval_amarel.slurm)
  printf 'smoke\t%s\t1\t%s\tnone\n' "$arm" "$smoke_job" | tee -a "$OUT_ROOT/jobs.tsv"
  echo "SMOKE_JOB arm=$arm job=$smoke_job"

  for seed in 1 2 3; do
    eval_out="$OUT_ROOT/full/${arm}_s${seed}"
    full_job=$(sbatch --parsable --dependency="afterok:$smoke_job" --kill-on-invalid-dep=yes \
      --job-name="arch_${short}_s${seed}" --partition=main --array=0-35 \
      --cpus-per-task=21 --mem=64G --time=02:00:00 \
      --export="ALL,NAMO_REPO=$REPO,SAGE_REPO=$SAGE,NAMO_BINDINGS=$BIND,CKPT=$CKPT_ROOT/${arm}_s${seed}.ckpt,OUT=$eval_out,N1SH=378,N2SH=378,WORKERS_PER_TASK=21,REFUSE_OVERWRITE=1,HMAX=2,SIM_BUDGET=900,PRIOR=model,SEED_BASE=7000" \
      scripts/slurm/aquaman_eval_amarel.slurm)
    printf 'full\t%s\t%s\t%s\t%s\n' "$arm" "$seed" "$full_job" "$smoke_job" | tee -a "$OUT_ROOT/jobs.tsv"
    echo "FULL_JOB arm=$arm seed=$seed job=$full_job smoke=$smoke_job"

    agg_job=$(sbatch --parsable --dependency="afterok:$full_job" --kill-on-invalid-dep=yes \
      --job-name="archa_${short}_s${seed}" --partition=main \
      --cpus-per-task=1 --mem=8G --time=00:20:00 \
      --export="ALL,NAMO_REPO=$REPO,SAGE_REPO=$SAGE,NAMO_BINDINGS=$BIND,EVAL_ROOT=$eval_out" \
      scripts/slurm/aggregate_search_eval_amarel.slurm)
    printf 'aggregate\t%s\t%s\t%s\t%s\n' "$arm" "$seed" "$agg_job" "$full_job" | tee -a "$OUT_ROOT/jobs.tsv"
    echo "AGG_JOB arm=$arm seed=$seed job=$agg_job full=$full_job"
  done
done

echo "ARCH EVAL QUEUED $(date) root=$OUT_ROOT"
