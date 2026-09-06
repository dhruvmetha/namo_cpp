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
PY=${NAMO_PYTHON:-python}
AMAREL_CPU_CAP=${AMAREL_CPU_CAP:-6720}
AMAREL_MAX_SUBMIT_TASKS=${AMAREL_MAX_SUBMIT_TASKS:-500}
SUBMIT_POLL_SECONDS=${SUBMIT_POLL_SECONDS:-60}
num_models=$((${#arms[@]} * 3))

# Fill the user's Amarel CPU cap across every seed that will evaluate concurrently. Keep one leaf
# evaluator per requested CPU; when the population is larger than the available width, allocate
# leaves between horizons in proportion to their episode counts. If there are fewer episode-seed
# pairs than the cap, use the exact useful maximum rather than reserving idle CPUs.
onepush_key=$("$PY" -m namo.eval_sets onepush_manifest)
n1=$("$PY" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$onepush_key")
n2=$("$PY" -c 'import json; from namo import eval_sets; print(len(json.load(open(eval_sets.PURE2PUSH))))')
total_episodes=$((n1 + n2))
total_episode_seed_pairs=$((num_models * total_episodes))
if [ "$total_episode_seed_pairs" -le "$AMAREL_CPU_CAP" ]; then
  if [ -z "${EVAL_WORKERS_PER_TASK:-}" ]; then
    for candidate in $(seq 21 -1 1); do
      if [ $((total_episodes % candidate)) -eq 0 ]; then
        EVAL_WORKERS_PER_TASK=$candidate
        break
      fi
    done
  fi
  [ $((total_episodes % EVAL_WORKERS_PER_TASK)) -eq 0 ] || {
    echo "EVAL_WORKERS_PER_TASK=$EVAL_WORKERS_PER_TASK does not divide $total_episodes episodes" >&2
    exit 2
  }
  n1sh=$n1
  n2sh=$n2
  tasks_per_model=$((total_episodes / EVAL_WORKERS_PER_TASK))
else
  EVAL_WORKERS_PER_TASK=${EVAL_WORKERS_PER_TASK:-21}
  tasks_per_model=$((AMAREL_CPU_CAP / (num_models * EVAL_WORKERS_PER_TASK)))
  [ "$tasks_per_model" -ge 1 ] || { echo "CPU cap is too small for one bundled task per model" >&2; exit 2; }
  leaves_per_model=$((tasks_per_model * EVAL_WORKERS_PER_TASK))
  n1sh=$((leaves_per_model * n1 / total_episodes))
  n2sh=$((leaves_per_model - n1sh))
fi
array_last=$((tasks_per_model - 1))
requested_full_cpus=$((num_models * tasks_per_model * EVAL_WORKERS_PER_TASK))
echo "FULL_WIDTH_PLAN models=$num_models tasks_per_model=$tasks_per_model workers_per_task=$EVAL_WORKERS_PER_TASK requested_cpus=$requested_full_cpus cap=$AMAREL_CPU_CAP N1SH=$n1sh N2SH=$n2sh populations=$n1+$n2"

wait_for_submit_slots() {
  local needed=$1 label=$2 queued
  while true; do
    queued=$(squeue -h -u "$USER" -o '%i' | wc -l)
    if [ $((queued + needed)) -le "$AMAREL_MAX_SUBMIT_TASKS" ]; then
      echo "SUBMIT_SLOTS_READY label=$label queued=$queued needed=$needed cap=$AMAREL_MAX_SUBMIT_TASKS"
      return 0
    fi
    echo "SUBMIT_SLOTS_WAIT label=$label queued=$queued needed=$needed cap=$AMAREL_MAX_SUBMIT_TASKS $(date)"
    sleep "$SUBMIT_POLL_SECONDS"
  done
}

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
  wait_for_submit_slots 1 "smoke_${arm}"
  smoke_job=$(sbatch --parsable \
    --job-name="archsm_${short}" --partition=main --array=0-0 \
    --cpus-per-task=14 --mem=64G --time=00:30:00 \
    --export="ALL,NAMO_REPO=$REPO,SAGE_REPO=$SAGE,NAMO_BINDINGS=$BIND,CKPT_ROOT=$CKPT_ROOT,CKPT=$CKPT_ROOT/${arm}_s1.ckpt,OUT=$smoke_out,N1SH=7,N2SH=7,WORKERS_PER_TASK=14,ONEPUSH_LIMIT=7,TWOPUSH_LIMIT=7,REFUSE_OVERWRITE=1,HMAX=2,SIM_BUDGET=900,PRIOR=model,SEED_BASE=7000" \
    scripts/slurm/aquaman_eval_amarel.slurm)
  printf 'smoke\t%s\t1\t%s\tnone\n' "$arm" "$smoke_job" | tee -a "$OUT_ROOT/jobs.tsv"
  echo "SMOKE_JOB arm=$arm job=$smoke_job"

  for seed in 1 2 3; do
    eval_out="$OUT_ROOT/full/${arm}_s${seed}"
    wait_for_submit_slots "$tasks_per_model" "full_${arm}_s${seed}"
    full_job=$(sbatch --parsable --dependency="afterok:$smoke_job" --kill-on-invalid-dep=yes \
      --job-name="arch_${short}_s${seed}" --partition=main --array="0-$array_last" \
      --cpus-per-task="$EVAL_WORKERS_PER_TASK" --mem=64G --time=02:00:00 \
      --export="ALL,NAMO_REPO=$REPO,SAGE_REPO=$SAGE,NAMO_BINDINGS=$BIND,CKPT=$CKPT_ROOT/${arm}_s${seed}.ckpt,OUT=$eval_out,N1SH=$n1sh,N2SH=$n2sh,WORKERS_PER_TASK=$EVAL_WORKERS_PER_TASK,REFUSE_OVERWRITE=1,HMAX=2,SIM_BUDGET=900,PRIOR=model,SEED_BASE=7000" \
      scripts/slurm/aquaman_eval_amarel.slurm)
    printf 'full\t%s\t%s\t%s\t%s\n' "$arm" "$seed" "$full_job" "$smoke_job" | tee -a "$OUT_ROOT/jobs.tsv"
    echo "FULL_JOB arm=$arm seed=$seed job=$full_job smoke=$smoke_job"

    wait_for_submit_slots 1 "aggregate_${arm}_s${seed}"
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
