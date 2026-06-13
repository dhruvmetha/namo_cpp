#!/bin/bash
# M3 H=2 FORESIGHT feeler, KEY-GRADED (zero sims) — the headline-direction read, fast.
# Per array task = one ckpt (from CKPTS ;-list). Ranks first pushes on pure2push by Q(s0,.,H) and grades
# hit@k vs the exhaustive pure2push.json setups (valid_first_push ∪ valid_1push). Runs BOTH H=2 (foresight)
# and H=1 (reactive control) on the SAME scenes — the H2−H1 delta = does budget-conditioning buy foresight.
# NOTE: key-graded ≠ the 75.2/34.5 SIM bars (different grading); the sim-graded eval_m3 is the completion headline.
#
# Required env at submit:  CKPTS (semicolon-separated ckpts; --array=0-(n-1)), OUT_DIR
#SBATCH --job-name=m3-feeler
#SBATCH --partition=gpu,gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --output=/scratch/dm1487/logs/m3feel-%A_%a.out
#SBATCH --error=/scratch/dm1487/logs/m3feel-%A_%a.err
set -euo pipefail

cd /cache/home/dm1487/projects/namo/namo_cpp
export MJ_PATH=/scratch/dm1487/mujoco/mujoco-3.2.7
export LD_LIBRARY_PATH="$MJ_PATH/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/build_python:$PWD/python:$PWD/scripts:$PWD/scripts/sandbox:/cache/home/dm1487/projects/namo/sage_learning"

: "${CKPTS:?must set CKPTS (semicolon-separated)}"
: "${OUT_DIR:?must set OUT_DIR}"
mkdir -p "$OUT_DIR"
IFS=';' read -ra arr <<< "$CKPTS"
ck="${arr[${SLURM_ARRAY_TASK_ID:-0}]}"
tag=$(echo "$ck" | sed -E 's#.*/scorer/([^/]+)/.*/(epoch[0-9]+).*#\1_\2#')
echo "task ${SLURM_ARRAY_TASK_ID:-0}: $tag  ($ck)"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || true

for H in 2 1; do
    out="$OUT_DIR/${tag}_h${H}.json"
    /scratch/dm1487/envs/namo/bin/python -u scripts/sandbox/eval_m3.py \
        --ckpt "$ck" --grade key --h "$H" --start 0 --end 985 \
        --out "$out" --leaf-out "$OUT_DIR/${tag}_h${H}.jsonl"
    /scratch/dm1487/envs/namo/bin/python - "$out" <<'PY'
import json, sys
r = json.load(open(sys.argv[1])); h = r["hit_at_k"]
print(f"  {sys.argv[1].split('/')[-1]}: H={r['H']} grade={r['grade']} n={r['n_graded']} "
      f"hit@1={h['1']:.1f} @3={h['3']:.1f} @5={h['5']:.1f} @10={h['10']:.1f}")
PY
done
echo "M3 FEELER DONE: $tag"
