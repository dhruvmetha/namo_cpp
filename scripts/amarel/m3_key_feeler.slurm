#!/bin/bash
# M3 H=2 FORESIGHT feeler, KEY-GRADED (zero sims) — the headline-direction read, fast.
# Per array task = one ckpt (from CKPTS ;-list). Ranks first pushes on pure2push by Q(s0,.,H) and grades
# hit@k vs the exhaustive pure2push.json setups (valid_first_push ∪ valid_1push). Runs BOTH H=2 (foresight)
# and H=1 (reactive control) on the SAME scenes — the H2−H1 delta = does budget-conditioning buy foresight.
# NOTE: key-graded ≠ the 75.2/34.5 SIM bars (different grading); the sim-graded eval_m3 is the completion headline.
#
# Required env at submit:  CKPTS (semicolon-separated ckpts; --array=0-(n-1)), OUT_DIR
#SBATCH --job-name=m3-feeler
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --output=logs/m3feel-%A_%a.out
#SBATCH --error=logs/m3feel-%A_%a.err
set -euo pipefail

cd "$NAMO_REPO"
: "${MJ_PATH:?source env.<machine>.sh first}"
export LD_LIBRARY_PATH="$MJ_PATH/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/build_python:$PWD/python:$PWD/scripts:$PWD/scripts/sandbox:$SAGE_REPO"

: "${CKPTS:?must set CKPTS (semicolon-separated)}"
: "${OUT_DIR:?must set OUT_DIR}"
mkdir -p "$OUT_DIR"
# task index encodes (seed, H): idx//2 = seed, H = {2,1}[idx%2]. ONE H per task (no 50-min sequential
# wall; a transient on one cell can't take out the seed's other cell). Submit --array=0-(2*nseed-1).
IFS=';' read -ra arr <<< "$CKPTS"
idx=${SLURM_ARRAY_TASK_ID:-0}
ck="${arr[$(( idx / 2 ))]}"
H=$(( idx % 2 == 0 ? 2 : 1 ))
tag=$(echo "$ck" | sed -E 's#.*/scorer/([^/]+)/.*/(epoch[0-9]+).*#\1_\2#')
echo "task $idx: $tag H=$H  ($ck)"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || true

out="$OUT_DIR/${tag}_h${H}.json"
"$NAMO_PYTHON" -u scripts/sandbox/eval_m3.py \
    --ckpt "$ck" --grade key --h "$H" --start 0 --end 985 \
    --out "$out" --leaf-out "$OUT_DIR/${tag}_h${H}.jsonl"
"$NAMO_PYTHON" - "$out" <<'PY'
import json, sys
r = json.load(open(sys.argv[1])); h = r["hit_at_k"]; fl = r.get("random_floor_at_k", {})
print(f"  {sys.argv[1].split('/')[-1]}: H={r['H']} n={r['n_graded']} "
      f"hit@1={h['1']:.1f} @3={h['3']:.1f} @5={h['5']:.1f} @10={h['10']:.1f}  "
      f"floor@1={fl.get('1','?')} @10={fl.get('10','?')}")
PY
echo "M3 FEELER DONE: $tag H=$H"
