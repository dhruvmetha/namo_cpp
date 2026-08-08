#!/bin/bash
# Arjuna-0 v2 stage 3: aggregate the pulled shards into the canonical gate table and print the
# headline rows next to every registered reference, so the comparison needs no second step.
#   bash scripts/rl_loop/run_arjuna_v2_agg.sh
set -u
R0="${NAMO_SCRATCH:?source env.<box>.sh first}/aquaman/round0"
REPO=${NAMO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PY="${NAMO_PYTHON:-python}"; command -v "$PY" >/dev/null 2>&1 || PY=$(command -v python3)

for i in $(seq 1 400); do
  grep -q 'ARJUNA V2 EVAL PULLED' "$R0/arjuna_v2_eval.log" 2>/dev/null && break
  grep -q '^ABORT' "$R0/arjuna_v2_eval.log" 2>/dev/null && { echo "eval chain aborted -- nothing to aggregate"; exit 1; }
  sleep 60
done
grep -q 'ARJUNA V2 EVAL PULLED' "$R0/arjuna_v2_eval.log" || { echo "TIMED OUT waiting for eval"; exit 1; }

n=$(ls "$R0"/eval_bfix/{AJ2_s1,AJ2_s2,AJ2_s3,AJ2NR_s1,AJ2NR_s2,AJ2NR_s3}/{1push_hmax2,2push}/shard_*.jsonl 2>/dev/null | wc -l)
echo "pulled shard files=$n (expect 432)"

"$PY" "$REPO/scripts/rl_loop/aquaman_agg.py" "$R0/arms_aj2.json" "$R0/gate_aj2.json"

R0="$R0" "$PY" - <<'EOF'
import json, os
R0 = os.environ["R0"]
g = json.load(open(f"{R0}/gate_aj2.json"))
# every reference is the registered 3-seed pooled number from horizon_q_model_registry.md
REF = {"theta0": (22.6, 92.0, 39.7), "random": (1.7, 70.1, None), "A": (27.7, 91.5, 38.0),
       "Bfix": (28.9, 87.1, 41.8), "BNG": (32.1, 88.6, 38.4), "ARJ(v1)": (27.7, 91.0, 42.5)}
print(f"\n{'arm':<10} {'2p-hard@5':>10} {'2p-hard@900':>12} {'1p-hard@1':>10}")
for k, (a, b, c) in REF.items():
    print(f"{k:<10} {a:>10} {b:>12} {'-' if c is None else c:>10}")
for k in ("AJ2", "AJ2NR"):
    if k not in g:
        continue
    h2 = g[k].get("2push", {}).get("hard", {})
    h1 = g[k].get("1push", {}).get("hard", {})
    print(f"{k:<10} {h2.get('solve@5','-'):>10} {h2.get('solve@900','-'):>12} {h1.get('solve@1','-'):>10}")
print("\nfull splits (difficulty x horizon):")
for k in ("AJ2", "AJ2NR"):
    if k not in g:
        continue
    for hz in ("1push", "2push"):
        for t in ("easy", "medium", "hard"):
            d = g[k].get(hz, {}).get(t, {})
            if d:
                print(f"  {k:<7} {hz:<6} {t:<7} n={d['n']:<5} @1={d['solve@1']:<5} @5={d['solve@5']:<5} "
                      f"@30={d['solve@30']:<5} @900={d['solve@900']:<5} s2s={d['avg_sims_to_solve']}")
EOF
echo "ARJUNA V2 AGG DONE $(date)"
