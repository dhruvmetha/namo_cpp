#!/bin/bash
# rewrite_paths.sh — one-time, on a NEW box: rewrite hardcoded /scratch/dm1487 -> your NAMO_SCRATCH, in BOTH
#   (a) the ~57 scripts that predate the env-var contract (scripts/, python/), and
#   (b) the test-set label JSONs (which bake in absolute car_envs XML paths).
# WHY: the eval critical-path scripts (eval_reactive_argmax, scorer_beam, eval_m3, live_scorer) hardcode /scratch/dm1487
# in arg-defaults + module constants — they do NOT read NAMO_SCRATCH — so this rewrite is MANDATORY for eval on a new box.
# Run from the repo root AFTER `source env.<machine>.sh` and AFTER the data is in place. Idempotent.
#   bash scripts/portability/rewrite_paths.sh            # uses $NAMO_SCRATCH
#   bash scripts/portability/rewrite_paths.sh /my/base   # or pass the base explicitly
set -euo pipefail
BASE="${1:-${NAMO_SCRATCH:?set NAMO_SCRATCH (source env.<machine>.sh) or pass the base as arg1}}"
OLD="/scratch/dm1487"
[ "$BASE" = "$OLD" ] && { echo "base == $OLD (Amarel) — nothing to rewrite"; exit 0; }
echo "=== rewriting $OLD -> $BASE ==="
echo "-- (a) scripts (scripts/ python/) --"
mapfile -t FILES < <(grep -rl "$OLD" scripts/ python/ 2>/dev/null || true)
printf '  %d files\n' "${#FILES[@]}"
[ "${#FILES[@]}" -gt 0 ] && printf '%s\0' "${FILES[@]}" | xargs -0 sed -i "s#$OLD#$BASE#g"
echo "-- (b) test-set label JSONs --"
LJ="$BASE/datasets/namo_testset_v1/labels"
if [ -d "$LJ" ]; then
  n=0; for j in "$LJ"/*.json; do [ -f "$j" ] && { sed -i "s#$OLD#$BASE#g" "$j"; n=$((n+1)); }; done
  echo "  rewrote $n label json(s)"
else
  echo "  (no $LJ yet — rerun after the test set lands)"
fi
echo "=== done. sanity: remaining $OLD refs in scripts: $(grep -rl "$OLD" scripts/ python/ 2>/dev/null | wc -l) ==="
