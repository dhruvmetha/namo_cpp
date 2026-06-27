#!/bin/bash
# Portability guard — two checks that protect git-based portability (Amarel ↔ ilab ↔ …):
#   1. No machine-specific absolute path baked into committed CODE (use $NAMO_*/namo.paths).
#   2. No tracked file imports a git-IGNORED module (it won't exist on a fresh clone).
# Run standalone or as a pre-commit hook (scripts/githooks/). See docs/PORTABILITY.md.
#   bash scripts/portability/check_no_hardcoded_paths.sh
# Exit 0 = clean, 1 = problem found.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
rc=0

# ── Check 1: hardcoded machine paths in code ────────────────────────────────
# The two active box prefixes that must not appear in code.
PREFIXES='/scratch/dm1487|/cache/home/dm1487'
# Legitimately machine-specific files (per-machine env + machine cards + portability tooling): exempt.
EXEMPT=':!env.*.sh :!scripts/amarel/activate.sh :!scripts/portability/*'
hits=$(git grep -nE "$PREFIXES" -- '*.py' '*.sh' '*.slurm' '*.yaml' $EXEMPT 2>/dev/null)
if [ -n "$hits" ]; then
  echo "❌ hardcoded machine path(s) in code — use \$NAMO_* / namo.paths instead:"
  echo "$hits"; echo ""; rc=1
fi

# ── Check 2: tracked code importing a git-ignored module ─────────────────────
# scripts/sandbox/ is an ignore-all quarantine; keepers are force-added. If a tracked
# file imports a module that's still ignored, a fresh clone breaks (ImportError).
ignored_mods=$(git status --porcelain --ignored scripts/sandbox/ 2>/dev/null \
  | awk '/^!! /{print $2}' | grep '\.py$' | xargs -r -n1 basename | sed 's/\.py$//' | sort -u)
if [ -n "$ignored_mods" ]; then
  tracked_imports=$(git ls-files '*.py' \
    | xargs grep -hoE "^[[:space:]]*(from|import) [a-z_][a-z0-9_]*" 2>/dev/null \
    | sed -E 's/^[[:space:]]*(from|import) //' | sort -u)
  bad=$(comm -12 <(echo "$ignored_mods") <(echo "$tracked_imports"))
  if [ -n "$bad" ]; then
    echo "❌ tracked code imports git-IGNORED module(s) — they won't exist on a fresh clone:"
    for m in $bad; do
      echo "  '$m' imported by: $(git grep -lE "(from|import) $m\b" -- '*.py' | tr '\n' ' ')"
      echo "    → force-add the module (git add -f scripts/sandbox/$m.py) after converting its paths"
    done
    echo ""; rc=1
  fi
fi

# 2b: tracked shell/slurm HARD-referencing (non-comment) a git-ignored script.
shell_bad=$(git ls-files '*.sh' '*.slurm' | while read -r sf; do
  [ -f "$sf" ] || continue   # skip files deleted in the working tree
  grep -vE '^[[:space:]]*#' "$sf" | grep -oE "scripts/[a-zA-Z0-9_/]+\.(py|sh|slurm)" | while read -r ref; do
    [ -f "$ref" ] && ! git ls-files --error-unmatch "$ref" >/dev/null 2>&1 && echo "  $sf -> $ref"
  done
done)
if [ -n "$shell_bad" ]; then
  echo "❌ tracked shell references git-IGNORED script(s) (won't exist on a fresh clone):"
  echo "$shell_bad"
  echo "   → force-add the target (git add -f <path>) after converting its paths"
  echo ""; rc=1
fi

[ $rc -eq 0 ] && echo "✓ portability guard clean (no hardcoded paths; no ignored imports/refs)"
exit $rc
