#!/bin/bash
# Portability guard — three checks that protect git-based portability (Amarel ↔ ilab ↔ …):
#   1. No machine-specific absolute path baked into committed CODE (use $NAMO_*/namo.paths).
#   2. No tracked file imports a git-IGNORED module (it won't exist on a fresh clone).
#   3. No machine-specific absolute path baked into committed DOCS (they rot into stale
#      pointers — e.g. /scratch/dm1487/h5 was cited 23× while existing on no box at all).
# Run standalone or as a pre-commit hook (scripts/githooks/). See docs/PORTABILITY.md.
#   bash scripts/portability/check_no_hardcoded_paths.sh
# Exit 0 = clean, 1 = problem found.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
rc=0

# ── Check 1: hardcoded machine paths in code ────────────────────────────────
# Box prefixes that must not appear in portable code.
# KNOWN GAP (measured 2026-08-06): the CS prefix /common/users/dm1487 is NOT listed.
# It is an equally box-locking path — a CS-locked `scorer_ckpt:` passes silently while
# the identical Amarel-locked one is flagged. Adding it was tried and reverted: it also
# matches committed Hydra run-outputs under outputs/**/.hydra/config.yaml (~227KB of
# hits), which are generated records, not portable code. Enforcing it needs those
# excluded first. Until then, CS-locked paths are a manual review item.
PREFIXES='/scratch/dm1487|/cache/home/dm1487'
# Legitimately machine-specific / non-portable-by-design: exempt.
#   env.*.sh, scripts/amarel/*  — per-box launchers and env (naming box paths IS their job)
#   scripts/portability/*       — this guard itself
#   python/namo/paths.py        — THE resolver; _LEGACY_SCRATCH is its input constant
#   scripts/sandbox/*           — ignore-all ad-hoc quarantine (see check 2)
EXEMPT=':!env.*.sh :!scripts/amarel/* :!scripts/portability/* :!python/namo/paths.py :!scripts/sandbox/*'
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

# ── Check 3: hardcoded machine paths in LIVE-POINTER DOCS (WARN-ONLY) ───────
# Docs were never scanned, so box paths rotted undetected: /scratch/dm1487/h5 was cited
# 23x while existing on NO box (the real file is $NAMO_H5 on CS only) — found 2026-08-05.
# Warn-only on purpose: this flags drift for review, it does not gate commits. Scoped to
# docs that act as live "where the thing is" pointers. Exempt by design:
#   CLAUDE.<machine>.md + PORTABILITY.md (naming real box paths IS their job)
#   docs/experiments/archive/** + docs/experiments/log/** (historical records — frozen)
#   DATA_COLLECTION_GUIDE.md (documents the Amarel on-disk layout deliberately)
DOC_SCAN='docs/*.md docs/experiments/*.md docs/pipeline/*.md'
DOC_EXEMPT=':!docs/PORTABILITY.md :!docs/experiments/archive/* :!docs/experiments/log/*'
doc_hits=$(git grep -nE "$PREFIXES" -- $DOC_SCAN $DOC_EXEMPT 2>/dev/null)
if [ -n "$doc_hits" ]; then
  echo "⚠  possible stale box path(s) in live-pointer docs — prefer \$NAMO_SCRATCH / \$NAMO_H5 / \$NAMO_DATASETS:"
  echo "$doc_hits"
  echo "   → a literal box path is wrong on every other box and rots silently."
  echo "   → VERIFY the artifact still exists before trusting any such pointer."
  echo "   (warning only — does not fail the guard)"
  echo ""
fi

[ $rc -eq 0 ] && echo "✓ portability guard clean (no hardcoded paths in code; no ignored imports/refs)"
exit $rc
