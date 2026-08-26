#!/usr/bin/env bash
# Live state for the real-robot campaign. Everything here is DERIVED, so it cannot go stale.
#
# The card at docs/experiments/log/EXP-2026-08-25-real-robot-campaign.md deliberately holds no
# branch names, no ahead/behind counts, no "currently". Those live here, computed on demand.
# A tracking doc that repeats mutable facts drifts and then misleads; this is the fix.
set -u
NAMO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RC="$(dirname "$NAMO")/robot_control"

row() {   # repo, branch, path
  local d=$1 b ahead behind
  b=$(git -C "$d" branch --show-current 2>/dev/null) || return
  git -C "$d" rev-parse --abbrev-ref "@{u}" >/dev/null 2>&1 \
    && read -r behind ahead < <(git -C "$d" rev-list --left-right --count "@{u}...HEAD" 2>/dev/null) \
    || { ahead="-"; behind="-"; }
  printf "  %-34s %-28s %-8s +%s/-%s\n" "$(basename "$d")" "$b" \
         "$(git -C "$d" rev-parse --short HEAD)" "$ahead" "$behind"
}

echo "namo worktrees                     branch                       head     ahead/behind upstream"
git -C "$NAMO" worktree list --porcelain | awk '/^worktree /{print $2}' \
  | grep -v '\.claude/worktrees' | while read -r d; do row "$d"; done

if [ -d "$RC" ]; then
  echo
  echo "robot_control worktrees"
  git -C "$RC" worktree list --porcelain | awk '/^worktree /{print $2}' | while read -r d; do row "$d"; done
fi

echo
echo "uncommitted in the main namo tree:"
git -C "$NAMO" status --short | sed 's/^/  /' | head -20
echo
echo "agents: run ListAgents. Card: docs/experiments/log/EXP-2026-08-25-real-robot-campaign.md"
