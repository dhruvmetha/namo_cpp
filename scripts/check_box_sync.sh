#!/usr/bin/env bash
# Is THIS checkout safe to run a campaign from?
#
# Written 2026-08-08 after a near-miss: the Amarel checkout was 326 commits behind with a
# build_python/*.so from July 2 -- three weeks older than d6088d0, the "sticky-collision fix" to
# include/planning/namo_push_controller.hpp. A collection run there would have produced labels under
# DIFFERENT push physics than every CS-side result, silently, with nothing in the output to show it.
#
# Three checks, because the git SHA alone does not cover any of the ways this actually goes wrong:
#   1. tracked code   -- SHA matches origin AND the tree is clean (uncommitted edits change behaviour)
#   2. compiled code  -- the .so is NEWER than the newest commit touching C++ (a stale .so is the
#                        silent one: python looks right, physics is not)
#   3. driver files   -- md5 of the files that actually decide what a collection records
#
# Usage:  bash scripts/check_box_sync.sh [--ref <branch>]     (default origin/<current branch>)
# Exit 0 = safe to launch. Non-zero = do not launch.
set -uo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null)" || { echo "not in a git repo"; exit 2; }

REF=""
[ "${1:-}" = "--ref" ] && REF="${2:-}"
BR=$(git rev-parse --abbrev-ref HEAD)
[ -n "$REF" ] || REF="origin/$BR"

fail=0
note() { printf "  [%s] %s\n" "$1" "$2"; }

echo "checkout: $(pwd)"
echo "branch:   $BR   ref: $REF"
echo

# ---- 1. tracked code -------------------------------------------------------------------------
git fetch -q origin 2>/dev/null || note WARN "could not fetch origin (offline?) - comparing to local ref"
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse "$REF" 2>/dev/null || echo "")
if [ -z "$REMOTE" ]; then
  note FAIL "ref $REF does not exist"; fail=1
elif [ "$LOCAL" = "$REMOTE" ]; then
  note PASS "HEAD matches $REF (${LOCAL:0:7})"
else
  behind=$(git rev-list --count HEAD.."$REF" 2>/dev/null || echo "?")
  ahead=$(git rev-list --count "$REF"..HEAD 2>/dev/null || echo "?")
  note FAIL "HEAD ${LOCAL:0:7} != $REF ${REMOTE:0:7}  (behind $behind, ahead $ahead)"; fail=1
fi

dirty=$(git status --porcelain -- python/ scripts/ include/ src/ cpp_bindings/ config/ | wc -l)
if [ "$dirty" -eq 0 ]; then note PASS "working tree clean (code dirs)"
else note FAIL "$dirty uncommitted change(s) in code dirs - they will silently alter the run"; fail=1; fi

# ---- 2. compiled code ------------------------------------------------------------------------
# A .so older than the newest C++ commit is stale. This is the check that today's near-miss needed.
SO=$(ls build_python/namo_rl*.so 2>/dev/null | head -1)
if [ -z "$SO" ]; then
  note FAIL "no build_python/namo_rl*.so - build with NAMO_MARCH=x86-64-v3 ./build_python_bindings.sh"; fail=1
else
  INFO="build_python/BUILD_INFO"
  if [ -f "$INFO" ]; then
    # Exact: compare the C++ tree the .so was built from against HEAD's. Timestamps cannot do this --
    # build-then-commit leaves the .so mtime EARLIER than the commit describing it (a false alarm),
    # and neither ordering proves the binary actually contains a given change.
    B_CPP=$(grep '^cpp_tree=' "$INFO" | cut -d= -f2)
    B_SRC=$(grep '^src_tree=' "$INFO" | cut -d= -f2)
    B_DIRTY=$(grep '^dirty_cpp=' "$INFO" | cut -d= -f2)
    H_CPP=$(git rev-parse HEAD:include 2>/dev/null || echo x)
    H_SRC=$(git rev-parse HEAD:src 2>/dev/null || echo y)
    if [ "$B_CPP" = "$H_CPP" ] && [ "$B_SRC" = "$H_SRC" ]; then
      note PASS ".so built from THIS C++ tree (include=${B_CPP:0:7} src=${B_SRC:0:7})"
      [ "${B_DIRTY:-0}" != "0" ] && { note WARN ".so was built with $B_DIRTY uncommitted C++ change(s) - not reproducible"; }
    else
      note FAIL ".so built from a DIFFERENT C++ tree - REBUILD"
      printf "         built: include=%s src=%s (%s)\n" "${B_CPP:0:7}" "${B_SRC:0:7}" "$(grep '^built_at=' "$INFO" | cut -d= -f2)"
      printf "         HEAD:  include=%s src=%s\n" "${H_CPP:0:7}" "${H_SRC:0:7}"
      fail=1
    fi
  else
    # No stamp (built before this mechanism existed). Fall back to timestamps, which can only WARN --
    # they are not decisive in either direction.
    CPP_COMMIT=$(git log -1 --format=%H -- include/ src/ cpp_bindings/ 2>/dev/null)
    CPP_EPOCH=$(git log -1 --format=%ct "$CPP_COMMIT" 2>/dev/null || echo 0)
    SO_EPOCH=$(stat -c %Y "$SO" 2>/dev/null || echo 0)
    note WARN "no build_python/BUILD_INFO - cannot verify what this .so was built from"
    if [ "$SO_EPOCH" -lt "$CPP_EPOCH" ]; then
      printf "         .so %s predates last C++ commit %s (%s) - rebuild to be certain\n" \
        "$(date -d @"$SO_EPOCH" +%F)" "${CPP_COMMIT:0:7}" "$(date -d @"$CPP_EPOCH" +%F)"
    fi
    printf "         rebuild once to stamp it: NAMO_MARCH=x86-64-v3 ./build_python_bindings.sh\n"
  fi
fi

# ---- 3. driver files -------------------------------------------------------------------------
# Content hashes of the files that decide what a collection records. Printed rather than compared to a
# baked-in list: a hardcoded expectation rots, whereas two boxes printing the same four lines is proof.
echo
echo "  driver-file md5 (must be identical on every box in a campaign):"
for f in python/namo/planners/opening/region_opening.py \
         python/namo/data_collection/modular_parallel_collection.py \
         python/namo/data_collection/region_opening_tree_car.yaml \
         config/namo_config_complete_skill15_car_1x.yaml; do
  [ -f "$f" ] && printf "    %s  %s\n" "$(md5sum "$f" | cut -d' ' -f1)" "$f"
done

echo
if [ "$fail" -eq 0 ]; then echo "SAFE TO LAUNCH"; else echo "DO NOT LAUNCH - fix the FAIL lines above"; fi
exit "$fail"
