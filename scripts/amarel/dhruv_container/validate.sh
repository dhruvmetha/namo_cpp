#!/usr/bin/env bash
# Inside the image: invoke the already-reviewed bounded validation unchanged.
set -euo pipefail
source "${1:?usage: validate.sh container.env}"
recipes=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$recipes/common.sh"
activate_native_runtime
refuse_existing "$CONTAINER_ROOT/results/fixed" "$CONTAINER_ROOT/results/search" \
    "$CONTAINER_ROOT/results/focused-tests.log" "$CONTAINER_ROOT/results/native-identity.json"
test -f "$CONTAINER_ROOT/provenance/build-ready"
cd "$NAMO_REPO"
"$NAMO_PYTHON" -B "$recipes/native_identity.py" \
    --output "$CONTAINER_ROOT/results/native-identity.json" \
    --reference "${NATIVE_REFERENCE:-$CONTAINER_ROOT/artifacts/dhruv-native-identity.json}"
"$NAMO_PYTHON" -B -m pytest -q -rs -p no:cacheprovider \
    python/tests/test_two_movable_doorway_adjacency.py \
    python/tests/test_best_first_sandbox_contract.py \
    python/tests/test_full_namo_strict_bfs.py \
    python/tests/test_full_namo_goal_clearance.py \
    python/tests/test_region_opening_targeted_contract.py \
    python/tests/test_solvability_runner.py \
    python/tests/test_best_first_protocol_defaults.py \
    python/tests/test_full_namo_budget_and_config.py \
    python/tests/test_push_budget.py \
    scripts/amarel/dhruv_container/tests | tee "$CONTAINER_ROOT/results/focused-tests.log"
"$NAMO_PYTHON" -B scripts/pipeline/validate_full_namo_alignment.py \
    --artifacts "$FROZEN_ROOT" --output "$CONTAINER_ROOT/results/fixed" --phase fixed
"$NAMO_PYTHON" -B scripts/pipeline/validate_full_namo_alignment.py \
    --artifacts "$FROZEN_ROOT" --output "$CONTAINER_ROOT/results/search" --phase search
touch "$CONTAINER_ROOT/provenance/validation-ready"
