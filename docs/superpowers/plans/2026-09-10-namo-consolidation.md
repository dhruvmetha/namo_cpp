# NAMO Selective Consolidation Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` to execute the checkboxes sequentially. Do not use subagents in this side conversation. Execution and integration into the checked-out dhruv-linux branches were explicitly authorized by the user on 2026-09-10. Follow the recorded scope and preserve ongoing work.

**Goal:** Integrate the missing Full-NAMO runtime and reusable testbed changes into the branches currently checked out on dhruv-linux, while preserving their newer functionality and existing experiment provenance.

**Architecture:** Use one temporary integration worktree based on `namo_cpp`'s `feat/horizon-q-redesign`. Port the goal-clearance feature and the selected testbed tools as separate, reviewable commits; then fast-forward the destination only after validation and explicit approval. Leave `robot_control` on `real-robot` unless the inventory identifies a specific additional user-authored change that actually needs integration.

**Tech Stack:** Git, Python, C++/pybind11, MuJoCo, pytest, and the existing NAMO environment/build scripts.

**Status:** Complete on 2026-09-10. Both implementation stages were promoted to the checked-out NAMO branch; its binding was rebuilt and all 108 focused tests passed. Robot-control and historical work were preserved. See the execution record for revisions and validation evidence.

## Engineering Standards

- Follow `plan-coding-standards`: focused responsibilities, existing naming conventions, reuse rather than duplicated planner logic, and updated docstrings for changed public interfaces.
- Preserve the current search behavior outside the explicitly integrated feature. Do not introduce new planners, learned-model changes, difficulty rules, or large refactors.
- Keep configuration in the existing configuration/environment interfaces. Do not add machine paths, checkpoint paths, or historical campaign quotas to runtime logic.
- Use existing diagnostics and specific failure reasons. Keep occupied-goal evidence separate from the reason a search stopped; do not turn exceptions into ordinary search failures.
- Commit the runtime feature and testbed tooling independently, with source-commit provenance. Each implementation commit must be coherent, buildable where applicable, and pass its relevant automated tests.
- Reuse existing tests. Add only the focused combined-feature regression specified below; do not build a new broad testing framework or rerun a complete benchmark to validate a merge.

## 1. Verified starting point and scope

These observations were checked directly on dhruv-linux. Recheck them at execution time because another session may advance a branch.

| Component | Repository / branch | Observed commit | Treatment |
|---|---|---|---|
| NAMO backend and Full-NAMO planner | `/home/dhruv/projects_dhruv/namo/namo_cpp`, `feat/horizon-q-redesign` | `f8be9ff402e2ab32be7356155c0e73d1d7ece726` | Integration destination |
| Robot runtime | `/home/dhruv/projects_dhruv/namo/robot_control`, `real-robot` | `ae522814b2647d53a571bb5d9bd7182852267a0c` | Preserve existing checkout and local edits |
| Goal-clearance implementation | NAMO commit `8bc97d99e11853ada7af53eb99e3600ea968d100` | `fix: recover movable-occupied goals in Full NAMO planning` | Source patch, not replacement tree |
| Aug9 independent certification | NAMO commit `15fc85963e8822c791dc84428b46452dfd5bd491` | `feat: certify sampled Aug9 two-hop scenes with independent K2 removal` | Source feature plus required snapshot dependencies |

`full_namo` is a package inside `namo_cpp`, not a third Git repository. The destination is not the branch named `main`.

The current NAMO branch already contains hop-aware routing (`dbf7c27f`), multi-blocker policy ranking (`92e9c689`), and multi-blocker search ranking (`6a2f9001`). Its tracked worktree was clean during inspection.

The goal-clearance source branch has **no common ancestor** with the destination. Its root, `851fb6bf0a9aa30a17f162bbf27c72d3d7cb9c0b`, imported a source snapshot. A later commit, `f897ac3c`, replaced portions of that snapshot with a deployed runtime. Neither is an ordinary incremental feature patch suitable for blindly applying to the destination.

The current robot-control branch contains the rebased versions of the identified Trian27/Tri-An refactors (`a1851d6f`, `692acc8d`, `e383a997`) and `a366f545`. Do not duplicate their older equivalent commits. It also has uncommitted changes in `config/objects.yaml`, `real_exp/EXPERIMENT_STATUS.md`, `real_trials/trials.csv`, the `rvg` submodule, and `src/robot_control/planner/navigation_baseline_planner.py`; do not stash, reset, commit, or otherwise manipulate these as part of NAMO consolidation.

### Explicit exclusions

- No whole-tree replacement, `--allow-unrelated-histories` merge, or rebase of published experiment branches.
- No changes to learned weights, local episode definitions, primitive inventories, navigation margins, collision-check cadence, or registered evaluation protocols.
- No copying or overwriting datasets, certificates, frozen manifests, model checkpoints, evaluation outputs, or running campaign code.
- No branch/worktree deletion or automatic cleanup. Consolidating source does not authorize removing historical sources.
- No robot movement, real-robot test session, Slurm submission, or full evaluation campaign.

## 2. Preflight and one isolated worktree

Run the following only after implementation is explicitly authorized. All shell blocks below are intended for a Bash shell on dhruv-linux, unless marked otherwise.

- [x] Read the destination's `CLAUDE.md`, `docs/problem_and_approach.md`, and `env.robotlearning.sh`. Use `superpowers:using-git-worktrees` when creating the integration worktree.
- [x] Recheck branch heads, tracked/untracked edits, and the existing worktree registry. Stop if the named destination changed branches or if a concurrent writer would be affected.

```bash
cd /home/dhruv/projects_dhruv/namo/namo_cpp
git branch --show-current
git rev-parse HEAD
GIT_OPTIONAL_LOCKS=0 git status --short
git worktree list --porcelain
git cat-file -t 8bc97d99e11853ada7af53eb99e3600ea968d100
git cat-file -t 15fc85963e8822c791dc84428b46452dfd5bd491
git merge-base HEAD fix/full-namo-goal-clearance-20260909
```

Expected: branch `feat/horizon-q-redesign`, both source objects reported as `commit`, and no common ancestor from the final command (exit status 1 at the inspected revisions). A changed destination commit requires reviewing its new changes, not resetting it to the SHA in this plan.

- [x] Inventory the relevant source families and classify each as `already present`, `port`, or `historical/out of scope`: goal-clearance, hop-aware random5, corrected testbed, two-push certification, Aug9 generation, labeled random5, and the earlier multiobject/four-object branches. Compare patches and resulting behavior, not just author strings or whether a commit SHA appears in history. Do not call every outstanding branch an outstanding feature.
- [x] Record the verified destination SHA and this classification in this plan when it is copied into the integration repository at `docs/superpowers/plans/2026-09-10-namo-consolidation.md`. No separate inventory framework is needed.
- [x] Check that `.worktrees` is already ignored, and that the proposed branch/path do not exist. If either exists, inspect and reuse it only if it belongs to this integration; do not overwrite it.

```bash
git check-ignore .worktrees
git branch --list integration/full-namo-consolidation-20260910
git worktree add \
  -b integration/full-namo-consolidation-20260910 \
  /home/dhruv/projects_dhruv/namo/namo_cpp/.worktrees/consolidate-full-namo-20260910 \
  feat/horizon-q-redesign
```

- [x] Configure and build only inside this new worktree. The machine script derives `SAGE_REPO` from the checkout's parent; override that derivation because this is a nested worktree. Keep this override in the execution environment, not in tracked source.

```bash
cd /home/dhruv/projects_dhruv/namo/namo_cpp/.worktrees/consolidate-full-namo-20260910
source env.robotlearning.sh
export SAGE_REPO=/home/dhruv/projects_dhruv/namo/sage_learning
export PYTHONPATH="$PWD/build_python:$PWD/python:$PWD/scripts:$PWD/scripts/sandbox:$PWD/scripts/pipeline:$SAGE_REPO"
PYTHON_BIN="$NAMO_PYTHON" ./build_python_bindings.sh
"$NAMO_PYTHON" -c 'import namo_rl; print(namo_rl.__file__)'
```

Expected: the imported binding is from the integration worktree's `build_python`, not another checkout. Missing libraries or data are an explicit blocker for the affected validation gate, not a passing skipped test. Do not replace the live checkout's build directory.

- [x] Establish the baseline by running the existing planner regression command from Stage A below, omitting `test_full_namo_goal_clearance.py` because that file is not present yet. Record any pre-existing failure before porting code.

Preflight creates no artificial checkpoint commit. The first implementation commit follows Stage A's validation.

## 3. Stage A — integrate goal clearance without losing multi-blocker handling

### Files and exact source

The complete source implementation and its focused tests are the patch introduced by `8bc97d99`, not the entire source-branch tree. Inspect it with:

```bash
git show --stat 8bc97d99
git show --format=fuller 8bc97d99 -- \
  include/ src/ python/namo/ python/tests/test_full_namo_goal_clearance.py
```

Files in that patch:

- Backend: `include/planning/push_primitive_executor.hpp`, `include/skills/namo_push_skill.hpp`, `include/wavefront/wavefront_grid.hpp`, `src/planning/push_primitive_executor.cpp`, `src/skills/namo_push_skill.cpp`, `src/wavefront/wavefront_grid.cpp`.
- Bindings: `python/namo/cpp_bindings/bindings.cpp`, `python/namo/cpp_bindings/rl_env.cpp`, `python/namo/cpp_bindings/rl_env.hpp`.
- Planner: `python/namo/planners/__init__.py`, `python/namo/planners/full_namo/full_namo_planner.py`, new `python/namo/planners/full_namo/goal_clearance.py`, `python/namo/planners/opening/best_first_region_opening.py`.
- Runner and tests: `python/namo/solvability_runner.py`, new `python/tests/test_full_namo_goal_clearance.py`; extend the existing multi-blocker regression in `python/tests/test_full_namo_strict_bfs.py`.

### Ordered work

- [x] Bring the existing goal-clearance regression file from the source commit into the integration worktree with `apply_patch`. Run it before the implementation to confirm the missing-feature failures. Reuse its existing occupied-goal, inaccessible-blocker, alternative-blocker, scorer-mask, terminal-diagnostics, and backend-state tests.
- [x] Port the implementation patch with `apply_patch`. Where the destination already has equivalent code, retain that implementation. Do not import `851fb6bf`, `f897ac3c`, or the historical whole-file CMake replacement merely to make patches apply.
- [x] Keep the C++ ownership query read-only, retain all overlapping movable owners, and preserve current grid, margin, and snapshot behavior. Extend existing binding signatures additively; do not discard later destination-side APIs.
- [x] Keep `full_namo_goal_clearance` as the existing explicit opt-in, with its supported-backend/local-search checks. Integrating the feature does not itself enable it for all callers or change existing campaign configurations.
- [x] Resolve the main planner conflict by distinguishing ordinary boundary searches from a specific goal-clearance task. At the existing blocker-selection site, preserve the destination's pooled behavior for normal routes:

```python
if choice.clearance_target is None:
    target_object_ids = tuple(
        other.object_id
        for other in all_route_choices
        if other.clearance_target is None
        and other.boundary == choice.boundary
        and (other.boundary, other.object_id) not in blocked_choices
    )
else:
    target_object_ids = (choice.object_id,)

opener_kwargs = {
    "target_object_id": (
        target_object_ids[0]
        if len(target_object_ids) == 1
        else target_object_ids
    ),
    "require_push": True,
}
if self.goal_clearance:
    opener_kwargs["accept_robot_goal"] = True
if choice.clearance_target is not None:
    opener_kwargs["clearance_target"] = choice.clearance_target
if self.local_search == "best_first" and len(path) == 2 and choice.reaches_goal:
    opener_kwargs["opening_predicate"] = (
        lambda candidate_env: candidate_env.is_robot_goal_reachable()
    )
```

- [x] Retain `_mark_blockers_tried`, the candidate-membership checks, `candidate_blockers` trace fields, and bookkeeping based on the object actually pushed. Do not replace those with the older source branch's single-selected-object assumptions.
- [x] Port the source's clearance/access route distinction and `graph_hops` accounting. A clearance task is not a fabricated free-space region. After a committed physical change, recompute the graph and continue even if its hop count has not decreased; preserve the destination's route-deferral/reset rules.
- [x] Preserve complete-episode success as actual end-goal reachability. Partial clearance of one overlapping blocker is progress, not complete success. Do not reinterpret the separate greedy-policy service's one-action-return contract as a completed episode.
- [x] Preserve simulation accounting, including the caller-specified 20,000-call Full-NAMO cap. Do not change local search defaults, action-validity checks, model weights, score combination, or local keyhole labeling predicates.
- [x] Preserve the source's runner output for unsuccessful runs: committed actions, terminal state, total calls, failure reason, and separate goal-occupancy diagnostics.

### One focused combined-feature regression

Extend `test_full_namo_searches_every_blocker_on_the_boundary_in_one_call` in `python/tests/test_full_namo_strict_bfs.py`, rather than adding another suite:

- [x] Parameterize its `goal_clearance_enabled` input over `[False, True]` with `pytest.mark.parametrize`.
- [x] Let its fake opener accept `**_kwargs` in addition to its current arguments, so the enabled variant accepts `accept_robot_goal` without masking target-object assertions.
- [x] Replace this test's `make_planner(...)` call with a construction that exercises the real feature configuration while still using the fake opener:

```python
opener = FakeOpener()
monkeypatch.setattr(
    FullNAMOPlanner,
    "_initialize_algorithm",
    lambda self: setattr(self, "region_opener", opener),
)
planner = FullNAMOPlanner(
    env,
    PlannerConfig(algorithm_params={
        "full_namo_local_search": "best_first",
        "full_namo_goal_clearance": goal_clearance_enabled,
    }),
)
```

- [x] Attach the following ordinary, unoccupied goal snapshot data:

```python
snapshot["goal_clearance"] = {
    "goal_xy": [0.0, 0.0],
    "resolution": 0.01,
    "cells": [{
        "xy": [0.0, 0.0],
        "grid": [0, 0],
        "static_blocked": False,
        "objects": [],
        "region": "goal",
    }],
    "access_regions": {},
    "reachable_objects": ["box_a", "box_b"],
}
```

Retain the test's exact existing assertions: the first call receives `("box_a", "box_b")`, the returned chain on `box_b` is accepted, no boundary exhaustion occurs, and `candidate_blockers` records both objects. The added parameterized case must fail if the integration accidentally restores single-blocker dispatch. Keep the existing greedy-policy pooled-blocker and pooled-exhaustion/rerouting tests unchanged as additional guards.

### Validation and commit

- [x] Rebuild the integration binding with the preflight build command. Run the following focused regression set and inspect skips as well as failures:

```bash
"$NAMO_PYTHON" -m pytest -q -rs \
  python/tests/test_full_namo_strict_bfs.py \
  python/tests/test_full_namo_budget_and_config.py \
  python/tests/test_full_namo_greedy_dfs.py \
  python/tests/test_best_first_edge_blacklist.py \
  python/tests/test_best_first_protocol_defaults.py \
  python/tests/test_solvability_runner.py \
  python/tests/test_full_namo_goal_clearance.py
```

Expected: all relevant cases pass; `test_backend_goal_ownership_is_read_only` executes with the real binding and verifies unchanged `qpos` and `qvel`. This is not satisfied by a stub-only pass.

- [x] Run the existing real multi-blocker fixture when its checkpoint and scene are present. It uses simulation, not the real robot:

```bash
NAMO_TEST_DEVICE=cpu "$NAMO_PYTHON" -m pytest -q -rs \
  python/tests/test_full_namo_policy_multi_blocker_scene.py
```

Expected: both doorway blockers reach the opener together, the returned action targets one of them, and policy scoring does not silently become simulated search. If the fixture is unavailable, record the missing artifact; do not claim this gate passed.

- [x] Reuse one already-saved recovered blocked-goal episode and its recorded seed/config as a bounded physics regression. Resolve the exact row from the goal-clearance experiment record at execution time; do not pick a new random case or overwrite its old result. Record scene identity, seed, current code/binding revision, call count, terminal diagnostics, and actual goal reachability in a new integration-only output directory. Do not assume identical call counts across different code or hardware. Do not submit a campaign to satisfy this check.
- [x] Review `git diff --check` and the staged diff. Stage only the Stage A files and the copied plan's execution record.
- [x] Commit with subject `fix(full-namo): integrate goal clearance with pooled boundary search` and body `Port the goal-clearance feature from 8bc97d99 onto feat/horizon-q-redesign. Preserve multi-blocker ranking, route retries, exact episode completion, and shared-budget accounting. Record focused unit and physics validation in the consolidation plan.`

## 4. Stage B — integrate selected testbed tooling, without reviving old campaigns

### Source and file map

The missing tools are present in the tree at `8bc97d99`, including the Aug9 certification change from `15fc8596`. Some dependencies came from the initial snapshot, so cherry-picking `15fc8596` alone is insufficient.

| Files | Purpose / integration treatment |
|---|---|
| `scripts/pipeline/certify_keyhole_horizons.py`, `python/tests/test_certify_keyhole_horizons.py` | Port independent certification, exhaustive one-push evidence, and generated-scene support together |
| `scripts/pipeline/compose_keyhole_modules.py` | Existing destination file: adapt only helpers required by the imported certifier; retain current fixes |
| `scripts/slurm/multihop_aug9_generate.slurm`, `scripts/slurm/multihop_aug9_eval.slurm`, new `scripts/slurm/multihop_aug9_pipeline.sbatch` | Review individual launcher changes; preserve current campaign settings and environment discovery |
| `scripts/pipeline/assemble_keyhole_random5.py`, `scripts/pipeline/freeze_keyhole_random5.py`, `scripts/pipeline/tests/test_freeze_keyhole_random5.py` | Port the mutually dependent result-assembly/freezing tools and their existing tests as a unit |
| `scripts/pipeline/build_balanced_keyhole_testbed.py` | Historical mixed-context selector: preserve only as an explicitly historical/reproducibility tool, not the new default testbed policy |

- [x] Inspect exact source files with `git show 8bc97d99:PATH`, using the paths above, and review the feature delta with `git show 15fc8596`. Apply only reviewed source with `apply_patch`; do not replace entire directories.
- [x] Check all certifier calls into `compose_keyhole_modules`: `get_region_snapshot`, `shortest_region_path`, `geom_sig`, and `REPO`. Reuse the destination's implementations when compatible. Preserve its later geometry, sampled-contact, and margin fixes.
- [x] Keep K2 certification's existing removed-K1 shortcut, with original scene files unchanged. Do not substitute post-K1-state certification or add a new requirement that the first independent opening preserve a predetermined second gate.
- [x] Preserve the distinction between independent gate evidence and full-episode success. Do not promote partial/unknown labels to definitive labels, or use witness length alone as proof that one-push options were exhausted. This consolidation is not authorization to relabel historical data.
- [x] Preserve five-seed aggregation semantics, including failed runs and the separate unresolved category. Preserve recorded thresholds as protocol-specific configuration; do not replace them with new bins or change a frozen testbed's membership.
- [x] Keep historical assumptions visible. The source assembler contains a campaign-specific `535 / 458 / 177` count assertion, and the old balanced selector contains `428`-scene / `280`-discovery-job assumptions. These are not general requirements for the current diverse-template testbed. Label their scope in docstrings/help or retain them as documented historical entrypoints; do not silently wire them into current generation. Generalizing these programs is a separate decision, not a prerequisite for consolidation.
- [x] Preserve source provenance and output schemas. New verification outputs go to an unused integration-specific path. Do not invoke packaging or publishing modes on an existing frozen directory.

### Validation and commit

- [x] Run the existing certifier, composer, and freezing tests:

```bash
"$NAMO_PYTHON" -m pytest -q -rs \
  python/tests/test_certify_keyhole_horizons.py \
  python/tests/test_compose_keyhole_modules.py \
  scripts/pipeline/tests/test_freeze_keyhole_random5.py
bash -n scripts/slurm/multihop_aug9_generate.slurm
bash -n scripts/slurm/multihop_aug9_eval.slurm
bash -n scripts/slurm/multihop_aug9_pipeline.sbatch
"$NAMO_PYTHON" scripts/pipeline/certify_keyhole_horizons.py --help
"$NAMO_PYTHON" scripts/pipeline/freeze_keyhole_random5.py --help
```

Expected: tests pass, shell parsing succeeds, and CLI help exits without starting generation, certification, freezing, or a scheduler submission. Existing tests must continue to reject failed/colliding pushes as openers, verify exhaustive one-push trial accounting, preserve K2 removal scope, and keep failed seeds in difficulty calculations.

- [x] Review the final file list for unrelated runtime changes, old quotas accidentally promoted to defaults, and machine-specific paths. Re-run Stage A's planner regression set if composer/snapshot/runner code changed during this stage.
- [x] Commit with subject `feat(pipeline): consolidate independent keyhole certification and testbed tools` and body `Port reviewed tooling from 15fc8596 and the 8bc97d99 source tree. Preserve current composer fixes, independent K2 removal, five-seed failure accounting, and historical campaign provenance. Keep legacy selectors explicitly scoped and leave existing datasets and jobs unchanged.`

## 5. Final review, promotion, and reproducibility handoff

- [x] Ensure all in-scope source families from preflight have an explicit disposition. If additional user-authored runtime work remains unresolved, list it; do not report that every branch is consolidated.
- [x] Confirm that the integration changes only NAMO source, relevant tests, selected tools, and this plan's provenance/validation record. Confirm that `robot_control` is unchanged by this work.
- [x] Review the destination-to-integration diff and confirm the two implementation commits contain no snapshot replacement, model artifact, generated result, broad configuration reset, or unrelated deletion.

```bash
git diff --check feat/horizon-q-redesign..HEAD
git diff --stat feat/horizon-q-redesign..HEAD
git log --oneline feat/horizon-q-redesign..HEAD
GIT_OPTIONAL_LOCKS=0 git status --short
```

- [x] Use the user’s explicit instruction to integrate into the checked-out branches as promotion authorization. Recheck the destination branch, its cleanliness, and whether another session is using its runtime. If the destination advanced, first bring those new commits into the integration branch and revalidate the overlap. Do not overwrite or reset the destination, and do not change a live runtime while it is in use.
- [x] After approval and a quiet destination checkout, promote with a fast-forward only:

```bash
git -C /home/dhruv/projects_dhruv/namo/namo_cpp \
  merge --ff-only integration/full-namo-consolidation-20260910
```

Expected: a fast-forward. If Git refuses, stop and reconcile the changed destination in the integration worktree. Do not substitute an automatic force operation or unrelated-history merge.

- [x] Before the promoted checkout is used for planning, rebuild its own `build_python` through `env.robotlearning.sh` and `build_python_bindings.sh`, verify `namo_rl.__file__`, and run the focused backend/goal-clearance regression there. A source merge alone does not update the compiled simulator interface. Coordinate this activation with the user; until it is done, report `source integrated; live binding activation pending`.
- [x] Record the final commit SHA, source patch SHAs, test outcomes/skips, binding location, and any remaining exceptions in this plan. Preserve all old branches, worktrees, immutable campaign clones, results, and certificates. Copying the validated revision to Amarel, rerunning evaluations, pushing branches, and deleting old worktrees require separate direction.

## Completion criteria

- The approved runtime and tooling changes are present on dhruv-linux's current NAMO branch without removing its newer multi-blocker behavior.
- Existing local opening semantics and registered evaluation settings are unchanged; Full-NAMO episode completion still depends on reaching the final goal.
- Relevant automated regressions and the bounded real-physics checks have recorded outcomes; a required skipped test is not called a pass.
- The promoted runtime is either rebuilt and validated or explicitly reported as awaiting activation.
- Robot-control edits, running work, historical data, and published experiment provenance remain intact.
- Any unintegrated source family is named explicitly. No blanket claim is made that every historical branch has been merged.

## Execution record — 2026-09-10

The user explicitly authorized integration into the checked-out branches on dhruv-linux. Implementation, commits, and fast-forward promotion are within that instruction. Work proceeded sequentially without subagents. No scheduler or real-robot commands were invoked.

### Verified preflight and source-family disposition

NAMO destination: `feat/horizon-q-redesign` at `f8be9ff402e2ab32be7356155c0e73d1d7ece726`. Tracked files were clean; the untracked `HIGH_LEVEL_PLANNER_README.md` was preserved. The integration worktree is `/home/dhruv/projects_dhruv/namo/namo_cpp/.worktrees/consolidate-full-namo-20260910` on `integration/full-namo-consolidation-20260910`. `.worktrees` was already ignored. The source remains unrelated in Git history; only selected feature changes were ported.

Robot-control remains `real-robot` at `ae522814b2647d53a571bb5d9bd7182852267a0c`. All four identified refactors (`a1851d6f`, `692acc8d`, `e383a997`, `a366f545`) are ancestors of this checkout. Existing edits and untracked experiment material were preserved; the initial tracked-diff SHA-256 was `2fd1fe51c27eeea842fe63ef3d54af871a3a305ed2bfd3cdb9584114f0541d0f`. The camera service was active; no planning process was active at preflight.

| Source family | Disposition | Evidence and scope |
|---|---|---|
| Goal clearance | Port | Runtime delta and existing tests from `8bc97d99`; preserve destination multi-object topology APIs, pooled blocker dispatch, and actual-pushed-object bookkeeping |
| Hop-aware random5 | Already present for runtime | `dbf7c27f`, `92e9c689`, and `6a2f9001` are in destination history; historical campaign outputs remain archival |
| Corrected testbed | Historical/reproducibility only | Retain the destination composer; port the legacy balanced selector with explicit 428-scene/280-job scope; do not activate old correction campaigns or donor policies |
| Two-push certification | Port | Complete independent certifier and its existing tests; do not import unrelated same-template donor-replay changes |
| Aug9 generation | Extend existing tooling | Preserve existing generation/evaluation defaults and environment discovery; add reviewed campaign overrides and independent-certification stages |
| Labeled random5 | Port | Assembly/freezing tools and existing tests; scope old quotas and difficulty thresholds to their recorded protocols |
| Earlier multiobject/four-object work | Existing runtime or historical/out of scope | Preserve current multi-object connectivity and all scene/experiment sources; no additional robot-control runtime patch identified in the approved feature scope |

Composer dependencies `get_region_snapshot`, `shortest_region_path`, `geom_sig`, and `REPO` already exist in the destination with compatible signatures. No composer replacement is needed. Geometric-prior additions in the historical source runtime, old CMake replacements, and unrelated feature branches are outside this integration.

### Stage A validation

All build and verification artifacts are isolated at `/home/dhruv/projects_dhruv/namo/scratch_namo/outputs/consolidation-20260910`. The worktree uses `env.robotlearning.sh` with the nested-worktree `SAGE_REPO` override supplied only in the execution environment. The imported binding path was verified to be inside this worktree’s `build_python`.

- Baseline build: passed; existing focused regression set: **51 passed** (`baseline.log`, exit 0).
- Imported occupied-goal tests before implementation: **8 failed, 14 passed** when run with strict-BFS tests; failures identify the missing clearance module, routing, binding argument, occupied-goal recovery, and runner opt-in (`stage-a-red.log`, expected exit 1).
- Rebuilt runtime and combined-feature regression: **60 passed**, no skips (`stage-a-green.log`, exit 0). The real binding executed `test_backend_goal_ownership_is_read_only`; qpos and qvel were unchanged.
- Existing real multi-blocker policy fixture: **1 passed**, no skips, on CPU with the installed HY5U checkpoint. Both doorway blockers remained pooled and policy returned without simulated search.
- Saved recovered episode `5e8e1e445ecea6136aac`, geometry `65c97f9654906b9d1db44ae2974073be`, HY5U S2, sampler seed 42, shuffle seed 7000, full-problem cap 20,000: the initial-scene replay reached the goal in **8 calls / 4 committed actions**. Scene, primary config, checkpoint, and all three primitive tables matched the original SHA-256 values. The historical fixed run used 37 calls; this is provenance, not a cross-revision performance claim. The newer destination solved this replay through ordinary openings.

The archived no-fix terminal state of the same episode was restored exactly: goal reachability began false, the trace explicitly exercised a goal-clearance task, and the goal became reachable in **14 simulator calls** (`physics-terminal-replay.json`, exit 0). Original evidence is preserved in `physics-source-record.json`; replay scripts, terminal states, traces, binding hash, and working-diff/source hashes are in the integration output directory. The validated integration binding SHA-256 is `0418c1c0a69618ac60812fd3d23c670a052170ee2bfddd31a8ccb3eed84bdf6c`. No historical record was overwritten.

### Stage B validation and reviewed differences

Runtime commit: 5febbe7684c833d8c3d86f584e573ad57c2963bf. Tooling comes from source tree 8bc97d99e11853ada7af53eb99e3600ea968d100, including independent generated-scene certification commit 15fc85963e8822c791dc84428b46452dfd5bd491.

- Existing freezing regressions failed before the tools were added (2 failures); the certifier tests could not collect because its module was missing. These expected pre-port outcomes are preserved in stage-b-freeze-red.log and stage-b-certifier-red.log.
- Certifier, unchanged composer, and freezing tests: **47 passed**, no skips (stage-b-tests.log, exit 0).
- All three Slurm scripts passed bash parsing. Certifier, freezing, assembly, and historical balanced-selector help exited successfully; none launched a pipeline or wrote a dataset (stage-b-cli.exit = 0, saved help outputs).
- The composer was retained byte-for-byte. Its existing snapshot, geometry, sampled-contact, and margin behavior needs no adaptation for the certifier. Stage B changes no planner, runner, backend, or configuration file, so the Stage A regression set was not needlessly repeated here.
- Both existing launchers retain their resource, sampling, and budget defaults. Generation retains env.amarel.sh as its default activation and its existing Python fallback; campaign overrides remain optional. The source's unconditional PYTHONHASHSEED=0 default was not ported. The evaluator retains default keyhole budget scope while permitting the source's explicit full-problem override.
- Historical assembly counts, the 428-scene/280-job selector, and the fixed-template freezing thresholds are explicitly scoped in docstrings/help. No new quotas were wired into generation.

A sequential source/diff review checked additive binding signatures, retained multi-object fields, pooled ordinary-boundary dispatch, clearance-only object selection, actual-pushed-object bookkeeping, scorer restoration, strict final-goal completion, K2 removal scope, complete one-push evidence, and failed-seed accounting. No unrelated snapshot, learned artifact, configuration reset, or historical campaign output is included.

### Promotion and live activation

The checked-out `feat/horizon-q-redesign` branch was fast-forwarded from `f8be9ff402e2ab32be7356155c0e73d1d7ece726` to the final implementation revision **`6a71d2f91d27d728e9d51afa2465ba09a21ab8a9`**, following runtime commit **`5febbe7684c833d8c3d86f584e573ad57c2963bf`**. This final documentation update records verification performed after that promotion; it changes no runtime or tooling source.

The destination rebuilt its own binding through `env.robotlearning.sh` and `build_python_bindings.sh`. Its actual import path is `/home/dhruv/projects_dhruv/namo/namo_cpp/build_python/namo_rl.cpython-312-x86_64-linux-gnu.so`, SHA-256 `0418c1c0a69618ac60812fd3d23c670a052170ee2bfddd31a8ccb3eed84bdf6c`, identical to the isolated validated binding. `BUILD_INFO` records implementation revision `6a71d2f9`, include tree `ef9f4e79c09383e704f04344f3a2d8f193a4f4eb`, source tree `4697fac7e3d392416ca09aa93049f87bf57f4ee6`, and `dirty_cpp=0`.

All eleven focused test files were then run together against the promoted checkout and rebuilt binding: **108 passed, no skips**, including the real ownership-state check, pooled-blocker policy fixture, planner/budget regressions, certification, unchanged composer, and five-seed freezing tests (`activation.log`, exit 0). Live binding activation is complete. Goal clearance remains the explicit opt-in `full_namo_goal_clearance=True` / runner `--goal-clearance`.

Final preservation checks matched robot-control’s branch, commit, and tracked-diff SHA-256 to preflight. The camera process remained alive. The untracked NAMO README retained SHA-256 `0e6637a5837d03f90fce10f4908f817b2cf6a856a99d689ad6d2693114b89985`; no tracked NAMO edits remain outside the committed integration. The historical goal-clearance source branch remains at `d28e09ca0213a29ae6cbb99dc163280b4a986e99`. All old branches, worktrees, frozen sources, certificates, datasets, and results were retained. The named historical/out-of-scope families in the disposition table were not merged wholesale.

Machine-readable final evidence is in `final-activation.json` beside the saved build logs, test results, original replay evidence, and bounded physics outputs. The same completed plan is mirrored to the original local planning document for handoff.
