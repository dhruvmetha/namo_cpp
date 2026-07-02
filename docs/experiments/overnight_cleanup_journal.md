# Overnight cleanup + wavefront-speed run

**Branch:** `feat/wavefront-cleanup-and-docs` (worktree `namo-cleanup`, forked from `feat/horizon-q-redesign` @ `df62137`).
**Operator:** Claude (autonomous overnight). **Human:** Dhruv, back in the morning.
This is the OPERATIONAL log of the run — what was done, why, and the numbers. Read top-to-bottom.

---

## ⛔ RESUME STATE (updated live — READ THIS FIRST after any compaction)

**Where I am:** working in worktree `/common/home/dm1487/robotics_research/ktamp/namo-cleanup`, branch `feat/wavefront-cleanup-and-docs`. Main checkout (`/common/home/dm1487/robotics_research/ktamp/namo`) is untouched. Env: `set -a; . ../.env; set +a` (arrakis; python `$NAMO_PYTHON`, MuJoCo 3.2.8; see memory `reference_arrakis_env`).

**THE GATE (run after every C++/behavior change; must stay 180/180, qpos 0):**
```
cd /common/home/dm1487/robotics_research/ktamp/namo-cleanup && set -a; . ../.env; set +a
./build_python_bindings.sh   # rebuild namo_rl after C++ edits
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 "$NAMO_PYTHON" scripts/sandbox/test_region_equiv.py --mode compare --n-per-tier 10 \
  --ref /common/users/dm1487/scratch_namo/eval/region_equiv/region_equiv_ref.json
```
Golden REF regenerable from `df62137` via `--mode capture`. Baseline profiler: `scripts/sandbox/profile_push.py`.

**DONE (committed, all gated 180/180):** doc INDEX+linter (`df62137`, on main branch too) · behavior gate (`9d56ab2`) · snapshot double-rebuild drop 8.9→5.5ms (`e1f7ffc`) · snapshot ctor-print silence (`7835584`) · docs compression -5/+archive (`775dc9a`) · morning-summary+recs (`216a9d0`) · **rec B: delete dead RegionAnalyzer subtree -3924 LOC (`e0bf976`)** · 10 memory notes written (backlog 0).

**IN PROGRESS:** rec A — reachability dirty-cache in `WavefrontPlanner::update_wavefront` (`src/wavefront/wavefront_planner.cpp:111`). Design = state fingerprint (start_pos + all movable poses); skip `recompute_wavefront` when unchanged. PROVEN SAFE: `reachability_grid_`/`dynamic_grid_` are written ONLY in recompute/rebuild/init (grep-verified), so the fingerprint is airtight. Add members to `include/wavefront/wavefront_planner.hpp` (~line 236). Then build + gate + profile before/after + commit.

**NEXT QUEUE (all gate-covered, do each then gate):** rec C rasterizer dedup (center-vs-corner trap) · rec D bbox-window for `WavefrontGrid::rebuild_grids` · more dead/dup code (a background agent is hunting) · then re-profile + update morning summary. Hourly Slack DMs to Dhruv (U07N1DR8S94). Do NOT touch `region_opening.py` search logic blind (not gate-covered).

---

## Mandate (from the user)

1. **Clean up doc/journal bloat** — compress to simplest, shortest forms; control future bloat; structure journaling better. No external tool required.
2. **Wavefront de-bloat + speed** — the wavefront logic is bloated and spread across ~40 files; optimize for speed and cleanliness **without changing behavior**.
3. **Safety net first** — record an eval set from the existing database that replicates how the codebase runs push skills for region-opening; gate every wavefront change against it (bit-identical, like the render-speedup "158/158 diff=0" gate). Make any eval sets **before** touching code.
4. **Profile** the region-opening hot path (push primitives, pushing sim, everything around it) to get an optimized, honestly-measurable baseline — because ML speedup ratios are meaningless if the base is slow.

## Confirmed research spine (the organizing axis)

- **Robotics:** push-primitive skill → **region-opening** (solved by **SEARCH**) → full **NAMO** pipeline.
- **Learning:** make that search **fast** — a sims-minimizing ranker that proposes the skill sequence for region-opening.

## Plan (safety-first ordering, each stage gated)

- **S0 — Ground** (4 parallel agents): wavefront-bloat map · region-opening hot-path trace · eval-harness/data scout · docs+journal compression plan. *(running)*
- **S1 — Golden eval set** (BEFORE any code change): record deterministic region-opening/push-skill outputs on ~20–50 scenes from the existing DB; store as the behavior oracle. Gate script must assert diff=0 on an unchanged tree first (prove the harness is sound).
- **S2 — Baseline profile**: time the hot path; record where wall-time goes. Numbers in this log.
- **S3 — Wavefront refactor**: dedup + avoid rebuild-every-call, one change at a time, each gated against S1 (must stay bit-identical) and re-profiled against S2.
- **S4 — Docs + journal compression**: execute the S0 doc plan; write the memory notes.
- Commit incrementally on the worktree branch. Nothing merged to the main branch without the user.

## Decisions ledger

- **[CLAUDE] Isolated worktree** so the main checkout is safe; branched from the commit that already has `docs/INDEX.md`.
- **[CLAUDE] Behavior gate before speed** — no wavefront edit lands until the golden eval set exists and the gate passes diff=0 on the unmodified tree.
- **[USER→CLAUDE] Obsidian needs no key** — local app opens the vault directly; only paid Sync/Publish needs a license (not needed).
- **[USER→CLAUDE] Write the 9 memory notes** — they are the anti-bloat mechanism (atomic auto-recalled facts vs 1000-line journals). One line each.

---

## PROGRESS LOG (append-only)

- **S0 launched** — 4 background grounding agents dispatched; awaiting results before any code change.
- **S0 DONE** — all 4 agents reported. Key findings:
  - **Wavefront:** 6 grid/BFS engines; 2 HOT. The canonical `WavefrontPlanner` rebuilds the grid from scratch on *every* reachability query, and the region-opening loop triggers **~8 rebuilds per candidate where ~2 distinct grids suffice** (`push_primitive_executor.cpp:196` funnels `is_robot_goal_reachable`/`get_reachable_edges`/`count_reachable_points`; fired 6× inside one `env.step` + 2× validation). `get_region_snapshot` rebuilds the grid **twice** (`rl_env.cpp:670-671`: ctor rebuild thrown away, then `update_dynamic_grid` redoes it). `RegionAnalyzer` subtree + sage `_compute_distance_field` are dead. → the "sim bottleneck" is largely redundant BFS, not `mj_step`.
  - **Gate design:** drive FIXED pushes through `RLEnvironment` directly (zero RNG; `set_full_state` zeros qvel, physics/wavefront carry no RNG, `PrimitiveGoalStrategy(shuffle_edges=False)` deterministic, region snapshot fixed seed).
  - **Docs:** verified redundancy → concrete 58→50 plan; journaling collapses to 3 live files (execution log + decision ledger + registry) + frozen evidence.
- **S2 DONE — baseline profile (median ms, 6 car scenes × 40 reps, C++ stdout suppressed).** `scripts/sandbox/profile_push.py`. `step(push)`=0.33 · `is_robot_goal_reachable`=0.23 · `get_reachable_edges`=0.23 · `get_reachable_objects`=0.82 · **`get_region_snapshot`=8.86** (26× the next op). ⇒ my prior "6 rebuilds/step dominate" guess was WRONG (step is cheap on these small 71×71 grids); the real hog is `get_region_snapshot`, which region-opening calls once per search node. Note: these test scenes are small — on larger deploy scenes the per-rebuild cost grows, so the reachability-cache win (#1) matters more there than the 0.33 ms step here suggests.
- **S3a DONE — wavefront win #1 (gate-covered, verified).** Removed the redundant second full-grid rebuild in `get_region_snapshot` (`rl_env.cpp:671`: `WavefrontGrid` ctor already calls `rebuild_grids(env)`, and `update_dynamic_grid()` is literally `rebuild_grids()` again on the same unchanged env — verified: 2 rebuilds/call → 1). **Gate: 180/180 discrete-identical, qpos max|diff|=0. `get_region_snapshot` 8.86 → 5.51 ms (−38%).** Behavior-identical by construction. Committed.
  - GATE SCOPE NOTE: the gate drives `RLEnvironment` directly (`step`/`get_reachable_*`/`get_region_snapshot`), so it covers C++ wavefront changes. It does NOT drive the `region_opening.py` planner, so Python search-loop edits (e.g. hoisting loop-invariant validation) are NOT gate-covered — those are left as evidence-backed recommendations, not applied blind overnight.
- **S3b DONE — wavefront cleanup #2 (gate-covered).** Moved the existing `CoutSilencer` in `get_region_snapshot` ABOVE the `WavefrontGrid` construction so the ctor's `Initialized wavefront grid`/`Grid rebuild took`/`Found N components` stdout no longer spams once per search node. **Gate 180/180, qpos 0.** `get_region_snapshot` 5.51 → 5.27 ms; bigger real-world win is removing per-node log I/O. Committed `7835584`.
- **S4 DONE — docs compression (reversible, git-preserved).** Deleted 5 archive slices that are strict subsets of `jan20_FULL_EVALUATION_REPORT` (`jan20_1push_results`, `eval_2push_1push_test_results`, `eval_2push_1push_test_consistency`, `dec25_crossattn_results`, `MCTS_TRAINING_DATA_STRUCTURE`). Archived 3 stale docs (`research_prompt`, the 2 uniform-rollout-sampler superpowers files) + removed empty `docs/superpowers/`. Reverted an over-eager archive of `results_design_report` (2 active docs cite it). Rewrote `INDEX.md`: de-hardlinked the gitignored personal doc, retagged `build_journal` FROZEN, linked this journal, recorded deferred merges. Lint: 0 broken doc→doc, 0 orphans. Committed `775dc9a`.
- **S5 DONE — 9 (+1) memory notes written.** The 9 `[[slug]]` the journals asked for now exist in `~/.claude/.../memory/` (backlog 0): `feedback_slurm_first`, `feedback_search_nosearch_lens`, `feedback_reuse_baselines`, `feedback_journal_attribute_decisions`, `project_minsnr_divergence`, `project_ro_single_object`, `project_policy_value_not_q`, `project_canonical_testset`, `reference_namo_value_learning_litmap`, plus a bonus `reference_arrakis_env`. `MEMORY.md` index updated.

---

## RECOMMENDATIONS — NOT applied (need review / higher risk than an overnight autonomous run should take)

### Wavefront (further speed/cleanup)
- **[A] Reachability dirty-cache (the canonical "big win").** `WavefrontPlanner` rebuilds the whole grid on every reachability query; the executor fires ~8/candidate where ~2 distinct grids suffice (4 identical pre-push + 2 identical post-push; `push_primitive_executor.cpp:84/125/161/166` + `namo_push_skill.cpp:163/192`). Fix: a monotonic state-version counter in `NAMOEnvironment` bumped on `step`/`set_full_state`/`set_robot_se2`; `WavefrontPlanner` caches `(last_version, last_start_pos)` and skips `recompute_wavefront` when unchanged (reachability is a pure function of state → bit-identical). **Deferred because:** (1) correctness risk if a state-mutation path is missed (stale grid → wrong reachability) — the gate WOULD catch it, but getting versioning complete is subtle; (2) low absolute payoff on the current small (71×71) test scenes where `step`=0.33 ms — measure on larger deploy scenes first. Gate-covered when done.
- **[B] Delete the `RegionAnalyzer` subtree (biggest LOC win).** A whole 3rd grid+flood-fill engine (`src/planners/region/region_analyzer.cpp:219`, + `region_based_planner`, `region_tree_search`, `region_path_planner`, `high_level_planner`, `examples/high_level_planning_demo.cpp`) is COLD — used only by standalone C++ binaries, NOT the Python `namo_rl` module (verified). Removing it is zero behavior-risk to the active path but touches CMake → verify the `high_level_planning_demo`/`namo_planner` binaries are unused, delete, rebuild, gate 180/180.
- **[C] Extract one shared rasterizer.** `is_point_in_rotated_rectangle` + `calculate_rotated_footprint` are duplicated in `wavefront_planner.cpp:127/198` and `wavefront_grid.cpp:137/208`, with a **center-vs-corner sampling divergence** (planner samples cell center `+0.5*res`, grid samples corner). Parameterize the sample-point offset and share — removes ~150 lines + the silent-divergence trap. Gate-covered; fiddly (must preserve each call site's convention exactly).
- **[D] bbox-window speedup for `WavefrontGrid::rebuild_grids`** (`wavefront_grid.cpp:45`): the O(W×H×objects) double loop tests every cell against every object; cull to each object's bounding box (the render-speedup already did this for the Python exporter). Would cut the remaining ~2.6 ms rebuild inside the 5.3 ms snapshot. Gate-covered.
- **[E] Dead-code deletions (viz-only, safe):** sage `visualizer.py:_compute_distance_field` (no callers); fix `wavefront_viewer.py:153` inflation from sqrt-diagonal to canonical `max(hx,hy)`.

### Docs (finish the compression — modest value, research-content edits, do with review)
- Merge (lossless append) `model_comparison_report.md` (prose) + `informative_1push_training_study.md` (epoch curves) → `informative_1push_results.md`; then delete the two sources.
- Edit-in-place dedups (replace duplicated text with a pointer): car-effect paragraph `horizon_q_datasets.md §1` → `canonical_testset.md`; RA@K `namo_pipeline.md §12.4` → `grounding_with_reachability.md`; H1–H5 wording `research_notes_F_characterization.md` → `F_problem_formulation.md`.

### Journaling restructure (proposed; ~10 journals → 3 live files)
- **Live:** `horizon_q_redesign_execution.md` (append-only execution log) + NEW `horizon_q_decisions.md` (decision ledger: `Dn/Hn — claim` / Hypothesis / Evidence / Verdict / [USER|CLAUDE]) + `horizon_q_model_registry.md` (ckpt catalog — never merge).
- **Freeze as evidence:** `horizon_q_build_journal`, `policy_framework_journal`, `informed_2push_journal` (+ledger), `scorer_hacman_journal`, `results_design_report_2026-06-15`, `scorer_1push_results`; lift their accept/reject verdicts as one-liners into the decision ledger.
- **Keep as non-journals:** `horizon_q_HANDOFF` (brief), `horizon_q_overview` (hub), `multipush_learning_primer` (reference), `ILAB_RESUME` (resume pointer).

---

## MORNING SUMMARY (2026-07-01 overnight)

**Branch `feat/wavefront-cleanup-and-docs`** (worktree `namo-cleanup`), forked from `feat/horizon-q-redesign@df62137`. Your main checkout is untouched. Nothing merged to any main branch.

**Shipped + verified (behavior-identical, gate 180/180, qpos diff 0 on every step):**
1. `test_region_equiv.py` — a reusable behavior gate: drives fixed pushes through the sim on 29 stratified scenes → 180 golden states. REF at `/common/users/dm1487/scratch_namo/eval/region_equiv/region_equiv_ref.json` (gitignored-sandbox script, like the render gate; regenerable from `df62137`).
2. Wavefront win: `get_region_snapshot` (the dominant per-search-node op) **8.86 → 5.27 ms (−40%)** — dropped a redundant 2nd full-grid rebuild + silenced per-node stdout spam.
3. Docs: −5 files (redundant), 3 archived; `INDEX.md` clean (0 broken links, 0 orphans); doc-link linter (`scripts/docs_lint.py`).
4. 10 memory notes (auto-recalled across sessions).

**Baseline (honest framing for the ML-speedup story):** on these car test scenes `step(push)` is already cheap (0.33 ms) — the per-sim cost is NOT the bottleneck; `get_region_snapshot` (once per search node) is. So ML's value is in reducing the *number* of sims/nodes, and the node-cost is what to keep shrinking (recs A–D). The base is now cleaner, so reported speedups won't be against a bloated snapshot.

**To verify anything I claim:** `cd namo-cleanup && set -a; . ../.env; set +a` then
`CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 $NAMO_PYTHON scripts/sandbox/test_region_equiv.py --mode compare` (expect 180/180) and `... profile_push.py` (per-op ms).

**Biggest open lever:** rec [A] reachability dirty-cache — deferred for correctness caution; worth measuring on larger scenes. Rec [B] (delete dead `RegionAnalyzer`) is the biggest cleanup LOC win.
- **S1 DONE — golden behavior gate built + validated.** `scripts/sandbox/test_region_equiv.py` (model-free; captures reachable-objects, reachable-edges, `is_robot_goal_reachable`, C++ `get_region_snapshot` graph/labels, and qpos fingerprint per push). Harness soundness confirmed: `compare` on the UNCHANGED build = **20/20 discrete-identical, qpos max|diff|=0**. Full golden captured on frozen `df62137`: **29 scenes (easy/med/hard) × ~6 pushes = 180 states, 0 errors, 8.7 s wall**. REF at `/common/users/dm1487/scratch_namo/eval/region_equiv/region_equiv_ref.json` (regenerable from this commit; deterministic scene selection). Gate must stay 180/180 through the refactor.
