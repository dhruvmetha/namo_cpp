# Overnight cleanup + wavefront-speed run

**Branch:** `feat/wavefront-cleanup-and-docs` (worktree `namo-cleanup`, forked from `feat/horizon-q-redesign` @ `df62137`).
**Operator:** Claude (autonomous overnight). **Human:** Dhruv, back in the morning.
This is the OPERATIONAL log of the run — what was done, why, and the numbers. Read top-to-bottom.

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
- **S1 DONE — golden behavior gate built + validated.** `scripts/sandbox/test_region_equiv.py` (model-free; captures reachable-objects, reachable-edges, `is_robot_goal_reachable`, C++ `get_region_snapshot` graph/labels, and qpos fingerprint per push). Harness soundness confirmed: `compare` on the UNCHANGED build = **20/20 discrete-identical, qpos max|diff|=0**. Full golden captured on frozen `df62137`: **29 scenes (easy/med/hard) × ~6 pushes = 180 states, 0 errors, 8.7 s wall**. REF at `/common/users/dm1487/scratch_namo/eval/region_equiv/region_equiv_ref.json` (regenerable from this commit; deterministic scene selection). Gate must stay 180/180 through the refactor.
