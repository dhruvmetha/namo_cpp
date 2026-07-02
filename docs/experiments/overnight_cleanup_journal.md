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
