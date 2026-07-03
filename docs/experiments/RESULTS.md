---
status: hub
tags: [results]
updated: 2026-07-03
---
# Compiled Results — experiment ledger

> One row per **finished** experiment, newest first. Claude appends a row when an experiment reaches
> `status: done` (auto, from the run output). Active experiment notes live in [log/](log/); finished ones move
> to [archive/](archive/) but stay on the board. Model-level ckpt/number catalog:
> [horizon_q_model_registry.md](horizon_q_model_registry.md). Plain markdown — reads on GitHub + for Claude.

| Date | Experiment | Hypothesis (1-line) | Metric | Verdict | Source |
|------|------------|---------------------|--------|---------|--------|
| 2026-07-03 | Reactive: random floor vs **NoHz-v3** (car) | reactive-random *baseline* vs the main model (NoHz-v3), 1push+2push, by difficulty | **2push** all 4.7±0.6 → **42.1**±1.7 · **1push** all 37.0±1.1 → **82.3**±0.2 (open%, region; easy/med/hard in card) | ✅ NoHz-v3 ≫ random in every cell (2push +37.4, 1push +45.3) | [_reactive_search](../../_reactive_search.md) |
| 2026-06-29 | Render speedup | model-input render can be ~20× faster, bit-identical | 2019→101 ms · gate 158/158 diff=0 | ✅ accept (no retrain) | sage `c0a00f7` |
| 2026-06-27 | NoHorizon vs Horizon @2 | dropping the horizon input doesn't hurt reactive/search | reactive 40.7 / best-first 37.8; NoHz ≥ Hz | ~ tie (NoHz ≥ Hz) | [redesign_execution](horizon_q_redesign_execution.md) |
| 2026-06-15 | M2b (+ dead-ends), 1-push | dead-end negatives sharpen the 1-push ranker | hard@1 32.86 ±2.4 · 2-push e2e 61.9% | ✅ best 1-push model | [registry](horizon_q_model_registry.md) |

*Seeded from prior work for continuity; the loop appends new rows going forward.*
