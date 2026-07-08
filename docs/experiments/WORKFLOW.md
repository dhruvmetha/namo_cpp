---
status: hub
tags: [workflow]
updated: 2026-07-03
---
# How we run experiments

> The operating loop. Claude reads this each session (pointed to from CLAUDE.md). Human-facing too.

## The loop
1. **You** create a stub note in [log/](log/) (new note → Insert Template → "experiment") and write the **Hypothesis**. Leave `status: idea`. Sync (pull-before-write).
2. **Claude** picks it up: flips `status: live`, fills a concrete **Plan**, and **git-commits before launching** (stamps `commit:`).
3. **Claude** runs it on SLURM/GPU (compute-resources skill; submit `gpu,gpu-redhat`, never wait >1h).
4. On finish, **Claude** auto-fills **Run + Result + Verdict** and sets `metric` + `status: done`. **Two reporting depths:**
   - **The card (`_*.md`)** holds the **DETAILED results + highly verbose analysis** — every table, the full difficulty×horizon breakdown, all plots, the a/b/c/d diagnostics, the caveats. This is the working record.
   - **[RESULTS.md](RESULTS.md)** reads like a **paper's Results section**: for each experiment, the **MAIN table(s) + MAIN figure + a tight key-finding paragraph** — curated, not exhaustive. Deep detail stays in the card; RESULTS.md is the polished, scannable compilation. Always split by difficulty × horizon. Then `git mv` the note to [archive/](archive/) and update the [model registry](horizon_q_model_registry.md) if a model trained.
5. **You** read the paper-style entry in RESULTS.md / the board and spin the next stub.

## Role separation (so two writers never collide)
- **You write:** idea-note Hypotheses, **Discussion** questions, + your own notes. **You read** everything else.
- **Claude writes:** the Plan/Run/Result of each note, RESULTS.md, the registry, the journals.
- Sync rule: **pull before you write.**

## Talking in a card (Discussion)
Ask a question in the card's **`## Discussion`** section: drop `**[you YYYY-MM-DD]** …` and I answer inline `**[Claude YYYY-MM-DD]** …`, newest at the bottom. It's our per-experiment channel, logged in git — the reasoning lives *with* the experiment, not in a lost chat.

## When a card grows into a folder (on demand)
A card stays a **single file** until it earns a folder — a plot, a long Q&A, or multiple runs. Then **Claude** converts `EXP-….md` → `EXP-…/index.md` + artifacts (`discussion.md`, `results/…`) beside it; the board still finds it (`index.md` keeps `type: experiment`). Don't pre-make folders.

## Status enum (never other spellings)
`idea` → `live` → `done`. (Non-experiment docs use `live` / `ref` / `hub` / `frozen` / `snapshot` / `archive`.)

## Verdict rule
Accept/reject **on numbers only** (Hypothesis → Evidence → Verdict). No vibes.

## Reporting conventions [USER]
- **Splits — ALWAYS:** every result split by **difficulty (easy/med/hard)** AND **horizon (1push/2push)**, never aggregate-only. If only one horizon ran, run/aggregate the other. Binning mechanics (tertiles, `pure2push_divisions.json`, `agg_react_search.py`) → [difficulty_stratification.md](../pipeline/difficulty_stratification.md); canonical table shape = `_reactive_search.md`.
- **Testset = BOTH tiers, explicitly [USER 2026-07-07]:** a model is not "testset-evaluated" until it has BOTH `namo_testset_v1` rows — `labels/onepush_episodes.json` (1323 eps) AND `labels/pure2push.json` (1018 eps). A same-family dev eval does NOT substitute for the 1push tier. (Lesson: the RL-loop gen evals ran 2push-only; the missing 1push row delayed the composition-vs-mechanical diagnosis by a day.)
- **Framing by regime:** REACTIVE = success only → **open-rate** (open@1 / open@2) by difficulty×horizon, no time/sim axis. SEARCH = **wall-time FIRST** (`avg t_wall`, `solve@1s/@5s/@30s`), THEN **sims** as the diagnostic (sims-to-solve, rank-of-winner).
- **Depth:** see step 4 above — card (`_*.md`) = full verbose detail; RESULTS.md = curated paper-style.

## Measuring time (wall-clock) — MUST be consistent
`t_wall` is hardware-dependent; **sims / episode counts are machine-independent** (compare those across any box freely). To compare TIME, measure on IDENTICAL hardware the same way: `time_bestfirst.py` **interleaved** (every method hits the same episode back-to-back on one node), `--exclusive`, CPU-microarch-**pinned** (`--constraint=emeraldrapids`/`icelake`) so times pool. Re-time a shared baseline (e.g. `random`) as an **anchor** to prove a new run pools with prior ones. **NEVER put wall-times from different boxes (arrakis vs Amarel vs westeros) on the same success-vs-time axis** — re-time on the baseline's exact setup instead.

## Must-do's [Claude]
1. **Commit before every run** — stamp the SHA in `commit:`.
2. **On finish** — append RESULTS.md + update the registry (if a model trained).

## Entrypoints (the real commands)
- **Train:** sage `scripts/train_h5_sampling.slurm` / config `train_scorer_edge`.
- **Eval:** `eval_scorer.py` on `namo_testset_v1` (canonical) / `scripts/amarel/eval_afterok.slurm`.
- **Collection:** `python/namo/data_collection/modular_parallel_collection.py`.
- **Baseline to beat:** NoHorizon-v3 reactive 40.7 / best-first 37.8 @2.
