# Docs Index — NAMO

The human front door to the docs. `CLAUDE.md` (repo root) is the *agent's* router; this is *your* map. Every markdown doc in the repo is listed below with a one-line purpose and a status tag. Keep it current with `python scripts/docs_lint.py`.

**Tags:** `[ROUTER]` schema/entry · `[HUB]` links many others · `[LIVE]` actively updated · `[REF]` stable reference · `[SNAPSHOT]` dated results, not maintained · `[FROZEN]` closed/superseded · `[ARCHIVE]` superseded, do not cite.

> **⛔ Current framing (2026-07-06): horizon/budget-conditioning is DROPPED.** The live model is a **single value (NoHz)** whose job is **first-push (setup) ranking**; there is no budget input, no `Q(s,a,H)`, no per-horizon heads. The `horizon_q_*` docs are **historical** — read [`docs/problem_and_approach.md`](problem_and_approach.md) for the current framing.

---

## ▶ If you read nothing else (written for a human)

Most docs below are written for the *agent* to reload context after compaction — that's why they feel like logs. These are written for *you* to actually read:

1. [`docs/problem_and_approach.md`](problem_and_approach.md) — **the canonical plain-English north-star: the problem, and the shape of the approach (search is expensive → learn a ranker to make it cheap → beat the random ranker).** Start here. `[HUB]` `[LIVE]`
2. [`docs/research/research_compass.md`](research/research_compass.md) — why the research matters; come back when the work feels uncertain. `[HUB]`
3. **The experiment loop** — [`WORKFLOW.md`](experiments/WORKFLOW.md) (how it works) + the [board](experiments/DASHBOARD.md) & [`RESULTS.md`](experiments/RESULTS.md) (where we are / what's next).
4. `docs/personal/researcher_mentality.md` — how you operate; includes a note to AI assistants. *(personal, gitignored — present only in your main checkout, not in fresh clones/worktrees.)*
5. [`docs/horizon_q_overview.md`](horizon_q_overview.md) — deeper map + the **historical** Horizon-Q record (framing superseded by #1; kept for detail/history).

The active work now lives in experiment **cards** ([`experiments/log/`](experiments/log/)); the journals below are reference/history.

> **🌙 Overnight cleanup+speed run (2026-07-01):** see [`archive/overnight_cleanup_journal.md`](archive/overnight_cleanup_journal.md) for what changed tonight (wavefront speedups, the behavior gate, doc cleanup, and the deferred recommendations).

---

## Entry points & setup (repo root)

- [`CLAUDE.md`](../CLAUDE.md) — the agent router/schema: invariants, read-first pointers, architecture cheatsheet. `[ROUTER]`
- [`README.md`](../README.md) — project README: eval scripts + C++/Python architecture, build, API. `[REF]`
- [`DATA_COLLECTION_GUIDE.md`](../DATA_COLLECTION_GUIDE.md) — end-to-end region-opening data collection + mask generation. `[REF]`
- [`CLAUDE.amarel.md`](../CLAUDE.amarel.md) / [`CLAUDE.ilab.md`](../CLAUDE.ilab.md) — machine cards (paths/env/SLURM per box). `[REF]`
- [`docs/PORTABILITY.md`](PORTABILITY.md) — stand the pipeline up on any machine. `[REF]`

## docs/ (loose)

- [`problem_and_approach.md`](problem_and_approach.md) — **canonical plain-English north-star** (search is expensive → learn a ranker to speed it up → beat the random ranker). Start here. `[HUB]` `[LIVE]`
- [`horizon_q_overview.md`](horizon_q_overview.md) — deeper map + **historical** Horizon-Q record (budget-conditioning framing superseded by `problem_and_approach.md`). `[HUB]`
- [`cluster_resources.md`](cluster_resources.md) — Amarel SLURM partition/GPU guidance (append-only dated log). `[LIVE]`
- [`planner_contract_drifts.md`](planner_contract_drifts.md) — confirmed config-key drift in the legacy collection path. `[REF]`

## docs/experiments/ — the active research ledger

**⚙️ Experiment loop (the system — how we propose/run/record experiments):**
- [`WORKFLOW.md`](experiments/WORKFLOW.md) — the operating loop, roles, status enum, and reporting + timing conventions. `[HUB]`
- [`ORCHESTRATION.md`](experiments/ORCHESTRATION.md) — running the parallel-experiment fleet: forking, worktrees, file-partition, merge-back, tiering. `[HUB]`
- [`RESULTS.md`](experiments/RESULTS.md) — compiled results sheet (one row per finished experiment). `[HUB]`
- [`DASHBOARD.md`](experiments/DASHBOARD.md) — Obsidian board: Bases (`experiments.base`) + Dataview fallback. `[REF]`
- [`log/`](experiments/log/) — active experiment cards (`idea`/`live`), one note each. `[LIVE]`
- [`archive/`](experiments/archive/README.md) — finished experiments + closed snapshots. `[ARCHIVE]`
- stub template: [`_templates/experiment.md`](_templates/experiment.md).

**Horizon-Q arc** (⚠ **HISTORICAL** — budget/horizon-conditioning was dropped, see [`problem_and_approach.md`](problem_and_approach.md); these are the frozen record of why; live work → [`log/`](experiments/log/) cards):
- [`ILAB_RESUME.md`](experiments/ILAB_RESUME.md) — old "pick up here" note (targeted the now-parked qboot); superseded by the board. `[FROZEN]`
- [`horizon_q_redesign_execution.md`](experiments/horizon_q_redesign_execution.md) — staged execution journal (Stage 0–4), append-only log. `[FROZEN]`
- [`horizon_q_HANDOFF.md`](experiments/horizon_q_HANDOFF.md) — self-contained brief: problem + arch + algorithm + results. `[REF]`
- [`horizon_q_search_redesign_journal.md`](experiments/horizon_q_search_redesign_journal.md) — the pivot: model = sims-minimizing ranker; D1–D5 decision ledger. `[REF]`
- [`horizon_q_build_journal.md`](experiments/horizon_q_build_journal.md) — pre-redesign empirical record (v2/v3/v4, ExIt); §9 log. `[FROZEN]` — evidence archive; active log is `horizon_q_redesign_execution.md`.
- [`multipush_horizonQ_journal.md`](experiments/multipush_horizonQ_journal.md) — multi-push / horizon-Q design spec ledger (37 decisions); self-parked 2026-06-10. `[FROZEN]`
- [`horizon_q_model_registry.md`](experiments/horizon_q_model_registry.md) — authoritative model / ckpt / headline-number catalog. `[HUB]` — read for paths, never glob.
- [`horizon_q_related_work.md`](experiments/horizon_q_related_work.md) — related-work / novelty positioning. `[REF]`
- [`policy_value_search_hypothesis.md`](experiments/policy_value_search_hypothesis.md) — newest (2026-07-01) falsifiable π+V split hypothesis (NOT committed). `[REF]`

**1-push / scorer arc (earlier):**
- [`scorer_hacman_journal.md`](experiments/scorer_hacman_journal.md) — overnight HACMan-faithful 1-push scorer journal (E0–E9); top FINAL SYNTHESIS supersedes numbers below it. `[FROZEN]`
- [`hacman_comparison.md`](experiments/hacman_comparison.md) — slide-ready: our region-opening ↔ HACMan parallel. `[REF]`
- [`multipush_learning_primer.md`](experiments/multipush_learning_primer.md) — plain-language map of model families + training schemes + case studies. `[REF]`
- [`policy_framework_journal.md`](experiments/policy_framework_journal.md) — 1-push architecture hypotheses (H0a/H1/H2/H5). `[FROZEN]` — all closed.
- [`informative_1push.md`](experiments/informative_1push.md) — informative-hard 1-push experiment checklist (embedded result stale). `[FROZEN]`
- [`informative_1push_results.md`](experiments/informative_1push_results.md) — feasibility results, corrected (has ARCHIVED/CONTAMINATED note). `[SNAPSHOT]`
- [`results_design_report_2026-06-15.md`](experiments/results_design_report_2026-06-15.md) — dated results + design rationale (later overtaken). `[SNAPSHOT]`
- [`informed_2push_journal.md`](experiments/informed_2push_journal.md) — 2-push hypothesis journal (leaf-vs-search). `[FROZEN]`
- [`informed_2push_data_ledger.md`](experiments/informed_2push_data_ledger.md) — running ledger of files/manifests/ckpts/jobs for informed-2-push. `[FROZEN]`

**experiments/archive/ — closed, moved 2026-07-02 (`[ARCHIVE]`):**
- [`archive/model_comparison_report.md`](experiments/archive/model_comparison_report.md) — plain-English model comparison (CORRECTED v2).
- [`archive/informative_1push_training_study.md`](experiments/archive/informative_1push_training_study.md) — 500-epoch + annealing follow-up study.
- [`archive/scorer_1push_results.md`](experiments/archive/scorer_1push_results.md) — clean 1-push scorer results snapshot.

## docs/pipeline/ — data-pipeline reference

- [`multi_episode_rooms.md`](pipeline/multi_episode_rooms.md) — **THE per-episode invariant** (one xml = many episodes; never key on the room) + failure modes. `[HUB]` — critical gotcha.
- [`namo_pipeline.md`](pipeline/namo_pipeline.md) — paper-ready, filename-free full-pipeline method. `[REF]`
- [`canonical_testset.md`](pipeline/canonical_testset.md) — spec/build of the canonical car test set `namo_testset_v1`. `[LIVE]`
- [`horizon_q_datasets.md`](pipeline/horizon_q_datasets.md) — datasheet for the v4 push-value datasets (the data the NoHz scorer trains on; budget-conditioning dropped). `[LIVE]`
- [`data_collection_phases.md`](pipeline/data_collection_phases.md) — per-phase parameter cookbook for the region-opening cascade. `[REF]`
- [`difficulty_stratification.md`](pipeline/difficulty_stratification.md) — how `eval_2push` bins problems easy/med/hard. `[REF]`
- [`full_namo_collection.md`](pipeline/full_namo_collection.md) — human mirror of `full_namo_collection.yaml`. `[REF]`
- [`grounding_with_reachability.md`](pipeline/grounding_with_reachability.md) — grounding ML goals to reachable primitive slots + RA@K metric. `[REF]`
- [`scorer_dataset.md`](pipeline/scorer_dataset.md) — data lineage: F-scorer vs DiT datasets from the same corpora. `[REF]`

## docs/research/ — problem framing & literature

- [`research_compass.md`](research/research_compass.md) — orientation hub: framings, what's novel, the two-paper plan. `[HUB]`
- [`F_problem_formulation.md`](research/F_problem_formulation.md) — paper-style formal definitions (F = C ∩ R), baselines B0–B4, hypotheses. `[REF]`
- [`research_notes_F_characterization.md`](research/research_notes_F_characterization.md) — empirical results + hypotheses (the "notes" twin of the formulation). `[LIVE]`
- [`scene_conditioned_sampler_design.md`](research/scene_conditioned_sampler_design.md) — unified scene-conditioned sampler design (successor doc). `[LIVE]`
- [`reading_list.md`](research/reading_list.md) — broad ~61-paper annotated bibliography. `[REF]`
- [`reading_list_F_characterization.md`](research/reading_list_F_characterization.md) — focused F-characterization reading list. `[REF]` *(`research_prompt.md` moved to `archive/` 2026-07-01 — superseded `F = T ∩ A` framing.)*

## docs/personal/

- `researcher_mentality.md` — how you operate as a researcher; includes a note to AI assistants. `[LIVE]` *(personal, gitignored — local only, not in fresh checkouts.)*

## docs/algorithms/ — code-tethered reference

- [`ML_DRIVEN_ASYNC_ALGORITHM.md`](algorithms/ML_DRIVEN_ASYNC_ALGORITHM.md) — ML-driven async N-push search spec (GPU/CPU overlap). `[REF]`
- [`region_opening_primitive.md`](algorithms/region_opening_primitive.md) — region-opening + primitive-goal planner walkthrough. `[REF]`
- [`push_pruning_and_aborts.md`](algorithms/push_pruning_and_aborts.md) — the 3-layer push prune/abort pipeline. `[REF]`
- [`REGION_CONNECTIVITY_SNAPSHOT.md`](algorithms/REGION_CONNECTIVITY_SNAPSHOT.md) — C++ region/connectivity snapshot API. `[REF]`
- [`WAVEFRONT_CELL_SEMANTICS.md`](algorithms/WAVEFRONT_CELL_SEMANTICS.md) — canonical `-1/0/1` wavefront cell encoding contract. `[REF]`
- [`nav_issues.md`](algorithms/nav_issues.md) — diff-drive navigation outstanding-issues punch list. `[LIVE]`

## docs/evaluation/

- [`ML_vs_GT_F_evaluation.md`](evaluation/ML_vs_GT_F_evaluation.md) — plan: score diffusion goals vs ground-truth feasible set F (point robot). `[SNAPSHOT]`
- [`ML_vs_GT_F_results_round1.md`](evaluation/ML_vs_GT_F_results_round1.md) — round-1 results for that evaluation. `[SNAPSHOT]`

## docs/archive/ — superseded, do not cite `[ARCHIVE]`

- [`jan20_FULL_EVALUATION_REPORT.md`](archive/jan20_FULL_EVALUATION_REPORT.md) — umbrella Jan-2025 eval; kept for its paper-ready tables (§13).
- [`research_prompt.md`](archive/research_prompt.md) — old lit-search prompt (superseded `F = T ∩ A` framing; moved here 2026-07-01). `[FROZEN]`
- [`2026-05-19-uniform-rollout-sampler-design.md`](archive/2026-05-19-uniform-rollout-sampler-design.md) — UniformRolloutSampler design spec (closed May; moved here 2026-07-01). `[FROZEN]`
- [`2026-05-20-uniform-rollout-sampler-implementation.md`](archive/2026-05-20-uniform-rollout-sampler-implementation.md) — its TDD implementation plan (closed May; moved here). `[FROZEN]`

> **Deleted 2026-07-01** (strict subsets of `jan20_FULL_EVALUATION_REPORT`, recoverable via git):
> `jan20_1push_results.md`, `eval_2push_1push_test_results.md`, `eval_2push_1push_test_consistency.md`,
> `dec25_crossattn_results.md` (superseded by jan20), `MCTS_TRAINING_DATA_STRUCTURE.md` (dead data-format spec).

## Code-adjacent READMEs (not in docs/)

- [`python/README.md`](../python/README.md) — Python bindings package guide (RL API, state mgmt).
- [`python/namo/visualization/mask_generation/README.md`](../python/namo/visualization/mask_generation/README.md) — 9-mask NPZ generation module.
- [`scripts/amarel/README.md`](../scripts/amarel/README.md) — Amarel SLURM quick reference.
- [`scripts/sandbox/README.md`](../scripts/sandbox/README.md) — sandbox (gitignored one-offs) vs committed `scripts/` convention.
- [`test_xml/little-car-modeling-package/README.md`](../test_xml/little-car-modeling-package/README.md) — standalone little-car MuJoCo model package (Chinese).

---

## Maintenance

`scripts/docs_lint.py` checks this corpus for: broken doc→doc links, broken doc→code links (file existence only, not line numbers), and orphan docs (nothing links to them). It also lists **unwritten memory notes** — `[[<slug>]]` wikilinks the journals ask for but that don't exist yet in `~/.claude/.../memory/` (a "capture this fact" backlog).

```
python scripts/docs_lint.py            # link health (exits 1 if broken)
python scripts/docs_lint.py --orphans  # also list orphan docs
python scripts/docs_lint.py --json     # machine-readable
```

Link health (2026-07-03): **0 broken doc→doc, 0 broken doc→code** — the earlier 14 doc→code were relative-depth rot (`../` → `../../`) plus one refactored-away reference (`diff_drive_navigation.cpp` → now `push_path_follower.cpp`), all fixed 2026-07-03. Remaining: 9 unwritten memory notes (see `archive/overnight_cleanup_journal.md`).

Pending doc merges (deferred from the overnight run — modest value, research-content edits, do with review): fold `model_comparison_report.md` (prose) + `informative_1push_training_study.md` (epoch curves) into `informative_1push_results.md`; and the edit-in-place dedups (car-effect text `horizon_q_datasets` → `canonical_testset`; RA@K `namo_pipeline` → `grounding_with_reachability`; H1–H5 `research_notes` → `F_problem_formulation`). Full plan in the overnight journal.
