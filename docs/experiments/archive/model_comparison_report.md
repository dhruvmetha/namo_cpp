---
status: snapshot
tags: [experiment]
updated: 2026-06-06
---

# NAMO push-prediction — model comparison (plain English) — CORRECTED v2

> **v2 correction (this version):** the v1 numbers were scored on test files whose difficulty label
> was per-*room*, but a room has many episodes (different pushed object/goal) — so ~24% of the "hard"
> file was actually easier episodes that leaked in, *inflating* the model on hard (v1 said hard @1 =
> 11.3–17.2). Fixed: every sample is matched to its **own episode** (by `object_center`) and **re-binned
> by that episode's true solve_rate**, deduped across files. See
> [docs/pipeline/multi_episode_rooms.md](../../pipeline/multi_episode_rooms.md). Clean test set:
> **413 hard / 491 med / 752 easy**, `gt_in_valid = 1.0`, `bad_match = 0`.

## The task
Given a cluttered scene, predict *where to push the target object* to open a path. `success@k` = chance
a working push is in the top-k. Difficulty = fraction of *reachable* pushes that work: HARD ~3%, MED
~17%, EASY ~65% (now measured per episode).

## Headline — deployment-faithful (reachable-masked), @1 / @20 success (%)
| | HARD | MED | EASY |
|---|---|---|---|
| random floor (reachability-aware) | 2.7 / 41.3 | 16.8 / 92.7 | 65.4 / 100 |
| **informative ep500** (hard specialist) | **5.9 / 55.2** | **28.9 / 94.5** | 64.6 / 99.9 |
| annealing ep400 | 5.4 / 54.7 | 29.5 / 95.5 | 62.3 / 99.9 |
| general (older safe_fp32) | 3.1 / 39.7 | 19.9 / 91.6 | 63.4 / 99.7 |

(@20 = empirical any-of-20 = coverage; @1 = mean single-shot. Floor is analytic-exact 1−(1−sr)^k.
n = 413 / 491 / 752 hard/med/easy.)

## What we're seeing — is the model learning? Yes, and we can say *why*.
- **It beats a reachability-aware floor, so it's discriminating *which* reachable push works** (the floor
  already gets reachability for free). HARD: **5.9 vs 2.7 @1 (~2.2×)**, **55.2 vs 41.3 @20 (+14pp)**.
  MED: 28.9 vs 16.8 @1 (~1.7×). EASY: ~tie (random-reachable already wins; nothing to learn).
- **The *informative training* is the cause, not the architecture.** The **general model sits at the
  floor on hard** (3.1 / 39.7 ≈ 2.7 / 41.3 — even below at @20). Same architecture, random-difficulty
  training → no hard-scene skill. So hard-scene discrimination comes from *what it was trained on*.
- **Annealing ≈ no-anneal** (5.9/55.2 vs 5.4/54.7 on hard; trades tenths elsewhere). Caveat: ep499 vs ep399.
- The edge is biggest at **single-shot on hard/med** — exactly where understanding the scene matters.

## What limits the model
- **Coverage is the ceiling.** Informative cracks only **55% of hard** scenes within 20 samples (cov:
  hard 55%, med 95%, easy ~100%). **~45% of hard scenes are uncrackable** inside its distribution — the
  single biggest gap, and the target for a discriminative scorer / value model.
- **It's a real signal, not a strong model:** ~6% right on the first try on hard. Better than chance,
  far from solved.
- Reachability masking is still essential (deployment already does it); all-edges scoring under-credits
  every model by matching predictions to impossible edges.

## Training-data caveat (affects how hard the "hard specialist" really is)
"informative_le10" was meant to be solve_rate ≤10% but the per-pkl manifest leaked easier episodes in:
actual mix is **34% hard / 46% med / 19% easy** (only ~76% ≤10%). A de-leaked H5
(`v3_1push_le10_clean`, 21,316 samples, all ≤10%) is built; a retrain on it (with the room-grouped
val split) is the cleaner specialist. See [multi_episode_rooms.md](../../pipeline/multi_episode_rooms.md).

## Implications for next steps
- **"Diffusion + reachability masking" is the baseline to beat** — bar is now honest (hard 5.9/55.2,
  not the inflated 11.3/44.1).
- **A discriminative scorer over reachable primitives** (HACMan-critic-style, supervised on exhaustive
  labels) remains the path: reachability-safe by construction, directly trains edge selection, aimed at
  the coverage ceiling. It must beat *reachable-masked diffusion*, and on 1-push also clear the free
  **geometric+wavefront oracle** (render footprint → wavefront → reachable?) — measure that first.

## Why v1 was wrong (so it doesn't recur)
Difficulty was assigned per room, but NPZ-gen emits a sample per *episode*; multi-episode rooms put
easier episodes into the "hard" file (24%), and samples were scored against one stored episode's answer
key. Both fixed by per-episode matching + re-binning. The `namo-data-pipeline` skill now gates this.

_Source JSONs: `/scratch/dm1487/eval_grounding/{informative_ep500,annealing_ep400,older_safe_fp32}_rebin.json`.
Harness: `eval_grounding.py` (+`.slurm`); per-episode key: `build_episode_validsets.py`._
