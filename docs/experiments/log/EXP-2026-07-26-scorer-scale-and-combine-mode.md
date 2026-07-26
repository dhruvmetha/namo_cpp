---
type: experiment
status: done
created: 2026-07-26
thread: rl_loop
robot: car
parent: EXP-2026-07-24-failure-discount-search
tags: [experiment, search, scorer, sigmoid, hl-gauss, combine-mode, deploy-scale]
---

# Deploy scoring scale: the inference sigmoid crushes the trained value, and `--combine q` beats `blend` on every tier

**⛔ Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** The model stays a frozen ranker throughout. Both parts below are about how the SEARCH reads and combines its scores at deploy time, not about retraining anything.

## The one sentence

The scorer trains an unbounded-in-[0,1] HL-Gauss value but the deploy path folds it through a sigmoid into [0.5, 0.731] — crushing the deliberate opener/setup label gap and making the adopted confidence discount nearly flat — and, separately, dropping the state-value term `V` and searching on raw `q` alone beats the default `blend` combine on every difficulty tier for both models.

## Part 1 — an inference-time sigmoid squashes the trained scale

### What's trained vs what's served

The scorer is trained with NO sigmoid. `scripts/rl_loop/train_q2_rankaux.py:158` sets `head_mode="hl_gauss", value_vmin=0.0, value_vmax=1.0`, and line 140 computes the ranking auxiliaries on `self._hl_gauss.value(logits)` — the differentiable E[bin] in [0,1]. Grep finds ZERO occurrences of `sigmoid` in that trainer or in `src/model/hl_gauss.py` (verified by grep, 2026-07-26).

At INFERENCE, `scripts/sandbox/live_scorer.py:184` applies a sigmoid on top of that expectation, folding [0,1] into [0.5, 0.7311] (verified: the code reads `return 1.0 / (1.0 + np.exp(-logits))` at that line, guarded by `if raw and is_hl: return logits` one line above at 182-183).

### Measured span

Over 361,755 real candidates on the ceiling model: sigmoid path spans **[0.5025, 0.7291]** (mean 0.5667, median 0.5380); the `--raw` path spans **[0.0098, 0.9899]** (mean 0.2751, median 0.1525). The hard model's raw path spans **[0.0098, 0.9900]** with median **0.0496** over 412,720 candidates — most candidates are confidently bad, which the sigmoid hides.

The trained label spacing is crushed by this: an opener target of 1.0 arrives at the search as 0.731 and a setup target of 0.9 arrives as 0.711 — the deliberate 0.10 opener-vs-setup gap becomes 0.02, and nothing ever reads below 0.5.

### Ordering is unaffected — verified, not assumed

A sigmoid is monotone, and this was checked rather than trusted on that fact alone: `sigmoid(raw_q)` reproduces the non-raw `q` to a maximum absolute error of **6.1e-08** on matched `(edge, depth)` cells, and all 25 paired episodes checked had identical pop sequences. The `--raw` flag itself was verified wired correctly through all six hops from CLI to scorer, with the caveat that it only bypasses the sigmoid when the head is HL-Gauss (`live_scorer.py:182`) and would be silently ignored on a non-HL-Gauss checkpoint.

### Consequence: every magnitude consumer is affected, ordering-only results are not

Every ordering-only result previously published (top-1, hit@k, rank-of-first-good-push) is immune to this. What IS affected is every deploy-search consumer of magnitude: the `blend` combine (Part 2), the `conf` failure discount, and `free_strike_q`.

- **`conf` discount quantified:** the adopted deploy rule demotes a board by `(1-q)^tau`, tau=0.15. Under the squashed scale that factor only ranges **0.821 to 0.901** across the entire observed q range — within 8% of a flat 0.87 multiplier. Under raw values it would range about **0.50 to 0.99**. So `--discount conf` is, in practice, barely confidence-weighted, and behaves close to a fixed-gamma discount.
- **⚠ OPEN QUESTION, not a conclusion:** the recorded `conf tau=0.15` win (sims-to-solve 46 → 27.8, card [EXP-2026-07-24](EXP-2026-07-24-failure-discount-search.md)) may be substantially a failure-COUNTING effect rather than a confidence effect. The discriminating experiment is a `--discount gamma --gamma 0.87` control arm, which has **NOT been run**.
- `--free-strike-q` defaults to 2.0 (verified: `eval_bestfirst.py` argparse default), and `q` under the sigmoid can never exceed 0.7311, so the free-strike allowance never fires at default settings.

### Confirmation on the 2-push test set

1018 episodes, `--combine q --discount off`, hmax 2, sim-budget 30. Raw and sigmoid arms are identical on every tier, as monotonicity requires:

| model | tier | solve@30 | avg sims | median sims-to-solve |
|---|---|--:|--:|--:|
| ceiling | easy | 84.9 | 9.82 | 3 |
| ceiling | medium | 80.9 | 10.06 | 3 |
| ceiling | hard | 59.6 | 16.01 | 3 |
| hard model | easy | 81.1 | 11.63 | 4 |
| hard model | medium | 80.2 | 11.25 | 4 |
| hard model | hard | 60.1 | 15.89 | 4 |

Identical cell-for-cell between the raw and sigmoid arms for both models on every tier.

## Part 2 — `--combine q` beats the default `--combine blend` on every tier

### Design

Same 1018-episode 2-push test set, hmax 2, sim-budget 30, both models (ceiling, hard), both discount settings (`conf tau=0.15`, `off`). `blend` (current default) = `0.5*q + 0.5*V`, where `V` is the board's mean top-5 `q`; `q` = the raw action score alone.

### Result

Solve% and avg sims, `blend → q`:

| arm | tier | solve% blend → q | avg sims blend → q |
| --- | --- | --- | --- |
| ceiling conf tau=0.15 | easy | 90.3 → 93.3 | 9.28 → 8.20 |
| ceiling conf tau=0.15 | medium | 83.9 → 87.5 | 10.12 → 9.19 |
| ceiling conf tau=0.15 | hard | 66.8 → 70.6 | 15.48 → 14.61 |
| ceiling off | easy | 80.3 → 84.9 | 11.10 → 9.82 |
| ceiling off | medium | 79.0 → 80.9 | 10.96 → 10.06 |
| ceiling off | hard | 55.8 → 59.6 | 16.80 → 16.01 |
| hard conf tau=0.15 | easy | 84.5 → 89.9 | 11.42 → 9.66 |
| hard conf tau=0.15 | medium | 79.0 → 84.1 | 11.49 → 10.58 |
| hard conf tau=0.15 | hard | 59.3 → 64.4 | 16.64 → 15.43 |
| hard off | easy | 73.5 → 81.1 | 13.40 → 11.63 |
| hard off | medium | 75.6 → 80.2 | 12.19 → 11.25 |
| hard off | hard | 55.8 → 60.1 | 17.07 → 15.89 |

**12 of 12 cells improve on both solve rate and sims** — both models, both discount settings, every tier.

### Reading (hypothesis, flagged as such — not a conclusion)

The state-value term `V` mixes two boards' score scales when comparing across boards, and dropping it removes that distortion. This is consistent with Part 1's point that the scorer's scale is exactly weakest at cross-board magnitude comparison (see also [EXP-2026-07-25](EXP-2026-07-25-search-depth-horizon.md)'s cross-board AUC discussion).

### Limitation

Single seed, one test set.

### Recommendation

Worth considering a change to the deploy default from `blend` to `q` — **NOT recorded as adopted here**, that is the user's call.

## Verdict

**Part 1 — accept as a measured artifact, not a bug that changes past ordering-only results; the discount-mechanism question stays open** pending the `gamma 0.87` control arm. **Part 2 — accept the 12/12 result at face value (single seed); recommend, do not adopt, switching the deploy combine default.**
