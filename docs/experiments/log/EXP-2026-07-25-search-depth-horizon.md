---
type: experiment
status: done
created: 2026-07-25
thread: rl_loop
robot: car
parent: EXP-2026-07-24-failure-discount-search
commit: 84475ca
tags: [experiment, search, horizon, hmax, ceiling, supervision, ranker-limits]
---

# Search depth vs the label horizon — the ranker's value is horizon-local

**⛔ Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** The model is a frozen ranker; this experiment changes only how DEEP the search is allowed to go (`--hmax`), with no search adjustments at all (`--discount off`). Sibling of [EXP-2026-07-24](EXP-2026-07-24-failure-discount-search.md) (which changed the search's *trust*, not its *depth*); grandparent [EXP-2026-07-21](EXP-2026-07-21-colossus-data-scaleup.md).

## The one sentence

Letting the search go deeper than 2 pushes makes a RANDOM searcher substantially better and the trained ranker worse — at hmax≥3 the ranker falls BELOW random on solve@900 — because the training set contains no state deeper than one push, so the model emits confident scores on states it has never seen and the queue commits to them.

## Why this was run

Question from user: does capping the search at `hmax=2` cost us anything, and is `hmax=4` even viable?

Claude's first answer was WRONG and is recorded here as a retraction (see §Retractions). The argument was "every eval episode is 2-push solvable by construction (`pure2push` 1018/1018, `twopush` 2341/2341 `is_2push_solvable`), so deeper search can add zero solves." True but irrelevant — the objective is **sims-to-solve**, not solve count. Depth can pay by exposing a denser solution manifold even with the solve set fixed.

## Design

- **Population:** 180 episodes, 60 easy / 60 medium / 60 hard, seeded `random.Random(20260725)`, drawn only from single-episode XMLs so tier attribution is unambiguous (pool sizes easy 216 / medium 390 / hard 342). Manifest `round3/eval/hmax_depth/subset180.txt`, tiers `subset180_tiers.json`. Tier field is `division` in `pure2push_divisions.json`.
- **Arms:** {model, random} × {hmax 2, 3, 4}. Random gets 3 seeds (7000/8000/9000) because `--prior uniform` is stochastic while the model arm is deterministic.
- **Flags (all arms):** `--prior {model,uniform} --discount off --hmax H --sim-budget 900 --agg mean5 --combine q`, ckpt `d20_plus_setup_only_splitloss/epoch011-val_loss1.6952.ckpt`.
- **hmax=2 controls REUSED, not re-run** — model from `round3/eval/2push/setuponly_splitloss` and random from `round0/eval/2push/random`. Both verified to reproduce their parent-card rows exactly on the full 1018 (model 97.5 / 46.0; random 89.9 / 115.6).
- **Compute:** model arms ilab job `189302` (60 tasks, partition `unlimited`); random arms Amarel jobs `59137935`–`59137943` (9 arrays × 30 shards) reusing the committed template `scripts/amarel/bestfirst_eval.slurm`.
- **Cross-box confounds closed before launch:** zero C++ delta between Amarel's checkout (`243c6c7`) and HEAD (no rebuild needed), and `eval_bestfirst.py` md5-matched on both boxes (`f7d5f630e27f8779dbf238368ed50c00`) so model and random ran byte-identical search code. Physics is already verified bit-identical across the two boxes.

## Result — depth helps RANDOM, not the model

Hard tier, % of the 60 episodes solved within k sims (random = mean ± sd over 3 seeds):

| arm | @5 | @30 | @100 | solve@900 |
|---|--:|--:|--:|--:|
| model hmax=2 | 31.7 | 60.0 | 68.3 | **98.3** |
| model hmax=3 | 40.0 | 63.3 | 76.7 | 85.0 |
| model hmax=4 | 36.7 | 58.3 | 71.7 | 78.3 |
| random hmax=2 | 4.4 | 16.7 | 37.8 | 73.9 ±6.8 |
| random hmax=3 | 6.1 | 35.0 | 62.2 | **87.8** ±1.6 |
| random hmax=4 | 10.0 | 48.9 | 67.8 | 85.6 ±2.1 |

All tiers pooled (n=180), solve@900: model 98.3 / 93.3 / 87.2 vs random 89.3 ±2.5 / 94.8 ±0.9 / 93.9 ±0.9 for hmax 2/3/4.

- **Random climbs monotonically with depth** (hard @30: 16.7 → 35.0 → 48.9, ~3×; solve@900 73.9 → 87.8). **The model does not** (@30: 60.0 → 63.3 → 58.3; solve@900 98.3 → 85.0 → 78.3).
- **At hmax≥3 the ranker is WORSE than random on solve@900** — hard 85.0 vs 87.8 (h3) and 78.3 vs 85.6 (h4); pooled 93.3 vs 94.8 and 87.2 vs 93.9. The ranker's @30 margin over random collapses **3.6× (h2) → 1.8× (h3) → 1.19× (h4)**.
- **The model still dominates tight budgets at every depth** — hard @5 36.7 vs 10.0 even at hmax=4. Its first few picks stay excellent; the long-run budget allocation is what fails.
- **Easy tier: depth is strictly worse for the model at every budget** (s2s on the common solved set 12.0 → 20.8 → 45.3). Easy episodes are found immediately at depth 1; extra levels are pure dilution.
- **Nothing became unsolvable.** Every unsolved episode at every depth hit the 900-sim budget; **zero** exhausted their queue (model h2 3/3, h3 12/12, h4 23/23). A correctly-ordered queue would have found them — so the degradation is a ranking failure, not a combinatorial wall.
- **The deeper manifold is real:** ~40% of hard model-h3 solves return a 3-push plan (19 of 46 on the common solved set) despite a 2-push route provably existing. Longer plans are cheaper to *find*.

**Verdict.** The depth benefit is a structural property of the problem — more push sequences reach the goal, so a blind searcher gains a lot. The ranker extracts less of it than chance does. Its advantage is **horizon-local**: worth roughly +24 points of solve@900 on hard at its training horizon, and −7 points two levels past it.

## Why — the supervision runs out at one push

Verified on the deploy training set `round3/h5/d20_plus_setup_only.h5` (sampled 32,177 of 257,409 rows; training mask is `loss_mask = value_mask * r_mask`, per `q2_dataset.py:72`):

| | |
|---|--:|
| cells supervised at all | 22.4% |
| **ceiling (one-sided cap) share of supervised cells** | **48.1%** |
| exact cells == 1.0 | 37.9% |
| exact cells == 0.9 | 62.1% |
| **exact cells == 0** | **0.0%** |
| ceiling share, ROOT rows | 38.5% |
| ceiling share, CHILD rows | 94.7% |

- **The ceiling is a cap, not a target.** `censored_loss` = `-log P(V ≤ c)` (`hl_gauss_censored.py:38-53`) — purely one-sided. On a proven-dead cell labeled 0.81, predicting 0.0 costs nothing and predicting 0.80 costs nothing. Half the supervision expresses no preference.
- **There is no downward gradient anywhere on the ranked action space.** Zero exact zeros. The only zeros in the loss come from `NAMO_UNREACH_WEIGHT=1.0` on the *unreachable* band, which is `r_mask=0` — disjoint from `loss_mask` and from the candidate pool the search proposes.
- **No supervision past one push.** `build_beast2_exh_ceil.py` filters `node_kind ∈ {root, depth2}`, and the H5 schema cannot represent anything deeper (the only depth field is a binary `is_root`). At hmax=3 the search scores boards two pushes deep; at hmax=4, three deep. Both are fully out of distribution.
- **The negatives that do exist are softer than the grammar claims.** In `round2_raw.h5`, dead `depth2` boards are only **46.1%** full sweeps (median 98% of reachable candidates tried, p10 as low as 25 tried against ~70 reachable). The builder docstring says a full sweep "proves failure within the 2-push horizon"; that holds for under half the boards the 0.81 ceiling is stamped on. This is a **breadth** gap at depth 2, distinct from the **depth** gap the γ² framing describes.

Consistent prior evidence (parent card): within-board ranking is strong (hard setup hit@1 ~47–50%, hard finish hit@1 ~63–66%, both 3–5× random) while cross-board comparability is weak (root-setup-vs-dead AUC 0.583).

> **⚠ CORRECTION 2026-07-26 — the `0.583` anchor is retired; see [`auc_metrics_reconciliation.md`](../auc_metrics_reconciliation.md).** It does not reproduce (the same setup-cell-vs-moved-dead-board-max construction gives hard **0.515**; unrestricted it is the 07-24 REPORT's 0.469). The *conclusion* survives but the *mechanism* is narrower than "the model can't compare boards": setup cell vs a random dead post-push **cell** is **0.892 all / 0.864 hard** — per cell the model is fine. It only loses to the dead board's **max**, and a dead board offers ~70 draws. So the weakness is a right-tail / order-statistic problem, and the cross-board claim must be quoted as such. Also corrected: root setup separation is **tier-flat** (V1 0.809/0.825/0.814), not hard-degraded — the old tier slope was a `valid_first_push` label artifact.

## The design tension this exposes [USER framing, 2026-07-25]

The γ-ladder (1.0 opener / 0.9 setup / 0.9 dead-finish / 0.81 dead-root) is the semantics of an **unbounded-horizon** value: 0.81 = γ² means "provably not 1 or 2, could still open at 3+." The deployed search is **fixed-horizon**. Under `hmax=2` that reserved value can never be cashed, and the model is never taught to evaluate the future it is reserving it for.

The deeper constraint is that **the only truth we can cheaply obtain is horizon-bounded** (exhaustive GT at scale is ruled out by construction). Verified positives are facts (1.0 = watched it open; 0.9 = found a finish). Negatives are only ever "swept to depth d and found nothing." So there is no configuration with complete unbounded supervision on the negative class. The real choice:

- **Commit to a horizon** — negatives become complete within it and labels sharpen, at the cost of a ranker that is horizon-specific and must be retrained if scope widens.
- **Stay horizon-agnostic** — negatives stay permanently one-sided and the model never receives a "this is bad" signal on anything it ranks.

We have been paying the second option's cost while running a fixed-horizon search. The failure-discount `w` of the sibling card works precisely because it manufactures the missing negative **at runtime from the one exact negative available for free** — a verified failed push — and uses it only as evidence to demote, never as a claim the board is dead (the floor ε means a benched board is never buried).

## Retractions [CLAUDE, same session]

Recorded so they are not carried forward:

1. **"hmax=4 is not viable / has zero upside."** Wrong metric — measured upside as solves (correctly zero, since all eval episodes are 2-push solvable) when the objective is sims. Depth can pay by exposing a denser manifold, which it demonstrably does for random.
2. **"Depth cuts hard sims 3× (47.3 → 15.5)."** Real for the restricted "solved by all three arms" set, but that set drops the 10 episodes h3 lost, which were expensive for h2 (median 358 sims). Excluding them pulls h2's mean down 104.4 → 47.3 while barely moving h3's. The anytime curves are the unbiased instrument and show a much more modest, budget-dependent effect.
3. **"In our deterministic setting MCTS-Solver backup is exact, so the tree derives deadness with no labeler."** False — proof-number/solver backup requires terminal states. Our problem has none: a state is never declared dead by the rules, you can always push again. "All children proven dead" decays to "proven dead to depth d," which is the same horizon bound, derived at exhaustive cost (~40–50 candidates/board → ~2.5k nodes at depth 2, ~100k at depth 3). What survives: **exactness is available at the immediate-effect level** ("this push does not open the goal" — one sim, free, exact) **and evaporates the moment the value spans future pushes.**
4. **"The pipeline collapses censored and proven-dead cells to the same 0.81."** Withdrawn earlier the same session after reading `build_beast2_exh_ceil.py` — root dead → 0.81, post-push dead → 0.9. The all-0.81 observation was an artifact of setup-only dropping finish boards.

## Artifacts

- Eval outputs: `round3/eval/hmax_depth/h3/`, `h4/` (model, ilab); `round3/eval/hmax_depth/random_amarel/random_h{2,3,4}_s{7000,8000,9000}/` (random, Amarel).
- Plots: `round3/eval/plots/hmax_depth_model_vs_random.png` (ranker vs random × 3 depths × 3 tiers, seed band), `hmax_depth_success_vs_sims.png` (model depths vs random h2 reference). Scripts alongside as `plot_hmax_depth.py`, `plot_hmax_model_vs_random.py`.
- Submission: `round3/eval/run_hmax_depth.sbatch` (ilab); `scripts/amarel/bestfirst_eval.slurm` driven by env vars (Amarel, reused unmodified).
- Timing anchor (single box, arrakis, 6 episodes, model arm): hmax=2 **74s**, hmax=3 **282s**, hmax=4 **402s**. Extra wall time is model forwards, not sims — at hmax=2 a depth-1 board is a leaf and never triggers `candidates()`; deeper, every depth-1 failure spawns a board and pays a scoring call. Not comparable to the ilab/Amarel run times (different hardware).

## Follow-ups this motivates (none run)

1. **Separate the two model failures.** Score the exhaustive `testset_gt.h5` depth-2 boards and compare live/dead discrimination against the depth-1 AUC 0.583 anchor. Collapse at depth 2 → the OOD-depth problem dominates; merely matching → the permissive ceiling dominates. The depth sweep is consistent with both.
2. **No-op boards at deploy.** `node_kind='depth2_noop'` rows (setup push did not move the object) are dropped from training as duplicates, but the search still expands them, and they score *higher* than real dead boards (top-score median 0.813 vs 0.676). A free pose check before expanding would refuse them. One sweep against the adopted `conf τ=0.15` config.
3. **Budget sensitivity of the depth trade.** All depth losses were budget starvation, so a larger `--sim-budget` should recover them; that would separate "depth is worse" from "depth needs more budget."
4. **Two-siding the dead cells is NOT a free fix.** It writes a number we have not verified (see §The design tension) and bakes hmax=2 into the label semantics. If attempted, it must be a stated design decision, and note it addresses the ceiling problem while leaving the past-one-push blindness untouched.

## Where the model can improve — the mechanical cause, and the two candidate directions (2026-07-25, post-result analysis)

**The ranking pressure has always been within-board.** `certain_order_rank_aux_losses` in [scripts/rl_loop/train_q2_rankaux.py](../../../scripts/rl_loop/train_q2_rankaux.py) aggregates per ROW (`row_sum.index_add(0, valid_idx, ce)`), and one row IS one board's 60×5 candidate grid. So every ranking term the model has ever received compares cells *inside a single board*. Cross-board ordering — the thing best-first actually needs, and the thing measured weak at AUC 0.583 — has never appeared in any loss.

That resolves what looked like two problems into one plus one:

- **"Dead post-push boards outrank true setups" and "make the correct setup rank higher" are the SAME axis** — score comparability across boards. Within its own board the model already places the true setup well (hard setup hit@1 ~47–50%, median rank 2; hard finish hit@1 ~63–66%, both 3–5× random). The setup only ranks "too low" relative to cells on *other* boards.
- **The genuinely separate second problem is depth** (this card's result): no supervision past one push, so the ranker is below random at hmax≥3. Fixing cross-board calibration at depth 1 does nothing for it.

**Candidate A — cross-board pairs in the rank-aux** (small change). Within a batch, form (cell on a live board, cell on a dead board) pairs and enforce the ordering, reusing the existing `RANK_LAMBDA`/`RANK_TEMP` plumbing. Directly optimizes the measured metric.

**Candidate B — board-level live/dead head** (cleaner supervision). Predict "does this board contain an opener" as its own scalar. That IS the quantity best-first needs for cross-board comparison, rather than hoping per-cell values become globally comparable as a side effect. Label = `n_win > 0`.

**Constraint on either:** restrict the DEAD side to the 46.1% of boards that were full sweeps. The rest are "tried ~98% and failed"; training a hard negative on an unproven one is exactly the error the ceiling exists to avoid.

## Bootstrapping a value function — what it would concretely do here [USER question, 2026-07-25]

**Mechanically it changes one thing: the ceiling stops being a cap and becomes a number.**

We already do half of it. `region_label_topk` (commit `243c6c7`) uses the model to CHOOSE which finishes to simulate, capping the sweep at k failures — pilot-validated at k=15 retaining **97.7% of the successes an exhaustive sweep finds at ~30% of the sim cost**. What we do NOT do is use the model to VALUE the candidates we skipped: when the top-k all fail the board is stamped censored, and the untried remainder contributes no gradient.

| | today | bootstrapped |
|---|---|---|
| top-k finishes all fail | ceiling 0.81, one-sided, no gradient | target = γ · (model estimate over the untried remainder) — a definite low value |
| sim cost per dead board | median **61 of ~70** candidates tried | k ≈ 15, then a forward pass |
| depth reachable | 1 push (the labeler stops there) | any — the recursion is depth-agnostic |

So the concrete payoff: **it dissolves the "only 46% of dead boards are real sweeps" problem**, because producing a negative no longer requires a sweep. And it is the only route to supervision past one push that does not need exhaustive GT — which the project rules out by construction.

**Why it is not circular**, despite backing up through a model measured at AUC 0.583: the base case is VERIFIED, not estimated. At the finish level "does this push open the goal" is decided exactly by one simulator call, so the backup chain terminates in truth one level down. That is what makes bootstrapping viable here and not in a domain without a cheap perfect verifier. Grounding is further helped by plentiful exact anchors (verified 1.0 openers).

**Why policy-dependence is acceptable here specifically:** bootstrapped targets measure "expensive for MY searcher at budget B", not true value. For a calibrated probability that would be a defect; for a **search heuristic** whose objective is minimum sims (see [problem_and_approach.md](../../problem_and_approach.md)) it is arguably more aligned than the true γ^k value.

**Limits, stated plainly:** (1) it cannot create information the search never found — systematic blind spots get frozen into the next round's labels, so it only pays if you ITERATE; (2) targets are relative to budget B, which becomes a knob trading bound-tightness against compute; (3) the further you back up, the more of the chain rests on model opinion rather than sims — a finish board is well-grounded, a depth-3 board leans on two layers of estimate. Trustworthy depth therefore grows with iterations, NOT immediately, which is the opposite of the "just raise hmax" intuition this card started from.

**Relevant history:** this is the ExIt/search-and-learn loop that was parked. Per memory `project_levints_abandoned`, the LevinTS overnight run was a STATIC-DATA shortcut and never the real search+learn loop — so this direction is untested, not falsified.

**Recommended staging [CLAUDE]:** run Candidate B as a cheap one-round probe first (labels already on disk) to confirm cross-board calibration is the live bottleneck, and design the bootstrap loop in parallel. If B moves deploy numbers, the bootstrap loop is well-motivated and the signal is identified; if it does not, the bottleneck is depth coverage — which points at bootstrapping too, for the other reason.
