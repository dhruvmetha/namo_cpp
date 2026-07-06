---
type: experiment
status: live
created: 2026-07-06
commit:
metric:
tags: [experiment]
---
# Reactive MPC to depth 5 — how much of the test set falls to zero-search control?

## Hypothesis
_(you, via chat 2026-07-06)_ Why don't we do reactive up to 5 levels deep? How much of the test set can then be solved for both 1push and 2push — instead of search, we do MPC.

_(Claude framing, 2026-07-06)_ The reactive-vs-search gap on 2push is huge (reactive@2 42.1 vs best-first ~95 solve@900). This experiment asks **which part of that gap is the push budget vs the lookahead**: MPC can't undo a bad push, but it CAN keep pushing. If open@5 recovers most of the gap → greedy mistakes are self-correctable by acting more, and deploy barely needs search. If open@k plateaus by k=3 → greedy mistakes are largely *irreversible*, which is the cleanest evidence yet that search (simulate-and-undo) is load-bearing. Either outcome is signal; the anytime curve open@1..5 is the figure. Caveat stated up front: this is NOT compute-matched to search (search spends sims, MPC spends real pushes) — it answers the deploy question "how much falls to zero-simulation control," not "which is cheaper."

## Plan
_(Claude, 2026-07-06)_ **CAR only**, testset `namo_testset_v1` (2push key n≈1018 · onepush key n≈1323), region-open criterion, protocol = the `_reactive_search.md` forced-dive setup extended to a depth-k loop.

- **Script:** extend `scripts/sandbox/eval_reactive_argmax.py` with `--max-pushes` (default 2 → backward compatible). Loop: at state s_i, rank candidates (`restrict_obj` = labeled object, matching the existing protocol), take argmax (or random for the floor), `env.step`, check `goal_open_pts`, stop early on open or empty pool. Leaf record gains `opened_at` (push index 1..5, or 0 = never) so open@k for all k≤5 comes from ONE run.
- **Object restriction stays ON** (all ≤5 pushes on the labeled object) — keeps difficulty bins + episode semantics comparable to reactive@2 and the random floor. Free-object MPC is a follow-up variant, not this card.
- **Arms:** NoHz-v3 (3 seeds, registry ckpts — reuse, no retrain) and random floor (10 seeds, `--prior uniform`, model-free), both horizons.
- **Compute:** SLURM CPU per feedback_slurm_first — random floors anywhere; model arm on Amarel (iLab `sage_learning` had `fast_scorer` skew per `_reactive_search.md` Run notes; re-verify before assuming). Shard as in the prior run.
- **Aggregate:** reuse `agg_react_search.py` binning (2push = `pure2push_divisions.json` divisions; 1push = solve_rate tertiles). Tables = open@1..5 × difficulty × horizon, mean±std across seeds. Figure = anytime curve (open@k vs k) per difficulty, model vs random. Anchor check: open@1/@2 must reproduce the `_reactive_search.md` numbers (82.3/42.1 model, 37.0/4.7 random) within seed noise.

## Run
_(auto on launch)_

## Result + Verdict
_(auto)_ — headline: open@5 for 1push and 2push, by difficulty. Read against reactive@2 (42.1) and best-first (~95): does the extra push budget close the search gap or plateau?

## Next
_(tbd)_

## Discussion
_(you ↔ Claude — ask here; I answer inline, dated. Newest at the bottom.)_
