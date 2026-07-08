---
type: experiment
status: idea
created: 2026-07-01
commit:
metric:
tags: [experiment]
thread: scorer-search
---
# Pure-V + recall-π vs combine=q (sims-to-solve)

> Full spec: [../policy_value_search_hypothesis.md](../policy_value_search_hypothesis.md). This is the atomic
> experiment card for that hypothesis. `status: idea` — not yet launched (qboot parked by USER 2026-07-02).

## Hypothesis
_(you)_ Splitting RO best-first search into a **policy π** (action proposal / branch ranking) + a **pure grounded value V** (frontier / state selection) — V trained on FIXED max-existence targets (NOT bootstrapped, NOT findability/density) — **beats the single `combine=q` head on sims-to-solve at a fixed solve-rate.** Honest prior [CLAUDE]: at H=2 the split may re-derive what `combine=q` already encodes; the one clean testable claim is **2 heads (calibrated V + recall π) vs `combine=q`** on sims-to-solve.

## Plan
_(fill on launch)_ Shared crop-encoder, 2 heads (π: 60×5 push logits; V: scalar). ExIt loop per the hypothesis doc (V = γ^sims-to-solve grounded target; π = masked ranking-CE over reachable). Eval: sims-to-solve at fixed solve-rate on `namo_testset_v1`, reactive@2 + best-first@2, vs `combine=q` (NoHorizon-v3 baseline: reactive 40.7 / best-first 37.8). Entrypoints: sage `train_scorer_edge`; eval `eval_scorer.py` / `eval_afterok.slurm`.

## Run
_(auto on launch)_

## Result + Verdict
_(auto)_ — accept iff the 2-head split beats `combine=q` on sims-to-solve outside seed noise.

## Next
_(tbd)_

## Discussion
_(you ↔ Claude — ask here; I answer inline, dated. Newest at the bottom.)_
