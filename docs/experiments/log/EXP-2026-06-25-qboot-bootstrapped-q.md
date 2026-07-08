---
type: experiment
status: idea
created: 2026-06-25
commit:
metric:
tags: [experiment, parked]
thread: scorer-search
---
# qboot — bootstrapped single-Q setup value (PARKED)

> **PARKED [USER 2026-07-02]** — not running until reactivated. Kept as an idea card so the context isn't
> lost (it was stripped from CLAUDE.md). Built but never launched. See [../ILAB_RESUME.md](../ILAB_RESUME.md).

## Hypothesis
_(you)_ Drop the Horizon input; train a single Q that labels each setup s0 with γ·V(s1) read from exhaustive finish labels (no re-sim). Two arms: `depth` (existence 1/0) vs `density` (finish findability). Prediction [CLAUDE]: matches-not-beats NoHorizon (finish is near-oracle ⇒ V(s1) ≈ status-quo setup label); depth ≥ density.

## Plan
Built: sage `feat/horizon-q` ilab launcher `train_bootstrap_ilab.slurm` (unpushed as of 2026-07-02). Gate: reactive@2 + best-first@2 (combine=q) vs NoHorizon-v3 40.7 / 37.8. Superseded-in-spirit by the pure-V hypothesis ([EXP pure-v](EXP-2026-07-01-pure-v-vs-combineq.md)), which argues the bootstrap is a moving target.

## Run
_(never launched)_

## Result + Verdict
_(none — parked)_

## Next
Reactivate only if the pure-V direction stalls and a simpler bootstrapped value is worth a gate.
