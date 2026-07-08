---
type: experiment
status: idea
created: 2026-07-08
commit:
metric:
thread: rl-loop
tags: [experiment, rl, self-imitation, data-growth]
---
# RL growth arms: does the self-imitation loop climb when the pool grows — and on which diet?

> Successor to [[EXP-2026-07-06-rl-only-self-imitation]] (falsified at gen-1; walls re-attributed 2026-07-07: (1) flat per-generation slope, (2) task-COMPOSITION shift — the pool was 1push-dominated while pure2push demands setup→finish; room-family shift RETRACTED). This experiment removes the data excuse and disentangles the two walls with two arms. Calibrating precedent: ReST^EM reports "most of the gains come from the first iteration" on FIXED pools — our bet is that per-round data NOVELTY (which their setup lacked) restores the slope.

## Hypothesis
_(you, via chat 2026-07-07/08)_ **The gen-1 flatline was data starvation, not a method failure: with a pool that GROWS each generation, the same self-imitation loop climbs the canonical testset.** Arm-resolved form: (N) fresh episodes of the same 1push-dominated kind restore the slope via novelty alone, vs (C) genuine-2push episodes teach the missing setup→finish skill. Falsifiable: testset climbing per generation toward NoHz-v3 parity (2push 40.8/25.3; 1push 82.3) at data parity (~300k rows).

## Arms (same seed policy, same budget, same loop — ONLY the growth diet differs)
- **Arm N (novelty):** pool grows +~4k/gen with UNTAPPED existing-manifest episodes of the same composition as before (v4_hq_h1 deadends remainder + sibling h1 manifests). Tests: does any fresh data restart the climb?
- **Arm C (composition):** pool grows +~4k/gen with GENUINE-2push episodes (exhaustively-verified F=∅, from the v4_hq_h2 labels on Amarel — already labeled, no new sweeps). Tests: was the missing skill just missing practice?
- **Readout:** only C climbs 2push → composition was the wall. Both climb → novelty was the wall. Neither climbs across two growth gens → the ReST^EM first-iteration plateau holds despite novelty — RL-only frame goes back on trial with no data excuse (search-in-loop next).

## Design (all pieces validated in the predecessor)
- **Seed:** both arms warm-start from the predecessor's arm-A gen-1 π (pure-RL lineage; testset 14.3 all / 6.7 hard 2push, 57.9 all 1push) + its buffer. Same seed in both arms isolates the diet.
- **Loop per generation (per arm):** grow pool (disjointness gate vs testset per batch — geometry-hash machinery) → collect on Amarel main-redhat (R=16 rollouts/ep on NEW episodes + refresh on a sample of old, T=0.1, ε=0.10, forced sweeps AFTER difficulty stamping) → harvest with --expected-shards → train π on CS GPU (first-free: arrakis / iLab A4500 / rlab A100; all four infra fixes carried; V DISABLED — hl_gauss hang open bug) → eval → report.
- **Difficulty stamping:** arm C episodes carry labels; any unlabeled admission uses the validated two-axis probe (first-push rate → legacy bins, 87-88% agreement; full-rollout rate → chain difficulty). Probe rollouts double as collection data; sweeps enable only post-stamp.
- **Per-generation dashboard [BOTH testset tiers — WORKFLOW rule 2026-07-07]:** testset 1push open@1 AND pure2push open@2, by canonical tiers; setup-hit@1/@8 + median rank; dev on held-out new-batch rooms; buffer stats (unique solves per tier, hard coverage); cumulative-rows counter vs the ~300k baseline.
- **Ops discipline [from predecessor lessons]:** agents use BLOCKING poll loops, never stop-and-wait waiters; orchestrator watchdogs armed whenever anything is in flight; announce-before-destructive; every status claim verified against sacct/filesystem before relay.

## Pre-registered gates
1. **Per-arm kill:** BOTH testset tiers flat (within noise) across two consecutive growth generations → that arm's mechanism is dead; stop that arm.
2. **Success bar:** monotone 2push-testset climb; parity with NoHz-v3 (40.8/25.3) at data parity confirms the hypothesis.
3. **Expectation calibration (registered):** given ReST^EM precedent + the predecessor's ~25pt quality gap on 1push, a realistic good outcome for ~4 growth generations is 2push-all in the mid-20s to low-30s with hard clearly off the 7-8 floor; hitting 40+ would be upside.
4. Coverage signals (kill-1/kill-3 analogues) inherited from the predecessor per generation.

## Plan
_(fill on launch)_

## Run
_(auto on launch)_

## Result + Verdict
_(auto)_ — H→E→V, verdict on numbers only; every table by difficulty × BOTH horizons.

## Next
_(tbd)_

## Discussion
_(you ↔ Claude — ask here; I answer inline, dated. Newest at the bottom.)_
