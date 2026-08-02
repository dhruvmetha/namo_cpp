---
type: experiment
status: live
created: 2026-08-02
thread: rl_loop
robot: car
parent: EXP-2026-07-25-search-depth-horizon
tags: [experiment, bootstrap, fitted-q, exit-loop, search-as-collector, claude-active]
---

# EXP-2026-08-02 — Aquaman: bootstrap value loop (fitted-Q / ExIt): guesses fill the gaps facts leave

**Lineage: DC (alphabetical, one letter per method revision) — this is `aquaman`.** Models: `aquaman-0` = round-0 zero-sim relabel, `aquaman-1/2/3` = crank rounds. Next revision (e.g. the from-scratch clean-room loop) = `batman`. The Marvel lineage (antman/beast/colossus) is the truth-only curriculum; DC is the bootstrap loop. The 2M-XML clean pool keeps its historical disk paths but the method is not "colossus" [USER 2026-08-02].

**⛔ Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** The model is a ranker; search + free perfect verifier; objective = fewer sims to solve than random, every tier. **This card is what Claude is actively working on.**

**Worktree:** branch `exp/bootstrap-value-loop`, checkouts `.claude/worktrees/bootstrap-value-loop` (namo) + `bootstrap-value-loop-sage` (sage). Card lives on the main branch; results land here after each step.

## The one sentence

Where a capped sweep left a mute one-sided ceiling, write a two-sided target `min(ceiling, γ·V̂(child))` with V̂ from the current model over the child board's untried cells, retrain, and iterate the loop ≥2 turns — testing whether model-opinion-capped-by-facts supplies the missing downward gradient and depth supervision that truth-only labels cannot.

## Why (evidence that forced this)

- [EXP-2026-07-25](EXP-2026-07-25-search-depth-horizon.md): ranker value is horizon-local — at hmax≥3 the model falls BELOW random (hard solve@900 85.0 vs 87.8 at h3; 78.3 vs 85.6 at h4); cause verified: no supervision past one push + **0.0% exact-zero targets** (48.1% of supervised cells are mute one-sided ceilings).
- The only cheap truth is horizon-bounded; exhaustive GT at scale is ruled out by construction — bootstrapped backup is the only route to supervision past one push.
- Prior ExIt attempt (EXP-2026-07-10) turned ONCE with dead=0 labels and direct setup regression — its failure indicts those deviations, not the loop.
- Standard grounding: fitted-Q iteration (Ernst '05) + Expert Iteration (Anthony '17)/AlphaZero, adapted to inverted costs (sim ≈ 1s ≫ net forward). Non-circular because every backup bottoms out in a sim-verified fact one level down; the cap = opinion never contradicts a proven sweep.

## The label rule (the entire experiment)

| cell class | today | this card |
|---|---|---|
| verified opener / setup | 1.0 / 0.9 exact | unchanged, weight 1.0, NEVER relabeled |
| simmed, failed, child state stored | ceiling (0.9 / 0.81), one-sided, mute | `min(cap, γ·V̂(child))`, two-sided, **weight 0.5** |
| simmed, failed, no stored child (old data finishes) | ceiling 0.9 | unchanged (nothing to look at) |
| reachable untried | masked | masked (θ grading itself = no info) |
| unreachable | outside loss | unchanged |

`V̂(s′) = top5-mean of θ over the child board's UNTRIED cells` (tried = verified failures, excluded). γ=0.9. Guesses are **never stored** — buffer holds facts only; guesses derived fresh at every H5 build with the current θ (refresh = FQI iteration; stale opinions cannot fossilize).

## The loop

```
θ ← d20_plus_setup_only_splitloss/epoch011      BUFFER ← d20_plus_setup_only.h5 (257k rows, facts)
ROUND 0 (zero sims): precheck → rebuild H5 (root capped cells only) → train aquaman-0 scratch → gate
ROUND 1..3: collect(θ, ~20k eps) → BUFFER += raw traces → rebuild (uniform rule) → train aquaman-r scratch → gate
VERDICT after round 3, pre-committed (one-turn results are not verdicts — the 07-10 lesson).
```

**From-scratch clean-room reference (= `batman`, only after aquaman proves the mechanism):** θ random → round 0 collects with RANDOM best-first, trains facts-only (guesses are noise at θ-random); rounds 1+ = the same loop, curriculum emerges from the sims-to-solve quota (no hand-built stages), linkage-complete data from the first sim. Buys purity + the strongest claim; costs re-spending the Marvel climb (~4–6 rounds × ~3M sims before the depth question is even testable) and cannot isolate the bootstrap question — hence seeded-first.

Collector (round ≥1) = the deploy best-first search itself, budget-capped; NO setup sweeps, NO top-20 rule, NO mass audits — zero sims spent proving deadness. Every simmed push's resulting state is persisted (makes the guess rule uniform).

## Locked defaults

| knob | value | rationale |
|---|---|---|
| budget B | 150 sims/episode | enough to cap boards meaningfully, ~2.5 min/ep |
| collection hmax | 2 (round 1) → 3 (rounds 2+) | θ must earn two-sided values before deep trust |
| exploration | 1-in-5 episodes random-ordered | polices wrong-LOW guesses; live random baseline; feeds buried-winner meter |
| easy quota | solved ≤5 sims → keep 1/10 | sims-to-solve under current θ IS the difficulty meter; frontier ratchets |
| audit slice | ~5% of unsolved eps: exhaust final board | fresh on-distribution answer key, measurement ONLY, never trained on |
| guess weight | 0.5 (facts 1.0) | pseudo-label down-weighting; facts win gradient wars |
| aggregator | top5-mean (not max) | softened backup, anti-overestimation |
| training | from scratch each round, same rankaux recipe + per-cell weight column | clean attribution |
| pool | colossus 2,031,481 clean geometry-disjoint XMLs | no new generation needed |
| eval | canonical testset BOTH tiers × difficulty, + hmax=3 subset-180 vs random | the target wall |

## Meters (per round, pre-registered)

1. **No-regression:** canonical testset both tiers × difficulty vs d20 + random.
2. **The target:** hmax=3 subset-180 — stop losing to random (today hard 85.0 vs 87.8).
3. **Buried-winner rate** (random slice + audit slice): winners found that θ ranked below top-k. Falling = loop heals blind spots (headline evidence); rising = bootstrap buries truth → STOP.
4. **Guess-quality:** V̂ separation on audited boards (proven-dead vs buried-winner) + target-distribution histogram drift (should migrate down round-over-round — the γ-contraction ratchet).

## Risks & rails

| risk | rail |
|---|---|
| wrong-HIGH guess | self-correcting: high score attracts next round's search → sims → facts overwrite |
| wrong-LOW guess | random slice + refresh + half weight; residual: systematic blind class (irreducible, metered) |
| deadly triad divergence | cap clamp + exacts never relabeled + half weight + top5-mean + scratch retrain |
| correlated skip-list bias (guess grades d20's own skips) | measured small: k=15 keeps 97.7% of winners; colossus audit 257/258 episodes had top-20 route; ~2% tail accepted, metered |
| one-turn misread | verdict only after round 3 |

## Round 0 plan (zero sims) — discuss results with user after EVERY step

1. **Precheck** — θ₀ forwards over child boards already in the H5: (a) V̂ separation audited-dead vs audited-buried-winner boards (no separation → STOP before training); (b) real target-distribution histogram (predicted: hump ~0.3–0.6, confident-dead tail near 0, clipped spike at 0.81).
2. **Rebuild** — root capped cells only (old data stores child states only for setups): `min(0.81, 0.9·V̂)`, weight 0.5; all else byte-identical.
3. **Train θ₁** from scratch, same recipe + weight column.
4. **Gate** — testset both tiers × difficulty vs d20 + hmax=3 subset. Pre-committed: clear regression → stop; flat → proceed (flat expected: no iteration, majority ceiling class untouched — round 0 is the safety check, not the experiment).

Paths (arrakis): H5 `$SCRATCH/curriculum2/beast/round3/h5/d20_plus_setup_only.h5` (995M); ckpt `$SCRATCH/curriculum2/beast/round3/models/d20_plus_setup_only_splitloss/checkpoints/epoch011-val_loss1.6952.ckpt`.

## Log

- **2026-08-02 [Claude]** Card created; worktrees `exp/bootstrap-value-loop` (namo @ fb02310, sage @ 6f90dc6). Design converged in chat (brainstorm with user): label rule, locked defaults, meters, round schedule. Next: round-0 step 1 precheck.

## Discussion

_(you ↔ Claude — ask here; answers inline, dated. Newest at the bottom.)_
