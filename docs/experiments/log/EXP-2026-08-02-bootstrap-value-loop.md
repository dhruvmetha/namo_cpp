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

## Hypotheses (H→E→V; drafted by Claude 2026-08-02, user to veto/edit)

- **H-main:** the binding constraint is missing negative + depth supervision (not volume/arch); fact-capped model guesses, iterated, hold hmax=2 and close the hmax=3 gap to random with zero new exhaustive labeling.
- **H0 (precheck):** θ₀'s top5-mean over an unswept remainder separates remainder-has-winner from remainder-dead boards on the 41.5k near-exhaustive child boards (52/48 classes). Bar: AUC ≥0.75 proceed; ≈0.5 stop.
- **H1 (aquaman-0):** two-sided targets on ~200k colossus setup-root capped cells leave every testset slice within seed-noise of d20 (flat expected; regression = stop). Not the payoff round.
- **H2 (rounds 1–2):** buried-winner rate falls round-over-round; refreshed target histogram migrates down (ratchet). Rising rate = stop rule.
- **H3 (rounds 2–3, payoff):** at hmax=3 the model reaches ≥ random's solve@900 band on hard (today 85.0 vs 87.8±1.6) while keeping hard @5 ≥3× random.
- **H4 (economics, feeds batman):** search-as-collector ≥ sweep labeling in deploy-delta per sim (rows/sim + testset delta per Msim).

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
| scores | **RAW HL-Gauss E[bin] everywhere — never the post-sigmoid** [USER 2026-08-02] | targets need true [0,1] magnitudes; sigmoid squashes to [0.5,0.73] → min(0.81, 0.9·V̂) could never go low, downward gradient destroyed. Collector runs `--raw`; precheck/rebuild use HLGauss.value (already raw) |
| eval | canonical testset BOTH tiers × difficulty, + hmax=3 subset-180 vs random | the target wall |
| eval depth | **1-push tier at hmax=2 ALWAYS** [USER 2026-08-02]; 2-push at hmax=2; depth arm hmax=3 | deploy search always has depth-2 freedom; hard-1p@h1 is partly a depth artifact. Registry 1p rows are h1 → USER is adding h2 baseline rows to the registry directly [2026-08-02]; this card's sweep runs only the 6 aquaman ckpts |

## Meters (per round, pre-registered)

1. **No-regression:** canonical testset both tiers × difficulty vs d20 + random.
2. **The target:** hmax=3 subset-180 — stop losing to random (today hard 85.0 vs 87.8).
3. **Buried-winner rate** (random slice + audit slice): winners found that θ ranked below top-k. Falling = loop heals blind spots (headline evidence); rising = bootstrap buries truth → STOP.
4. **Guess-quality:** V̂ separation on audited boards (proven-dead vs buried-winner) + target-distribution histogram drift (should migrate down round-over-round — the γ-contraction ratchet).
5. **Score-drift panel (round 0+):** diff d20-vs-successor score distributions on (a) relabeled cells (should drop — intended), (b) base-block capped cells (spillover meter — should hold), (c) exact cells (should hold). Directly separates "new labels' bad generalization" from seed noise; motivated by the val_loss↮deploy disconnect (the capped region is UNSUPERVISED freedom — part of d20's deploy strength is emergent, never selected for) [USER question 2026-08-02].

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

## Round-0 results — COMPLETE 2026-08-02 evening

**Offline AUC panel (exhaustive-GT `twopush_gt_h5`, 1,152 eps; canonical variants; A/B = 3-seed mean [min,max]).** All 6 aquaman ckpts + θ₀ scored same-run (`$SCRATCH/aquaman/round0/auc/`).

| metric | d20+su (θ₀) | arm A | arm B |
|---|--:|--:|--:|
| V6 live-vs-dead board, all | 0.753 | 0.777 [.774,.779] | **0.785** [.763,.807] |
| V6, hard | 0.725 | 0.757 [.741,.779] | **0.766** [.734,.802] |
| hard setup hit@1 | 21.2 | 22.9 [21.2,24.6] | **24.6** [20.3,27.1] |
| hard finish hit@1 | 51.2 | 54.8 [53.3,55.8] | **56.3** [51.2,60.3] |
| all setup hit@1 | **57.3** | 53.2 [52.4,54.4] | 55.4 [55.0,56.1] |
| all V1 | 0.817 | 0.798 | 0.800 |
| all V5 | 0.557 | 0.543 | 0.533 |


**Deploy gate (hmax=2, discount off, canonical 1322/1012; θ₀ = `deploy-nodiscount-hmax2-v1` control; A/B 3-seed mean±sd).** Full table `$SCRATCH/aquaman/round0/gate.json`; curves `success_vs_sims.png`.

| slice | θ₀ | arm A | arm B | verdict |
|---|--:|--:|--:|---|
| hard-2p @2 / @5 / @30 | 9.5 / 22.6 / 50.4 | **13.6±1.5 / 27.7±1.2 / 54.5±0.3** | **13.1±1.6 / 28.7±3.7 / 54.3±1.9** | **WIN, both arms, beyond noise** |
| medium-2p @5 / @30 | 57.4 / 80.1 | 51.4±0.9 / 77.2±0.5 | 52.1±2.2 / 78.3±1.3 | the tax; recovers by tail |
| hard-1p @1 / @5 | 39.7 / 82.4 | 39.2±2.4 / 79.9±1.7 | 39.9±0.5 / 78.9±1.7 | held / slight @5 softness |
| ceilings @900 (all slices) | — | — | — | held within noise everywhere |

**Depth attribution (the honest correction):** aquaman@h3 hard@900 = 95.6–96.1 LOOKED like H3 confirmed — but θ₀ re-run with current search defaults (dedupe_noop + jam-prune, added post-July by round-4 work) scores **98.3**, and random-with-new-defaults 90.0. **The July depth wall was substantially search hygiene, not missing supervision.** H3's motivating symptom no longer exists under current defaults; aquaman adds nothing at depth (≈−2, near jitter). Recorded in the artifacts table (`aquaman0-depth-h3`).

**Hypothesis scoreboard:** H0 PASS (quiz AUC 0.853). H1 MIXED — not "every slice within noise": hard-2p improves beyond noise, medium-2p@5 and hard-1p@5 regress beyond noise, ceilings hold. No stop-rule condition fired. H3 RETIRED as motivation (symptom was artifact); surviving rationale = the negative-gradient/separation mechanism — measurably real (V6 dose-response) and deploy-visible (hard-2p). H2/H4 untested (need rounds 1+).

**Round-0 verdict [numbers]:** mechanism validated; net deploy effect = budget shift medium→hard at zero ceiling cost; recommend ROUND 1 with `medium-2p@5` promoted to a named per-round stop-metric. Round-1 go/no-go = user's call.


**Read:** the bootstrap's target quantity — board-level live/dead separation (V6, the metric EXP-2026-07-25 said might need a dedicated head) — improved with clean A→B dose-response, +3–4 pts. Hard-tier within-board ranking up. Watch-item: pooled setup hit@1 dips 2–4 pts (easy/med only; hard improves) — deploy sweep arbitrates whether it costs sims. No disqualifying regression offline.

## Log

- **2026-08-02 [Claude]** Card created; worktrees `exp/bootstrap-value-loop` (namo @ fb02310, sage @ 6f90dc6). Design converged in chat (brainstorm with user): label rule, locked defaults, meters, round schedule. Next: round-0 step 1 precheck.
- **2026-08-02 [Claude] Census (30k sample, true loss mask `vm×rm`):** base-root 23.5% cells supervised / 37.3% ceiling; colossus-root 13.4% / 52.5% (exact all 0.9 setups, untried 22.2/row); child 23.9% / **94.6%** ceiling (exact all 1.0 winners, tried median 70, untried median 0). Reproduces EXP-2026-07-25's 38.5/94.7. Zero episode ambiguity: 0/215,856 multi-root `(xml,object)` groups. Linkage sparse: only 6.0% of roots have an in-file child; colossus block's 157,310 children live in the 200k source H5 → **aquaman-0 scope = colossus block (~26k setup roots, ~157k linkable capped cells)**.
- **2026-08-02 [Claude] Rebuild v1 (aquaman0_train.h5) — mechanically sound, dose too small.** `scripts/rl_loop/aquaman_rebuild.py`: pose-match linkage verified excellent (median err 1.4cm, p99 5.8cm, gate_fail 2,957); 67k "collisions" = jammed-push duplicate children (same resulting pose — dedupe correct); source multi-root groups = 0 (no cross-episode pooling). **But guessed cells = 15,869 of 559,322 capped (2.8%)**: the 200k source H5 is a curated selection — its 157,310 children are mostly winner boards (46,273 matched exact setup cells, skipped by design) + the ~33k negative dose; capped-cell children ≈ 16k. The full capped-children set never left Amarel. Guessed-target quartiles [0.34/0.54/0.69] — high vs precheck's base-children tails because colossus untried cells are the OLD ranker's ranks 21–70 (mid-pack), and θ₀ disagrees with d20's order there; capped at 0.81 regardless. **Decision pending: (A) accept 15.9k-cell dose (weak H1) vs (B) pull Amarel raw shards (full dose, exact linkage) — scout dispatched to verify raw schema.** Artifacts: `$SCRATCH/aquaman/round0/aquaman0_train.h5` + `.report.json` (v1, superseded if B).
- **2026-08-02 [Claude] Rebuild v2 DONE — full dose, two arms, exact linkage.** Amarel raw (`colossus0_200k_4322a98/h5_ccbb2d1_224996_32/`, 32 shards, 24GB, parent_edge/parent_depth columns) rsynced to CS (`$SCRATCH/aquaman/round0/raw/`); `aquaman_rebuild_v2.py` map×32 (5 GPUs, ~25min) + per-arm reduce. 687,616 children kept (141,581 capped-class + 546,035 censored-masked; 96k exhausted-skips). **Arm A (capped-only): 45,469 cells**, target quartiles [0.014/0.036/0.115], 72% <0.1. **Arm B (+censored): 591,504 cells**, quartiles [0.085/0.232/0.468], 28% <0.1, clip 0.4%. Distributions match precheck physics (A = audit dregs, B = ranks-21-70 remainders where θ₀ sees promise). Sanity: exact cells byte-identical 60/60 rows both arms; guessed cells valid; weight 0.5/1.0 flows through patched `q2_dataset` (guess_mask). Both-arms decision [USER]: run A and B in parallel (3 seeds each) — dose-response isolates cap-sharpening vs silence-filling. Files `aquaman0_train_{A,B}.h5`. Smoke train job ilab 198360 (arm B seed 1, 1 epoch) before the 6-job fleet. `scripts/rl_loop/aquaman_precheck.py` (worktree), θ₀ raw E[bin] via `eval_auc.score_h5`, 41.5k child boards, counterfactual sweep-stop quiz. **AUC(live vs dead remainder) = 0.853 / 0.853 / 0.855 at K=10/20/30** (bar ≥0.75), V̂ medians live 0.585/0.477/0.415 vs dead 0.147/0.085/0.053. True-untried remainders (19,323 boards): target `min(0.81,0.9·V̂)` quartiles **[0.011, 0.028, 0.115]**, clip@0.81 binds 0.2% — far LOWER than the predicted 0.3–0.6 hump (untried tails are the collection ranker's skip-list; θ₀ scores them near zero) → the downward gradient arrives at full strength. Honest overlap: AUC 0.85 ≠ 1.0, wrong-low tail exists as priced. Artifacts: `$SCRATCH/aquaman/round0/precheck.{json,log}`. Next: step 2 rebuild (colossus-block linkage via `action_motion` pose match + 200k-source child scoring), pending user go.

## Discussion

_(you ↔ Claude — ask here; answers inline, dated. Newest at the bottom.)_
