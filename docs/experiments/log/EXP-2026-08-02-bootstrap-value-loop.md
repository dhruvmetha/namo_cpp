---
type: experiment
status: done
created: 2026-08-02
updated: 2026-08-12
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
| round-1 extra arm | **depth-token arch × 3 seeds on the same round-1 H5** [USER 2026-08-02]; NEVER used as collector (B_s1 collects) | re-tests the rejected architecture on on-policy depth-linked data; clean arch-only comparison |
| collection task shape | **rounds 2+: `--cpus-per-task=12` + in-process worker pool** [USER 2026-08-02] | Amarel submit cap = 500 jobs; fat tasks reach ~6,000 CPUs (colossus shape). Round-1 collector was 2-cpu single-process |
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

**⛔ CORRECTION 2026-08-07 [Claude] — arms A and B were the SAME experiment; read every A-vs-B contrast below as seed noise.** `aquaman_rebuild_v2.py` `run_reduce` wrote `value_target`, `ceiling_mask`, and `guess_mask` but never `value_mask`. The dataset builds `loss_mask = value_mask * r_mask` (`q2_dataset.py:73`) and the rank-aux reuses that same mask (`weighted_module.py:54`), so arm B's 546,035 class-1 (masked-but-simmed) cells — which have `value_mask=0` by definition, that is what made them "masked" — never entered the regression loss or the ranking competition. Verified directly on the two files: across 6,129 sampled rows every in-loss cell is byte-identical between `aquaman0_train_A.h5` and `aquaman0_train_B.h5` (413,869 cells compared, max delta 0.000000), and guess cells inside the loss are A=845 / B=845.

**Consequences of that correction:** (a) round 0 is ONE condition with SIX seeds, not two arms at different doses — pool the A and B columns into a single band; (b) "V6 dose-response A→B" (0.777→0.785) and "13× the cells, same deploy result ⇒ saturated at 45k" are both artifacts of there being no dose difference at all; (c) the hard-2p gain of +4.8/+5.6/+5.1 over θ₀ came entirely from arm A's 45,469 capped cells and is therefore 6-seed robust — a stronger result than originally claimed, from 0.5% of the labels; (d) arm B's actual hypothesis — that recovering the simmed-but-unrecorded cells adds value — remains **UNTESTED**. Refired as arm `Bfix` (reduce now also sets `value_mask=1` on every relabeled cell).

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

**Hypothesis scoreboard:** H0 PASS (quiz AUC 0.853). H1 MIXED — not "every slice within noise": hard-2p improves beyond noise, medium-2p@5 and hard-1p@5 regress beyond noise, ceilings hold. No stop-rule condition fired. H3 RETIRED as motivation (symptom was artifact); surviving rationale = the negative-gradient/separation mechanism — measurably real (V6 +3–4 pts over θ₀ across all six seeds; the claimed A→B dose response was an artifact, see the 2026-08-07 correction) and deploy-visible (hard-2p). H2/H4 untested (need rounds 1+).

**Round-0 verdict [numbers]:** mechanism validated; net deploy effect = budget shift medium→hard at zero ceiling cost; recommend ROUND 1 with `medium-2p@5` promoted to a named per-round stop-metric. Round-1 go/no-go = user's call.


**Read:** the bootstrap's target quantity — board-level live/dead separation (V6, the metric EXP-2026-07-25 said might need a dedicated head) — improved +3–4 pts over θ₀ (0.753 → 0.777/0.785; the A-vs-B spread is seed noise, see the 2026-08-07 correction above, NOT a dose response). Hard-tier within-board ranking up. Watch-item: pooled setup hit@1 dips 2–4 pts (easy/med only; hard improves) — deploy sweep arbitrates whether it costs sims. No disqualifying regression offline.


## Round-0 REDONE + loss anatomy — 2026-08-07/08

Arm B was never trained (see the correction at the top of Round-0 results). This is the refire plus everything it turned up. All arms: 3 seeds, canonical gate (1322 1push@hmax2 + 1012 2push, budget 900, `combine=q`, **discount off**, dedupe+jam on), zero unmatched episodes. Mean [min,max].

| arm | what it is | 2p h@2 | 2p h@5 | 2p h@30 | 2p h@900 | 2p med@5 | 1p h@1 | 1p h@5 |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| random (3 seeds) | uniform ranker | 0.0 | 1.7 | 11.2 | 70.1 | 5.9 | 3.3 | 28.1 |
| θ₀ | deployed `d20+setup_only` | 9.5 | 22.6 | 50.4 | **92.0** | **57.4** | 39.7 | **82.4** |
| arm A | 45,469 guessed cells, λ=0.05 | 13.6 | 27.7 | 54.5 | 91.5 | 51.4 | 39.2 | 79.9 |
| `ANR` | arm A, ranking aux **off** | 3.7 | 10.7 | 31.1 | 82.9 | 28.7 | 15.4 | 57.3 |
| `aL10` | arm A, λ_lower=0.10 | 13.6 | 27.0 | 55.5 | 91.3 | 52.3 | 37.6 | 79.9 |
| `Bfix` | arm B done properly, 591,504 cells | 13.4 | 28.9 | 50.4 | 87.1 | 53.4 | **41.8** | **83.5** |
| `BfixNR` | Bfix, ranking aux **off** | 5.8 | 11.2 | 28.2 | 81.5 | 31.6 | 15.2 | 58.1 |
| `BNG` | Bfix + guesses barred as rank positives | **14.6** | **32.1** | **55.7** | 88.6 | 52.6 | 38.4 | 81.4 |
| **`ANG`** | **arm A + guesses barred as rank positives** | **14.6** | **30.7** | 53.1 | **91.8** | 50.9 | 39.0 | 81.0 |

Offline panel (`twopush_gt_h5`, 1,152 eps), mean [sd]:

| metric | arm A | Bfix | BfixNR (aux off) | ANR (aux off) | BNG | ANG-era ref |
|---|--:|--:|--:|--:|--:|--:|
| V1 root separation | 0.798 | 0.796 | 0.797 | 0.810 | 0.798 | — |
| V4 cell vs cell | **0.896** | 0.866 | 0.847 | 0.869 | 0.880 | — |
| V5 cell vs dead board-max | 0.543 | 0.527 | **0.625** | **0.616** | 0.538 | — |
| V6 board live/dead | 0.777 | 0.786 | 0.707 | 0.688 | **0.786** | — |
| F2 finish within-board | **0.906** | 0.902 | 0.735 | 0.736 | 0.902 | — |
| setup hit@1 hard | 22.9 | 23.2 | 17.5 | 17.2 | **24.0** | — |
| finish hit@1 hard | 54.8 | 56.1 | 32.5 | 30.0 | **57.0** | — |

### What these say [verdicts on numbers]

**1. The ranking auxiliary carries roughly half the deployed performance.** Removing it costs hard-2p@5 27.7→10.7 (arm A) and 28.9→11.2 (Bfix), 1p-hard@1 39.2→15.4 and 41.8→15.2, finish hit@1 hard 54.8→30.0. Every band non-overlapping, both doses. For scale, the depth-token architecture moved ±3–4 pts and was rejected; a 12× label dose moved ±4–5. **The term doing the most work had two settings ever tried: on, and default.**

**2. It is the ONLY consumer of the ceilings, which are 46–48% of supervised cells.** A ceiling has no point target — regression can only penalise exceeding the cap, never order it. The aux uses them as competitors that facts must outrank. That is why deleting the aux and deleting the ceilings (round-2 arm B2, hard-2p@5 → 5.1) cost about the same: two ways of severing one pathway.

**3. λ_lower saturates immediately.** 0 → 0.05 is worth +24 pts on hard-2p@2; 0.05 → 0.10 is worth nothing (13.6→13.6 @2, 27.7→27.0 @5, all bands overlapping). Matches the registry's earlier "split vs opener-only indistinguishable". **The λ question is closed — stop tuning it.**

**4. The aux BUYS within-board order and COSTS cross-board comparability.** Turning it off *raises* V5 0.543→0.616 and 0.527→0.625, while destroying F2 (0.906→0.736) and finish hit@1 (54.8→30.0). Since 88–94% of deploy pops happen on child boards, the finish collapse dominates and deploy craters — but the cross-board damage is real and measured.

**Mechanism, verified on cached scores (not inferred):** the aux is `log_softmax(dim=1)` over one board, so it is shift-invariant per board — add a constant to every cell and the loss is unchanged, hence zero pressure to keep boards on a common scale. Measured consequence: within-board spread 0.516 (aux off) → **0.666** (aux on), while the spread *across* board maxima shrinks 0.227 → 0.204, and **dead-board maxima inflate 0.625 → 0.720** against live 0.768 → 0.880. V5 is precisely "setup cell vs dead board max", so inflating dead maxima is a direct hit. Normalised by spread the separation is *worse* with the aux: 0.239 vs 0.277. My earlier "spends capacity" phrasing is retired in favour of this.

**5. Guessed cells were being treated as CERTAIN tiers — fixed, and it was worth points.** `run_reduce` clears `ceiling_mask` on relabeled cells, and the aux treats "in-loss and not a ceiling" as certain, looping over `torch.unique(labels[exact])`. Bootstrapped targets are continuous floats, so each becomes its own tier: **θ₀ 2 levels per 256-row batch, arm A 27, Bfix 593.** The half-weight never reached the aux (it only weights the regression), so 4th-decimal model noise was enforced as known ordering at full strength — and cost ~300× the loop iterations (Bfix ~20 min/epoch vs arm A ~5 on the same a4000).
`NAMO_RANK_EXCLUDE_GUESS=1` (opt-in) bars guesses from being positives while keeping them as competitors in `lower`. Unit-verified: tiers 106→2, opener loss bit-identical, guessed cells still in the competition mask. **Result: +3.2 hard-2p@5 on Bfix (28.9→32.1) and +3.0 on arm A (27.7→30.7), both non-overlapping, and 4× faster training.** `BNG` also recovered the @30 ceiling Bfix had lost (50.4→55.7).

**6. Arm B's hypothesis is REJECTED, and we now know why.** Its 546,035 extra cells are *exactly* the children whose sweeps were **censored** — the one population whose answer is unrecoverable from this data (see [EXP-2026-08-08-arjuna](EXP-2026-08-08-arjuna-hard-labels.md) census). Guessing there cost 4–5 pts of hard-2p reach (@900 91.5→87.1, @30 54.5→50.4). Arm A's population, by contrast, is fully resolved — it succeeded partly by accident of which cells the H5 happened to record. **Not "more labels hurt" but "labels on the unknowable hurt."**

**7. Deploy recommendation: `ANG`** — arm A + guess-exclusion. Hard-2p@5 **30.7 vs θ₀'s 22.6 (+8.1)** with reach intact (@900 91.8 vs 92.0). `BNG` is faster still (@5 32.1) but pays 3.4 pts of ceiling. Neither is registered as deployed pending user call.

### Structural facts measured 2026-08-07 (durable; do not re-derive)

- **Targets are 96–100% ≥ 0.8**; zero of 143,705 exact facts is a 0. The regression fits a near-constant — this is why ordering supervision dominates it. Not a focal-loss problem: there are no negatives to down-weight.
- **Ceilings: 129,026 = 46–48% of supervised cells**, essentially untouched by aquaman (arm A converted 602 of them; Bfix converted zero additional — its cells came from the masked pool).
- **Loss reduction is a flat per-cell mean over the batch** (`(ce*mask).sum()/mask.sum()`), so a board with 70 supervised cells gets 70× the gradient of one with a single cell. Boards with **zero openers are 28% of boards but receive 21% of the gradient**. Per-board normalisation would move that to 28% — a 1.35× reweight, worth an arm, but second-order next to the target compression.
- **`solve@2 ≈ setup hit@1 × finish hit@1`**, verified to ~1 pt across all conditions (arm A 22.9×54.8=12.6 vs measured 13.6). 2-push@2 is low because it demands two consecutive first-guesses, and the binding term is setup@1 at ~23% on hard.
- **`--raw` vs sigmoid is inert under `combine=q` + `discount=off`** — verified byte-identical on 13 episodes (solve 100.0/100.0, avg_sims 33.69/33.69, per-episode sims identical). Raw is now the default; `--sigmoid` keeps the legacy scale. It is NOT inert under `blend`/`product` or `--discount conf`.
- **Fine eval sharding gave 1.4×, not the ~3–4× projected** (256 shards 11 min vs 72 shards 15 min): concurrency, not shard granularity, was binding once 400 tasks were queued.
- **`depth-token` is push-depth attention, unrelated to horizon conditioning** — it expands each contact into five motion-grounded depth tokens. Horizon conditioning remains untested; the nearest prior is the old Hz embedding result.

## 📌 PINNED — round-3+ design discussion (resume after round-1 gate) [2026-08-02 evening, USER+Claude]

1. **Re-rooting + compositional chain verification (the round-3 core; NO arms — one design).** Post-push boards become fresh episode roots (stored qpos, `set_full_state`); an h2 search from a depth-1 board verifies a 2-chain suffix; determinism composes it with the verified parent link into a fully-verified 3-chain → parent cell gets **exact γ² = 0.81** (the label class that cannot exist today). Recurse for γ³+. Requirements: (a) exact state identity; (b) lineage tag (parent xml, object, parent-cell) on every re-rooted episode; (c) builder join step propagating exact γ^(k+1) onto parent cells (re-rooted failures tighten parents too); (d) values are found-chain lower bounds — same semantics as 0.9 setups. **Doctrine: collection search stays hmax=2 at every depth forever; depth = laddered roots + composition; hmax=3 is eval-only.** **[USER-confirmed 2026-08-02] Scaffolding→masonry conversion: composed chains RELABEL guessed/capped cells with TRUE values (exact γ^k), one-directional (opinion→fact, never back) — bootstrap = dense provisional scaffolding at the frontier, composition progressively replaces it with verified masonry; buffer monotonically hardens toward truth. Bootstrapped γ·V̂ stays the volume workhorse; composed exacts are the calibration spine (one verified chain re-calibrates thousands of similar guesses via generalization).**
2. **Retry list ("redo failed stuff") [USER-confirmed 2026-08-02].** Unsolved episodes carry into the next round's manifest (better θ, fresh budget); cap ≤20% of manifest. Meter: retry solve-rate = cleanest round-over-round progress number.
3. **Deep-audit exhaustion tier.** Fail-twice residue (~≤500 eps/round) gets full-tree exhaustion (budget=∞, stop on queue-empty; ~2,800 sims/ep) → mints proven-dead-within-h2 (exact-zero anchors). NOTE: tonight's 5% audit slice = 900-budget extension, NOT exhaustion.
4. **Budget at depth.** B=150 fine for h2 (solves mean ~29 sims, cap non-binding); re-rooted rounds keep h2 so B=150 likely stands; recalibrate from round-1 sims histograms.
5. **Collector v2 — HARD REQUIREMENT before any round-3 wave [USER 2026-08-02 22:20]:** (a) 12-cpu fat tasks + in-process worker pool over rooms (~6× throughput, 500×12≈6,000 cpus); (b) **collect+render FUSION** — render ctx at capture time inside the collector, eliminating the separate render wave (a full stage, ~30 min/round). Build during round-2 train window; smoke 5 rooms local + 1 fat Amarel task. Target: collection+render ≤10 min at 60k rooms. Also: 1-line stubs for quota-dropped episodes; collector-mix hedge (later).
6. **Exploration dial policy:** 20% random slice = detection-sized; if buried-winner meter stalls, prefer TARGETED exploration of flagged classes over raising the global dose.
7. **Round-2 (backfill) already authorized:** replay old root caps (5M) + child caps (2.8M, 2-push replay chains via raw linkage — no PKLs needed) → guesses by round-1 model → mute caps 44%→~2%; doubles as ~8M-push lineage audit. Colossus PKLs (35GB Amarel) thereby deletable.


8. **Speed-lever ledger [USER 2026-08-02 ~22:45, pinned]:** DONE tonight: /dev/shm train staging (in template, NFS fallback); eval fine-shards (SHARD 4-6) in two half-waves + hard-first ordering (straggler elimination — same CPUs, ~3× less wall); fat 12-cpu shape locked for v2. QUEUED for v2: collect+render fusion; cost-balanced sharding (shard by expected sims, not scene count). DECLINED: **torch.compile — user call, not worth the churn**; Amarel-GPU scoring (10-min stage, plumbing > payoff); sub-900 eval budgets & episode subsetting (protocol-breaking). BATMAN-ERA: streaming actor fleets (kills inter-stage troughs). Post-v2 utilization ≈85-90% of the 6,720-cpu quota during waves; residual floors = slowest single unit + stage boundaries + CS-GPU train epochs.

## References (the named ancestry of each component)

- **Fitted Q-Iteration** — Ernst, Geurts & Wehenkel 2005, *Tree-Based Batch Mode RL* (JMLR): batch Bellman backups with targets recomputed per iteration → our guess-refresh.
- **Expert Iteration** — Anthony, Tian & Barber 2017, *Thinking Fast and Slow with Deep Learning and Tree Search* (NeurIPS): search improves net, net improves search → the crank.
- **AlphaZero** — Silver et al. 2017/2018 (Nature/Science): self-play data generation, gating, replay buffer; our cost-inverted adaptation (sims expensive, net cheap).
- **DAgger** — Ross, Gordon & Bagnell 2011 (AISTATS): on-policy state aggregation D₀∪…∪Dᵢ → collection base loop; expert = the simulator.
- **Prioritized Level Replay** — Jiang, Grefenstette & Rocktäschel 2021 (ICML): replay worst-performing environments → the retry list.
- **Go-Explore** — Ecoffet et al. 2019/2021 (Nature): return-to-stored-state then explore → re-rooting from stored boards.
- **Reverse curriculum from start states** — Florensa et al. 2017 (CoRL): laddered start-state curricula → effective-depth-1 rungs, mechanized.
- **KataGo** — Wu 2019: mixed self-play data diet (fresh openings + salvaged positions) → the three-stream collection budget.
- **Deadly triad** — Sutton & Barto, *RL: An Introduction* (2nd ed.), §11.3: the divergence risk our caps/anchors/half-weight are rails against.
- **Censored regression / survival analysis** — one-sided ceiling losses (beast-0a lineage, in-house validated).
- **Verified-subproof composition** (analogy) — HTPS/AlphaProof-style theorem-proving RL: composing verified sub-results → our deterministic chain composition (the domain-specific piece; enabled by bit-identical physics).

## Log

- **2026-08-08 [Claude] Round-0 refire COMPLETE + loss anatomy; branch forked to `arjuna`.** Seven arms × 3 seeds trained and gated overnight (Bfix, BfixNR, ANR, aL10, BNG, ANG, plus arjuna's ARJ on the sister card), ~1,500 canonical eval tasks, 4 AUC panels. Headlines: the ranking aux carries ~half of deployed performance and is the only consumer of the 46% of cells that are ceilings; λ_lower saturates at 0.05; guessed cells were being enforced as *certain* tiers (593 per batch on Bfix vs 2 on θ₀) and barring them is worth +3.0/+3.2 hard-2p@5 and 4× training speed; arm B's 546,035 cells are exactly the censored children, so its hypothesis is rejected on the sharper reading "labels on the unknowable hurt". **Deploy candidate `ANG`: hard-2p@5 30.7 vs θ₀ 22.6, reach 91.8 vs 92.0.** Full tables + verdicts in § Round-0 REDONE. Two of my own predictions were wrong and are recorded as such (guess-exclusion expected negligible on arm A, gave +3.0; fine sharding expected ~4×, gave 1.4×). Code: `NAMO_RANK_EXCLUDE_GUESS`, raw scorer default (verified byte-identical), parameterised eval sharding, all committed and pushed.

- **2026-08-07 [Claude] ARM B WAS NEVER TRAINED — round-0 correction + refire.** Reading `run_reduce` while planning the arjuna branch: it writes `value_target`/`ceiling_mask`/`guess_mask` and never `value_mask`, while `q2_dataset.py:73` builds `loss_mask = value_mask * r_mask` and `weighted_module.py:54` reuses it for the rank-aux. Arm B's class-1 cells are `value_mask=0` by construction (that is what "masked" means), so all 546,035 of them were inert. File-level proof: 6,129 rows sampled, 413,869 in-loss cells byte-identical A vs B (max delta 0.000000), guess cells in loss A=845 / B=845. **Round 0 = one condition, six seeds.** The +4.8/+5.6/+5.1 hard-2p gain therefore comes from arm A's 45,469 relabeled root cells alone — 0.5% of the labels, on the ~26k colossus setup roots only — and is 6-seed robust rather than 3. Retired claims: "V6 dose-response A→B" and "13× cells, same result ⇒ saturation" (no dose existed). Fix shipped in `aquaman_rebuild_v2.py` (`vm[e,d]=1.0` alongside the target write, arm name `Bfix`); rebuilt `aquaman0_train_Bfix.h5` from the banked 32 `edits_*.npy` (reduce only, no GPU map). Pre-registered prediction before training [Claude]: flat-to-slightly-negative on hard-2p, since arm A's cells are 72% below 0.1 (low-end signal) while the class-1 cells sit at quartiles 0.085/0.232/0.468 — mid-scale mass, which is what blurs cross-board comparison. Rounds 1-2 parked meanwhile; note the same `value_mask` question applies to the round-2 refresh (591,504 edits, 20,567 rows) and is NOT yet checked.

- **2026-08-03 03:40 [Claude] B3 (POS_WEIGHT=2): ≈ B1 — no rescue** (hard-2p@5 23.1±0.9 vs B1 23.4; 1p-hard@5 recovers to 81.1). Balance theory ELIMINATED; fault isolated to what B1/B3 share: **r1-as-guesser quality and/or frontier-heavy diet**. Morning queue: (1) teacher audit — r1 vs B_s1 quizzed on the 447-episode audit key; (2) recovery test — rebuild B1 file with B_s1 guesses, 3 seeds; if green → permanent loop rule "AUDIT EVERY TEACHER BEFORE IT TEACHES"; (3) diet-mix ablation. Night closed: rounds 0-1 clean, round-2 red caught by pre-registered brake, caps-essential finding permanent, mining doctrine validated (wave held), ~5M sims of facts banked, deploy-best unchanged.

- **2026-08-03 03:15 [Claude] ROUND-2 GATE: RED — STOP-RULE TRIPPED, CHAIN PARKED.** Collection: 67,370 rooms + 7,470 retries → 39,561 kept traces (3.8×), 4,919 verified setups; build: 639,059-board files (60% on-policy), guesses by r1_std_s1. Nine seeds, three arms, canonical eval (zero unmatched). **B2 (old caps deleted): CATASTROPHIC** — hard-2p@5 5.1, 1p-hard@1 16.7 → *the old one-sided ceilings are massively load-bearing; permanent keep* (the composition question answered decisively). **B1 (everything): REGRESSED** — med-2p@5 48.8±1.3 vs r1 53.2 (stop-metric DEEPENED), hard-2p@30 47.4 vs 55.5. Suspects: (a) r1-as-guesser quality (its refreshed old-cell guesses drifted UP — 15.6% <0.1 vs θ₀-era 28% — noted pre-hoc); (b) frontier/retry-heavy diet drowning anchors. B3 (POS_WEIGHT=2) eval in flight = diet-vs-guesser arbiter. **Actions: round 3 NOT launched; 50k mining wave HELD (pilot itself validated: 4.5 wins/ep, 319 exact-zeros, 54s/room); deploy-best unchanged (r1_std/r0-B). All facts intact; r2 opinions discarded.** Mining pilot data + 1k-room traces banked. Diagnosis queue: score-drift panel on r2-B1, guess-quality audit of r1-as-teacher vs B_s1 on the 447-audit answer key, diet-mix ablation.

- **2026-08-02 [Claude] ROUND-1 GATE (std arm) — crank turns clean.** First full search-as-collector cycle: 20k rooms (positions 300k-320k), 21,191 episodes derived, 938,807 sims, 10,408 kept traces (quota-filtered DAgger-style); 92,645 new boards + 643k new guess cells; old buffer refreshed by B_s1; merged 349,839-board file (census: 9.03M exact / 1.24M guesses / 8.53M caps / 5.75M masked). 3 std seeds (val 1.818-1.826) evaluated canonical (1322/1012, zero unmatched). **vs θ₀ / r0-B: hard-2p @2 14.3±3.4 (held), @5 28.2±2.7 (held), @30 55.5±3.4 (held); med-2p@5 53.2±0.3 (stop-metric: improved vs r0's 52.1, tax vs θ₀ persists); 1p-hard@5 81.4±2.5 (r0 softness healed); hard-2p@900 90.3±0.7 (−1.2 vs r0, watch).** No stop-rule tripped → round 2 (backfill) proceeding per standing authorization. **Incident:** dt eval arms failed ×204 on Amarel — pinned checkout's sage predates depth-token arch (`action_motion_dim` unknown); fixed by shipping worktree sage as SAGE_REPO override; ~25min delay; RULE PINNED: checkpoints travel with their code version. Artifacts: `aquaman/round1/{aquaman1_train.h5,gate_std.json,eval_amarel/}`, collector/build scripts committed on `exp/bootstrap-value-loop`.

- **2026-08-02 [Claude]** Card created; worktrees `exp/bootstrap-value-loop` (namo @ fb02310, sage @ 6f90dc6). Design converged in chat (brainstorm with user): label rule, locked defaults, meters, round schedule. Next: round-0 step 1 precheck.
- **2026-08-02 [Claude] Census (30k sample, true loss mask `vm×rm`):** base-root 23.5% cells supervised / 37.3% ceiling; colossus-root 13.4% / 52.5% (exact all 0.9 setups, untried 22.2/row); child 23.9% / **94.6%** ceiling (exact all 1.0 winners, tried median 70, untried median 0). Reproduces EXP-2026-07-25's 38.5/94.7. Zero episode ambiguity: 0/215,856 multi-root `(xml,object)` groups. Linkage sparse: only 6.0% of roots have an in-file child; colossus block's 157,310 children live in the 200k source H5 → **aquaman-0 scope = colossus block (~26k setup roots, ~157k linkable capped cells)**.
- **2026-08-02 [Claude] Rebuild v1 (aquaman0_train.h5) — mechanically sound, dose too small.** `scripts/rl_loop/aquaman_rebuild.py`: pose-match linkage verified excellent (median err 1.4cm, p99 5.8cm, gate_fail 2,957); 67k "collisions" = jammed-push duplicate children (same resulting pose — dedupe correct); source multi-root groups = 0 (no cross-episode pooling). **But guessed cells = 15,869 of 559,322 capped (2.8%)**: the 200k source H5 is a curated selection — its 157,310 children are mostly winner boards (46,273 matched exact setup cells, skipped by design) + the ~33k negative dose; capped-cell children ≈ 16k. The full capped-children set never left Amarel. Guessed-target quartiles [0.34/0.54/0.69] — high vs precheck's base-children tails because colossus untried cells are the OLD ranker's ranks 21–70 (mid-pack), and θ₀ disagrees with d20's order there; capped at 0.81 regardless. **Decision pending: (A) accept 15.9k-cell dose (weak H1) vs (B) pull Amarel raw shards (full dose, exact linkage) — scout dispatched to verify raw schema.** Artifacts: `$SCRATCH/aquaman/round0/aquaman0_train.h5` + `.report.json` (v1, superseded if B).
- **2026-08-02 [Claude] Rebuild v2 DONE — full dose, two arms, exact linkage.** Amarel raw (`colossus0_200k_4322a98/h5_ccbb2d1_224996_32/`, 32 shards, 24GB, parent_edge/parent_depth columns) rsynced to CS (`$SCRATCH/aquaman/round0/raw/`); `aquaman_rebuild_v2.py` map×32 (5 GPUs, ~25min) + per-arm reduce. 687,616 children kept (141,581 capped-class + 546,035 censored-masked; 96k exhausted-skips). **Arm A (capped-only): 45,469 cells**, target quartiles [0.014/0.036/0.115], 72% <0.1. **Arm B (+censored): 591,504 cells**, quartiles [0.085/0.232/0.468], 28% <0.1, clip 0.4%. Distributions match precheck physics (A = audit dregs, B = ranks-21-70 remainders where θ₀ sees promise). Sanity: exact cells byte-identical 60/60 rows both arms; guessed cells valid; weight 0.5/1.0 flows through patched `q2_dataset` (guess_mask). Both-arms decision [USER]: run A and B in parallel (3 seeds each) — dose-response isolates cap-sharpening vs silence-filling. Files `aquaman0_train_{A,B}.h5`. Smoke train job ilab 198360 (arm B seed 1, 1 epoch) before the 6-job fleet. `scripts/rl_loop/aquaman_precheck.py` (worktree), θ₀ raw E[bin] via `eval_auc.score_h5`, 41.5k child boards, counterfactual sweep-stop quiz. **AUC(live vs dead remainder) = 0.853 / 0.853 / 0.855 at K=10/20/30** (bar ≥0.75), V̂ medians live 0.585/0.477/0.415 vs dead 0.147/0.085/0.053. True-untried remainders (19,323 boards): target `min(0.81,0.9·V̂)` quartiles **[0.011, 0.028, 0.115]**, clip@0.81 binds 0.2% — far LOWER than the predicted 0.3–0.6 hump (untried tails are the collection ranker's skip-list; θ₀ scores them near zero) → the downward gradient arrives at full strength. Honest overlap: AUC 0.85 ≠ 1.0, wrong-low tail exists as priced. Artifacts: `$SCRATCH/aquaman/round0/precheck.{json,log}`. Next: step 2 rebuild (colossus-block linkage via `action_motion` pose match + 200k-source child scoring), pending user go.

## Discussion

_(you ↔ Claude — ask here; answers inline, dated. Newest at the bottom.)_

## Status reconciliation (2026-08-12)

**Closed as `done`.** Rounds 0/0-redone/1 fully gated and registered (registry rows for A/B/Bfix/BNG/ANG/ANR/aL10); round 2 hit its own stop-rule (RED) and the chain was parked by design. The thread forked into [EXP-2026-08-08-arjuna-hard-labels](EXP-2026-08-08-arjuna-hard-labels.md) and then [EXP-2026-08-09-crossboard-ranking](EXP-2026-08-09-crossboard-ranking.md), which is the live successor.

**Dangling, deliberately unexecuted:** the PINNED round-3+ design (re-rooting/composition, retry list, deep-audit exhaustion, collector v2 fusion) and the `batman` clean-room reference run. Anyone reviving these should first read the 2026-08-12 corpus-composition findings on the crossboard card — the opener-bearing-root-fraction result changes what a re-rooting round should target.
