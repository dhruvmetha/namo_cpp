---
type: experiment
status: live
created: 2026-08-08
parent: EXP-2026-08-02-bootstrap-value-loop
thread: rl_loop
robot: car
tags: [experiment, arjuna, hard-labels, policy-relative, floor, claude-active]
---

# EXP-2026-08-08 — Arjuna: facts where the bootstrap guessed (hard labels, policy-relative value)

**Lineage: INDIAN MYTHOLOGY (alphabetical, one letter per method revision) — this is `arjuna`; next is `bhima`** [USER 2026-08-07].
The split from DC is philosophical, not incremental: **arjuna masks what it does not know; DC guesses it.**
DC (aquaman/batman) fills every unresolved cell with `min(cap, γ·V̂)` at half weight — see [EXP-2026-08-02](EXP-2026-08-02-bootstrap-value-loop.md).
Arjuna writes a value only where the simulator settled it, leaves the rest masked, and defines value **relative to the searcher we deploy** rather than to the world.

**⛔ Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** The model is a ranker; the simulator is a perfect free verifier; success is fewer sim calls than random on every tier.

## Semantics (the fork, decided 2026-08-07 [USER])

Value is **policy-relative**: what a push is worth to the hmax=2 searcher we actually run.
Under that definition the scale is closed and tiny — `1.0` opener, `0.9` setup, `0` dead — and `0.81 = γ²` cannot occur, because it denotes a two-step setup and the deployed search has no third push.
A failed push-2 is therefore **0 exact**, not a censored bound: the searcher can never cash a deeper win, so its value to us is zero.
This is not a claim about the world. A cell scored 0 may well open at depth 4; that is out of scope by the problem definition (one object, one region pair, 1-push or 2-push) and recoverable later, since labels are rebuilt from raw every round.
The world-value alternative (ceilings that tighten by one γ per level exhausted, never reaching a floor) stays with DC.

## Why (the measurement that forced it, 2026-08-07)

**96–100% of every training target sits ≥ 0.8, and not one of 143,705 exact facts is a zero.**

| file | in-loss cells | median target | frac < 0.5 | frac ≥ 0.8 |
|---|--:|--:|--:|--:|
| θ₀ `d20_plus_setup_only` | 272,389 (sampled) | 0.900 | **0.000** | **1.000** |
| aquaman arm A | 272,389 | 0.900 | 0.002 | 0.998 |
| aquaman Bfix | 280,793 | 0.900 | 0.025 | 0.968 |

The regression term is fitting a near-constant: a model emitting 0.87 everywhere scores well on it.
That is the mechanical reason the listwise ranking auxiliary carries roughly half the deployed performance ([EXP-2026-08-02](EXP-2026-08-02-bootstrap-value-loop.md) § 2026-08-07), and it is not a focal-loss problem — focal down-weights easy negatives so rare positives dominate, but here there are no negatives at all.

## arjuna-0 — the build (zero sims, zero GPU)

`scripts/rl_loop/arjuna_build.py`. Same base file as aquaman arm A (`round3/h5/d20_plus_setup_only.h5`), one rule changed, on the same cells arm A guessed.

The answer was already in the Colossus raw shards, unused: every child board carries `f_grid`, `n_win`, `n_tried`, and `finish_sweep_censored`.

| child board status | parent root cell |
|---|---|
| `n_win > 0` — has an opener | already recorded as an exact 0.9 setup; **never overwritten** |
| `n_win == 0` **and not censored** — swept clean | **0.0 exact, full weight** |
| `n_win == 0` and `finish_sweep_censored` | genuinely unknown — **ceiling left alone** |

Census over all 32 raw shards (`arjuna_build.log`):

| | rows |
|---|--:|
| raw child rows walked | 3,250,482 |
| not ours (no join to the 26k Colossus roots) | 2,068,961 |
| parent already exact (opener/setup) — skipped | 493,896 |
| **censored — declined** | **546,035** |
| **proven dead → exact 0** | **141,581** |
| collisions | 0 |

**The 546,035 censored children are exactly the class-1 population aquaman arm B/Bfix guessed.**
So Bfix's extra half-million labels were opinions about the one group whose answer is unrecoverable from this data — and they cost 4–5 points of hard-2p reach.
Arm A's population, by contrast, resolves completely: all 141,581 of its children were swept clean, which is why arm A could be relabeled with facts at 3× the cell count arm A itself guessed (45,469).

Verification gate before any GPU was touched (all four must pass, and did): written values ∈ {0, 0.9}; 216,733 pre-existing exact facts unchanged; nothing outside the write mask changed; `r_mask` unchanged.

**Dose is thin: 141,581 zeros ≈ 0.8% of in-loss targets** (the file holds ~17.5M). For scale, arm A's 45,469 guesses were ~0.26% and moved hard-2p by ~5 points. No flood risk — the 2026-07-25 regression came from zeroing *unknowns*, which arjuna refuses to do.

## Result — canonical gate, 3 seeds, mean [min,max]

Protocol identical to round-0 (1322 1push@hmax2 + 1012 2push, budget 900, `prior=model`, `agg=mean5`, `combine=q`, **discount off**, dedupe+jam on). Zero unmatched episodes.

| slice | θ₀ | arm A (guess) | **ARJ (facts)** |
|---|--:|--:|--:|
| 2push hard @2 | 9.5 | 13.6 [11.7,15.3] | 14.1 [10.9,16.1] |
| 2push hard @5 | 22.6 | 27.7 [26.3,29.2] | **27.7** [25.5,32.1] |
| 2push hard @30 | 50.4 | 54.5 [54.0,54.7] | 54.7 [51.8,56.2] |
| 2push hard @900 | 92.0 | 91.5 [90.5,92.7] | 91.0 [89.8,92.0] |
| 2push medium @5 | 57.4 | 51.4 [50.2,52.3] | 51.5 [50.6,52.7] |
| **1push hard @1** | 39.7 | 39.2 [37.3,42.6] | **42.5 [41.7,43.6]** |
| 1push hard @5 | 82.4 | 79.9 [77.5,81.4] | 80.9 [78.9,82.4] |

Offline panel (`twopush_gt_h5`, 1,152 eps), mean [sd]:

| metric | arm A (guess) | **ARJ (facts)** |
|---|--:|--:|
| V1 root separation | 0.798 [0.006] | 0.801 [0.007] |
| V4 cell vs cell | **0.896** [0.002] | 0.885 [0.002] |
| **V5 cell vs dead board-max** | 0.543 [0.007] | **0.533** [0.021] |
| V6 board live/dead | 0.777 [0.002] | **0.790** [0.008] |
| F2 finish within-board | 0.906 [0.003] | 0.906 [0.003] |
| setup hit@1 hard | 22.9 [1.4] | **24.6** [3.8] |
| finish hit@1 hard | 54.8 [1.1] | 52.6 [2.9] |

## ⛔ SCOPE CORRECTION (2026-08-08, after the verdict below was written)

**v1 changed 1.4% of the bounded cells. The verdict below is true of that dose and must not be read as a test of "does the model need a floor".**

| file | in-loss | bounded ("at most X") | converted vs θ₀ | zeros |
|---|--:|--:|--:|--:|
| θ₀ | 17,428,630 | 8,294,162 (47.6%) | — | 0 |
| aquaman arm A | 17,428,630 | 8,255,643 | 38,519 (**0.5%**) | 0 |
| aquaman Bfix | 17,966,355 | 8,255,643 | 38,519 (**0.5%**) | 0 |
| **arjuna-0 v1** | 17,428,630 | 8,179,502 (46.9%) | **114,660 (1.4%)** | 114,660 |

Every label intervention this project has run moved ≤1.4% of the bounded population. After all of them, ~47% of supervised cells still say "at most X" with nothing pulling them down.

**Why so little was reachable — a join limit, not a choice:**

| region | rows | in-loss | bounded | bound value |
|---|--:|--:|--:|---|
| base-root | 188,467 | 13.28M | 37.2% | **≤0.81** |
| base-child | 42,885 | 3.00M | **94.8%** | **≤0.90** |
| colossus-root | 26,057 | 1.06M | 54.2% | ≤0.81 |

Only the 26,023 Colossus setup roots have child boards in the raw shards, so only they can be resolved by linkage. **94% of bounded cells sit on d20-base rows for which no child board was ever stored** — unresolvable as archived, at any effort. Within the reachable population v1 actually converted **20%**, a real dose; it is 1.4% only against a file that is 94% out of reach.

**And the two bounds are inverted relative to what the deployed searcher can cash:**

| cell | current bound | correct under hmax=2 |
|---|---|---|
| root, simmed, failed, child unresolved | ≤0.81 — *below* setup grade, so undiscovered setups are pre-emptively buried | ≤0.9 (a setup is still possible) |
| child, simmed, failed | ≤0.90 — the loosest bound in the system | **= 0** (no third push exists) |

The mis-set one is the large one: **2.84M child cells that should be exact zeros are bounded at 0.9**, on the tier carrying 88–94% of deploy pops, and every one was already simmed. That is the floor experiment v1 did not run — and is what v2 below does.

## Verdict [numbers] — the floor hypothesis is FALSIFIED **at the 1.4% dose**

**Pre-registered before training** [Claude, 2026-08-07]: *"if the floor theory is right the gain shows up first in cross-board — V5 above its stuck 0.53 — and if V5 doesn't move, the theory is wrong regardless of what the solve rates do."*

**V5 = 0.533 against arm A's 0.543.** It did not move. Adding the first floor this project has ever had — 141,581 exact zeros at full weight — bought nothing in cross-board comparability.

So the standing explanation for the cross-board hole (no zeros → no bottom → dead boards float → best-of-70 beats a real setup) is **wrong**, and should not be repeated. The surviving explanation is mechanical and measured: the ranking auxiliary is `log_softmax(dim=1)` over a single board, hence shift-invariant per board, and it inflates dead-board maxima 0.625 → 0.720 (see the DC card's 2026-08-07 section). No label change touches that.

**What the facts did buy, consistently across both panels:** best 1push-hard@1 in the entire line (**42.5** vs θ₀ 39.7), best setup hit@1 hard (24.6), best V6 (0.790).
That fits a narrower mechanism than the one proposed: a proven zero says *this specific push fails* — direct evidence for opener ranking, which is exactly 1-push top-1 — while adding nothing about comparing one board to another, because arm A's guesses on those same cells already sat at median 0.029.

**2-push is flat** (27.7 vs 27.7 @5), and finish hit@1 hard *fell* 54.8 → 52.6. Since `solve@2 ≈ setup@1 × finish@1` (verified to ~1 pt across conditions), the better setup pick was cancelled by the worse finish pick.

## arjuna-0 **v2** — the big-dose floor test [USER 2026-08-08, COMPLETE]

v1 is superseded as the floor test (its artifacts stay registered; nothing was deleted — `arjuna0_train.h5` is a standalone copy and the ARJ models are cited in the registry).

**Label scheme [USER]** — needs no linkage at all, so it reaches the whole file, not just the 6% Colossus block:

| cell | v2 label |
|---|---|
| opener (finish) | **1.0** |
| setup | **0.5** (γ drops 0.9 → 0.5, widening the finish ≫ setup ≫ rest ladder) |
| **every bounded cell** (≤0.81 root and ≤0.90 child) | **0.0 exact, two-sided, full weight** |
| untried | masked — unchanged; we never invent a value for a push nobody executed |

Build `scripts/rl_loop/arjuna_build_v2.py` → `round0/arjuna0v2_train.h5`. Census: openers 3,422,785 · setups 5,592,004 · **bounded→zero 8,372,367** · untried still masked 1,016,234. Gate (no GPU touched until it passed): distinct values exactly `[0.0, 0.5, 1.0]`, **zeros 48.2% / setups 32.2% / openers 19.5%**, bounds remaining **0**, untried preserved.

**Zeros go 0.66% → 48.2% — a 70× dose.** This is the first label file in the project's history whose targets genuinely vary; every earlier result was measured inside a regime where 96–100% of targets sat ≥0.8 and the regression could satisfy itself by emitting ~0.87 everywhere.

**Arjuna v2 is NOT arm A or arm B — it is a third condition, and the distinction is the whole fork.** All four files below are copies of the same base, `round3/h5/d20_plus_setup_only.h5` (θ₀'s own training data); they differ only in what is written on top. Measured directly, ~4,000 rows sampled per file:

| file | in-loss cells | untried (masked) | bounds left | **guessed cells** | zeros | distinct values |
|---|---|---|---|---|---|---|
| base `d20_plus_setup_only` | 272,389 | 16,266 | 129,628 | 0 | **0.0%** | `{0.81, 0.9, 1.0}` |
| **arjuna v2** | 272,389 | 16,266 | **0** | **0** | **47.6%** | `{0.0, 0.5, 1.0}` |
| aquaman arm A | 272,389 | 16,266 | 129,026 | 602 | 0.0% | continuous |
| aquaman Bfix | **280,793** | **7,862** | 129,026 | 9,006 | 0.0% | continuous |

Three things this settles:
- **Guesses belong to the DC/bootstrap lineage only.** A and B invent values (`min(cap, 0.9·V̂)` from the model's own top-5) for cells the simulator never resolved. v2 invents nothing — bounds are the only population it touches, and its file is **178 bytes** larger than the base because it rewrites values in place.
- **The base data contained no zeros at all** — every label was 0.81, 0.9, or 1.0. The model had never once been shown "this push is worthless." v2 is the first file in which it is, for 47.6% of its supervision.
- **The round-0 `value_mask` bug and its fix are visible in the table:** Bfix's in-loss count is 8,404 higher than every other file's and its untried count correspondingly lower — those are the class-1 parents the missing mask had locked out of the loss.

**Where it runs** — `round0/run_arjuna_v2.sh` (gated build→verify→train; no GPU is touched unless the label census passes) → `run_arjuna_v2_eval.sh` (ship to Amarel, 6 × 72 shards in two waves at the submit cap, pull back) → `run_arjuna_v2_agg.sh` (`arms_aj2.json` → `gate_aj2.json` + the difficulty × horizon table against every registered reference). CS jobs **203552–203557**, launched 15:00.

**Arms:** 3 seeds × {ranking ON, ranking OFF}, `RANK_LAMBDA = LOWER_RANK_LAMBDA = 0.1` (equal budget [USER]) → `AJ2_s{1,2,3}` / `AJ2NR_s{1,2,3}`. The aux fires on exactly three tiers — openers above setups+zeros, setups above zeros, level-0 skipped for want of anything lower — i.e. finish > setup > rest, with no fictional tiers (this file contains no guesses).

**What each contrast decides:**
- **v2-ON vs arm A / ARJ** — does a real floor achieve what 1.4% could not? Watch **V5**, stuck at 0.53–0.54 through every intervention to date.
- **v2-ON vs v2-OFF** — the sharper one. Every previous aux ablation ran on a file that was ~47% bounds, where regression had almost nothing to learn and the aux carried it by default. This is the first fair test of whether ordering supervision is *intrinsically* dominant or was compensating for degenerate labels. **Pre-registered prediction [Claude]: the gap narrows but the aux still wins**, because ranking optimises the top of the list while regression averages over ~68 cells in which one setup matters.

**⚠ Honest asymmetry, recorded BEFORE results:** zeroing *child* bounded cells is correct under hmax=2 (a failed push-2 has no successor). Zeroing *root* bounded cells asserts "this is not a setup" for **~5.5M cells** whose children were never resolved — some are genuine setups, so those labels are wrong. This is the move that regressed 2026-07-25 (sims-to-solve 46.0 → 53.8), differing in one respect: that run also zeroed **untried** cells, inventing facts about pushes nobody executed; v2 only zeroes cells that were simmed. **If v2 regresses, this is the first suspect**, and the follow-up arm is the split — child→0, root→masked — which separates the safe half from the risky half.

## v2 RESULT — canonical gate, 3 seeds, mean [min,max] (2026-08-08 16:40)

Aggregate `round0/gate_aj2.json` (spec `arms_aj2.json`), 432/432 shards, zero unmatched. Reference rows are the registered 3-seed pooled numbers.

| arm | labels | aux | 2p-hard@5 | 2p-hard@900 | 1p-hard@1 | 2p-hard s2s |
|---|---|---|---|---|---|---|
| θ₀ | base | on | 22.6 | 92.0 | 39.7 | — |
| arm A | +602 guesses | on | 27.7 | 91.5 | 38.0 | — |
| Bfix | +9,006 guesses | on | 28.9 | 87.1 | 41.8 | — |
| **BNG** | +9,006 guesses | on, guesses barred | **32.1** | 88.6 | 38.4 | — |
| ARJ v1 | 1.4% floor | on | 27.7 | 91.0 | **42.5** | — |
| **AJ2** | **47.6% floor** | on | 26.8 [24.8, 28.5] | 90.0 [86.9, 92.0] | 38.1 [35.8, 39.7] | 105.0 |
| **AJ2NR** | **47.6% floor** | **off** | 19.9 [17.5, 21.9] | 87.8 [86.9, 88.3] | 29.4 [28.4, 30.9] | 134.5 |

### Verdict 1 [numbers] — the floor hypothesis is FALSIFIED at FULL dose, not just at 1.4%

v2's 2p-hard@5 band [24.8, 28.5] contains arm A (27.7), ARJ v1 (27.7) and Bfix (28.9): **no gain, and no regression either.** BNG's 32.1 sits above v2's entire band and remains the best model in the line. 1p-hard@1 is *below* both ARJ v1 (42.5) and Bfix (41.8), and its band's top just touches θ₀ (39.7).

This closes the question the scope correction reopened. v1's null was dismissible as a 1.4% dose; **72× more floor changes nothing.** Giving the regression a real zero to predict is not what the ranker was missing.

### Verdict 2 [numbers] — the aux and the labels are SUBSTITUTES, and this is the session's real finding

The rank-off arms are where the dose shows up. Same ablation, two label regimes:

| labels | aux off | aux on | aux's marginal value |
|---|---|---|---|
| bootstrap guesses (`BfixNR` → `Bfix`) | 11.2 | 28.9 | **+17.7** |
| hard floor (`AJ2NR` → `AJ2`) | 19.9 | 26.8 | **+6.9** |

Real labels nearly **double** the no-ranking model (11.2 → 19.9 on 2p-hard@5; 15.2 → 29.4 on 1p-hard@1) and cut the aux's marginal contribution by ~60%. The earlier reading — "everything is ranking, the labels barely move anything" [USER 2026-08-08] — was true *only because the labels were degenerate*: with 96–100% of targets ≥0.8 the regression had nothing to learn, so the aux was carrying the model alone. Fix the labels and most of that gap closes by itself.

The aux still wins by ~7 points and the bands do not overlap ([24.8, 28.5] vs [17.5, 21.9]), so ordering supervision is **not** purely a workaround for bad labels — it contributes on its own. **The pre-registered prediction ("the gap narrows but the aux still wins") is confirmed on both halves.**

### Splits (difficulty × horizon) — required reporting

```
AJ2    1push  easy   n=2091  @1=96.9  @5=99.5  @30=100.0  @900=100.0  s2s=1.1
AJ2    1push  medium n=1263  @1=80.0  @5=96.4  @30=99.7   @900=100.0  s2s=1.8
AJ2    1push  hard   n=612   @1=38.1  @5=78.8  @30=94.6   @900=100.0  s2s=9.6
AJ2    2push  easy   n=1155  @1=0.1   @5=69.2  @30=90.7   @900=99.8   s2s=15.2
AJ2    2push  medium n=1464  @1=0.0   @5=50.2  @30=75.3   @900=97.7   s2s=43.7
AJ2    2push  hard   n=411   @1=0.0   @5=26.8  @30=53.3   @900=90.0   s2s=105.3
AJ2NR  1push  easy   n=2091  @1=97.7  @5=99.8  @30=100.0  @900=100.0  s2s=1.0
AJ2NR  1push  medium n=1263  @1=80.3  @5=95.8  @30=99.6   @900=100.0  s2s=1.9
AJ2NR  1push  hard   n=612   @1=29.4  @5=73.5  @30=92.2   @900=99.7   s2s=13.6
AJ2NR  2push  easy   n=1155  @1=0.2   @5=62.1  @30=86.9   @900=99.7   s2s=19.2
AJ2NR  2push  medium n=1464  @1=0.0   @5=43.2  @30=68.7   @900=97.9   s2s=58.4
AJ2NR  2push  hard   n=411   @1=0.0   @5=20.0  @30=40.6   @900=87.8   s2s=134.5
```

Two things to read off. **The aux's contribution is uniform across difficulty, not concentrated in the hard tail** — AJ2 beats AJ2NR by +7.1 / +7.0 / +6.8 points at 2push easy/medium/hard @5. And **on 1push-easy the aux is very slightly harmful** (96.9 vs 97.7 @1), the same easy-tier inversion the wall-clock campaign saw. Neither arm reaches θ₀'s 2p-hard@900 of 92.0.

### Plot — `round0/v2_success_vs_sims.png` (script `round0/plot_v2_curves.py`)

Success vs simulator calls, 2×4 horizon × difficulty, line = 3-seed mean, band = seed min–max. Encoding is deliberate: **colour = label regime** (blue bootstrap guesses, orange hard floor), **line style = aux** (solid on, dashed off). The substitution result is then legible without reading the legend — **the blue pair is far apart, the orange pair is close**, in every one of the eight panels. Random is the dotted grey reference.

Two things visible in the plot that the scalar table hides: the aux's benefit is concentrated in the **1–30 sim range** and closes by ~300 sims on every tier (it buys ordering, not reach), and on **1push-easy all four models are indistinguishable from each other** and only the random baseline separates — that tier is saturated and carries no signal about labels or loss at all.

### v2 AUC panel — `round0/auc_aj2.json`, 1,152 episodes, 3 seeds (script `round0/auc_compare.py`)

```
arm             V1        V2        F1        F2        V4        V5       V5m        V6   setup@1  finish@1
Bfix         0.796     0.790     0.848     0.902     0.866     0.527     0.563     0.786      56.2      69.9
BfixNR       0.797     0.784     0.761     0.735     0.847     0.625     0.693     0.707      59.9      46.0
ANR          0.810     0.791     0.774     0.736     0.869     0.616     0.685     0.688      60.7      44.1
BNG          0.798     0.797     0.837     0.902     0.880     0.538     0.582     0.786      55.2      69.6
ARJ          0.801     0.801     0.859     0.906     0.885     0.533     0.576     0.790      55.2      69.5
AJ2          0.785     0.792     0.826     0.882     0.900     0.543     0.593     0.760      55.1      64.9
AJ2NR        0.785     0.789     0.823     0.877     0.882     0.642     0.723     0.734      56.7      63.4
```

**1. V5 is confirmed immovable by labels — the pre-registered watch resolves.** `AJ2` V5 = 0.543 [0.529, 0.562], overlapping BNG [0.522, 0.551] and ARJ [0.505, 0.556]. Seven rank-on arms across three label regimes — bootstrap guesses, 1.4% floor, 47.6% floor — all sit in 0.527–0.543. A 72× label dose does not touch cross-board comparability. **This closes the label route to V5 for good; it is a loss-structure problem, and the board-ordering head is the only remaining lever.**

**2. `AJ2NR` posts the highest V5 ever measured — 0.642 [0.631, 0.648]**, above BfixNR (0.625) and ANR (0.616). Cross-board comparability is *best* with honest labels and no aux, and the aux costs a full 0.10 of it (bands nowhere near overlapping). The aux does not merely fail to help V5 — it actively suppresses it, exactly as the `log_softmax(dim=1)` shift-invariance predicts.

**3. The F2 collapse is CURED by labels, and this is the mechanism behind verdict 2.** Removing the aux used to destroy finish separation: 0.902 → 0.735. Under hard labels it does not: **0.882 → 0.877, bands overlapping.** So the aux was never teaching finish ordering *per se* — it was compensating for labels in which finish and setup were nearly the same number. Give the model a real 1.0 / 0.5 / 0.0 ladder and it learns that ordering from the regression alone. This is why `AJ2NR` (19.9) nearly doubles `BfixNR` (11.2).

**4. V4 improves while V5 does not — the order-statistic gap, isolated.** `AJ2` posts the best V4 in the line (0.900 vs ARJ 0.885, BNG 0.880): it beats the *typical* dead cell better than any model we have. But V5 — the same comparison against each dead board's **maximum** — is flat. With a median of 75 cells per dead board, the max is an extreme order statistic, and being better than average buys nothing against it. **This is the cleanest statement yet of what "cross-board weakness" actually is.**

**5. The aux's whole remaining value is SETUP ranking on HARD boards — and the pooled table inverts this.** Per-difficulty blocks (`easy`/`med`/`hard`) are in the same panel; the table above is the `all` row, which is dominated by easy+med (865 of 983 setup boards).

```
hard tier only          setup@1  finish@1      V2      F2      V5      V6   2p-h@2   2p-h@5
Bfix     bootstrap on      23.2      56.1   0.773   0.915   0.385   0.762     13.4      28.9
BfixNR   bootstrap OFF     17.5      32.5   0.722   0.747   0.483   0.679      5.8      11.2
BNG      bootstrap on      24.0      57.0   0.800   0.913   0.411   0.762     14.6      32.1
ARJ      1.4% floor on     24.6      52.6   0.802   0.919   0.404   0.766     14.1      27.7
AJ2      47.6% floor on    25.1      50.8   0.779   0.890   0.455   0.740     12.7      26.8
AJ2NR    47.6% floor OFF   14.4      49.8   0.755   0.888   0.535   0.705      4.9      20.0
```

**`AJ2NR` is not failing at finish — it is failing at setup, only on hard boards.** Finish is a tie (49.8 vs 50.8, F2 0.888 vs 0.890); setup hit@1 is **14.4 vs 25.1, a 43% relative loss.** The pooled row said the opposite (`AJ2NR` 56.7 *better* than `AJ2` 55.1) because per tier it goes **easy 80.8 vs 72.7 (+8.1, 377 boards) · med 52.9 vs 53.3 (tie, 488) · hard 14.4 vs 25.1 (−10.7, 118)** — it wins where boards are plentiful and loses where they are few, so the average inverts the truth. Same aggregation trap that retired the 0.583 anchor ([project_auc_reconciliation]).

Seed bands on the hard tier settle which of those two is signal: **setup@1 `AJ2` [22.0, 27.1] vs `AJ2NR` [11.0, 16.9] — disjoint**, while **finish@1 [48.8, 53.7] vs [47.9, 50.8] — overlapping**. The setup collapse is real; the finish tie is real.

The arithmetic closes on the independently verified identity `solve@2 ≈ setup@1 × finish@1`:

| arm | setup@1 × finish@1 | predicted 2p-hard@2 | actual |
|---|---|--:|--:|
| `AJ2` | .251 × .508 | 12.8% | **12.7** |
| `AJ2NR` | .144 × .498 | 7.2% | **4.9** |

**Mechanism.** Hard boards are *defined* by low setup density — a few valid setups among ~68 candidates. The listwise aux optimises the top of the list; the regression optimises mean error across all cells. When positives are rare the mean is dominated by negatives, so a regression can be numerically accurate and still bury the one setup. On easy boards positives are everywhere and getting the general level right suffices — which is exactly why `AJ2NR` *beats* `AJ2` there.

**This qualifies verdict 2.** Hard labels substitute for the aux on **finish** ranking (the cured F2 collapse) but **not for setup ranking on sparse boards**. That residual is the entire remaining 6.9 points, and it says the next lever is a loss that up-weights rare positives — not more labels.

> **⛔ CORRECTION [Claude 2026-08-08].** This section first read "the AUC panel does NOT explain v2's deploy gap" and floated V6 board-triage as an unverified hypothesis. **That was wrong**, and wrong for an avoidable reason: I read only the pooled `all` block when the panel had `easy`/`med`/`hard` all along. The panel explains the gap precisely. The V6 hypothesis is withdrawn — V6 does move (0.760 vs 0.734) but it is not the mechanism.

### The asymmetry flagged before launch is the surviving suspect

Zeroing *child* bounded cells is provably correct under hmax=2. Zeroing *root* bounded cells asserts "this is not a setup" for ~5.5M cells whose children were never resolved, and the flat-to-slightly-down result is exactly what that predicts: real information added on the child half, false labels added on the root half, netting to zero. **This is now the pre-registered next arm** — child→0, root→masked — and it is the only way to tell "the floor doesn't help" from "the floor helps, and 5.5M wrong labels cancel it out." Until it runs, verdict 1 should be read as *the floor as applied* rather than *any floor*.

## v3 and v4 — the ladder width, and the guesses restored [USER 2026-08-08 evening]

Two follow-ups, both single-variable against v2, all four cells sharing the 47.6% floor:

| file | setup label | guesses on censored children |
|---|---|---|
| `arjuna0v2_train.h5` | 0.5 | none |
| `arjuna0v3_train.h5` | **0.9** (the base value) | none |
| `arjuna0v4_train.h5` | 0.9 | **546,035 cells restored from Bfix, verbatim** |

**Why v3 exists.** v2 moved *two* things at once versus the base file — it added the floor AND narrowed setup 0.9 → 0.5. v3 keeps setup at 0.9, so v3-vs-base isolates the floor and v3-vs-v2 isolates the ladder width. Under the 51-bin HL-Gauss head that width is ~25 bins (v2) against ~5 (v3).

**Why v4 exists.** It completes the 2×2 at setup 0.9: `base/θ₀` (no floor, no guesses) · `v3` (floor) · `Bfix` (guesses) · `v4` (both). Guessed VALUES are copied from `aquaman0_train_Bfix.h5` rather than recomputed — they are a deterministic function of a fixed checkpoint, so copying reproduces arm B exactly at zero GPU cost. Verification it hit the right population: restored **exactly 546,035** cells, landing on Bfix's in-loss (280,793) and masked (7,862) counts to the digit, while non-guess labels stayed `{0, 0.9, 1.0}`.

**v4's aux-on arm runs with `NAMO_RANK_EXCLUDE_GUESS=1`, deliberately.** Measured on the v4 file, 256-row batch: **386 rank tiers without exclusion, 3 with** (`[0.0, 0.9, 1.0]`). Feeding the restored cells to the aux would recreate the 593-tier pathology — grading 4th-decimal model noise as certain ordering — and would confound "do guesses help?" with "reintroduce a known bug."

### Results — canonical gate, 3 seeds, mean [min,max]

| arm | setup | aux | 2p-hard@5 | 2p-hard@900 | 1p-hard@1 |
|---|---|---|--:|--:|--:|
| θ₀ | 0.9 | on | 22.6 | 92.0 | 39.7 |
| Bfix | 0.9 | on | 28.9 | 87.1 | 41.8 |
| **BNG** | 0.9 | on | **32.1** | 88.6 | 38.4 |
| ARJ (v1) | 0.9 | on | 27.7 | 91.0 | 42.5 |
| AJ2 | 0.5 | on | 26.8 | 90.0 | 38.1 |
| AJ2NR | 0.5 | OFF | 20.0 | 87.8 | 29.4 |
| AJ3 | 0.9 | on | 25.5 | 90.3 | 39.5 |
| AJ3NR | 0.9 | OFF | 15.1 | 84.7 | **11.6** |
| **AJ4** | 0.9 | on | **29.9** [28.5, 31.4] | 89.3 | 37.6 |
| AJ4NR | 0.9 | OFF | *(seeds 1–2 still training)* | | |

### Verdict 1 [numbers] — arm B's "labels on the unknowable hurt" was CONDITIONAL, not general

**AJ4 vs AJ3 — add the guesses, floor held fixed: 25.5 → 29.9 (+4.4), and AJ3 sits below AJ4's entire band.** The identical guessed values that cost arm B 4–5 points now *help*, because they are barred from defining rank tiers. **The guesses were never the problem; letting the ranker treat them as certain grades was.** This is the second time the same distinction has decided a result — BNG beat Bfix by exactly this mechanism.

**AJ4 vs BNG — add the floor, guesses held fixed: 32.1 → 29.9 (−2.2).** Consistent with everything else here. Caveat: AJ4's band [28.5, 31.4] overlaps BNG's 32.1 at the top, so BNG > AJ4 is suggestive, not established; the +4.4 is the solid claim.

### Verdict 2 [numbers] — LABEL SPACING governs the regression, and only when the aux is off

| labels | aux OFF | aux ON | aux's marginal value |
|---|--:|--:|--:|
| bootstrap (`BfixNR`→`Bfix`) | 11.2 | 28.9 | +17.7 |
| floor, setup 0.5 (`AJ2NR`→`AJ2`) | 20.0 | 26.8 | +6.9 |
| floor, setup 0.9 (`AJ3NR`→`AJ3`) | 15.1 | 25.5 | +10.4 |

**This RETRACTS the v2 reading.** v2's headline was "hard labels substitute for the aux — they nearly double the no-aux model and cut its marginal value 60%." Holding the labels honest and moving *only* the spacing collapses it again: `AJ3NR` posts **1p-hard@1 = 11.6, the worst number in the project**, below bootstrap-label rank-off (15.2) against random's 3.3. So what rescued v2's aux-off arm was **25 bins of separation, not label honesty**.

With the aux ON, spacing is irrelevant — AJ3 vs AJ2 F2 is .884 vs .882, finish@1 65.9 vs 64.9. **The aux makes the model insensitive to label spacing; without it, spacing IS the ordering signal.**

Mechanism note, stated as hypothesis: F2 compares finish (1.0) against dead (0.0) on child boards, and *both those labels are identical in v2 and v3*. So the spacing effect must act through the shared representation — a target distribution with mass at 0.9 and 1.0 compresses the model's high range differently than 0.5 and 1.0 — rather than through the comparison F2 measures directly. Untested.

## Open / next

- **Next lever is a rare-positive loss, not more labels [new 2026-08-08].** The aux's entire surviving contribution is setup ranking on hard boards (25.1 vs 14.4 hit@1, disjoint bands), and hard boards are exactly those with a few valid setups among ~68 candidates. Regression's mean-error objective cannot see rare positives; the listwise aux can. Anything that up-weights the rare positive directly — focal/ranking hybrids, positive-weighted CE, top-k losses — is now better motivated than another label scheme. Note this is *not* the 2026-08-02 read that "there are no negatives to down-weight, so it isn't a focal-loss problem": that was true of the **old** labels, where 96–100% of targets sat ≥0.8. `arjuna0v2_train.h5` is 47.6% zeros, so the class imbalance is now real and the objection no longer applies.
- **The cross-board hole is unaddressed by labels.** It needs a signal spanning boards: a board-ordering head ([EXP-2026-08-02-board-live-head](EXP-2026-08-02-board-live-head.md), labels already balanced 22,271 live / 19,282 dead), or a ranking competition list that is not `dim=1`. Within-episode comparability suffices — the deploy queue only ever holds one episode's boards.
- **Why finish hit@1 dropped** is unexplained and is the reason 2-push stayed flat. First thing to chase.
- **546,035 censored children** remain unresolvable from stored data. Grounding them means re-rooting: replay the parent push (1 sim, physics is deterministic), then sweep the child's untried remainder. Colossus stores no qpos and `chain_depth ∈ {1,2}`, so no grandchildren exist and the world-value route is closed without new collection.
- **Not attempted, deliberately:** the γ² grandchild bootstrap (needs states we never stored) and any re-collection.

## Log

- **2026-08-08 (PM) [Claude] v2 COMPLETE — floor falsified at full dose; the aux/label substitution is the real finding.** Six models trained and evaluated (432/432 shards), numbers in the v2 RESULT section. Two ops notes worth keeping. **(1) `--gres` on the sbatch command line is rejected by CS `unlimited`** ("Please do not specify the number of CPUs") — pick the GPU type with `--nodelist` and let the template's own `--gres=gpu:1` stand. **(2) Racing beat migrating.** The seed-1 pair landed on ilab1 (a4500, 6.7 min/epoch) while seed 2 got rlab2 (a100, 1.5 min/epoch) — 4.5×. Killing the laggards to move them would have forfeited 45 min of real progress on an unverifiable bet that a fast GPU was free (rlab2's GPUs are not visible from its login shell). Duplicating seed 1 onto rlab2/rlab7 and taking whichever COMPLETED first cost nothing and won: the a100 duplicate finished in **18:09** and cancelled the ilab1 original at 64 min. The rlab7 pair died at 54 s and didn't matter. **On a shared cluster with free capacity, race — don't migrate, and don't wait.**

- **2026-08-08 (PM) [Claude] 432 eval tasks died in 1 s with 0-byte logs; three stacked causes, each hiding the next.** Cost ~30 min. **(a) `NAMO_REPO` unset:** the template derived the checkout from `${BASH_SOURCE[0]}`, but **SLURM copies the batch script to `/var/lib/slurm/slurmd/job<N>/slurm_script`**, so `REPO` resolved to `/var/lib/slurm`. **(b) `set -e` + `source env.amarel.sh >/dev/null 2>&1`:** the resulting failure was discarded, leaving no trace anywhere. **(c) `NAMO_BINDINGS` unset:** `namo_bfix` has no `build_python`, and `BIND` is computed *before* the env is sourced, so past (a) it would have died on `ModuleNotFoundError: namo_rl`. Every earlier wave worked only because an interactive shell had both exported and `sbatch --export=ALL` carried them in; submitting over ssh does not. Template fixed to fall back to `SLURM_SUBMIT_DIR` (which survives spooling) and to stop silencing the `source`. **My own probes cost half that time:** I tested with `sbatch --wrap`, which runs under `sh`, where `source` does *not* search the current directory, and briefly concluded `env.amarel.sh` was missing when it was present. **Lesson: reproduce with the real interpreter and `set -x`, not with a convenience wrapper.**

- **2026-08-08 (PM) [Claude] eval waiter would have shipped epoch-0 checkpoints — caught before it ran.** The stage-2 chain gated on "6 `epoch*.ckpt` files exist AND the queue is empty". Both were already true minutes after launch: Lightning writes an `epoch000` file almost immediately, and the chain runs on **arrakis, which has no `squeue`**, so the queue test was vacuously 0. Had the script been running it would have evaluated six barely-trained models and produced a clean-looking, entirely false result table. It had never actually started, which is the only reason nothing was lost. Rewritten to gate on SLURM **job state** (`sacct -X` over ilab2, a submit host) and to abort unless all six report `COMPLETED`; parser dry-run verified against the live jobs. **General lesson: artifact-existence is not a completion signal for anything that checkpoints per epoch — poll the scheduler, not the filesystem.** (The `scaled-run` skill's "monitor by artifact count, not buffered logs" advice is about *liveness*; for *doneness* it is actively wrong here.)

- **2026-08-08 (PM) [Claude] SCOPE CORRECTION + v2 launched.** Measuring the bounded population showed v1 converted only **114,660 of 8,294,162** bounded cells (1.4%), and that aquaman arm A / Bfix converted 38,519 (0.5%) — i.e. *every* label intervention in this project has moved ≤1.4% of the cells that carry no value. Cause is a join limit: only the 26,023 Colossus setup roots have child boards in the raw shards, and **94% of bounded cells sit on d20-base rows with no child stored at all**. Also found the bounds are inverted: root cells are capped ≤0.81 (below setup grade, burying undiscovered setups) while child cells — where a failed push-2 is worth exactly 0 under hmax=2 — get the loosest cap in the system, ≤0.90, across **2.84M cells on the tier holding 88–94% of deploy pops**. The v1 verdict is therefore true only of a 1.4% dose and has been re-titled accordingly; it was overclaimed as written. **v2 [USER]:** opener 1.0 / setup 0.5 / every bounded cell 0.0 / untried masked — needs no linkage, reaches the whole file, takes zeros from 0.66% → **48.2%**. Six runs launched (3 seeds × ranking on/off, λ=0.1 both levels). Nothing deleted; v1 artifacts remain registered.

- **2026-08-08 [Claude] arjuna-0 built, trained, evaluated; floor hypothesis falsified on its own pre-registered test.** Build `arjuna_build.py` over 32 raw shards → 141,581 exact zeros (`arjuna0_train.h5`), verification gate passed (4/4 invariants), 3 seeds trained on CS SLURM (val_loss 1.6825/1.6811/1.6964 — *lower* than arm A's 1.7056–1.7230, so facts fit better than guesses), canonical eval on Amarel (216/216 shards, zero unmatched), AUC panel on westeros. Headline: **1push-hard@1 42.5 (best in line, bands [41.7,43.6] barely overlapping arm A), 2push flat, V5 0.533 unchanged.** Chain ran unattended and gated: training would not have started had the census failed.
