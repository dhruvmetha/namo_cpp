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

## Verdict [numbers] — the floor hypothesis is FALSIFIED

**Pre-registered before training** [Claude, 2026-08-07]: *"if the floor theory is right the gain shows up first in cross-board — V5 above its stuck 0.53 — and if V5 doesn't move, the theory is wrong regardless of what the solve rates do."*

**V5 = 0.533 against arm A's 0.543.** It did not move. Adding the first floor this project has ever had — 141,581 exact zeros at full weight — bought nothing in cross-board comparability.

So the standing explanation for the cross-board hole (no zeros → no bottom → dead boards float → best-of-70 beats a real setup) is **wrong**, and should not be repeated. The surviving explanation is mechanical and measured: the ranking auxiliary is `log_softmax(dim=1)` over a single board, hence shift-invariant per board, and it inflates dead-board maxima 0.625 → 0.720 (see the DC card's 2026-08-07 section). No label change touches that.

**What the facts did buy, consistently across both panels:** best 1push-hard@1 in the entire line (**42.5** vs θ₀ 39.7), best setup hit@1 hard (24.6), best V6 (0.790).
That fits a narrower mechanism than the one proposed: a proven zero says *this specific push fails* — direct evidence for opener ranking, which is exactly 1-push top-1 — while adding nothing about comparing one board to another, because arm A's guesses on those same cells already sat at median 0.029.

**2-push is flat** (27.7 vs 27.7 @5), and finish hit@1 hard *fell* 54.8 → 52.6. Since `solve@2 ≈ setup@1 × finish@1` (verified to ~1 pt across conditions), the better setup pick was cancelled by the worse finish pick.

## Open / next

- **The cross-board hole is unaddressed by labels.** It needs a signal spanning boards: a board-ordering head ([EXP-2026-08-02-board-live-head](EXP-2026-08-02-board-live-head.md), labels already balanced 22,271 live / 19,282 dead), or a ranking competition list that is not `dim=1`. Within-episode comparability suffices — the deploy queue only ever holds one episode's boards.
- **Why finish hit@1 dropped** is unexplained and is the reason 2-push stayed flat. First thing to chase.
- **546,035 censored children** remain unresolvable from stored data. Grounding them means re-rooting: replay the parent push (1 sim, physics is deterministic), then sweep the child's untried remainder. Colossus stores no qpos and `chain_depth ∈ {1,2}`, so no grandchildren exist and the world-value route is closed without new collection.
- **Not attempted, deliberately:** the γ² grandchild bootstrap (needs states we never stored) and any re-collection.

## Log

- **2026-08-08 [Claude] arjuna-0 built, trained, evaluated; floor hypothesis falsified on its own pre-registered test.** Build `arjuna_build.py` over 32 raw shards → 141,581 exact zeros (`arjuna0_train.h5`), verification gate passed (4/4 invariants), 3 seeds trained on CS SLURM (val_loss 1.6825/1.6811/1.6964 — *lower* than arm A's 1.7056–1.7230, so facts fit better than guesses), canonical eval on Amarel (216/216 shards, zero unmatched), AUC panel on westeros. Headline: **1push-hard@1 42.5 (best in line, bands [41.7,43.6] barely overlapping arm A), 2push flat, V5 0.533 unchanged.** Chain ran unattended and gated: training would not have started had the census failed.
