---
type: experiment
status: active
created: 2026-08-09
updated: 2026-08-09
metric: "TBD — gate: XB/RP must lift V5 ≥0.60 (from 0.543) with F2 ≥0.87 on the offline panel; canonical eval only for survivors."
tags:
  - experiment
  - loss
---
# Cross-board ranking + rank-pure — is the per-board list the V5 hole, and was regression ever needed?

**⛔ Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** The model is a ranker; the simulator is a perfect free verifier; success is fewer sim calls than random on every tier.

## Plain-language key [USER: define everything]

| term | plain meaning |
|---|---|
| **board** | one snapshot of the room; one H5 row; up to 300 candidate pushes (60 contacts × 5 depths) |
| **episode / family** | the root board + every child board its pushes spawned; the deploy heap only ever holds ONE episode's boards |
| **dead board** | a child board with no verified finish anywhere on it — every push on it is junk |
| **champion / impostor** | a dead board's HIGHEST-scored cell — the only cell of that board the heap ever compares against others |
| **burial** | count of dead-board cells scored above the best true setup of the episode (measured on exhaustive GT; hard-tier median ~150–200 for every round-1 arm) |
| **per-board rank aux** | round-0 loss term: softmax ordering of cells WITHIN one board (`dim=1`) — blind to everything cross-board |
| **XB** | round-1 arm: AJ2 base + the SAME softmax ordering over one batch-flat list (256 random boards, "strangers") — crowd-suppressor, all rivals pushed a little |
| **RP** | round-1 arm: rank-pure — regression and ceiling terms DELETED, per-board + batch-flat rank terms only, weights 1.0 |
| **MM** | round-2 arm: AJ2 base + margin-vs-MAX on the batch-flat list — hinge, full gradient on the single tallest rival until beaten by `MM_MARGIN=0.2`, zero on the rest — champion-hunter aimed at strangers |
| **EG** | round-2 arm: AJ2 base + the softmax ordering over the EPISODE FAMILY list (grouped sampler seats root+children together) — right classroom, soft penalty |
| **EGMM** | round-2 arm: AJ2 base + margin-vs-MAX WITHIN the family list — "your setup must beat the champion of every dead board in its episode"; the deploy duel as a loss |
| **RPB** | round-2 arm: RP + `RP_BRAKE=0.02` of the exact regression term — tests whether regression's only surviving job is keeping the score scale from stretching |
| **V4 / V5** | setup vs a TYPICAL dead cell (crowd) / setup vs each dead board's CHAMPION (max) — V4 can be a record while V5 sits at coin-flip, because max-of-70 is a tail statistic |
| **shovel vs grave** | within-board sharpness (how fast search recovers after entering a wrong board) vs cross-board burial (how many wrong boards outrank the setup) — independent axes, measured independently |

## Why now (evidence that forced this)

Within-board supervision is saturated: 7 rank-on arms across 3 label regimes all sit at V5 0.527–0.543, a 72× label dose moved nothing ([EXP-2026-08-08](EXP-2026-08-08-arjuna-hard-labels.md)).

The deploy residual is cross-board: 88–94% of hard-tier pops happen on child boards, ranked by a magnitude no loss supervises; the flat heap compares cells across boards while the aux's `log_softmax(dim=1)` is shift-invariant per board ([EXP-2026-08-02-2push-failure-audit](EXP-2026-08-02-2push-failure-audit.md)).

Literature anchor: Chrestien et al. NeurIPS 2023 prove search efficiency ⇔ perfect ranking over the WHOLE open list and beat regression everywhere with rank-only training; their path-only-list ablation (our per-board analog) clearly lost. Kim & Shimanuki CoRL 2019 hit the same two-scope problem in GTAMP and hand-coded the cross-state half — the learned version is untried.

## Hypotheses (H→E→V; drafted [Claude] 2026-08-09, user to veto/edit)

**H1 [Claude]: the per-board list is the V5 mechanism.** Adding a batch-flat cross-board ranking term (same certain-order construction, list = whole batch) removes per-board shift freedom and lifts V5 from 0.543 to ≥0.60 without F2 dropping below 0.87.

**H2 [Claude]: regression is scaffolding, not signal.** With both ranking scopes present, deleting the regression + ceiling terms entirely (rank-pure) matches the full loss on the gate. Watch item: score pile-up at the [0,1] endpoints (histogram meter) — if RP wobbles, regression's one surviving job is the optimization brake.

**H3 [Claude, deploy-conditional]: if V5 recovers but canonical deploy stays flat @30–300, the flat heap is convicted** as the binding constraint and the two-level scheduler inherits a proven mandate. Not a failure mode — a decision outcome.

## Arms

| arm | loss | seeds |
|---|---|---|
| `XB_s{1,2,3}` | AJ2 loss (regression + censored + per-board rank λ=0.1/0.1) **+ cross-board rank `XB_LAMBDA=0.1`** | 3 |
| `RP_s{1,2,3}` | ranking only: per-board (opener+setup) + cross-board (opener+setup), all weights 1.0, **no regression, no censored fence** | 3 |
| `AJ2_s{1,2,3}` / `AJ2NR_s{1,2,3}` | registered baselines — reused, NOT retrained | — |
| `BNG_s{1,2,3}` | best-ever bar (2p-hard@5 32.1) [USER 2026-08-09] — registered, reused for the panel and any canonical comparison | — |

RP checkpoints on validation ranking loss (its monitor cannot be regression — it doesn't train it). XB keeps the standard monitor for apples-to-apples ckpt selection vs AJ2.

## Locked defaults (one variable: the loss)

- Data: `/common/users/dm1487/scratch_namo/aquaman/round0/arjuna0v2_train.h5` (995M, the AJ2 file; zero guesses, no ceiling cells → EXCLUDE_GUESS and cap rules dormant on this file).
- Head 51-bin HL-Gauss [0,1], `RANK_TEMP=0.15`, 12 epochs, batch 256, lr 3e-4, 3 seeds — all matching AJ2.
- Parked behind the gate, deliberately: episode-grouped lists (needs root↔child linkage scout on the 26k Colossus setup roots), γ-native honest-cap labels, unbounded head, margin-shape loss, more bins.

## Plan

1. Unit test: two-board toy (live+dead board) — dead-board junk must receive downward gradient under the flat call, zero under the per-board call. Commit before run.
2. Smoke: 1 XB seed, 2 epochs, arrakis — loss decreases, no NaN, epoch time sane (one-big-softmax cost check).
3. Fleet: 6 runs (XB×3, RP×3) raced on the a100 boxes (~20 min each; box-sync + GPU check first).
4. Panel readout (`auc_compare_arms.py`), per-tier as always, + BNG column + score histograms. The V5 ≥0.60 / F2 ≥0.87 line is a pre-registered REFERENCE, not a kill-switch — **no accept/reject at this step [USER 2026-08-09: "don't accept/reject, we might be onto something"]**; smoke-level bugs are the only thing that stops an arm here.
5. Canonical eval overnight for ALL arms (Amarel pulls the pushed branch; `check_box_sync.sh` before launch), difficulty × horizon splits, registry entry on completion. Verdicts on the full numbers are the user's call.

## Outcome fork (pre-registered readings)

- V5 up, F2 held → mechanism real; next arms = episode-grouped sampler, and RP-vs-XB decides regression's fate.
- V5 flat → per-board float was not the cap; board-live head becomes the main route; no sampler is built.
- V5 up, F2 collapsed → one scalar refuses both scopes at this τ/λ → tune or split heads (AlphaZero shape).
- V5 up, deploy flat (post-canonical) → scheduler convicted with a controlled experiment.

## Ops notes

- No worktree: worktrees don't propagate to Amarel — pushed branches do. Develop on `feat/horizon-q-redesign`, push after every commit, Amarel pulls the branch for the canonical round only.
- No C++ changes → no rebuild, no `.so` sync concern.

## Round 2 — target the MAX, and the family (launched 2026-08-09 evening; user: "test everything sensible, list reasoning" )

**Evidence that forced round 2** (2-seed early panel + score autopsy, full tables in Log):

V5 flat under batch-flat softmax: XB 0.550 vs AJ2 0.543 (target was ≥0.60); RP 0.518. Yet RP posts the best V4 ever (0.915 all / 0.887 hard) — best-ever AVERAGE suppression of dead cells, near-worst on the MAX. The V5 hole is an order statistic, not average comparability — the third independent measurement saying so.

Autopsy (child-board score geometry): XB left dead-board maxima undeflated (0.39–0.42 vs AJ2 0.415); RP inflated the whole scale (spread 0.45→0.67, dead max 0.68) — the no-brake stretch, mild form. AJ2NR still owns the V5-friendly geometry (dead max 0.302).

**Mechanism 1 — softmax stalls before the tallest rival sinks.** The listwise CE has floor ≈ log(#positives) (positives share one probability pie); as positives dominate, TOTAL rival gradient shrinks toward zero, and the tallest dead cell keeps only a sliver. A margin-vs-max loss cannot stall: any rival above the gap gets full gradient — and "positive vs tallest rival" is literally the statistic V5 measures and the heap pops.

**Mechanism 2 — training never contained the pair V5 grades.** V5 compares a setup against dead boards of its OWN episode (same room, near-identical images — the hard negatives). Batch-flat lists are 256 random strangers; measured in the train H5: only 12,868/215,909 episodes (6%) have ≥2 stored boards (54,368 rows, 21%), and random batching almost never co-locates a family. The model aced strangers (V4 record) and never saw family.

**Arms (each = one mechanism, all else locked to round 1):**

| arm | change vs AJ2 base | mechanism |
|---|---|---|
| `MM_s{1,2,3}` | + batch-flat margin-vs-max term (hinge, margin `MM_MARGIN=0.2`, `MM_LAMBDA=0.1`) | 1 alone |
| `EG_s{1,2,3}` | + episode-grouped softmax term (family lists via grouped batch sampler) | 2 alone |
| `EGMM_s{1,2,3}` | + margin-vs-max WITHIN family lists — "your setup must beat the tallest cell of every DEAD board in the episode (children of failed pushes, dead-end children) and the junk cells of live children; finishes stay above it by tier" [phrasing sharpened by USER 2026-08-09] — the deploy objective verbatim | 1 + 2 |
| `RPB_s{1,2,3}` | RP + small regression brake (`RP_BRAKE=0.02` × exact HL-Gauss) | H2's remaining piece: brake-not-signal |

Margin 0.2 reasoning: bounded [0,1] head, live–dead gap measured ≈0.2 — the margin demands that gap against the tallest rival specifically; env knob for the follow-up sweep if needed.

Known limit, logged up front: the EG/EGMM terms fire only on the 21% of rows with stored siblings — per-term loss logging keeps their contribution visible; if the dose is too small the arm reads flat and says so.

Round-2 arms get the FULL canonical treatment like round 1 [USER 2026-08-09]: offline panel + score autopsy on completion, then the same Amarel canonical sweep (1push+2push, difficulty × horizon) beside AJ2/AJ2NR/BNG via `run_fleet_eval.sh`. No accept/reject at any readout; verdicts on the user's read of the full tables.

## Round 3 — rank-pure EGMM (launched 2026-08-09 late night, USER: "start this only for now")

`RPE_s{1,2,3}` (`train_q2_rankpure_egmm.py`): the fully value-free stack — RP's two softmax rank terms + EGMM's family margin-vs-max at `RPE_FAM=1.0`, NO regression anywhere, categorical labels only, monitor = RP's val rank loss. Rationale: if it lands near EGMM's profile, value machinery exits the project entirely. Pre-registered watch items: score-stretch (RP's, no brake) COMPOUNDING with the V6 board-scramble (EGMM's) — meters are the per-tier score histogram and V6; front-curve @2/@5 is where compounding would show.

## Round-1 RESULTS (canonical complete 2026-08-09 evening; 3 seeds pooled, 432/432 shards, zero unmatched)

| arm | 2p-h@2 | 2p-h@5 | 2p-h@30 | 2p-h@900 | 2p-med@5 | 1p-h@1 | 1p-h@5 | s2s-hard |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| AJ2 (ctrl) | 12.7 | 26.8 | 53.3 | 90.0 | 50.2 | 38.1 | 78.8 | 105.3 |
| AJ2NR | 4.9 | 20.0 | 40.6 | 87.8 | 43.2 | 29.4 | 73.5 | 134.5 |
| XB | 12.4 | 28.7 | 54.7 | 91.0 | 49.5 | 38.2 | 78.4 | 99.6 |
| RP | 12.9 | 24.3 | 50.9 | 89.3 | 50.3 | 36.9 | 80.9 | 104.6 |

References: BNG 32.1 @5 · Bfix 28.9 · A 27.7 · θ₀ 22.6 · random 1.7. Aggregate `round0/gate_xbrp.json` (spec `arms_xbrp.json`), panel `round0/auc_xbrp.json` (3 seeds).

Observations [numbers, no verdicts — user's call]: XB gains sit in the predicted mid-curve zone (+1.9 @5, +1.4 @30, −5.7 s2s) with V5 FLAT (0.456 vs 0.455 hard) — the deploy gain traces to finish ordering (offline finish@1 +2.3), not cross-board repair. RP (regression deleted) lands within ~2 pts of the full loss everywhere and takes best-in-table 2p-h@2 (12.9) and 1p-h@5 (80.9), against the historic no-aux collapse (4.9/20.0) — regression reads as largely replaceable at deploy, its absence costing ~2 mid-curve pts consistent with the un-braked stretch in the autopsy. H1's V5 mechanism remains open; that is round 2's target.

## Round-2 RESULTS (canonical complete 2026-08-09 night; EGMM_s1 eval trailing, 2-seed rows marked)

| arm | 2p-h@2 | 2p-h@5 | 2p-h@30 | 2p-h@900 | 2p-med@5 | 1p-h@1 | s2s-hard |
|---|--:|--:|--:|--:|--:|--:|--:|
| AJ2 (ctrl) | 12.7 | 26.8 | 53.3 | 90.0 | 50.2 | 38.1 | 105 |
| XB | 12.4 | 28.7 | 54.7 | 91.0 | 49.5 | 38.2 | 100 |
| MM | 13.4 | **31.6** [27.7, 38.0] | 53.3 | 89.3 | 50.4 | **39.4** | 108 |
| EG | 10.9 | 29.7 [26.3, 31.4] | **57.4** | 89.1 | 46.8 | 34.2 | 89 |
| EGMM (3s final) | 12.2 | 27.3 | 50.4 | 90.0 | 45.9 | 36.9 | 93 |
| RPB | 12.4 | 23.4 | 48.7 | 88.3 | **51.4** | 38.1 | 88 |

References: BNG 32.1 · Bfix 28.9 · random 1.7. Aggregates `gate_r2.json` (spec `arms_r2.json`), offline panel `auc_round2.json` (12 seeds), plots `r2full_success_vs_sims.png`.

Observations [numbers, no verdicts]:

**The 2×2 personality grid held at deploy.** MM (sharpener): best early curve of the round — @5 31.6 pooled, BNG-class WITHOUT BNG's guess labels, 1p-h@1 39.4 best-in-table; its famous 38.0 was the s2 seed, band [27.7, 38.0]. EG (thinner): most consistent BNG-class @5 ([31.4, 31.4, 26.3]), best @30 (57.4), s2s 89 — the crowd-thinning cashes mid-curve exactly as the burial analysis predicted. EGMM (digger): s2s 93 as 3 seeds (the flashy 81 was the 2-seed subset; still the round's cheapest digging beside EG) and the only robust offline V5 mover (0.492 3-seed) + best F2 (0.920), but worst front-curve — its V6 drop (board-vs-board scrambling) is the standing suspect. RPB: weakest at deploy (@5 23.4) — the 2% brake did NOT recover RP's mid-curve; the leash story is falsified at this dose.

**Offline meters anti-predicted the deploy podium** (burial/V5 crowned EGMM; sims crowned MM/EG) — front-curve tracks within-board sharpness (the @2 identity setup@1×finish@1 verified again: MM 0.252×0.536≈13.5 vs measured 13.4), while V5/burial improvements cash out as cheaper DIGGING (s2s 81–89 for the family arms), not earlier solves.

**Oracle probe [same night]: inputs are NOT blind.** Fresh 4-layer CNN separates live/dead children at held-out AUC 0.795 (climbing) — resolution-blindness falsified; the trained rankers' V6 0.74–0.79 ≈ a 6-epoch toy, so liveness headroom exists and the missing-children collection is well-founded. Probe log `probe_livedead.log`, script in session scratchpad (to be landed in scripts/experiments).

**Data census that motivates the collection:** children 16% of training rows vs 94% of deploy pops; dead children of non-setup pushes ~absent (children exist only under the 26k Colossus setup roots; "94% of bounded cells sit on rows with no child stored"). Bfix/BNG added LABELS to these same rows, never new boards — no label scheme could touch this hole.

## Log

- 2026-08-09 late [Claude, USER-requested] BNG re-evaluated from original ckpts on tonight's stack, fresh dirs `BNGre_s*` (registered artifacts untouched): @2 14.6/@5 31.9/@30 55.5/@900 89.3 vs registered 14.6/32.1/55.7/88.6 — reproduction within sim jitter, eval stack drift-free, all cross-campaign comparisons clean. Aggregate `gate_bngre.json`.

- 2026-08-09 [Claude] Card created; design discussion (loss structure, literature deep-read, bounded-vs-unbounded, weight semantics) in session `ranking_loss`. Code next: XB reshape + RP subclass + unit test.
- 2026-08-09 evening [Claude] Round-1 trained (6/6), panel + autopsy + canonical COMPLETE (tables above). Round-2 fleet (MM/EG/EGMM/RPB × 3) launched after 2-epoch smokes passed; canonical for round 2 queued on completion.
