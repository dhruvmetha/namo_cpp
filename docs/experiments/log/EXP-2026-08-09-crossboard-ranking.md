---
type: experiment
status: active
created: 2026-08-09
updated: 2026-08-12
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

### ⭐ FAMILY CORPUS VERDICT (2026-08-11, first seed — THE result of the card)

**EGMMF_s1 (EGMM loss on the family corpus, 1.23M rows / 78% children): the V5 wall FELL.** Offline hard: **V5 0.455→0.492→0.721**, setup@1 28.0 (above the all-time 25–27 ceiling), finish@1 55.2, F2 0.898. Deploy [CORRECTED at 3 seeds]: @900 per seed [94.9, 89.0, 88.1], pooled **90.7** — the single-seed 94.9 was band-top; the record claim is RETRACTED (θ₀'s 92.0 stands). Pooled EGMMF ≈ par reach with the front-curve cost; RPEF deploys poorly (@5 15.3/@900 85.6) — offline wall-fall did not cash at deploy under the 85%-children mix. Trade: front-curve down (1p-h@1 23.9, @5 23.7) — the 85%-children mix starved root-board supervision; a sampler-ratio fix (blend old-corpus roots), not a mystery. Pre-registered branch: STRONG verdict → the data hypothesis is crowned; V5 was hunger, as the elimination argument + plates QED said. Gen-2 recipe: proven-label exhaustive corpus (COLLECTED: 39,563 room-shards, lie-rate → ~0) + root/child rebalance — deploy conversion of the offline breakthrough rides on it. **BANDS CONFIRM (3 seeds each): EGMMF V5 0.695 [0.667, 0.721] — band floor clears the old all-time best by +0.13. RPEF V5 0.685 [0.662, 0.708] — the VALUE-FREE stack breaks the wall identically: the un-burial needed data, not anchors.** Known mix trades (F2 ~0.86, front-curve soft, RPEF setup@1 16.7) → gen-2 = root/child rebalance + proven labels. Canonical rows for remaining seeds folding in.

## Family corpus — pre-verdict data-quality audit (2026-08-11, BEFORE results by design)

Audit-slice lie-rate MEASURED: **12.8%** of capped-dead children are secretly live (211/1,647 checked over 246 episodes matched between the capped base and the exhaustive 2% audit) — in the predicted 10–15% band. Branch readings pre-registered: partial verdict → escalation pass cleans a known 12.8% of the duel diet; strong verdict → headroom remains; null verdict → data cannot carry the blame at 12.8%, the hypothesis takes the hit. Also on record: rendered dose = 5.5 children/episode (re-render from same pkls is the cheap dose-increase), champion selection = old-model scores guarded by 2 random dead/episode.

## Round-3b RESULTS (complete 2026-08-10 pre-dawn) — every axis closed

**RPL2 (unbounded, margin 0.2 = 0.41σ): FULL RECOVERY + deploy twins with RPE** — offline V5 0.467/F2 0.906/V6 0.719; deploy @5 26.0 vs RPE 25.9, all cells within band. RPL's entire failure attributes to ONE constant (margin 2.2σ). **Boundedness axis CLOSED: free choice; keep the simpler bounded head; size margins to the adopted scale (σ-units) if ever unbounded.** RPL itself deploys functional (@5 21.7, reach 90.3) — damage LOCALIZED to family-type comparisons (offline panel wrecked, root skills intact): loss pathologies localize to the comparison classes that generate them.

**RPEA (binary plates, USER-designed): the information-limit QED.** V5 0.429 ≈ RPE 0.431 (pre-registered lift absent); autopsy: dead-max 0.611→0.573 AND live-max 0.696→0.653 — **rigid −0.04 translation, gap unchanged (0.085→0.080)**. An ABSOLUTE anchor on dead cells cannot separate what the features cannot distinguish — it pushes live near-twins down equally. With this, every force class (softmax, hinge, plates) + 3 label regimes + 2 heads have failed to cleave the live/dead gap: **V5 is feature-knowledge-limited; the family corpus is the only door** (and the probe's 0.795 says the door is real). setup@1 19.9 — mild floating-setup sag, watch line grazed. Canonical (3s): @2 9.9 / @5 26.0 / @900 91.0 (ties ledger-best reach) / s2s 122 — a healthy RPE-class deploy despite the absent V5 lift; the plates cost nothing and prove everything. RPM canonical: @5 20.6 — the sag carries to deploy.

**RPM (hole 4): sharpener does NOT survive anchor removal** — setup@1 16.5 vs anchored MM's 25.2. The stranger-hinge's within-board pressure needs the anchor.

**RPG/RPEG canonical confirmed the grind at deploy:** @2 4.3/2.6, s2s 132/144 — offline collapse fully mirrored.

### Round-3 RESULT (complete 2026-08-10 early)

`RPE` 3 seeds: offline V5 0.431/V6 0.720/F2 0.916 hard; deploy 2p-h@2 11.4 / @5 25.9 / @30 52.1 / **@900 90.9** / s2s 104. **Both pre-registered tripwires CLEAN**: no histogram walls (top bins ~0.3% mass — the margin self-stops, Kim's lineage vindicated) and V6 0.720 — BETTER than parent EGMM's 0.685, so removing regression relieved rather than compounded the board-scramble. RPU (unbounded head) therefore does NOT fire — its trigger never appeared; the bound is not binding. Verdict-shaped observation [no accept/reject]: the fully value-free stack (categorical labels, ordinal losses only) deploys ≈ the AJ2 control with second-best reach in the ledger. Aggregate `gate_rpe.json`, panel `auc_rpe.json`.

## Model mini-registry [USER: keep every model + location here]

All under `$NAMO_SCRATCH/aquaman/round0/` (CS); ckpts `models/<ARM>_s{1,2,3}/checkpoints/epoch*.ckpt` (best-val = first alphabetically), eval dirs `eval_bfix/<ARM>_s*/{1push_hmax2,2push}/`, Amarel ckpt copies `/cache/home/dm1487/aquaman0/ckpts_bfix/`.

| arm | script + key env | gate json | panel json | status |
|---|---|---|---|---|
| XB | `train_q2_rankaux.py` `XB_LAMBDA=0.1` (+RANK 0.1/0.1) | `gate_xbrp.json` | `auc_xbrp.json` | complete |
| RP | `train_q2_rankpure.py` (defaults) | `gate_xbrp.json` | `auc_xbrp.json` | complete |
| MM | `train_q2_round2.py` `MM_LAMBDA=0.1` | `gate_r2.json` | `auc_round2.json` | complete |
| EG | `train_q2_round2.py` `EG_LAMBDA=0.1 NAMO_GROUP_EPISODES=1` | `gate_r2.json` | `auc_round2.json` | complete |
| EGMM | `train_q2_round2.py` `EGMM_LAMBDA=0.1 NAMO_GROUP_EPISODES=1` | `gate_r2.json` | `auc_round2.json` | complete (s1 = 3rd submission; 2 OOM crashes on oversubscribed ilab GPUs) |
| RPB | `train_q2_rankpure.py` `RP_BRAKE=0.02` | `gate_r2.json` | `auc_round2.json` | complete |
| RPE | `train_q2_rankpure_egmm.py` (margin 0.2, span [0,1]) | `gate_rpe.json` | `auc_rpe.json` | complete |
| ~~RPU~~ | wide-span bridge arm | — | — | CANCELLED before fleet [USER 2026-08-10: "only RPL"] — superseded by the honest head |
| RPEA | `train_q2_rankpure_egmm.py` `RPE_ANCHOR=0.1` — BINARY PLATES [USER 2026-08-10]: regress ONLY extremes (openers→1, dead→0), setups UNANCHORED, floating between plates via rank/hinge. Motivation: absolute non-fading down-force on dead champions (the -b-bug evidence: zeros-drag removal collapsed hard@1, so junk-pins provably suppress); no setup target → grind-immune + zero spacing engineering. Pre-reg: V5 > RPE's 0.431, dead-max deflated, setup@1 ≥21. | `gate_rpea.json` | `auc_rpea.json` | launched 2026-08-10 |
| RPM | `train_q2_rankpure_egmm.py` `RPE_FAM=0 RPE_MMSTR=1.0` — grid hole 4 [USER]: MM's stranger-hinge value-free; does the sharpener (MM 31.6 @5 anchored) survive no-anchor? | `gate_rpm.json` | `auc_rpm.json` | launched 2026-08-10 |
| RPL2 | `train_q2_rankpure_linear.py` `RPE_MARGIN=0.2` — the margin-recalibrated retry [USER 2026-08-10]: 0.2 ≈ 0.4σ of RPL's adopted scale, the demand ratio RPE proved safe. Isolates the HEAD axis properly (RPL-1.0 confounded head with miscalibrated margin). Pre-reg: ≈RPE closes boundedness as a free choice; >RPE earns unbounded real estate; still broken indicts free-scale dynamics themselves. | `gate_rpl2.json` | `auc_rpl2.json` | launched 2026-08-10 |
| RPL | `train_q2_rankpure_linear.py` (LINEAR 5-out head, margin 1.0, leash 1e-3, no bins/no regression — the Chrestien configuration + hinge-on-max + leash) | `gate_rpl.json` | `auc_rpl.json` | launching 2026-08-10 |
| BNGre | BNG ckpts re-evaluated (drift check) | `gate_bngre.json` | (= `auc_bng.json`) | complete |
| R1 | `train_q2_round2.py` `EGMM_LAMBDA=0.1 NAMO_GROUP_EPISODES=1 NAMO_ROOT_FRAC=0.5` on `family0_train_v2.h5` — 2×2 ratio dial | `gate_2x2_r1.json` | `auc_2x2.json` | complete 2026-08-11 |
| R2 | same recipe, `NAMO_ROOT_FRAC` unset, on `family1_train_v1.h5` (exhaustive proven labels) — 2×2 label dial | `gate_2x2_r2g2.json` | `auc_2x2.json` | complete 2026-08-11 |
| G2 | same recipe + `NAMO_ROOT_FRAC=0.5` on `family1_train_v1.h5` — both dials | `gate_2x2_r2g2.json` | `auc_2x2.json` | complete 2026-08-11 |
| EGMMF5 / R15 | EGMM recipe + `NAMO_GAMMA=0.5` on `family0_train_v2.h5` (R15 adds `NAMO_ROOT_FRAC=0.5`) | `gate_wladder_b.json` | `auc_wladder_b.json` | complete 2026-08-12 |
| R25 / G25 | EGMM recipe + `NAMO_GAMMA=0.5` on `family1_train_v1.h5` (G25 adds `NAMO_ROOT_FRAC=0.5`) | `gate_wladder_a.json` | `auc_wladder_a.json` | complete 2026-08-12 |
| **HY / HY5** | EGMM recipe on `hybrid_train_v1.h5` (old-corpus roots + family0 children; HY5 adds `NAMO_GAMMA=0.5`) — **1p 42.0/42.8, @900 94.9/96.3, V5 0.690 (HY): the campaign result** | `gate_hybrid.json` | `auc_hybrid.json` | complete 2026-08-12 |
| **HY5U** | HY5 + `NAMO_UNREACH_W=0.1` (unreachable cells = exact zeros, regression only) — **best model ever trained here: @5 39.5, @900 97.5, 1p 45.3, setup@1 32.2, finish@1 60.1** | `gate_hy5u.json`, `gate_common.json` | `auc_hy5u.json` | complete 2026-08-12 |
| HY5U3 | same, `NAMO_UNREACH_W=0.3` — dose probe (1 seed) | (pending 1 shard) | `auc_hy5u.json` | eval refilling |

Baselines reused, never retrained: AJ2/AJ2NR (`gate_aj2.json`), BNG (`gate_bng.json`), θ₀/random (registry). Full registry rows: [horizon_q_model_registry.md](../horizon_q_model_registry.md).

RPU design note [USER: margin 1 for both]: margin 1.0 on the bounded [0,1] head is degenerate (equals the whole range), so the pair is RPE at its natural 0.2/[0,1] (banked) vs RPU at 1.0/[-10,10] — same 51-bin head affinely re-spanned, 20 units of headroom, zero plumbing change, deploy order-preserving. Tests whether hard-tier family duels (RPE's weak spot, V5 0.431) improve with room for gaps beyond the unit box.

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

## Isolation 2×2 (launched 2026-08-11, USER: "let's isolate properly" / "Go")

EGMMF won offline (V5 0.695) but cratered 1p-h@1 (38→24) and didn't convert at deploy. The family corpus changed THREE dials at once vs the old corpus: (1) root share of rows 84%→15% (root gradient ÷5.7); (2) root sweep depth cap 20→12 tried pushes (`finish_topk_cap`), rest censored; (3) added 1.045M child boards whose capped "dead" labels lie 12.8% of the time (audit-measured). Loss is already controlled: same EGMM recipe on the old corpus scored 1p-h@1 36.9. This grid separates the remaining two axes — root **ratio** and label **truth** — one dial per cell.

| cell | train H5 | labels | root share | knob | status |
|---|---|---|---|---|---|
| EGMMF | `family0_train_v2.h5` (1.23M rows) | capped-12, lie-rate 12.8% | 15.2% | — | complete (baseline) |
| R1 | `family0_train_v2.h5` | capped-12 (same lies) | 50% via sampler | `NAMO_ROOT_FRAC=0.5` | training |
| R2 | `family1_train_v1.h5` (512k rows) | PROVEN (exhaustive, censored=0.0) | 15.7% natural | — | training |
| G2 | `family1_train_v1.h5` | PROVEN | 50% via sampler | `NAMO_ROOT_FRAC=0.5` | training |

All cells: `train_q2_round2.py` `EGMM_LAMBDA=0.1 NAMO_GROUP_EPISODES=1`, 12 epochs, seeds 1-3, ckpts `round0/models/{R1,R2,G2}_s{1,2,3}/`. Jobs 206702-206710 (CS, NFS reads — `NAMO_STAGE_SHM=0`; today's estate-wide /dev/shm purge killed two staged fleets in ~40s, plus one NFS-permission node failure; all three incidents were cluster-side, logged below).

**The rebalance mechanism** (`NAMO_ROOT_FRAC`, commit 86b9414): appends duplicated SINGLETON root families to each epoch until roots reach the target fraction. `_family_lists` needs ≥2 rows, so singletons never enter the family loss — the cross-board terms that broke the V5 wall are byte-identical; only per-board base-loss exposure changes. Smoke-verified: exposure meter 0.152→0.500, measured batch stream 49.5% roots, epoch 1.23M→2.08M rows.

**Gen-2 corpus** (`family1_train_v1.h5`, built+verified 2026-08-11): 511,657 rows = 80,551 roots + 431,106 children (44.4% live / 55.6% dead — mirrors family0's 46/54); `finish_sweep_censored` fraction **0.0** (label proof); per-row chunks + lzf; md5-verified CS copy. Amarel build: 200-shard array 13 min wall, pool 2.5 min (`pool_family_h5.py`). **Known confound riding the label dial:** exhaustive sweeps cost more per episode, so family1 has HALF the episodes of family0 (80.5k vs 188k roots) — if R2/G2 underperform, corpus SIZE and label truth are entangled; if they win, size can't be the explanation.

**Pre-registered readings** (branch logic fixed before results): R1 recovers 1p-h@1 toward ~38 with V5 held ≥0.66 → crater was root starvation, ratio dial confirmed. R1 stays low → starvation insufficient, suspicion moves to root label depth/lies. R2 lifts V5/setup@1 over EGMMF at SAME 15% mix → lies were biting offline too. G2 = the product; its deploy curve vs θ₀ 92.0 @900 and MM's 39.4 1p-h@1 is the conversion test. Readouts: offline panel (V5/V6/F2/setup@1/finish@1 via `eval_auc`) + full canonical deploy (difficulty × horizon), no accept/reject at any readout [USER standing rule].

### 2×2 RESULTS (complete 2026-08-11 night; canonical 432/432 + 216/216 shards, 3 seeds pooled; hard tier)

| arm | V5 | V6 | setup@1 | finish@1 | 2p-h@2 | 2p-h@5 | 2p-h@30 | 2p-h@900 | s2s-h | 1p-h@1 | 1p-h@5 | 2p-m@5 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| R1 (rebalance, lying big corpus) | 0.606 | 0.654 | **26.8** | 49.3 | 9.9 | 24.3 | 45.8 | 88.1 | 123.8 | **29.4** | 67.5 | 44.0 |
| R2 (honest small corpus, natural mix) | 0.539 | 0.692 | 11.6 | 44.5 | 6.6 | 19.1 | 39.4 | 88.9 | 133.8 | 21.8 | 64.6 | 38.2 |
| G2 (honest + rebalance) | 0.564 | 0.690 | 20.6 | 51.4 | 10.5 | 20.6 | 44.9 | 88.7 | **114.2** | 25.5 | 67.0 | 43.0 |
| EGMMF (baseline cell) | **0.682** | 0.642 | 23.8 | 49.2 | 11.0 | 24.6 | 41.8 | **90.7** | 124.4 | 24.7 | 60.1 | 36.5 |
| AJ2 (old-corpus control) | 0.455 | — | — | — | 12.7 | 26.8 | 53.3 | 90.0 | 105.3 | 38.1 | 78.8 | 50.2 |

Aggregates `gate_2x2_r2g2.json`, `gate_2x2_r1.json`; panels `auc_2x2.json` (+ early `auc_r2arm_early.json`, `auc_g2arm_early.json`); eval dirs `eval_bfix/{R1,R2,G2}_s*/`. Family-campaign gates share 42/36 unmatched episodes (hard n 591/350-354) vs AJ2-era matching (612/411) — within-campaign comparisons clean.

**Readings [numbers, no verdicts — user's call]:**

1. **Neither dial (nor both) recovers 1-push.** Ratio dial is real but small and consistent: +4.7 on the big corpus (24.7→29.4), +3.7 on the small one (21.8→25.5). Best family-corpus 1p-h@1 (29.4) still sits 8.7 under the old-corpus class (~38). The pre-registered "R1 stays low" branch fired: exposure was NOT the binding constraint.
2. **Honest labels: V6 up, V5 down at this size.** Label truth improved live-board-vs-dead-board separation (V6 0.69 vs 0.64) but both honest cells sit far under EGMMF's V5 0.682 — the V5 wall-fall tracks corpus volume (1.045M vs 431k children), not label truth. Entanglement (half the episodes) was pre-flagged; R1 partially de-confounds: it shows even the big corpus loses V5 under rebalance.
3. **Rebalance taxes V5** (0.682→0.606 on the same big corpus): extra root singleton batches dilute the family-loss gradient share per epoch, and best-val lands at epoch 3 (fewer family passes). Setup@1 moves opposite (23.8→26.8, band-top 27.1 — new all-time). V5 and setup@1 are NOT the same axis: one is cross-board altitude, the other within-board ordering.
4. **Where the fixes DID cash:** medium-2p @5 44.0/43.0 (R1/G2) vs EGMMF 36.5 — the rebalanced arms are the best medium-tier family models; G2 digs cheapest (s2s 114.2).
5. **The un-isolated dial is now prime suspect for the residual 1p gap: root sweep depth.** Family roots were swept at cap 12 (vs d20's 20 + Bfix/BNG label passes); no 2×2 cell varied this. The natural next cell is HYBRID: old-corpus root rows (deep exhaustive 1p labels) + family child boards — tests root-label depth without new collection.

## Hybrid + wide-ladder campaign (launched 2026-08-12, USER: "Goi" / "Retrain with {0, 0.5, 1}")

Two follow-ups to the 2×2, 18 models total, same EGMM loss (hinge-vs-max in family + base) everywhere:

**HYBRID (root-content dial, the one the 2×2 never varied):** `hybrid_train_v1.h5` = old-corpus rows (257,409; d20-deep 1p labels; setup tier harmonized 0.5→0.9 at build — mixed encodings would tier-split the rank losses) + family0 children (1,045,250). 1,302,659 rows; 215,856 roots. 66% of family child episodes share (xml, object_id) with an old-corpus row → smoke measured **78,196 families joining an old root with family children in one list**. Builder `scripts/pipeline/build_hybrid_h5.py`. Pre-registered: 1p-h@1 → ~38 AND V5 ~0.68 = root-content + volume stories both confirmed, recipe settled; 1p recovers but V5 sags = trade persists, tune mix; neither = the family-corpus direction has its answer.

**WIDE LADDER {0, 0.5, 1} (USER):** all four 2×2 cells + hybrid retrained with `NAMO_GAMMA=0.5` (load-time setup remap; zero rebuild). Rationale: under the 0.9 ladder the opener-setup label gap (0.1) is SMALLER than the hinge margin (0.2) — regression and hinge fight; under 0.5 every tier gap (0.5) clears the margin. Arms: EGMMF5/R15/R25/G25 (grid cells at 0.5) + HY (hybrid at 0.9) + HY5 (hybrid at 0.5). Jobs 206939-206950 + 206959 (R25_s2 resubmit — /dev/shm purge ate a DataLoader semaphore on rlab4; purge now also kills loky sems, not just staged files) + 206968-206973. Eval: three auto-waves (A=R25+G25, B=EGMMF5+R15, C=HY+HY5), each offline panel (`auc_wladder_a/b.json`, `auc_hybrid.json`) + canonical fleet.

### CAMPAIGN RESULTS (complete 2026-08-12; canonical, 3 seeds pooled, hard tier)

| arm | V5 | setup@1 | finish@1 | 2p-h@2 | 2p-h@5 | 2p-h@30 | 2p-h@900 | s2s-h | 1p-h@1 | 1p-h@5 | 2p-m@5 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **HY** (hybrid, 0.9) | **0.690** | **27.1** | 52.3 | 12.4 | 23.7 | 50.6 | **94.9** | **90.2** | **42.1** | 76.1 | 45.1 |
| **HY5** (hybrid, 0.5) | 0.616 | 24.6 | 52.3 | **13.6** | 28.5 | 50.6 | **96.3** | 98.3 | **42.8** | **81.4** | **51.4** |
| EGMMF5 (family0, 0.5) | 0.643 | 23.2 | 52.3 | 8.8 | 23.7 | 50.3 | 91.5 | 130.8 | 30.6 | 67.7 | 41.7 |
| R15 (family0, 0.5 + RF) | 0.587 | 20.9 | 48.0 | 8.2 | 23.2 | 44.4 | 89.5 | 128.6 | 29.9 | 67.3 | 44.4 |
| R25 (family1, 0.5) | 0.462 | 15.8 | 47.6 | 10.2 | 18.4 | 40.1 | 88.1 | 136.1 | 24.7 | 65.0 | 41.0 |
| G25 (family1, 0.5 + RF) | 0.481 | 20.0 | 50.3 | 10.7 | 23.7 | 48.6 | 88.7 | 115.1 | 26.6 | 71.9 | 46.7 |
| AJ2 (control) | 0.455 | 25.1 | 50.8 | 12.7 | 26.8 | **53.3** | 90.0 | 105.3 | 38.1 | 78.8 | 50.2 |
| MM (old-corpus champ @5) | 0.439 | 24.6 | **53.7** | 13.4 | **31.6** | 53.3 | 89.3 | 108.5 | 39.4 | 80.4 | 50.4 |

**Seed bands (retraction-proof):** HY 1p-h@1 [39.6, 41.7, 44.7] — band FLOOR above AJ2's 38.1; @900 = 94.9 on ALL THREE seeds. HY5 1p-h@1 [37.6, 44.7, 46.2]; @900 [94.9, 96.6, 97.5] — floor beats the all-time θ₀ 92.0 by +2.9. (HY_s3 1push refilled shard 1 of 40 post-pull — final 432/432 numbers: HY 1p-h@1 42.1 / @5 76.1, table updated; HY5 432/432 complete from the start.)

**Readings [numbers, no verdicts]:**
1. **Both pre-registered hybrid bars CLEARED, seed-robust: the campaign's central claim is confirmed.** Root-content (d20-deep labels × 216k roots) fixes 1p (24.7→42.0, EXCEEDING the old corpus's 38.1) while family children keep V5 at 0.690. No trade — the two skills coexist in one corpus.
2. **@900 94.9-96.3 = new all-time reach by a wide margin** (previous record θ₀ 92.0; band floors clear it). The family thesis finally cashed at deploy — on the corpus that ADDS children to the old corpus instead of replacing it.
3. **Ladder effect is corpus-dependent and real:** on family0 the {0,0.5,1} ladder alone bought +5.9 1p / +8.5 @30 / 91.5 @900 with V5 held (EGMMF5); on family1 it traded V5 down for small deploy gains; on the hybrid it trades V5 (0.690→0.616) for reach (@900 96.3), sharpness (@2 13.6), and medium tier (51.4). Balance fix (`NAMO_ROOT_FRAC`) is NOT additive with the ladder (R15 ≈ EGMMF5 minus V5) — rebalance is superseded by better root content.
4. **Remaining old-corpus edges:** MM keeps 2p-h@5 (31.6 vs HY5's 28.5) and AJ2/MM keep @30 (53.3 vs 50.6) — the early-mid hard curve is the surviving gap; everything else now belongs to the hybrids.
5. Gates `gate_hybrid.json` (+`gate_hybrid_perseed.json`), `gate_wladder_a/b.json`; panels `auc_hybrid.json`, `auc_wladder_a/b.json`.

## HY5U — unreachable cells as exact zeros: the campaign's largest deploy jump (2026-08-12) [USER: "force unreachable cells to be dead"]

**The change.** Deploy only ever scores REACHABLE pushes (`candidates()` in `scripts/sandbox/eval_bestfirst.py` builds its pool from `rank_first_pushes_h2`), so unreachable cells had always been masked out of training entirely (`loss_mask = value_mask * r_mask`). `NAMO_UNREACH_W` (commit on `train_q2_round2.py`) puts them back as **exact zeros in the REGRESSION ONLY**, at fractional weight. Two guardrails, both load-bearing: (1) they are barred from every ranking list via a separate binary `rank_mask` (set for train AND val through `on_after_batch_transfer`) — an unreachable cell must never become the family hinge's tallest rival nor pad a softmax denominator; (2) weight 0.1, because unreachable cells outnumber labeled ones ~3.4:1 (smoke: 225 unreachable at 0.1 vs 64 labeled at 1.0 = 26% of loss mass). Arms: `HY5U_s{1,2,3}` (w=0.1, hybrid corpus + `NAMO_GAMMA=0.5`), `HY5U3_s1` (w=0.3 dose probe).

**Cross-campaign comparability fix, applied FIRST.** Family-campaign evals and the arjuna/BNG-era evals ran different episode lists (measured: 32 episodes in AJ2 that HY5U never evaluated, 12 the other way; 1012 vs 992 two-push). `aquaman_agg.py` scores each arm on its own shards, so those solve@k rates were different populations. New tool `scripts/rl_loop/aquaman_agg_common.py` intersects every arm's episode set and re-bins — **common set 1314 one-push / 980 two-push, identical hard-tier n=591/354 for all arms.** Every number below is on that common set; earlier cross-campaign quotes in this card (and in Slack) used per-arm denominators and shift by 1-7 points.

| arm (common set) | 2p-h@2 | 2p-h@5 | 2p-h@30 | 2p-h@100 | 2p-h@900 | s2s-h | 1p-h@1 | 1p-h@5 | 2p-m@5 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| AJ2 | 13.0 | 26.6 | 51.7 | 63.6 | 90.4 | 114.1 | 38.7 | 80.2 | 50.6 |
| BNG | 15.0 | 30.2 | 53.7 | 68.6 | 88.4 | 97.1 | 39.1 | 82.9 | 53.1 |
| MM | 14.1 | 31.1 | 51.4 | 64.7 | 89.3 | 116.1 | 40.3 | 81.2 | 50.8 |
| EG | 11.3 | 28.5 | 55.6 | 67.8 | 89.3 | 95.9 | 34.3 | 76.1 | 47.3 |
| EGMMF | 11.0 | 24.6 | 41.8 | 62.4 | 90.7 | 124.4 | 24.7 | 60.1 | 36.5 |
| HY | 12.4 | 23.7 | 50.6 | 73.7 | 94.9 | 90.2 | 42.1 | 76.1 | 45.1 |
| HY5 | 13.6 | 28.5 | 50.6 | 74.0 | 96.3 | 98.3 | 42.8 | 81.4 | 51.4 |
| **HY5U** | **23.4** | **39.5** | **62.7** | **78.8** | **97.5** | **77.8** | **45.3** | **87.6** | **65.6** |

Seed bands (3 seeds): 1p-h@1 [44.7, 46.7, 44.7] · 2p-h@5 [40.7, 38.1, 39.8] · 2p-h@900 [95.8, 97.5, 99.2] — every seed beats every prior arm on every column. Offline: V5 **0.496** (DOWN from HY5's 0.616), V6 **0.803** (up from 0.749), setup@1 **32.2** and finish@1 **60.1** — both all-time highs by wide margins (previous: 27.1 setup@1 this campaign, 56.1 finish@1 from the earliest label generation). Gate `gate_hy5u.json` + `gate_common.json`, panel `auc_hy5u.json`.

**Readings [numbers, no verdicts]:**
1. **This closed the campaign's open front.** Early hard 2-push (@2/@5/@30) had resisted every corpus and loss change for two days, with BNG/MM unbeaten for months; teaching unreachability moved @5 from 31.1 to 39.5 (+8.4) and @2 from 15.0 to 23.4 (+8.4).
2. **It is the single largest deploy improvement of the campaign — larger than the hybrid corpus itself** — and it is NOT a data change. It partially revises the arc's "composition explains the big swings, loss knobs the small ones" conclusion: an input-space supervision change (where you *cannot* push) outperformed everything collected.
3. **Mechanism hypothesis, UNVERIFIED:** dense geometry supervision on 3.4× more cells teaches reachability structure explicitly, which the ranking sits on top of; previously the model had to infer that structure from sparse labeled cells alone. A clean test would ablate the auxiliary at fixed corpus (e.g. w ∈ {0, 0.1, 0.3, 1.0}) — the w=0.3 probe (`HY5U3_s1`) is the first point of that sweep, one shard refill pending.
4. **V5 anti-predicted deploy for the FIFTH time** (0.496, the worst of any hybrid arm, on the best-deploying model ever trained here). V5 is a burial diagnostic; it is not a model-selection criterion under any circumstances.

## CHANGE LEDGER — every change, why it was made, what it did [USER 2026-08-12: "record of every change, why, and the result, positive or negative"]

One row per change tried in this campaign, in order. **Verdict is against the change's own control**, not against the global best. Deploy numbers are 2p-hard unless noted; ⚠ marks rows measured on per-arm episode denominators (pre-dating the common-episode fix) — those are internally consistent within a round but shift 1-7 points when re-scored on the common set, so compare rows within a block, not across blocks. Rows marked ✅ are on the common set (`gate_common.json`).

### Block 1 — loss shape, old corpus (rounds 1-3b). Control: AJ2 (@2 12.7 / @5 26.8 / @30 53.3 / @900 90.0 / s2s 105 / 1p 38.1) ⚠

| change | why it was tried | result | verdict |
|---|---|---|---|
| **XB** — batch-flat cross-board softmax λ=0.1 | per-board softmax is shift-invariant, so nothing supervises across boards; make the list the whole batch | @5 28.7 / @30 54.7 / @900 91.0 / s2s 99.6 / 1p 38.2, **V5 FLAT 0.456 vs 0.455** | **mildly positive, wrong mechanism** — gain traced to finish ordering, not cross-board repair |
| **RP** — delete regression + censored entirely | test whether the value head was ever needed under a ranking deploy (GBFS consumes order only) | within ~2 pts everywhere (@2 12.9 best-in-round, 1p@5 80.9) | **neutral = the finding** — regression is scaffolding, not signal |
| **MM** — margin-vs-max on batch-flat list | softmax CE stalls at its floor ~log(#positives); a hinge on the single tallest rival cannot stall | @5 **31.6** / 1p **39.4** — best early curve of the round, BNG-class without BNG's guess labels | **positive** |
| **EG** — softmax over the EPISODE FAMILY list | V5 grades setup-vs-own-episode's dead boards, a pair that almost never co-occurs in a random batch (21% of rows had siblings) | @30 **57.4** / s2s **89.3** — best mid-curve | **positive** |
| **EGMM** — margin-vs-max WITHIN the family | the deploy duel written as a loss: beat the champion of every dead board in your episode | V5 0.492 (only robust offline mover), F2 0.920 — but worst front-curve of the round | **mixed** — offline yes, deploy no |
| **RPB** — RP + 2% regression brake | hypothesis that regression's one surviving job is stopping score-scale stretch | @5 23.4, weakest of round 2 | **negative** — leash story falsified at this dose |
| **RPE** — rank-pure EGMM | can the value-free stack carry the family hinge? | @5 25.9 / @900 90.9, V6 0.720 > parent EGMM's 0.685 | **neutral-positive** — removing regression relieved rather than compounded |
| **RPG / RPEG** — unanchored family softmax | grid-completion arms: does the family list work with a softmax instead of a hinge? | setup@1 **12.9 / 9.8**; deploy @2 **4.3 / 2.6**, s2s 132 / 144 | **strongly negative** — the grind law: unanchored family softmax presses setups into the junk pile, and it jams a co-trained hinge |
| **RPL** — linear head, margin 1.0 | truly unbounded scores, closest to the NeurIPS configuration | @5 21.7, offline family panel wrecked | **negative — but from a mis-sized constant**: margin 1.0 = 2.2σ of the adopted scale |
| **RPL2** — same head, margin 0.2 (0.41σ) | isolate the HEAD axis from the miscalibrated margin | @5 26.0 ≈ RPE's 25.9, all cells within band | **neutral = the finding** — boundedness is a free choice; size margins in σ |
| **RPEA** — binary plates (anchor openers→1, dead→0, setups float) [USER] | an absolute down-force on dead champions, with no setup target to grind | V5 0.429 ≈ RPE 0.431; autopsy: live-max AND dead-max both fell 0.04, **gap unchanged** | **null — and the QED**: an absolute anchor cannot separate what the features cannot distinguish → V5 is information-limited, data is the only door |
| **RPM** — MM's stranger-hinge without the anchor | does the sharpener survive going value-free? | setup@1 16.5 vs anchored MM's 25.2; @5 20.6 | **negative** — the stranger-hinge needs the anchor |

### Block 2 — corpus composition (family corpora, isolation 2×2) ⚠. Control: EGMMF

| change | why it was tried | result | verdict |
|---|---|---|---|
| **Family corpus (EGMMF)** — add 1.045M child boards of failed pushes | 94% of deploy pops are child boards; training had ~16%. RPEA proved labels/losses couldn't fix V5, so the hole had to be data | **V5 0.455 → 0.682: the wall fell.** But 1p **38 → 24.7** | **mixed, and the pivotal result** — first offline breakthrough, first deploy regression |
| **R1** — `NAMO_ROOT_FRAC=0.5` root rebalance | is the 1p crater just root exposure (84% → 15% of rows)? | 1p 24.7 → 29.4 (+4.7); V5 0.682 → 0.606; setup@1 **26.8** (then-record) | **small positive, hypothesis rejected** — exposure was not the cause |
| **R2** — exhaustive proven labels (family1) | 12.8% of capped "dead" child labels are lies; do they matter? | V6 0.642 → 0.692 ✔; V5 → 0.539 ✘; 1p 21.8 (worst) | **negative overall** — truth helped live/dead only; half the children cost V5 |
| **G2** — proven labels + rebalance | both dials together | V5 0.564, 1p 25.5, s2s 114.2 (best digging of the block) | **mixed** — no combination recovered 1p |
| **(diagnosis)** per-board label census | why did none of the above work? | roots are 95-98% labeled in EVERY corpus; what differs is **opener-bearing root fraction**: 49-55% (family) vs 73-76% (old), and deploy 1p orders identically | **the mechanism** — episode difficulty skew, not label thinness |

### Block 3 — label ladder {0, 0.5, 1} [USER call] ⚠ / ✅

| change | why it was tried | result | verdict |
|---|---|---|---|
| **EGMMF5** — ladder on family0 | under the 0.9 ladder the opener-setup gap (0.1) is SMALLER than the hinge margin (0.2), so regression and hinge fight | 1p 24.7 → **30.6** (+5.9), @30 41.8 → 50.3, @900 91.5, **V5 held 0.643** | **positive** — the best single-knob gain before HY5U |
| **R15** — ladder + rebalance | are the two additive? | ≈ EGMMF5 minus V5 (0.587) | **negative** — not additive; rebalance superseded |
| **R25 / G25** — ladder on family1 | same knob, honest corpus | V5 −0.08; modest deploy gains (G25 @30 48.6, medium 46.7) | **mixed** — the ladder's value is corpus-dependent |

### Block 4 — the two winners ✅ (common episode set; control BNG @2 15.0 / @5 30.2 / @30 53.7 / @900 88.4 / 1p 39.1)

| change | why it was tried | result | verdict |
|---|---|---|---|
| **HY / HY5** — hybrid corpus: old-corpus roots + family children | Block 2 localized 1p to root content and V5 to child volume; the hybrid supplies both instead of trading them. Setup tier harmonized 0.5→0.9 at build (mixed encodings would tier-split the rank losses) | HY: 1p **42.1**, @900 94.9, **V5 0.690 held**, setup@1 27.1. HY5 (+ladder): @900 **96.3**, @5 28.5, medium 51.4 | **strongly positive** — both skills coexist, no trade; first family-lineage model to beat the old corpus at its own game |
| **HY5U** — unreachable cells as exact zeros, regression only, weight 0.1 [USER] | deploy never scores unreachable pushes, so they were fully masked; teaching them as hard zeros adds geometry supervision on 3.4× more cells at zero inference cost. Kept OUT of ranking lists so no unreachable cell can become the hinge's tallest rival | **wins every column**: @2 23.4, @5 **39.5**, @30 62.7, @100 78.8, @900 **97.5**, s2s **77.8**, 1p **45.3**, 1p@5 87.6, medium **65.6**; setup@1 32.2 and finish@1 60.1 both all-time. V5 **fell** to 0.496 | **strongly positive — the campaign's largest single gain**, and the only change that moved the early hard curve (@5 31.1 → 39.5) |

### Block 5 — methodology changes (no model, but they changed what the numbers mean)

| change | why | result |
|---|---|---|
| `aquaman_agg_common.py` — common-episode aggregation | family-era and BNG/AJ2-era evals ran **different episode lists** (32 episodes in AJ2 that HY5U never saw, 12 the other way); solve@k rates were different populations | common set 1314 1push / 980 2push, identical hard-tier n for all arms. Cross-campaign numbers shift 1-7 pts; HY5U's dominance survives unchanged |
| 3-seed pooling discipline | a single-seed 94.9 was announced as an all-time record, then retracted at 3 seeds (pooled 90.7) | every headline since carries seed bands; one public retraction was the cost of learning this |
| per-board label census | the "sweep depth" diagnosis was a hypothesis stated as mechanism | measurement refuted it (roots 95-98% labeled everywhere) and replaced it with the opener-bearing fraction — correction banked in § POST-HOC CORRECTION |

### Standing scoreboard of ideas that FAILED (kept so they are not retried blind)

Unanchored family softmax (grind) · hinge without anchor (RPM) · regression brake at 2% (RPB) · absolute plates on dead cells (RPEA — and its null is a proof, not a shrug) · margins sized in raw units instead of σ (RPL) · root rebalancing as a fix for 1p (R1/G2) · exhaustive relabeling at the cost of corpus size (R2) · ladder+rebalance stacking (R15).

## Unreachable-rule follow-ups: dose sweep + corpus generality (complete 2026-08-13)

Two questions left open by HY5U, both answered on the COMMON episode set (`aquaman_agg_common.py`; 1314 1push / 980 2push, or 949 2push where AJ2U's episode list narrows it).

**(a) Dose — 0.1 is near-optimal; the effect is NON-MONOTONE.** Arms `HY5U` (w=0.1, 3 seeds), `HY5U3_s1` (0.3), `HY5U10_s1` (1.0), against `HY5` (w=0).

| unreachable weight | 2p-h@2 | 2p-h@5 | 2p-h@30 | 2p-h@900 | s2s-h | 1p-h@1 | 2p-m@5 |
|---|--:|--:|--:|--:|--:|--:|--:|
| 0 (HY5) | 13.6 | 28.5 | 50.6 | 96.3 | 98.3 | 42.8 | 51.4 |
| **0.1 (HY5U)** | 23.4 | **39.5** | **62.7** | **97.5** | 77.8 | **45.3** | **65.6** |
| 0.3 (HY5U3) | **24.6** | 35.6 | 57.6 | **97.5** | 98.4 | 43.7 | 59.2 |
| 1.0 (HY5U10) | 20.3 | 34.7 | 57.6 | 94.9 | **75.9** | 43.1 | 61.0 |

We were NOT underdosing. 0.1 wins nearly every column and 1.0 regresses toward the no-rule baseline. Caveat: 0.3 and 1.0 are single seeds vs 0.1's three — but 1.0's @5 (34.7) sits BELOW 0.1's worst seed (38.1), so the decline is real, not seed noise. The 0.1 default came from mass balance (unreachable cells ~3.4:1 vs labeled, so w=0.1 ≈ 26% of loss mass); that reasoning landed close to optimal, but the curve's shape was not predicted and only two interior points exist — anyone wanting the true optimum should sweep 0.05-0.2 at 3 seeds.

**(b) Corpus generality — the rule does NOT transfer to the root-heavy corpus.** `AJ2U` = AJ2's corpus + `NAMO_UNREACH_W=0.1`, 3 seeds (common set 980 2push, after refilling AJ2U_s1's missing shard 27 — an earlier reading on 949 episodes gave the same conclusion).

| arm | 2p-h@2 | 2p-h@5 | 2p-h@30 | 2p-h@900 | s2s-h | 1p-h@1 | 2p-m@5 |
|---|--:|--:|--:|--:|--:|--:|--:|
| AJ2 | 13.0 | 26.6 | 51.7 | 90.4 | 114.1 | 38.7 | 50.6 |
| AJ2U | 9.9 | 26.8 | 53.7 | 90.7 | **80.5** | 39.6 | 50.3 |

Solve rates are flat within noise (per-seed @5 [28.0, 27.1, 25.4]); the one real effect is **sims-to-solve 114.1 → 80.5** — it finds the same solutions faster — against a drop at @2. Nothing resembling the +8 to +12 across-the-board jump the same rule produced on the hybrid.

**Readings [numbers, no verdicts]:**
1. **The unreachable rule is corpus-dependent, not a general training fix.** Hybrid = ~83% child boards (rule worth +8 to +12 everywhere); AJ2 corpus = ~84% ROOT boards (rule ≈ neutral, sims-to-solve aside). Mechanism hypothesis, UNVERIFIED: reachability geometry changes most after a push, so explicit geometry supervision pays where child boards dominate. The clean test would hold corpus fixed and vary child fraction alone.
2. **Offline predicted this correctly — the first time all campaign.** AJ2U's offline panel fell on every metric (V5 0.455→0.344, setup@1 25.1→20.3, finish@1 50.8→48.0) and deploy indeed showed no gain. Against five prior anti-predictions this does NOT rehabilitate offline metrics as a selection criterion; it is one agreement in six.
3. **Deployment guidance:** keep `NAMO_UNREACH_W=0.1`, apply to child-heavy corpora only. The [DAgger round-4 spec](EXP-2026-08-12-dagger-round4.md) produces exactly such a corpus, so its assumptions survive unchanged.

Gates `gate_dose.json`, `gate_aj2u.json`; panels `auc_dose_aj2u.json`, `auc_aj2u_early.json`.

## Deep budget: 4000 simulator calls (complete 2026-08-13) [USER: "increase the budget to 4000 sims"]

`HY5U` × 3 seeds vs uniform random × 3 seeds (7000/8000/9000), 2push, 1012 episodes each, **both arms re-run fresh through the same script** (`scripts/slurm/eval_budget_2push.slurm`) so key, raw score scale, agg/combine/discount and seeds are identical — the 2026-08-02 random baseline used a different key and the legacy sigmoid scale and must NOT be reused at a new budget. Driver `scripts/rl_loop/run_budget4k_waves.sh`, 250 shards/arm (~4 episodes each), waves under Amarel's 500-job cap. Aggregate `gate_b4k.json`, raw `eval_b4k/{HY5U_s*,rand_s*}/2push/`, plot `plots/hy5u_vs_random/success_vs_sims_2push_4000.png`.

| tier | n | model @900 | model @4000 | random @900 | random @4000 | median calls (model vs random) | mean calls (model vs random) |
|---|--:|--:|--:|--:|--:|--:|--:|
| easy | 1140 | 99.9 | **100.0** | 100.0 | 100.0 | 2 vs 12 (**6.0×**) | 7.5 vs 24.4 (3.2×) |
| medium | 1440 | 99.7 | **100.0** | 97.4 | 99.9 | 3 vs 51 (**17.0×**) | 22.3 vs 143.5 (6.4×) |
| hard | 354 | 97.5 | **100.0** | 66.7 | 97.2 | 12 vs 470 (**39.1×**) | 120.2 vs 790.2 (6.6×) |

**Call ratios above are PAIRED** — computed per (episode, seed) on the pairs BOTH arms solved (1140 / 1438 / 344 pairs). ⚠ An earlier version of this section quoted 3.3×/6.5×/**7.2×** from `avg_sims_all`, which charges every UNSOLVED episode the full 4000-call ceiling; since random leaves ~3% of hard episodes unsolved and the model leaves none, that statistic partly measured random's failures rather than its slowness, and it inflates automatically with any budget increase. Corrected on the user's challenge, 2026-08-13. The mean/median split is itself informative: on hard the model's median is 12 calls but its mean is 120 — a heavy tail of a few expensive episodes — whereas random is uniformly slow (470 median, 790 mean).

**Readings [numbers, no verdicts]:**
1. **At a large budget the SOLVE RATES converge — the ranker's value is entirely efficiency.** Hard tier: model 100% vs random 97.2% at 4000 calls, a 2.8-point gap, versus a 30.8-point gap at 900. This is the expected consequence of the simulator being a perfect verifier: enough random tries eventually find the answer. Any headline framed as "we solve more problems" collapses at large budget; the honest framing is **6.6× fewer calls on hard in the mean, and 39× at the median** (paired; 17× medium, 6× easy).
2. **The model reaches 100% on hard 2push** — first time any arm has cleared the tier outright, and it needs 4000 calls to do it (97.5 at 900), so the tail is real but reachable.
3. **Random's hard-tier average is 790 calls (paired) of a 4000 ceiling**; the model averages 120 with a median of 12. Per-episode paired scatter shows the ranker is cheaper on **87% / 90% / 93%** of easy / medium / hard problems individually — the win is per-problem, not an averaging artifact.
4. Method note: the previous 900-budget curves compared against a random baseline collected under different settings. This campaign supersedes those for any random comparison at any budget.

## Depth-selection bias — RETRACTED IN LARGE PART, then re-measured correctly (2026-08-13)

**⛔ RETRACTION.** An earlier version of this section reported that the model picks the deepest push 69.6% of the time, scores 48.0% top-1 overall, and is **worse than random (0.69×) on the 43% of boards without a depth-4 opener** — and claimed this explained setup@1/finish@1 with a ~17-point upside. **That was an artifact of the candidate set and is withdrawn.** It was posted to Slack before the error was found; the correction was posted immediately after.

**The error.** The analysis defined candidates as `r_mask > 0`. **`r_mask` is per-EDGE, not per-cell**: measured, it is identical across all 5 depths on 100% of (board, edge) pairs. It answers "can the robot reach this contact point", not "is this push executable". In the exhaustive GT only 60.4% of reachable depth-4 pushes are ever labeled (vs 100% at depth 0) because the deep push is often physically impossible — the object would hit a wall or another object, so no goal is generated and no trial happens. The old analysis therefore let the model "choose" cells that the deploy-time primitive generator would never offer, then scored it wrong when those were not openers.

**Corrected, on FEASIBLE cells (`value_mask & r_mask`) — what deploy actually offers:**

| metric | artifact | corrected |
|---|--:|--:|
| top-1 picks landing on depth 4 | 69.6% | **46.0%** |
| overall top-1 hit | 48.0% | **75.6%** |
| boards WITH a depth-4 opener | 79.2% (4.65× random) | 79.3% (4.56×) |
| boards WITHOUT a depth-4 opener | 7.0% (**0.69×**) | **70.7% (2.07× random)** |

**What stands after correction:** a real but modest depth tilt — 46.0% of top-1 picks at depth 4 against a 26.2% true-opener share — and a genuine but much smaller weakness on boards whose answer is not deep (70.7% vs 79.3%; lift 2.07× vs 4.56×). The model is **never worse than random**. The per-depth score offset is 0.48 SD (means 0.114 → 0.242, within-depth SD 0.266), directionally correct since the true opener rate does rise with depth; argmax over the candidate set amplifies it, but far less than the artifact suggested.

**What is withdrawn:** "worse than random on 43% of boards", "this explains setup@1 / finish@1", and the 48% → ~65% upside estimate. Do not cite them.

**Durable lesson (the reason this section is kept rather than deleted):** `r_mask` = reachable CONTACT, `value_mask & r_mask` = executable-and-tried PUSH. Any offline ranking analysis must use the latter as the candidate set, because that is what the deploy-time goal generator produces. Conflating them silently inflates every top-1-style number against the model. Two earlier claims in this same investigation were also wrong and are recorded so they are not repeated: (1) collection does NOT prefer depths (4,3,2) — that default belongs to `BeamPlanner` in `scripts/sandbox/scorer_beam.py` on the eval side, while collection's `region_opening.py` sorts `(depth asc, score desc)` or `(score desc, depth asc)` and runs root sweeps exhaustively; (2) the corpus does not "teach" the deep bias — training and exhaustive GT agree deep is better by a similar ratio (2.8× vs 2.4×), the difference in absolute opener rate being board easiness.

Where sort order DOES still matter, untested: child boards are capped at 12 tried finishes (`region_finish_topk_cap: 12`) and collection scores with `goal_strategy: scorer`, so the collecting model's preference decides which finishes are ever labeled — a possible model-bias feedback loop into the next corpus.

## WALL-CLOCK campaign at budget 4000 (complete 2026-08-13) [USER: "do the time evaluation", "4000 only"]

Protocol `scripts/slurm/eval_walltime.slurm`: every task takes a WHOLE node (`--exclusive`) of ONE fixed CPU generation, single-threaded, model scoring on CPU; timing accumulated inside `eval_bestfirst.solve_scene` so the timed search IS the canonical search. 12 arms (HY5U ×3 seeds, uniform random ×3 seeds, both legs), 20 shards each, budget 4000. Raw `eval_walltime4k/`, plots `plots/walltime4k/success_vs_time_{1push_hmax2,2push}.png`.

**Ops note:** the campaign was first pinned to `icelake` and crawled at 5 concurrent tasks — ZERO idle icelake nodes cluster-wide while 146 nodes sat idle on other generations. Moved wholesale to `cascadelake` (73 idle) and DISCARDED the icelake partials: mixing CPU generations inside a timing campaign invalidates it, and the protocol needs one fixed generation, not that specific one. 5 → 125 concurrent.

**TWO-PUSH — paired on episodes both arms solved:**

| tier | pairs | model wall | random wall | wall ratio | MEDIAN wall ratio | sim-call ratio | scoring share of model wall |
|---|--:|--:|--:|--:|--:|--:|--:|
| easy | 1140 | 2.46s | 7.01s | **2.85×** | 2.41× | 3.32× | 12.9% |
| medium | 1439 | 8.56s | 35.79s | **4.18×** | **7.60×** | 5.96× | 6.6% |
| hard | 343 | 35.97s | 177.20s | **4.93×** | **20.16×** | 6.42× | 3.7% |

**ONE-PUSH:**

| tier | pairs | model wall | random wall | wall ratio | sim-call ratio | scoring share |
|---|--:|--:|--:|--:|--:|--:|
| easy | 2091 | 0.49s | 0.51s | **1.03×** | 1.82× | 25.9% |
| medium | 1260 | 0.88s | 2.47s | **2.82×** | 5.77× | 17.3% |
| hard | 591 | 2.36s | 8.76s | **3.71×** | 6.45× | 11.4% |

**Readings [numbers, no verdicts]:**
1. **The sim-count advantage does NOT fully survive in seconds, but most of it does on 2push.** Hard 2push keeps 4.93× of a 6.42× sim ratio (77%); hard 1push keeps 3.71× of 6.45× (58%).
2. **Scoring overhead is not the main leak, and it shrinks with difficulty** — 3.7% of model wall on hard 2push vs 25.9% on easy 1push. The network cost is fixed per decision while simulation time grows with the problem.
3. **The bigger leak: the model's simulator calls are individually MORE EXPENSIVE** — 0.306s vs random's 0.244s on hard 2push (1.25×), 0.380s vs 0.246s on hard 1push (1.54×). The model chooses substantive pushes (contact, real motion, deeper chains); random frequently picks pushes that fail fast. **Simulator-call count is therefore not a neutral currency — it flatters the model by ~1.25-1.5×.** Wall-clock is the honest axis.
4. **Medians are far stronger than means** (hard 2push 20.2× median vs 4.93× mean) — same heavy tail as the sim counts: a few expensive episodes dominate the model's average.
5. **Easy 1push is a statistical tie (1.03×)** — the fixed scoring cost plus costlier calls cancel a 1.82× call advantage. The registered 2026-08-06 campaign found the then-model actually SLOWER there (0.85×), so this is an improvement, but "always run the ranker" is not the right deployment rule on trivial problems. Easy 2push has no such problem (2.85×).
6. Cross-campaign caution: these are budget-4000 numbers on cascadelake; the registered `walltime-nodiscount-hmax2-v1` rows are budget-900 on icelake. **Do not put them on one axis.**

### Tail structure of the wall-clock distribution (2026-08-13)

The mean/median gap is not noise — it is the model's failure mode made visible, and it scales with difficulty on BOTH horizons.

| horizon / tier | model mean | model median | mean÷median | worst 10% of episodes = X% of model's total time | median speed-up |
|---|--:|--:|--:|--:|--:|
| 1push easy | 0.49s | 0.46s | 1.1 | 18% | 0.9× |
| 1push medium | 0.88s | 0.63s | 1.4 | 38% | 1.4× |
| 1push hard | 2.36s | 0.82s | 2.9 | 65% | 3.2× |
| 2push easy | 2.46s | 1.34s | 1.8 | 47% | 2.4× |
| **2push medium** | 8.56s | 1.55s | **5.5** | **72%** | **7.6×** |
| **2push hard** | 35.97s | 4.53s | **7.9** | **70%** | **20.2×** |

Random's distribution is comparatively flat (hard 2push mean÷median 1.9; worst 10% = 42% of its time). Percentiles on hard 2push (model vs random): p25 1.31 vs 25.05s (**19.1×**), p50 4.53 vs 91.40s (20.2×), p90 91.82 vs 510.67s (5.6×), **p99 549 vs 1063s (1.9×)**.

**Reading:** the ranker is effectively BIMODAL — when the answer is near the top of its ordering it finishes almost immediately (a quarter of hard 2push episodes in under 1.3 s), and when it is not, it degrades toward random (ratio 1.9× at p99). Random has no fast mode at all, so its distribution is tight in log terms. This is why the quoted speed-up depends entirely on the statistic: **20× describes the typical hard problem, 4.9× describes expected compute.** Report both; the mean alone understates the typical case and the median alone hides the tail.

**Actionable:** the remaining headroom is CONCENTRATED — fixing the worst 10% of episodes recovers ~70% of the model's total runtime. And the best target is arguably **2push MEDIUM, not hard**: same concentration (72%), same bimodality (5.5), but 1439 episodes against hard's 343 and presumably more tractable failures. This is an argument for the DAgger round independent of the ones already on its card.

## Arc narrative — every decision, its reason, and what it concluded (2026-08-11 → 08-12) [USER: record everything in detail]

This section is the reasoning trail for the isolation 2×2 → hybrid arc, written so the choices are reconstructible without the chat.

**Starting point and the problem statement.** The family corpus had produced the first-ever offline V5 wall-fall (EGMMF 0.695) but deploy did not convert: 1p-h@1 cratered from the old-corpus ~38 to 24.7, and @900 pooled 90.7 sat below θ₀'s 92.0. The corpus switch had changed THREE things at once relative to the old corpus — root share of rows (84%→15%), root sweep depth (cap 20→12 tried pushes), and the addition of 1M capped-label children (12.8% of "dead" labels lies) — so no single cause could be read off. The methodological sin was bundling dials; the arc's job was unbundling them.

**Why a 2×2 and why these two axes.** Exposure (how often roots are seen) and label truth (are the children's dead labels honest) were the two axes we could vary WITHOUT new collection machinery: exposure via a sampler knob on the existing H5, truth via the already-collected exhaustive shards. Root sweep depth could NOT be varied cheaply (it is baked into collection), so it was explicitly flagged at launch as the un-isolated rider. Loss was already controlled: the identical EGMM recipe on the old corpus scored 36.9 — whatever broke, broke in the data.

**Why R1's rebalance was built as singleton families.** The family loss terms (`_family_lists`) skip lists with fewer than 2 rows. Appending duplicated SINGLETON root families therefore raises per-board root exposure while leaving the cross-board terms byte-identical — a true single-dial change. The exposure meter (0.152→0.500, measured 49.5% in the batch stream) was printed per the standing law: when you shift a training distribution, meter what loses exposure.

**Why the exhaustive corpus was accepted despite being half-size.** Exhaustive sweeps cost ~5.6k sims/episode, so the same compute bought 80.5k episodes vs family0's 188k. The entanglement (truth × size) was pre-registered as a known confound with its resolution written down in advance: if honest cells underperform, size and truth are entangled; if they win, size cannot be the explanation.

**2×2 conclusions (all seed-pooled, canonical):** (a) exposure is real but small — +4 to +5 1p points on both corpora, nowhere near the 13-point crater; (b) label lies are not the 1p wound — honest labels moved V6 up (0.69 vs 0.64, genuine live/dead improvement) and 1p not at all; (c) V5 tracks child VOLUME, not truth — both honest cells dropped to 0.54-0.56 against EGMMF's 0.68; (d) rebalance TAXES V5 on the same corpus (0.682→0.606) because the extra root batches dilute the family-loss gradient share, while simultaneously setting a then-all-time setup@1 (26.8) — proof that within-board ordering and cross-board altitude are different skills served by different data; (e) the residual 1p gap therefore pointed at the one un-varied dial: root supervision CONTENT (sweep depth × board count).

**Why the hybrid, and why it was built exactly this way.** The old corpus's root rows (257k, d20-deep exhaustive 1p labels) are the strongest root supervision we own; family0's 1.045M children are the V5-winning contrast class. The hybrid concatenates them instead of choosing between them. Two design decisions mattered: (1) 66% of family child episodes share (xml, object_id) with an old-corpus row, so the episode-grouped sampler joins an old deep-labeled root with family children in ONE ranking list (78,196 such families measured at smoke) — the cross-board duel the family thesis wanted, now anchored by a well-labeled root; (2) label harmonization: the old corpus stamps setups 0.5, family stamps 0.9, and the rank losses tier by VALUE — mixed encodings would have invented a phantom tier (0.9-setups forced above 0.5-setups), so old rows were remapped 0.5→0.9 at build time.

**Why the wide ladder {0, 0.5, 1} ran as a full second grid [USER call].** Under the 0.9 convention the opener-setup label gap (0.1) is smaller than the hinge margin (0.2), so regression and hinge pull against each other; under 0.5 every tier gap clears the margin. The remap is load-time (`NAMO_GAMMA=0.5`), so the entire second grid cost zero data engineering. Historical note: arjuna's own 0.5-ladder test had shown V5-flat, so this was not known-win chasing — it was a clean interaction test against the new corpora.

**Campaign conclusions (the durable ones):**
1. **The hybrid confirms both causal stories at once, seed-robust: root content fixes 1p (42.1, beating the old corpus's 38.1 with the band floor), child volume holds V5 (0.690), and they coexist in one corpus with no trade.** The 1p crater was never about how OFTEN roots are seen or whether children's labels lie — it was about how many distinct roots exist and how richly each is labeled.
2. **@900 94.9-96.3 is the new all-time reach, with seed floors above the old record (92.0).** The family thesis — children teach the ranker which boards are dead — finally cashed at deploy the moment it was ADDED to strong root supervision instead of replacing it.
3. **The ladder is corpus-dependent, not a universal win:** +5.9 1p on family0 alone (EGMMF5 30.6, V5 held), V5-negative on the small honest corpus, and on the hybrid it trades V5 (0.690→0.616) for reach (@900 96.3), sharpness (@2 13.6), and medium tier (51.4). Pick per deployment goal, not by doctrine.
4. **`NAMO_ROOT_FRAC` rebalance is superseded:** not additive with the ladder (R15 ≈ EGMMF5 minus V5); better root CONTENT beats re-showing thin roots. The knob stays in the code as an isolation tool, not a recipe ingredient.
5. **Offline V5 anti-predicted deploy for the fourth time this campaign** (EGMM crowned by V5 then out-deployed by MM; R25/G25 V5 down deploy up; EGMMF5 V5 down deploy up; HY5 V5 below HY yet the stronger deploy row). V5 remains a diagnosis meter for the burial mechanism, NOT a model-selection criterion; canonical deploy stays the only arbiter.
6. **Open front:** early-mid hard 2-push (MM 31.6 @5 / AJ2-MM 53.3 @30 vs hybrid 28.5/50.6). Natural next cells: MM's batch-flat stranger-hinge crossed onto the hybrid corpus; exhaustive wave-2 (~45k unused rooms + pool-gen filler) to grow honest children under the now-proven recipe.

**POST-HOC CORRECTION [2026-08-12, measured — supersedes the "root sweep depth / thin root labels" wording above].** The residual-1p diagnosis named "sweep depth" as the un-isolated dial. Direct measurement of the H5s refutes the *mechanism* while confirming the *conclusion* (root supervision content). Per-root-board census (sampled ~4k boards per corpus, in-loss reachable cells):

| corpus | reachable cells labeled | openers/board | setups/board | boards with ≥1 opener | deploy 1p-h@1 |
|---|--:|--:|--:|--:|--:|
| old (arjuna0v2) | 95% | 15.5 | 26.4 | **76%** | 38.1 |
| hybrid | — | 14.5 | 25.3 | **73%** | **42.1** |
| family0 | 98% | 17.4 | 17.6 | 55% | 24.7 |
| family1 (exhaustive) | 98% | 13.6 | 13.8 | 49% | 21.8 |

Root boards are ~95-98% labeled in EVERY corpus — label density was never the difference, and `region_finish_topk_cap` (20 vs 12) governs FINISH sweeps on child boards, not root cells. What differs is **episode selection**: the family campaigns sampled far harder episodes, where only 49-55% of root boards contain any verified opener (vs 73-76% for old/hybrid). Deploy 1p-h@1 orders exactly with that column. A ranker learns "which push opens this board" from boards where an opener exists; a corpus half-composed of opener-free boards supplies little of that contrast, and no amount of re-showing (`NAMO_ROOT_FRAC`) or re-labeling (exhaustive recollect) can manufacture it — which is precisely why both cheap fixes failed. Setup-label supply skews the same way (26 vs 14-18 per board), explaining the parallel setup@1 ordering. **Actionable form: when collecting for 1-push/setup ranking, measure and control the opener-bearing fraction of root boards; difficulty-skewed collection silently starves the signal.**

**Ops lessons this arc added (also in Log):** CS `/dev/shm` staging is unsafe estate-wide (RemoveIPC purge kills staged H5s AND loky semaphores; NFS + page cache is the safe default when the H5 fits job memory); teardown-noise FAIL events (FileNotFoundError after epoch 011) are benign — verify against sacct + final ckpt before acting; Amarel's evening queue swallows 432-task canonical waves in ~7 minutes, so eval is never the bottleneck; single missing shards are refilled with `sbatch --array=<id>` on the same env, never a full wave rerun.

## Log

- 2026-08-11 [Claude] Isolation 2×2 launched (section above). CS estate had a bad day, three distinct cluster-side failures in one hour: (1) jobs on rlab2 + later the ilab2 login node lost NFS access to `/common/home` ("Could not chdir to home directory: Permission denied") — submit host moved to ilab1; (2) staged `/dev/shm` H5s were deleted under running jobs within ~40s on BOTH rlab3 and rlab6 (RemoveIPC-style purge, now estate-wide — previously only rlab4/rlab7) — fleet resubmitted with `NAMO_STAGE_SHM=0`, NFS reads + page cache (both H5s ≪ 48G job memory, so only epoch 1 pays); (3) killed fleets left no SLURM output files at all (log write also blocked). Lesson: on CS, treat `/dev/shm` staging as UNSAFE until re-verified; page-cache NFS is the safe default for H5s that fit in job memory.
- 2026-08-11 [Claude] Gen-2 exhaustive H5 built on Amarel and landed on CS (verification in section above); `pool_family_h5.py` landed in `scripts/pipeline/`.

- 2026-08-09 late [Claude, USER-requested] BNG re-evaluated from original ckpts on tonight's stack, fresh dirs `BNGre_s*` (registered artifacts untouched): @2 14.6/@5 31.9/@30 55.5/@900 89.3 vs registered 14.6/32.1/55.7/88.6 — reproduction within sim jitter, eval stack drift-free, all cross-campaign comparisons clean. Aggregate `gate_bngre.json`.

- 2026-08-09 [Claude] Card created; design discussion (loss structure, literature deep-read, bounded-vs-unbounded, weight semantics) in session `ranking_loss`. Code next: XB reshape + RP subclass + unit test.
- 2026-08-09 evening [Claude] Round-1 trained (6/6), panel + autopsy + canonical COMPLETE (tables above). Round-2 fleet (MM/EG/EGMM/RPB × 3) launched after 2-epoch smokes passed; canonical for round 2 queued on completion.
