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

RP checkpoints on validation ranking loss (its monitor cannot be regression — it doesn't train it). XB keeps the standard monitor for apples-to-apples ckpt selection vs AJ2.

## Locked defaults (one variable: the loss)

- Data: `/common/users/dm1487/scratch_namo/aquaman/round0/arjuna0v2_train.h5` (995M, the AJ2 file; zero guesses, no ceiling cells → EXCLUDE_GUESS and cap rules dormant on this file).
- Head 51-bin HL-Gauss [0,1], `RANK_TEMP=0.15`, 12 epochs, batch 256, lr 3e-4, 3 seeds — all matching AJ2.
- Parked behind the gate, deliberately: episode-grouped lists (needs root↔child linkage scout on the 26k Colossus setup roots), γ-native honest-cap labels, unbounded head, margin-shape loss, more bins.

## Plan

1. Unit test: two-board toy (live+dead board) — dead-board junk must receive downward gradient under the flat call, zero under the per-board call. Commit before run.
2. Smoke: 1 XB seed, 2 epochs, arrakis — loss decreases, no NaN, epoch time sane (one-big-softmax cost check).
3. Fleet: 6 runs (XB×3, RP×3) raced on the a100 boxes (~20 min each; box-sync + GPU check first).
4. Gate: offline AUC panel (`auc_compare_arms.py`), per-tier as always. V5 ≥0.60 AND F2 ≥0.87 → pass. Score histograms per tier for the pile-up watch.
5. Only on pass: canonical eval overnight (Amarel pulls the pushed branch; `check_box_sync.sh` before launch), difficulty × horizon splits, registry entry on completion.

## Outcome fork (pre-registered readings)

- V5 up, F2 held → mechanism real; next arms = episode-grouped sampler, and RP-vs-XB decides regression's fate.
- V5 flat → per-board float was not the cap; board-live head becomes the main route; no sampler is built.
- V5 up, F2 collapsed → one scalar refuses both scopes at this τ/λ → tune or split heads (AlphaZero shape).
- V5 up, deploy flat (post-canonical) → scheduler convicted with a controlled experiment.

## Ops notes

- No worktree: worktrees don't propagate to Amarel — pushed branches do. Develop on `feat/horizon-q-redesign`, push after every commit, Amarel pulls the branch for the canonical round only.
- No C++ changes → no rebuild, no `.so` sync concern.

## Log

- 2026-08-09 [Claude] Card created; design discussion (loss structure, literature deep-read, bounded-vs-unbounded, weight semantics) in session `ranking_loss`. Code next: XB reshape + RP subclass + unit test.
