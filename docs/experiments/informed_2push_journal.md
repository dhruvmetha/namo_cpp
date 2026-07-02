---
status: live
tags: [experiment]
updated: 2026-06-09
---

# Informed 2-push — experiment journal

Goal: do **informed depth-2** for region opening (single object, [[project_ro_single_object]]) — guide
the depth-2 search so we solve 2-push scenes with FAR fewer MuJoCo sims. The scorer is a 1-step `Q(s,a)`.
The open question that gates everything: **is the depth-2 bottleneck the LEAF (scorer recall at
post-first-push states `s1`) or the SEARCH layer (can `V(s1)=max P` rank first pushes)?**

Method discipline: pre-register a hypothesis + prediction, accept/reject with numbers. Decisions tagged
[USER]/[CLAUDE]. All runs collisions-OFF (train-match — matches scorer training + how the
`*_2push_solvable` manifests were defined; [USER] confirmed collisions-allowed is the legit setting).

Refs/assets: champion scorer `sharp_s1 epoch017-val_loss0.2713.ckpt`; beam `scripts/sandbox/scorer_beam.py`;
manifest `test_2push_solvable_combined.txt` (1186 scenes, per-episode, feb+aug9). Lit map: [[reference_namo_value_learning_litmap]].

---

## Decisions
- [USER] Scope to SIM experiments first; ignore real-robot. Target depth-2 (not 1-push, not n-push yet).
- [USER] Train with v3 envs (feb_car + aug9_car, similar object sizes); reuse the 5-phase collection for pkls if data needed.
- [CLAUDE] First experiment = the leaf-vs-search diagnostic, before training anything. Cheap, reuses the beam, decides where to invest.

---

## H1 — LEAF is the bottleneck (scorer recall collapses at `s1`)
**Hypothesis [CLAUDE]:** the scorer was trained/graded on initial scenes `s0` (top-1 @1). The depth-2
leaves are post-first-push states `s1` (object displaced) — off-distribution. So the true 2nd push is
ranked much lower at `s1` than the `s0` recall would suggest.

**Pre-registered prediction:** recall@10 at `s1` **< 60%** (vs the `s0` hard recall@10 = 77.6% reported
for sharp). If recall@10 at `s1` ≥ 75% → REJECT H1 (leaf is fine, bottleneck is the search layer).

## H2 — `V(s1)` does not separate good first-pushes from dead-ends (calibration)
**Hypothesis [CLAUDE]:** `V(s1)=max P` is within-state ranking, not calibrated across states. So it
poorly distinguishes first pushes that LEAD to a solvable `s1` (good) from those that don't (dead-end).

**Pre-registered prediction:** AUC( `V(s1)`; good-vs-dead leaves ) **< 0.70**. If AUC ≥ 0.80 → REJECT H2
(the first-push ranking signal is fine).

**Decision rule:**
- H1 accepted → invest in `s1` leaf data (5-phase, exhaustive-1-push at `s1`) + retrain scorer for recall@`s1`.
- H1 rejected, H2 accepted → invest in a cost-to-go / first-push value (Bejjani-style distillation).
- both rejected → the beam is already near-optimal; the win is just engineering the sweep down.

**Method:** `scripts/sandbox/diag_leaf_s1.py` — reuses `BeamPlanner`. Per scene: sweep first pushes
(capped), sim `s0→s1`; if `s1` already open → count as 1-push-first (manifest-label sanity); else score
`s1`, verify 2nd pushes in scorer order up to top-K, record rank-of-first-success and `V(s1)`. Recall@k
over solvable leaves; AUC of `V(s1)` good-vs-dead.

**Status:** running on `test_pure2push_combined.txt` (genuine depth-2, 985 scenes; beam baseline 16%@1→56%@2)
at N=25, first_cap=20, top-K=15. Switched OFF the `*_2push_solvable_combined` manifest after a smoke run
showed its first scenes are 1-push-contaminated (first push opens the goal alone; every leaf solves at rank 0)
— exactly the journal's known caveat. Results ↓ TBD.

**Method caveats (honest interpretation):**
- **Censoring:** a leaf is called "solvable" only if a 2nd push opens within the scorer's top-K(=15). True
  2nd pushes ranked >15 are missed → recall@k is *conditional on success∈top-15* (biased high), and dead-end
  count is inflated. So: LOW recall@1–3 even within top-15 ⇒ leaf clearly weak (strong signal); HIGH recall
  within top-15 is necessary-not-sufficient for "leaf is fine." The honest leaf-quality number is the pair
  (solvable-leaf-fraction within top-15, recall@k among them).
- **First-push undersampling:** `first_cap=20` may miss a scene's true setup move → that scene logs only
  dead-end leaves. Affects solvable-leaf *counts*/AUC, not recall@k among found leaves.
- **No ground-truth `(a1,a2)`:** validsets store only 1-push `valid`/`tried` at `s0`; depth-2 solving pairs
  aren't persisted. A clean *uncensored* H1 test would need either deeper verification (top-60+) at promising
  leaves or re-running depth-2 search with `(a1,a2)` logging. Staged as the H1-confirmation follow-up if this
  pass flags the leaf.

### RESULT (N=25 pure-2-push, 436 leaves / 130 solvable, 5738 sims, ~61 min) — `diag_leaf_s1_pure2push.json`
```
leaf recall@s1:  @1 .254  @3 .415  @5 .600  @10 .877  @20 1.0   median success rank = 3
V(s1) AUC good-vs-dead = 0.534    V_good_median 0.9946   V_dead_median 0.9933   (130 good / 306 dead)
4/25 scenes had a first push that opens the goal alone (minor residual 1-push contamination)
```
- **H1 REJECTED (strong):** predicted recall@10 < 60%; got **87.7%** (≥75% = reject). The scorer's RANKING
  generalizes to `s1` — once you're at a completable leaf, it puts the finishing 2nd push in the top-10 ~88%
  of the time (≥ the s0 hard recall@10 of 77.6%). *The leaf is not the bottleneck. Do NOT retrain on `s1`.*
- **H2 ACCEPTED (strong):** predicted AUC < 0.70; got **0.534** (≈ random). `max P` **saturates at ~0.99 on
  solvable AND dead-end leaves** → it cannot rank first pushes. This is the cross-state calibration failure
  predicted from the property analysis: good within-state ORDER, useless absolute VALUE.
- (Censoring caveat holds but doesn't threaten either call: recall is conditional-on-top-15 yet still high;
  AUC≈0.5 is robust because both classes sit at V≈0.99 regardless of mislabeled-dead.)

**VERDICT → the informed-2-push problem = FIRST-PUSH SELECTION.** The leaf is reusable as-is; the broken piece
is "which first push leads to a solvable `s1`," and `max P` answers it at chance. Next: can a *calibrated*
first-push value beat `max P`?

---

## H3 — a better scalar / learned value ranks first pushes far above `max P`
**Hypothesis [CLAUDE]:** `max P` saturates, but the `s1` scorer map (or a learned head) contains a *calibrated*
signal that separates good first pushes from dead-ends.
**Two sub-tests, cheapest first:**
- **H3a (training-free):** some cheap aggregate of the `s1` (60,5) map — mean-top-k P, frac(P>0.99),
  top1−top2 margin, n-reachable — has AUC(good-vs-dead) **> 0.70**. (Computed from `pool2` already in hand,
  *no extra sims*.) If yes → training-free informed-2-push.
- **H3b (learned, if H3a fails):** a head `Q(s0, a1)` predicting "`a1` leads to a 2-push-solvable `s1`",
  trained on the per-leaf good/dead labels this diagnostic produces (masked BCE, same as the f_grid scorer
  but label = leads-to-solvable-`s1`). Target AUC **> 0.80** on held-out rooms.
**Method:** extend `diag_leaf_s1.py` to log per-leaf records (candidate scalars + good/dead label + obj/edge/depth/xml);
re-run larger (parallelized via SLURM array, collisions-allowed + target-region-goal).
**Status:** array job `55815040` RUNNING (8 tasks × 30 = 240 pure2push scenes, all started immediately on
main-redhat, ~73 min/task). Aggregator `diag_fpv_aggregate.py` ready. NOTE this node has nproc=1 → had to
go to SLURM for parallelism. Smoke hint: `maxP` saturates (~0.994) but `frac_ge_099` (≈0.016 on a dead leaf)
and `mean_all` (≈0.08) are NOT saturated — candidate separators. Verdict ↓ TBD on array completion.

### RESULT (4486 leaves, 233 scenes, 174 rooms; 3/8 shards hit the 2h wall but only ~7 scenes lost)
```
leaf recall@s1:  @1 .435  @3 .616  @5 .716  @10 .889   median success rank = 1   → LEAF GOOD (H1 reject reconfirmed, stronger)
first-push-value AUC (held-out 51 rooms, group-by-room, no leakage):
   maxP .690   mean_top5 .796   mean_all .701   n_pushes .706   frac_ge_099 .694   margin .703
   6-scalar logistic COMBO = 0.817   (coef: mean_all .54, frac_ge_099 .42, n_pushes .39, ... maxP ≈ 0)
```
- **H3a ACCEPTED:** a TRAINING-FREE scalar/combo ranks first pushes at **AUC 0.82 held-out** — vs the beam's
  current `V(s1)=max P` (≈0.53–0.69). `mean_top5` alone = 0.80. `max P` is the *worst* feature (coef ≈ 0):
  the saturation that sinks it is exactly H2. **Deployable beam change: replace `V(s1)=max P` with the
  combo (or `mean_top5`).**
- **Leaf reconfirmed GOOD** on 10× the data: median success rank = 1, recall@5 = 72%. The scorer is a fine
  leaf as-is; do not retrain it for the leaf role.

**Crucial nuance (what H3a does and doesn't buy):** these scalars are computed at `s1` (POST-sim). So the
combo improves *which swept first push to expand for the 2nd push* (cuts wasted 2nd-push expansions / raises
the N1 hit-rate) — it does **NOT** cut the first-push SWEEP, because you still sim each `a1` once to get `s1`.
**The real sim-reduction lever is H3b** below.

## H3b — learn `Q(s0, a1)` to predict a good first push WITHOUT simulating `s1` (the sweep-cut)
**Hypothesis [CLAUDE]:** a head over the `s0` crop can predict "`a1` leads to a 2-push-solvable `s1`"
directly, so we rank/prune first pushes BEFORE simulating them → cut the depth-2 sweep (the dominant sim cost).
Target: held-out-room AUC **> 0.80** (matching what the post-sim combo achieves, but from `s0` alone).
**Data:** the 4486 per-leaf good/dead labels (`diag_fpv_shard*.jsonl`) keyed by (`xml`, `obj`, `edge1`,
`depth1`) are the seed — but SMALL/SPARSE (first_cap=20 of ~125 reachable; 233 scenes, censored top-15
labels). Needs scale-up: bigger array (more scenes, higher first_cap, maybe top-30 labels) → render `s0`
crops via `live_scorer` → train a first-push-value head (masked BCE, label = leads-to-solvable-`s1`,
room-split). Status: NOT STARTED (next session).

---

## HANDOFF (for when you're back)
**Established with numbers this session:**
1. The depth-2 bottleneck is **first-push selection, not the leaf.** (H1 reject: leaf recall@5=72%, rank-1
   median; H2 accept: `max P` AUC 0.53–0.69.)
2. **Training-free win available now:** swap the beam's first-push ranker from `V(s1)=max P` → the
   `mean_top5`/combo scalar (held-out AUC 0.80–0.82). This is the obvious immediate change.
**Ranked next steps:**
- **(A) Validate the free win end-to-end** — wire `mean_top5`/combo into `scorer_beam` first-push ranking,
  re-run the pure-2-push eval, confirm it lifts solve@2 and/or cuts sims vs the `max P` baseline. (~1 beam eval.)
- **(B) H3b — the sweep-cut** — scale the array (e.g., 200–400 scenes, first_cap 40, top-30 labels; raise
  SLURM `--time`), build the `s0`-crop dataset, train `Q(s0,a1)`. This is the real "informed 2-push" model.
- **(C) Clean-label option** — to de-censor, log ground-truth `(a1,a2)` by re-running depth-2 with full
  verification at promising leaves; tightens H3b labels.
**Open caveats:** labels censored at top-15; 3 shards hit the 2h wall (use smaller shards / longer `--time`);
combo is a 6-param linear model (held-out AUC is honest but re-fit per train split).
