# Policy-framework journal — scorer → (policy + value)

**Why a new journal [USER+CLAUDE]:** this is a distinct *architecture/framework* line — "should the (60,5) head be a
per-action sigmoid value (scorer) or a softmax policy, and do we move to policy+value (actor-critic) à la AlphaZero?"
— separate from 1-push scorer tuning ([scorer_hacman_journal.md]) and the 2-push value diagnostic
([informed_2push_journal.md]). Keep it hypothesis → pre-registered prediction → accept/reject with numbers; tag
[USER]/[CLAUDE]; **paired matched-seed** comparisons (the E6 lesson: effects < seed-wobble ±1.7 → always pair).

**Thesis (from the 3-agent AlphaZero/MuZero sweep, see [[project_policy_value_not_q]]):** when you act via SEARCH, the
net should output a **policy prior** (which pushes) + a **value V(s)** (how good a state) — NOT a standalone Q (the
search computes Q). Soft/Gaussian action-smoothing is principled for a *policy* (cross-entropy over a distribution) but
biased for a *Q* (independent absolute values) — which retro-explains our sharp-beat-soft result (the scorer is Q-like).

**Translation feasibility:** same `EdgeCrossAttn` backbone, same (60,5) output. Only changes: output sigmoid→softmax,
loss BCE→cross-entropy, target {0,1 per cell}→{normalized distribution}. Reuse `train_classifier.py`,
`classifier_module.py` (its `soft_edge_sigma`/`soft_depth_sigma` already build soft targets), `scorer_data.py` (f_grid),
`eval_scorer.py` (hard@1/recall — argmax of a policy = argmax of the map, same metric), `resolve_robust.sh` (paired).
Add one flag: `head_mode ∈ {sigmoid_bce, softmax_ce}`. No new data for the framing study (uses the 1-push f_grid).

---

## STATE — 2026-06-09 PM (running record)
**Milestone: the measurement foundation exists.** Built the canonical car test set `namo_testset_v1`
([[project_canonical_testset]], `docs/pipeline/canonical_testset.md`) — geometry-verified 0-leak, 1-push tier
(1228 sc / 1671 eps, exhaustive) + 2-push tier (**808 pure-2-push eps** with exhaustive **F1′** in `labels/pure2push.json`).
Every hypothesis below now grades against this, not the old confusing/leaky manifests. The 1-push GATE already ran on it
(see below).

**Where we are in the ordered program** (cheapest-and-most-decisive first; each early step can kill the need for later ones):
- ✅ **H0a — GATE** (recall@k, 1-push) — DONE → hard recall@10 = 75.8% (misses are wrong-EDGE), med 97%, easy 100%.
- 🔄 **H0b — training-free first-push baseline (mean_top5)** (1-ply lookahead, graded vs F1′ on `pure2push.json`) — LAUNCHING. If a free
  1-ply lookahead surfaces the enabling first-pushes into top-k, a learned policy/value may be unnecessary. Reuses
  `scripts/sandbox/diag_leaf_s1.py` (logs per-first-push scalars + edge1/depth1 → post-hoc recall@k vs F1′).
- ⏳ H1 framing → H1.5 post-push probe → H2 self-attn → policy-only search → H3 value head → H4 deploy → H5 masked targets.
  H1+ need GPU + the `head_mode` flag (and [USER] design green-light); the free steps gate whether we go there at all.

---

## Ablation backlog (ordered; each makes one design choice concrete)
1. **[NOW] FRAMING — scorer (sigmoid-BCE) vs policy (softmax-CE), × sharp vs soft target.** Existing f_grid data.
2. **Edge self-attention** on / off / reachability-masked, × **label density** (100/25/10%). Existing data (subsample). [USER]: sparsity should favor independent edges.
3. **Value head** — present/absent; pool method; classification (Stop-Regressing) vs MSE. Needs value-target collection.
4. **Q vs policy+value in the search (deploy):** does policy-prior + V + search beat the current beam on solve@k / sims? Needs heads + search integration.
5. **Targets from SEARCH (partial, masked) vs exhaustive f_grid** — the "label what you tried, mask the rest" scheme for depth ≥2. Needs the search-target collection.

---

## H5 — SAMPLED+MASKED labels vs exhaustive f_grid [PROMOTED to after H0b — [USER] question 2026-06-09]
**[USER] framing:** "for 2-push we cannot possibly get exhaustive data — stick to that framing even for 1-push. Does
masking help given sampled data? Should we continue sampling? Is this some loop training thing?"
**Why promoted:** every later step (H3 value collection, H4, ExIt) trains on SAMPLED data; H5 decides whether that's
viable and how much sampling is enough. 1-push is the ONLY place with an oracle (exhaustive f_grid + namo_testset_v1)
— measure the lesson here, apply it on 2-push where it can never be measured. Lit grounding: this is
positive-unlabeled / partial-label learning (Xu & Denil PU-reward; iterative PU constraint learning) — unsampled
cells are UNKNOWN, not negative; treating them as negative = false-negative pessimism (the known RO failure mode).

**Design (5 conditions × 3 matched seeds, champion sharp recipe; loss mask/labels are the ONLY difference):**
| cond | labels | loss on |
|---|---|---|
| Aexh | exhaustive f_grid | all reachable (ceiling CONTROL, retrained with bce_reachable_only) |
| B5/B15/B30 | k sampled cells/scene (outcome known only there) | the k sampled cells (MASKED) |
| C15 | 15 sampled; unsampled FORCED to 0 (false negatives) | all reachable (the PU-BUG baseline) |
Confound control: ALL conditions (incl. Aexh) use `bce_reachable_only=true` so BCE scope is identical; sampling
applies to TRAIN only (val stays exhaustive); `sample_seed=seed` so seed-variance includes sampling variance.
Impl: `scorer_data.py` (`sample_k`/`unsampled_negative` → `loss_mask`) + `classifier_module.py` (training uses
`loss_mask`); data-path unit-asserts PASS (row0: 36 positives → bug-mode creates 27 false negatives).
Runner: `sage_learning/scripts/train_h5_sampling.slurm` (15 runs). Eval: `eval_scorer.py` on namo_testset_v1.

**Pre-registered predictions:**
- **H5a [CLAUDE]:** masked-B15 lands within a few pp of Aexh on easy/med recall@10 but loses meaningfully on HARD
  (hard scenes have few positives; sampling 15/75 rarely hits them → weaker hard signal).
- **H5b [CLAUDE, high confidence]:** C15 ≪ B15 on recall@k (false negatives suppress valid pushes) AND C15 scores
  saturate low/miscalibrated — masking is necessary. If C15 ≈ B15, masking doesn't matter and the PU framing is moot.
- **H5c [CLAUDE]:** monotone in k (B5 < B15 < B30 ≤ Aexh); the B30→Aexh gap tells us how much MORE sampling buys
  (Aexh ≈ k=75 avg). Gap closed by B30 → "sample more" beats "loop". Large persistent gap at B30 → justifies the
  ExIt-style relabel loop (round-2 samples where round-1 proposes).
**Decision rule:** B15 ≈ Aexh on hard → sampled collection is fine as-is for 2-push (just mask). Gap that shrinks
with k → spend sims on more samples. Gap that doesn't shrink with k → the loop (on-distribution resampling) is the
mechanism, design it for H3 collection.

**Status:** smoke PASSED (job 55856466: B15 val_loss 0.884→0.749 healthy; C15 1.92→2.12 — the bug baseline already
diverging on the exhaustive val after 2 epochs, early corroboration of H5b). **Full 15-run matrix = job 55856898**
(launched 2026-06-09 23:40, gpu,gpu-redhat, ~2.6 h/run). Next: eval all ckpts on namo_testset_v1 → paired verdict.

---

## H1 — FRAMING: policy vs scorer, and is action-smoothing a *policy* thing? [NOW]
Controlled 2×2 on the `sharp` backbone, E4 data, fixed room-split, matched seeds {1,2,3}, paired.

```
                       sharp target            soft target (Gaussian σ over edge×depth)
 sigmoid-BCE (scorer)  C1 = current champion   C2  (journal: soft was REJECTED here)
 softmax-CE  (policy)  C3                       C4  (hypothesized winner)
```
Targets: sigmoid sharp = f_grid {0,1}; sigmoid soft = Gaussian-spread f_grid (existing `_build_soft_target`).
policy sharp = normalize(f_grid) (uniform over solving cells); policy soft = Gaussian-smoothed multimodal distribution.
Metric: **hard@1** (does argmax solve) + recall@{5,10,20}, per difficulty (`eval_scorer.py`).

**Pre-registered predictions:**
- **H1a [CLAUDE] framing alone is ~a wash at sharp:** C3 ≈ C1 on hard@1 (within ±1.7), because argmax is argmax. (If C3 ≪ C1 → softmax framing hurts; if C3 ≫ C1 → it helps even sharp.)
- **H1b [CLAUDE] action-smoothing is a POLICY thing (the key test):** soft HELPS the policy but HURTS/neutral the sigmoid → **C4 > C3** AND **C2 ≤ C1**. ACCEPT if the soft×policy gain is positive (paired) while soft×sigmoid is ≤0 → confirms the theory and makes **policy+soft the new champion**. REJECT if soft hurts both framings (smoothing is just bad) or helps both (framing-independent).

**Decision rule → architecture:**
- H1b accepted → adopt **softmax-policy head + soft manifold targets**; carry into the value-head build.
- H1b rejected, H1a wash → keep the sigmoid scorer as the proposal head (simpler), revisit smoothing.

**Status:** designed, not launched (GPU compute — awaiting [USER] green-light). Implementation = `head_mode` flag in
`classifier_module.py` + relabel hook; 2×2 × 3 seeds = 12 runs, `gpu,gpu-redhat`, paired via `resolve_robust.sh`.

---

## H0a — GATE: recall@k of the champion scorer on the CLEAN test set [2026-06-09]
First measurement on the geometry-verified canonical test set (`namo_testset_v1`, `v3_test_episodes.json`, 0 train-leak —
see [[project_canonical_testset]]). Champion `sharp` ckpt (`epoch017-val_loss0.2713`), `eval_scorer.py`, 1656 episodes.

| bucket | n | success@1 | recall@5 | **recall@10** | recall@20 | rank-1st-valid median |
|---|---|---|---|---|---|---|
| hard (sr 2.8%) | 413 | 32.9 | 62.5 | **75.8** | 88.1 | 3.0 |
| med  (sr 16.8%)| 491 | 81.3 | 94.5 | **96.7** | 98.6 | 1.0 |
| easy (sr 65%)  | 752 | 99.6 | 99.9 | **100** | 100 | 1.0 |

**Verdict on the GATE (pre-registered: high recall → scorer already surfaces solutions → may not need policy+value;
poor recall on hard → learned policy justified):** SPLIT by difficulty.
- **easy/med: recall@10 ≥ 97%** → a ~10-sim search almost always contains the solving 1-push; per-action scorer is
  near-sufficient there, policy+value adds little. (Floor@10 is 98%/81% though — easy is near-saturated, weak signal.)
- **hard: recall@10 = 75.8% (vs floor 25.6%)** → real headroom; 24% of hard episodes don't even have the solving push
  in the top-10. **This is where first-push selection is the bottleneck — exactly the policy+value/search target.**

**Failure analysis (why hard misses) — actionable:** of hard fail@1, **90.3% are WRONG-EDGE** (contact point), only
6.5% right-edge-wrong-depth; depth-acc GIVEN right edge = 83.4%. So the model's depth head is fine; the gap is
**which contact edge** to push on hard scenes. ⇒ the next lever is better *edge/contact* ranking on hard (the
training-free `mean_top5` first-push ranker, then learned policy), NOT depth modeling. Pairs with the 2-push tier:
hard 1-push scenes (sr<5%) are the bridge to genuine depth-2.

Result JSON: `namo_testset_v1/stats/champion_1push_recall_gate.json`. This also end-to-end VALIDATES the test set
(eval ran clean against it).

---

## RED-TEAM RISKS (background agent, 2026-06-09; 1 of 3 reports — the other 2 pending)
**[USER] motivation: justify the machinery — don't over-apply. The red-team supports this.**

- **RISK-2 (biggest): few-sim search amplifies policy errors; per-action scorer may be the RIGHT minimal tool.**
  In the few-sim regime (our goal), the search visits ~k≪300 actions. The policy prior IS the search — if the solving
  push isn't in its top-k, the search never tries it, and V(s) can't rescue it (V doesn't score unvisited actions).
  The existing per-action scorer ranks ALL 300 (recall@10≈89% from the overnight diag) → can't miss that way.
  ⇒ **"policy+value not Q" is NOT settled for few-sim.** [[project_policy_value_not_q]] assumed "search computes Q",
  which is false for the unvisited 280 actions. DEMOTE policy-vs-Q from decision → hypothesis, adjudicated by H1 + the gate below.

- **GATE (cheap, run FIRST): policy recall@k from the existing f_grid.** Is the solving push in the model's top-k?
  High recall → per-action scorer already surfaces solutions → may not need policy+value (just a better value/ordering,
  e.g. the training-free `mean_top5` H0b). Poor recall on hard cases → learned policy justified. **This gates H3+.**

- **RISK-1: H1 tests the wrong distribution.** f_grid = INITIAL states; we deploy on POST-PUSH (mid-chain) states (OOD).
  H1 winner may not transfer. → **H1.5 [NEW]: post-push probe** — score ~50 post-push states exhaustively, re-check the
  framing/recall there. Without it H1's conclusion is contingent.

- **RISK-3: H3 bootstrap/ordering.** Value labels are search-collected, but search quality depends on the policy →
  early data biased. Fix ordering: H1/H2 → **policy-only search (no value)** intermediate → H3 collect → full H4. Likely
  needs DAgger-style re-collection, not one-shot.

- **2×2 confound [fix]:** sharp/soft sigma is a per-head hyperparam; a single σ conflates "amount" vs "goodness of"
  smoothing. Sweep σ ∈ {0,0.5,1,2 cells} (or verify the head×smoothing interaction holds across σ), don't fix one σ.

- **Missing baseline:** independent action scoring (scene cross-attn only, NO inter-action self-attn) — that's literally
  the current per-action scorer; it's the obvious control for "does inter-action attention help or hurt." Add to H2.

- **Metric gap:** hard@1 on 1-push is NOT predictive of solve@k/sims (different distribution; conflates policy/value/search).
  Add **policy recall@k** as the bridge metric; ablate policy-only vs +value separately in H4.

- **Negatives are heterogeneous:** dead-end from wrong-first-push vs impossible-config vs wrong-second-push are different;
  treating all as one "unsolvable" may teach "looks unfamiliar" not "is a dead end." Tag negative TYPE in the collection.

**Revised order:** H0a recall@k GATE (free) → H0b mean_top5 baseline (free) → H1+H1.5 (framing, +post-push probe) →
H2 (self-attn incl. independent-scoring control) → policy-only search → H3 (value collect, tag negative types) → H4.
