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

## Ablation backlog (ordered; each makes one design choice concrete)
1. **[NOW] FRAMING — scorer (sigmoid-BCE) vs policy (softmax-CE), × sharp vs soft target.** Existing f_grid data.
2. **Edge self-attention** on / off / reachability-masked, × **label density** (100/25/10%). Existing data (subsample). [USER]: sparsity should favor independent edges.
3. **Value head** — present/absent; pool method; classification (Stop-Regressing) vs MSE. Needs value-target collection.
4. **Q vs policy+value in the search (deploy):** does policy-prior + V + search beat the current beam on solve@k / sims? Needs heads + search integration.
5. **Targets from SEARCH (partial, masked) vs exhaustive f_grid** — the "label what you tried, mask the rest" scheme for depth ≥2. Needs the search-target collection.

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
  e.g. the training-free `mean_top5` Step-0). Poor recall on hard cases → learned policy justified. **This gates H3+.**

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

**Revised order:** recall@k GATE (free) → Step-0 mean_top5 swap (free) → H1+H1.5 (framing, +post-push probe) →
H2 (self-attn incl. independent-scoring control) → policy-only search → H3 (value collect, tag negative types) → H4.
