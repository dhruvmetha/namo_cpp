---
status: ref
tags: [experiment]
thread: scorer-search
updated: 2026-07-01
---

# HYPOTHESIS — Policy + Value decoupled search (research; to be tested)

> **Filed 2026-07-01 [USER research hypothesis — NOT a committed design].** A hypothesis to run experiments against, not
> a decision.
>
> **HYPOTHESIS (falsifiable):** splitting the RO best-first search into a **policy π (action proposal / ranking)** and a
> **pure grounded value V (state selection / frontier ordering)** — where V is trained on FIXED max-existence targets
> (NOT bootstrapped, NOT findability/density) — **beats the single `combine=q` head on sims-to-solve at a fixed
> solve-rate.**
>
> This doc = the hypothesis + the *exact* experimental setup to test it + a critical prior assessment of where it likely
> helps vs degenerates. Sits on the horizon-Q line ([horizon_q_redesign_execution.md](horizon_q_redesign_execution.md)
> Stage 0/1; [horizon_q_search_redesign_journal.md](horizon_q_search_redesign_journal.md) thesis). Tags [USER]/[CLAUDE].
> Constraint: **NO exhaustive GT** — targets are search-derived (ExIt), never enumerated.

## The idea [USER]
Best-first search where the two roles are split:
- **Policy π(a|s)** — ranks/proposes actions (which pushes to branch, and the within-state branch order). Used ONLY to select expansions.
- **Value V(s)** — orders the frontier: which reached state to expand next ("state selection").

This is the actor-critic / AlphaZero-PUCT / **PHS\*** decomposition (policy prior + value heuristic). We already cloned PHS\*/LevinTS during the earlier research.

## Exact TRAINING loop (ExIt form)
Shared crop-encoder, two heads: **π-head** → (60×5) push logits; **V-head** → scalar v(s)∈[0,1]. Env available. Per iteration i:
1. **Collect.** For each training scene s0, run the inference search with (π_i, V_i), budget B, until solve/exhaust. Log every reached state s_j (crop), every action tried at s_j, whether s_j's subtree solved, sims-to-solve from s_j.
2. **Targets** (from the search's own outcome — no oracle):
   - **V:** `v*(s_j) = γ^(sims-from-s_j-to-solve)` if the subtree solved, else `0`. (γ-discounted **existence** cost-to-go.)
   - **π:** at s_j, solving-path action(s)=1, other *tried* actions=0, *untried*=masked (recall/ranking over reachable only).
3. **Train.** V-head = HL-Gauss/BCE to `v*`; π-head = masked ranking-CE. Replay buffer.
4. Repeat. Cold start: π_0 = current scorer; V_0 seeded from 1-push exhaustive finish labels + one search pass.

## Exact INFERENCE loop
```
frontier = max-heap keyed by V;  push (s0, V(s0));  sims = 0
while frontier and sims < B:
    s = pop-max-V()                    # V = STATE selection
    for a in π.top_k(s, k):            # π = ACTION proposal / branching
        s' = env.step(s, a); sims += 1 # one sim per branch
        if goal_open(s'): return SOLVED, sims
        push (s', V(s'))
return FAIL
```
(reactive = k=1, B=depth; search = larger B/k.)

## ⭐ KEEP THE VALUE PURE [USER decision 2026-07-01]
- V trained on **FIXED grounded targets** (one-shot search returns / exhaustive-where-available), **NOT bootstrapped** (`V←r+γV(s')` self-reference), **NOT findability/density** (top-k-mean).
- **Why stable:** bootstrap = moving target = the classic value-divergence (Stage-1 bootstrap wobble was this). Pure grounded regression has no self-reference → stable by construction.
- **Why max/existence, not density:** (a) finish is near-oracle → need ONE continuation, not many; (b) Stage-3 measured depth(max) > density(top-k-mean) 34.1 vs 30.3; (c) an existence value is ~policy-independent → no staleness even when π improves (so no moving target from *either* end).
- **Net:** pure grounded max-backup depth value = the stable (non-bootstrapped) form of `qboot_depth`.

## Critical assessment [CLAUDE — don't-assume]
1. **A value for a *budgeted* best-first is budget-conditioned → it IS the Horizon.** `v*=solvable within B sims` depends on B. That's Hz, and we measured **Hz ≈ NoHz** (thin hard edge). Fix B → existence/depth value = `qboot_depth 34.1 < NoHz 40.7`. So this is largely two variants we already ran, both failed to cleanly beat NoHz. **Biggest red flag.**
2. **At H_max=2 the machinery degenerates.** The only frontier is {s1}; `V(s1)`=one-push-solvability=**the finish value, already near-oracle** (top1 58%). The only non-trivial value is `V(s0)`=**setup** (the known bottleneck). So it's not a general V^π — it's {finish (have it), setup (hard)}. **The design only earns its keep at H≥3.**
3. **Can't order the frontier by V(s′) without simming to reach s′** → V orders *already-reached* states; pre-sim signal is still π/q. Transition ~deterministic ⇒ **q(s,a) ≈ V(T(s,a))** — the current `combine=q` frontier *is* the next-state value at depth 2. So V-ordering vs q-ordering ≈ same signal; the real content is **two heads / two losses**, not the search structure.
4. **Circularity** (if bootstrapped) — mitigated by KEEPING V PURE (above).
5. **Coverage cap.** Loss masks unreachable (confirmed: `loss_mask=r_mask`, hl_gauss masked). V/π get zero gradient off the reachable/tried set → trained on the search's own trajectories, blind exactly where π avoids (where hard openers hide). Needs ε-branching / uncertainty or the frontier ordering degenerates. At H=2 the oracle finish hides this.

## Decision / where this stands [CLAUDE]
- Right shape for **deep** search (H≥3); at our **H=2 it re-derives {finish, setup-ranking, horizon}**.
- **The one honest testable claim:** separate calibrated V-head (grounded, pure, max-backup) + recall-tuned π-head **vs the single q-head**, measured on **sims-to-solve at fixed solve-rate** (`combine=q` baseline).
- The setup bottleneck is a **discrimination** problem (solvable vs dead first-pushes); the split doesn't change that — the lever is the **setup-ranking loss + exploration-driven coverage** (search discovering setups the labels miss).
- **NOT recommended: the full policy-iteration loop.** RECOMMENDED next if we pursue this: the minimal one-round ablation (two heads, two losses, grounded pure V, no iteration) against `combine=q`.
