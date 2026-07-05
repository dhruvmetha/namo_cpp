---
status: ref
tags: [experiment]
updated: 2026-06-10
---

# Region Opening ↔ HACMan: the problem-setting parallel, and what transfers

**Purpose [USER 2026-06-10]:** show people how our NAMO Region-Opening setting parallels HACMan's (Zhou et al., CoRL 2023, arXiv:2305.03942; HACMan++ RSS 2024, arXiv:2407.08585), and how their findings apply to our work. Slide-ready: each section is one talking point.

---

## 1. The shared problem shape: "WHERE to touch × HOW to push"

Both problems decompose a nonprehensile manipulation action into a **discrete spatial choice grounded on the object's geometry** plus a **parameterized motion**:

| | HACMan | Region Opening (ours) |
|---|---|---|
| Task | push/flip an object to a 6D goal pose | push ONE object so a goal region becomes reachable |
| WHERE (discrete) | contact point on the object's **point cloud** (~hundreds of points) | contact edge on the object's perimeter (**60 contact points**) |
| HOW (parameterized) | continuous 3D end-effector motion after contact | discrete push **depth** (5 primitive lengths) |
| Effective action space | ~hundreds × continuous | 60 × 5 = **300 discrete** |
| Action representation | per-POINT score map over the cloud | per-EDGE score map over the contact manifold |
| Success criterion | pose within threshold | region opens (wavefront reachability flips) |
| Horizon | up to 10 greedy steps | 1–3 pushes |

**The deep similarity:** the learned object is the same — a **spatially-grounded per-action critic**: a map that paints "how good is acting HERE" onto the object's geometry. Our `EdgeCrossAttn` scorer is HACMan's critic with edges instead of cloud points and a depth axis instead of a continuous motion head.

## 2. Why this representation is the load-bearing choice (their evidence + ours)

- **HACMan's ablation:** remove the spatial per-point action grounding (use a global state→action policy instead) → performance collapses from ~83% to near zero on their hardest tasks. The per-point map IS the method.
- **Their headline numbers:** 83.3% (train objects) / **89.1% (unseen instances)** / 82.7% (unseen categories) vs best non-spatial baseline ~22%. The spatial grounding is what generalizes across geometry.
- **Our independent echo (H2, 2026-06-10):** removing inter-edge self-attention from our per-edge map costs 4–5pp hard@1 in every data regime — and our H0a failure anatomy shows the residual challenge is exactly the WHERE choice (90% of hard misses are wrong-edge, not wrong-depth).

## 3. Where the settings genuinely differ (and what each difference buys)

| dimension | HACMan | ours | consequence |
|---|---|---|---|
| Training signal | online RL (TD3, ~1M sim steps, ms-fast sim) | **supervised labels from a perfect sim** (outcome of executed pushes) | our sim costs ~1s/push (full controller rollout) → RL-by-exploration is off the table; supervised/search-labeled is the right scheme (lit: ExIt/MC-targets at short horizon) |
| Labels per state | TD-bootstrapped Q | ground-truth binary outcomes; **sampled ~30/state + masked** (H5-validated) | we get calibrated probabilities + no deadly-triad machinery |
| Goal encoding | per-point "goal flow" (where each point must go) | goal region channel in the scene crop | theirs is a richer target spec — relevant if we ever do pose-targeted pushing |
| Action selection at deploy | greedy argmax per step, ~10 steps | top-k + **sim verification** (perfect sim as oracle), 1–3 pushes | we can verify before committing — they can't (real-world deployment) |
| Multi-step credit | implicit via TD through greedy rollouts | explicit: horizon-labels ("succeeds within budget", parked H3′ plan) | at horizon 2–3, explicit MC labels are lower-bias than TD |
| Robot | floating gripper / arm | differential-drive car (push = navigate + contact + track) | our "one push" is a long closed-loop maneuver — the 1s cost, and why primitives are precomputed |

## 4. What we already imported from HACMan (and validated)

1. **The per-action spatial critic architecture** — our scorer line was built HACMan-faithful from E2 on (per-edge tokens cross-attending to the scene); tonight's H1/H2 re-validated sigmoid per-cell scoring + inter-token attention against alternatives, with pre-registered ablations.
2. **Hybrid discrete/parameterized action thinking** — (edge, depth) ≅ (contact point, motion param).
3. **Boltzmann-over-Q contact selection (training-time exploration)** — their trick for sampling contacts ∝ exp(Q/β); ours appears as top-k proposal + verification at deploy.

## 5. What we can still take from HACMan / HACMan++ (the apply-next list)

1. **Goal-flow-style conditioning** — encode *where the object should end up* per contact point, not just which region should open. Relevant the moment Region Opening needs pose-aware placement (e.g., "open the region WITHOUT blocking the corridor").
2. **HACMan++'s primitive vocabulary** — their RSS'24 extension scores (primitive-type, location, params) jointly: one map per primitive type. Our analog: adding "navigate", "pull", or multi-object choice as primitive types under one scoring head — the natural n-push/multi-object generalization.
3. **Unseen-geometry generalization protocol** — their unseen-instance/unseen-category eval splits are a cleaner generalization story than our room-level holdout alone; worth adopting for the paper.
4. **Sim-to-real recipe** — their zero-shot transfer (50% real) came from the abstract point-cloud state. Our abstract 5-channel mask crops are the same bet; their result is evidence it can survive the gap.

## 6. What we add over HACMan (the contribution slide)

1. **Multi-step with certified labels:** HACMan's horizon is implicit (greedy TD). We build *exhaustive* 1-push answer keys + search-generated enabling-push labels (F1′) on a geometry-verified test set — accuracy claims per difficulty bucket that an RL-trained critic can't make.
2. **The data-cost result (H5):** a per-action critic trains to exhaustive-quality from **~30 sampled outcomes/state with masked loss** — and the naive alternative (untried=failure) is catastrophic (−15pp). HACMan never had to face this (their sim is free); for slow-sim domains this is the enabling recipe.
3. **Perfect-sim deployment loop:** Q-map orders → simulator *verifies* top-k → value-as-max prunes. HACMan must trust its argmax; we get to check ours for ~3 sims.
4. **A measured baseline ladder** on a public-able benchmark: random floor → geometric oracle → recycled 1-ply lookahead (49 sims, 34.5%@1) → learned map (0 sims) — each rung quantified.

## 7. One-line summary (the closing slide)

> **Region Opening is HACMan's problem setting transplanted to navigation-among-obstacles — same
> spatially-grounded per-action critic, same where×how action factorization — but with a perfect (slow)
> simulator instead of cheap RL rollouts. That swap changes the optimal training scheme from online TD to
> supervised search-labeled masked learning (our H5 recipe), and changes deployment from trust-the-argmax
> to verify-with-the-sim. Everything HACMan proved about spatial action maps transfers; everything about
> TD training does not — and we have the ablations for both claims.**

**Sources:** HACMan (arXiv:2305.03942, CoRL'23), HACMan++ (arXiv:2407.08585, RSS'24), our journals: [policy_framework_journal.md] (H0a/H1/H2/H5 verdicts), [multipush_horizonQ_journal.md] (H3′ plan), [multipush_learning_primer.md] (scheme taxonomy + case studies).
