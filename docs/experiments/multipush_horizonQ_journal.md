---
status: frozen
tags: [experiment]
updated: 2026-06-11
---

# Multi-push / horizon-Q journal — ⏸ PARKED [USER 2026-06-10]

**[USER] scope decision:** "Focus only on the architectural decisions on the 1-push problem. Don't worry about the 2-push problems yet." Everything 2-push / AlphaZero / policy+value / horizon-Q moved HERE from [policy_framework_journal.md](policy_framework_journal.md) (now the 1-push ARCHITECTURE journal) for future use. Nothing in this file drives current experiments. The arch journal's H5 (masking) and H2 (self-attn) verdicts feed back into the collection design below when this line un-parks.

---

## 📖 Companion primer [2026-06-10]
[multipush_learning_primer.md](multipush_learning_primer.md) — plain-language map of all model families + training schemes + 10 case studies (3-agent lit sweep). **Independent convergence:** the sweep's recommended path (spatial Q-map + value, search-generated MC labels, supervised fit, optional 2-3 ExIt rounds, shallow verify-search, never TD at horizon 2-3) IS the H3′ design below. Also: tonight's 1-push verdicts (arch journal) pin the architecture (sigmoid-sharp + self-attn) and the data price (~30 sampled labels/state, masked).

## 🟢 UN-PARKING BUILD SPEC — every choice grounded [2026-06-11, USER+CLAUDE]
**Status:** un-parks this line. Refines H3′ below; the **gamma decision (#9) REVISES the 2026-06-10 robustness-over-optimality call** (now: prefer shorter; robustness recorded, not trained — #12). [USER] locked the 9 load-bearing forks this session; [CLAUDE] defaults the tunables (vetoable). Each choice carries a reason + grounding (literature or our own H-series). Tag key: [USER] decided · [validated] = our experiment · [CLAUDE] = defaulted.

### What we're learning
1. **Budget-conditioned horizon-Q `Q(s,a,H)`** — one net, H as input; policy=top-k, value=pool. *Why:* one object acts + evaluates + handles unknown-H by querying H; H-input generalizes the framework. *Grounding:* finite-horizon DP (value is time-indexed); Pardo "Time Limits in RL" (ICML'18); Decision Transformer (timestep cond.); Fedus multi-horizon value (ICLR'19); UVFA (ICML'15); [[project_policy_value_not_q]]. [USER framing + CLAUDE]
2. **H_max = 2** — prove recursion/value-bootstrap on the horizon we have a test set for; extends to 3. *Grounding:* namo_testset_v1 2-push tier. [USER]
3. **One head** (map = policy = value) — simplest; split only if calibration tension bites. *Grounding:* AlphaZero shared trunk; [[project_policy_value_not_q]]. [USER]
4. **Single object, 1-hop RO** — the problem is a push-sequence on ONE object opening ONE adjacent region. *Grounding:* [[project_ro_single_object]]; generator require-adjacent default. [USER prior]

### Architecture
5. **EdgeCrossAttn spatial per-edge critic (60×5)** — spatial grounding generalizes across geometry; the WHERE choice is load-bearing. *Grounding:* HACMan (2305.03942) + its ablation; our H2. [validated]
6. **Self-attn ON + Fourier PE + per-edge embed** — *Grounding:* our H2 (+4–5pp); HACMan inter-point attn. [validated]
7. **H = learned embedding over {1,2}** — standard conditioning mechanism. *Grounding:* UVFA/FiLM; DT timestep embed. [CLAUDE default]

### Target / labels
8. **Target = "solvable within H, best play"; MC/search targets, NEVER TD** — perfect short-horizon sim ⇒ MC unbiased
   + cheap; TD's edge only past horizon ~10; avoids deadly triad. *Grounding:* TD-or-not-TD (1806.01175); AlphaZero final-outcome targets; primer. [validated-direction / USER prior]
9. **Gamma discounting (prefer shorter): 1.0 / γ≈0.9 / 0** — single-map argmax prefers the cheaper solution; one-query deploy; depth-readable value; zero extra sims. **REVISES 2026-06-10 robustness-over-optimality.** *Grounding:* discounted return = standard cost-aware value (Bellman). [USER 2026-06-11]
10. **Binary per-push success (20% reachable bar, FROZEN)** — per-push outcome is inherently binary; the bar is the wired criterion. *Grounding:* `region_opening._validate_opening`; test set calibrated to it. [USER]
11. **Recursion `Q(s,a,H)=open OR V(child,H−1)`; budget decrements through the transition** — budget is the scarce resource (expensive oracle); clean decrement is what enables truncated search. *Grounding:* Bellman finite-horizon; budget = direct consequence of the expensive-oracle setting. [derivation]
12. **Record success-fraction (robustness) alongside gamma; re-record per round** — reactive (no-verify) regime needs robustness; ~free to log (don't early-exit the second level); it's POLICY-CONDITIONED ⇒ must Reanalyze each round, never freeze. *Grounding:* journal "labels recorded per-horizon (reversible)"; Reanalyze; reactive deploy need. [USER + CLAUDE reconciliation]

### Value head / loss
13. **Pool = top-k mean, NOT raw max** — *Grounding:* H0b (mean_top5 34.5 > maxP 24.6 @1; max is fluke-dominated). [validated]
14. **Classification value head (HL-Gauss bins), not regression** — *Grounding:* Stop-Regressing (2403.03950); primer "we already follow this style". [CLAUDE default]
15. **PU masking (untried = UNKNOWN)** — untried ≠ failure; FNs catastrophic. *Grounding:* our H5 (untried-as-fail −15pp); PU learning. [validated]
16. **Loss: balanced masked BCE first; recall-tilt later** — ranking by calibrated probability is Bayes-optimal (PRP); tilt toward top-k only after measuring the 26→70 hard@1→@10 gap. *Grounding:* Probability Ranking Principle (Robertson'77); our H1 (sigmoid-sharp beat softmax). [validated + CLAUDE]

### Data generation
17. **Climb the horizon ladder (H=1 model-free → H=2 search-distilled)** — bottom rung needs no model (sim labels directly), so no cold-start chicken-and-egg; higher rungs reuse lower (bootstrap). *Grounding:* value iteration; curriculum; Bejjani RHP "planner→supervised→refine" (1803.08100); Contact-MCTS (2206.09023). [CLAUDE + USER cold-start]
18. **Regenerate H=1 @ 20% bar + KEEP dead-ends** — old f_grid is old-bar + 0 hopeless scenes ⇒ value can't represent "low"/unsolvable. *Grounding:* H0b diagnosis #2; bar-mismatch re-eval. [USER]
19. **Sample ~30 cells/scene, masked** — ≈ exhaustive at a fraction of sims. *Grounding:* our H5. [validated]
20. **H=2 only on the informative subset (no 1-push opener)** — concentrate sims on the setup signal; easy scenes' H=2 labels are free by monotonicity. *Grounding:* value-of-information; budget-Q monotonicity. [CLAUDE]
21. **H=2 leaf: verify early, bootstrap late** — Q₁ is OOD on post-push early (H0b) ⇒ verify; trust value only once in-distribution. *Grounding:* H0b; CQL/IQL pessimism (2006.04779 / 2110.06169). [CLAUDE]
22. **H=2 collection harvests post-push H=1 labels for free** — the s′ you generate ARE the post-push data H0b requires. *Grounding:* H0b requirement. [efficiency]
23. **Tag negative TYPE** (dead-end / useless / second-unsolvable) — heterogeneous negatives, don't collapse. *Grounding:* journal red-team item. [CLAUDE]

### Sampling / exploration (the bootstrapping discipline)
24. **First-push selection: uniform / uncertainty, NEVER policy-confidence** — the 1-push policy is STRUCTURALLY blind to setup pushes (warm-start paradox); confidence-gating starves the exact signal we need. *Grounding:* PUCT (guide-but-don't-gate); offline-RL coverage assumption. [CLAUDE, USER-prompted]
25. **Exploration floor ≥ 25–30% every round** — iteration heals only REVISITED regions; floors prevent policy-gated blind spots. *Grounding:* ε-greedy / Boltzmann (HACMan's trick); exploration lit. [CLAUDE]
26. **Acquire by disagreement, not confidence** — surface buried winners; doubles as the aleatoric-floor probe. *Grounding:* query-by-committee (Seung'92); Bootstrapped DQN (Osband'16); RND/ICM. [CLAUDE]
27. **Ramp policy-bootstrap explore→exploit over rounds** — bootstrap only as fast as the policy earns trust. *Grounding:* AlphaZero temperature annealing. [CLAUDE]

### Iteration / training
28. **ExIt/DAgger 2–3 rounds, Reanalyze, not one-shot** — search labels are biased by the current policy. *Grounding:* ExIt (NeurIPS'17); DAgger (Ross'11); Reanalyze (2104.06294); our RISK-3. [validated-direction]
29. **DAgger on the policy's own greedy-rollout states** — no-verify reactive visits policy-CREATED post-push states; must be in-distribution there. *Grounding:* DAgger covariate shift; reactive-extensible goal. [CLAUDE, USER Q8]
30. **Budget-cond training: H=1 first, then mixed batches + replay** — avoid H=1 drowning sparse H=2; trunk transfers; replay avoids forgetting. *Grounding:* continual-learning replay; multi-task training. [CLAUDE]
31. **Warm-start encoder from champion scorer; RE-INIT the value / H head** — reuse the validated 1-push representation, but the high-H value canNOT inherit the 1-push head (warm-start paradox poisons setup values). *Grounding:* transfer learning; the warm-start paradox. [CLAUDE]

### Deploy — BOTH regimes (standing lens [[feedback_search_nosearch_lens]])
32. **No-search:** query `Q(s,·,H)` at decrementing budget, top-k, execute/verify. Lookahead amortized; needs strong HIGH-H head. *Grounding:* HACMan greedy argmax deploy. [lens]
33. **Search:** net = prior (top-k breadth), sim = expand, value @ leaf at `H−1`, back up. Truncated explicit search + learned value covers the depth. *Grounding:* AlphaZero (value truncates the tree); TD-MPC2; Contact-MCTS. [lens]
34. **Unknown difficulty → iterative-deepen over H** (no-search trusts V(s,H) readout; search deepens + verifies). *Grounding:* iterative deepening; the difficulty-readout. [USER-derived]
35. **No-verify reactive = closed-loop greedy argmax** — re-ground on the real state each step so errors don't compound. *Grounding:* HACMan deploy; MPC / receding-horizon re-planning. [USER Q8]
36. **Verify = check COMPLETE proposals (not search); recall@k suffices** — perfect sim ⇒ propose-and-check; the map needs recall@k, not rank-1. *Grounding:* hacman_comparison verify edge; TAMP feasibility checking. [validated-framing]

### Evaluation
37. **hit@k (recall@k) on namo_testset_v1 (20% bar), BOTH regimes, + post-push slice + dead-end slice** — hit@k is the deploy metric; both regimes per the lens; slices probe H0b's blind spots (post-push reliability; does it say "low" on hopeless?). *Grounding:* deploy objective = success@k; H0b; the canonical test set. [CLAUDE]

**Still to pin by experiment (not yet grounded — open tunables):** γ exact value; k₂; informative-subset threshold; dead-end ratio; #ExIt rounds; recall-tilt timing; one-head-vs-split.

---

## Original thesis (now revised — see H3′ below)
From the 3-agent AlphaZero/MuZero sweep ([[project_policy_value_not_q]]): when you act via SEARCH, the net should output a **policy prior** + a **value V(s)** — NOT a standalone Q (the search computes Q). Soft/Gaussian action-smoothing is principled for a *policy* (cross-entropy over a distribution) but biased for a *Q* (independent absolute values) — retro-explains the sharp-beat-soft result (the scorer is Q-like). **Red-team demotion:** in the FEW-SIM regime the search visits k≪300 actions, so "search computes Q" is false for the unvisited ones; the per-action scorer ranks ALL 300 and can't miss that way ⇒ policy-vs-Q was demoted from decision to hypothesis. H0a/H0b then measured the free options (below).

## H3′ — HORIZON-Q: the converged design [2026-06-10 synthesis, USER+CLAUDE discussion]
**Decision trail (revises the thesis):** ONE function, not two — Q_H(s,a)="this push leads to success within the remaining budget", same EdgeCrossAttn + per-cell sigmoid training. Policy = top-k(map) [USER: "treat the ranker as a policy"]; Value = max(map) (calibration head only if max proves optimistic — H0b showed maxP is bias-prone, mean_all more honest → pooling may be the patch). WHY Q-not-policy-CE for sampled data: per-cell labels are ABSOLUTE facts that survive sampling+masking; softmax-CE targets are RELATIVE verdicts that sampling corrupts ("best of the 15 sampled" can be a lie). A policy also cannot express hopelessness (sums to 1) — the no-hopeless-scenes diagnosis requires it. H1 retains one card: if CE ordering wins big on exhaustive data, consider CE-finetune for ordering.

**Target definition (precise):** Q_h(s,a) = 1[region opens within h pushes, starting with a from s]. Training label = the collection search's empirical return: opened directly→1; ≥1 sampled follow-up opened→1 (graded variant: fraction of sampled follow-ups that worked = route robustness); all sampled follow-ups failed→soft 0; untried→masked. Monte-Carlo/search returns, NEVER the model's own predictions (no TD/deadly-triad); iterate = regenerate by fresh search (Reanalyze pattern). Known wrinkle: s0 rows carry h=2 labels, s1 rows h=1 (state doesn't encode budget) — fallback if miscalibrated: budget as input token.

**Data = the 5 species:** s0 direct openers, s0 ENABLERS (F1′-style), post-push s1 states with their 1-push labels, HOPELESS s1 states (all-zeros — mandatory, diagnosis #2), soft negatives (budget-limited). Sampled k per state (k = arch-journal H5 verdict) + masked loss; collection = the tagged depth-2 machinery pointed at TRAIN scenes.

**[USER] DECISION — robustness over optimality:** when a dense 2-push route and a rare 1-push needle coexist, PREFER the easy 2-push. ⇒ single blended head; labels still recorded per-horizon (reversible). Eval = success within push budget, NOT min-push. [CLAUDE honesty note: the binary label does NOT inherently encode route density — the dense-route preference emerges only via sampling noise/recognizability/calibration side-effects; if wanted BY DESIGN, use the graded (success-fraction) label. Verification-search makes the ranking choice non-critical either way.]

**Search compatibility:** Q orders, sim VERIFIES top-3 (vs the beam's blind ~49), V=max prunes; every verified push = a new training sample (the ExIt loop, if H5c says random sampling has a ceiling). **Target to beat (pre-registered):** H0b's 34.5% @1 at ~49 sims/scene — beat it at ZERO lookahead sims.

---

## H0b RESULT — training-free first-push baseline vs exhaustive F1′ [2026-06-10, FINAL]
**Setup:** 787 pure-2-push scenes; per scene, EVERY reachable first-push simulated (38,689 sims), the post-push state scored by the champion, first-pushes ranked by training-free scalars, recall@k graded vs the exhaustive F1′ (`labels/pure2push.json`). Graded on the 391 episodes where ≥1 enabling first-push was inside the swept candidate set (coverage filter — measures RANKING quality given coverage). Result: `namo_testset_v1/stats/fpv_step0_final.json`.

| ranker | @1 | @3 | @5 | @10 | @20 |
|---|---|---|---|---|---|
| mean_top5 | **34.5** | 52.9 | 63.4 | 72.6 | 90.3 |
| mean_all | 30.9 | 53.7 | 62.7 | **79.0** | 91.8 |
| maxP | 24.6 | 42.7 | 51.9 | 65.5 | 85.7 |
| random floor | 11.8 | 29.7 | 43.0 | 64.6 | 86.5 |

(95% CI ≈ ±4.7pp @1, ±4.3pp @10, n=391.)

**Verdicts:**
- **ACCEPT: real top-rank signal** (34.5 vs 11.8 @1 = ~9 SE; 3× random).
- **ACCEPT (operative): NOT sufficient for few-try selection** — ≈floor by @10-20. ⇒ learned first-push value justified by measurement. Costs ~49 sims/scene at deployment for a 34.5% pick — the baseline to crush at 0 sims.
- **Diagnosis #1:** champion SATURATES on post-push states (~0.99 on dead s1's — OOD; never trained there).
- **Diagnosis #2 [USER catch, verified]:** training data has **0 hopeless scenes** (all 98,387 rows have ≥1 valid push; mean 54% of reachable succeed, p10=12.5%) — the model never learned that "all-low" is a legal output. ⇒ HARD REQUIREMENT: value data must include dead-end/unsolvable states.
- **Surprise:** mean_all > mean_top5 at k≥10; maxP worst (single-cell flukes dominate the max).
- **Cost note [honest]:** verdict stable at ~300 episodes; full 787 bought CI tightness only. Next time: half. **H0 pair CLOSED** (H0a in the arch journal + this). Both free options measured; both insufficient where it matters.

---

## Parked backlog (was items 3-5 of the arch journal's backlog)
3. **Value head** — present/absent; pool method; classification (Stop-Regressing) vs MSE. Needs value-target collection. (Largely subsumed by H3′: V = max/pool of the Q map; separate head only if max is miscalibrated.)
4. **Q vs policy+value in the search (deploy):** does Q-ordered + sim-verified search beat the current beam on solve@k / sims? Needs H3′ model + search integration.
5. **Targets from SEARCH (partial, masked) vs exhaustive** for depth ≥2 — superseded by the arch journal's H5 (the 1-push oracle version of exactly this question).

## Parked red-team items (2-push / pipeline-ordering)
- **RISK-1:** 1-push arch verdicts may not transfer to POST-PUSH states (OOD). → H1.5 post-push probe before trusting any arch verdict for the multi-push build. (The depth-2 pkls already contain exhaustive F(s1) per expanded a1 — the probe's answer key needs only a parser, no new sims.)
- **RISK-3:** value labels are search-collected; search quality depends on the policy → early data biased. Ordering: arch verdicts → policy-only search → collect → iterate (DAgger-style, not one-shot).
- **Metric gap:** 1-push hard@1 is not predictive of solve@k/sims; use policy recall@k as the bridge metric; ablate policy-only vs +value separately at deploy.
- **Negatives are heterogeneous:** tag negative TYPE in collection (wrong-first-push / impossible-config / wrong-second-push) — don't collapse to one "unsolvable".

## Assets ready for un-parking
- `namo_testset_v1` 2-push tier: 808 pure-2-push episodes with exhaustive F1′ (`labels/pure2push.json`).
- The tagged depth-2 collection machinery (`region_opening.py` chain_depth/parent tags + `build_2push_validset.py`).
- H0b leaf dumps (38,689 scored post-push records) — seed data for a post-push probe (TEST scenes; do not train).
- `policy_value_v1` dataset home + collection-design rules (README).
