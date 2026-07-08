---
status: ref
tags: [experiment]
thread: scorer-search
updated: 2026-06-13
---

# Horizon-Q — Related Work (claim-by-claim, weekend lit pass 2026-06-13)

> **⚠ HISTORICAL framing (2026-07-06): budget/horizon-conditioning was DROPPED** (measured ≈ no-horizon, **NoHz** ahead — 40.7 vs 34.1; at ≤2 pushes the budget input has nothing to do). This threat/novelty audit still stands as related-work positioning, but where it defends "budget-Q / budget-conditioning" as our design, read that as the **historical** line — the live model is a single value/ranker (NoHz) whose job is first-push (setup) ranking. Current framing: [../problem_and_approach.md](../problem_and_approach.md); the positive-only through-line is in [../research/positive_only_value_learning_litmap.md](../research/positive_only_value_learning_litmap.md).

4-agent targeted pass (Sonnet finders → Opus synthesis). Each claim axis: nearest neighbor, what they did, **our delta**, threat to novelty. Numbers we defend with: reactive solve@1 **22.9** (8.4× the 2.7 random floor, 0 sims); H=2 query **5.5× floor** at finding setups / H=1 **at floor**; budget-Q@H1 beats the H=1 champion M2b **+5.5pp**; dead-end rows (51% of data) **+3.2pp** hard@1. (Search/sim-budget *curve* is NOT yet defensible — flawed beam being rebuilt as value-as-leaf-evaluator best-first.)

---

## Claim 1 — Amortized learned search (value-as-leaf-evaluator) for expensive-sim push manipulation

**Nearest neighbors:**
- **SAVE** — Hamrick et al., "Search with Amortized Value Estimates," ICLR 2020 (1912.02807). Learned Q priors guide MCTS; MCTS returns improved Q-targets; strong perf at *small* search budgets. **Delta:** physics-puzzle/Atari (cheap sims), no continuous push action space, no budget-conditioned Q(s,a,H), no reactive(0-sim) regime reported. **THREAT: HIGH** (closest *algorithm*).
- **Bejjani et al.**, "Planning with a Receding Horizon for Manipulation in Clutter using a Learned Value Function," Humanoids 2018 (1803.08100). Learned cost-to-go bootstrapped from a planner guides a physics-sim push search. **Delta:** multi-object clutter rearrangement, value is *unconditional* on budget, no discrete contact-edge catalog, no reactive vs search curve. **THREAT: HIGH** (closest *domain*).
- **DeepCubeA** — Agostinelli et al., Nature MI 2019. Learned cost-to-go as the leaf heuristic in batched **weighted A\*** (inadmissible, accepts near-optimal). **Delta:** combinatorial puzzles, perfect free transition (no sim cost), no reactive regime. **THREAT: MED** — but it's our **precedent** for using an inadmissible learned heuristic in search (cite positively).
- **AVO** — Chen et al., "Amortized Value Optimization for Contact Mode Switching," 2510.07548, 2025. Offline value as terminal cost in trajopt → 50% budget reduction on screwdriver-turning. **Delta:** dexterous multi-finger + gradient trajopt, not discrete-push best-first; no budget-Q. **THREAT: MED** (recent).
- Also: ExIt (NeurIPS'17), AlphaZero/MuZero, V-GPS (CoRL'24 value re-ranks policy samples), TD-MPC2 (LOW).

**Verdict:** No single work = budget-conditioned Q + reactive-vs-search + contact-edge push in an expensive-sim domain. SAVE (algorithm) + Bejjani (domain) are the pair to beat; DeepCubeA legitimizes the inadmissible-learned-heuristic search.

## Claim 2 — Conditioning a manipulation critic on REMAINING PUSH BUDGET H  ⚠ biggest threat

**Nearest neighbors:**
- **C-Learning** — Eysenbach et al., "Horizon-Aware Cumulative Accessibility," ICLR 2021 (2011.12363). Single network C(s,g,**H**) = prob of reaching g within H steps, integer-horizon input, *horizon- dependent behavior* (different H → different path). **Delta:** H = reachability **timesteps in navigation**, not a count of discrete manipulation **pushes**; their horizon-mismatch is "safer/slower" not "hunts setups & dilutes openers"; no manipulation. **THREAT: MED-HIGH** (closest structural analog).
- **TDM** — Pong et al., ICLR 2018 (1802.09081). Q(s,a,g,**τ**), τ = remaining timesteps, decremented per step. **Delta:** continuous clock-time, not push count; treats horizon as a hyperparameter, no non- subsumption *finding*. **THREAT: MED.**
- LOW: UVFA (goals not budget), Time-Limits/Pardo (aliasing fix), Decision Transformer/RvS (return-to-go), Qureshi time-indexing (absolute step, policy not critic), Ferber NN-heuristics (unconditional h).

**Verdict:** Remaining-**push**-budget conditioning of a manipulation critic, queried **budget-matched**, has no direct prior. Nearest = C-Learning (but timestep/navigation). **Our H=2-dilutes-H=1 non-subsumption finding appears unreported.** ⇒ The paper must explicitly contrast C-Learning (horizon *meaning* + domain).

## Claim 3 — Dead-end / unsolvability supervision sharpening a critic (measured ranking gain)

**Nearest neighbors (no HIGH):**
- **Steinmetz & Hoffmann**, "Search and Learn: Dead-End Detectors, Traps, and Trap Learning," IJCAI 2017. Learns conjunctive nogood representations of dead-end/trap states during *symbolic* planning search. **Delta:** symbolic planner + prune-and-skip, gain = task **coverage**, not a continuous push critic ranked by **precision@k**. **THREAT: MED** (nearest in spirit).
- VINS (ICLR'20, off-manifold negatives for conservative value), Contrastive-RL (ICLR'24, hard negatives from far states), Ståhlberg unsolvability heuristics (IJCAI'21), CQL (pessimism) — MED/LOW.
- Fatemi dead-ends (ICML'19), Leave-No-Trace, medical dead-ends — LOW (exploration-safety / different domain).

**Verdict:** Supervising a **push critic** with explicit dead-end rows and **measuring a precision@k gain** appears **novel** — prior dead-end work is exploration-safety, symbolic-coverage, or covariate-shift, never a robotics ranking gain. Lowest-threat axis. (Our +3.2pp is the headline support number.)

## Claim 4 — Masked / sampled partial-label supervision of a dense push-action critic

**Nearest neighbors:**
- **HACMan** — Zhou et al., CoRL 2023 (2305.03942). **Our architectural lineage** — per-contact-point critic. Trained by *online RL* (TD on the chosen point) → never faces the offline partial-label problem. **Delta:** offline planner data, only ~30/300 cells tried, masked **classification** loss, fixed discrete push-edge grid. **THREAT: HIGH** — but it's the **foundation we cite**, not a competitor.
- **Where2Act** — Mo et al., ICCV 2021. Per-point affordance, **masks untried points**, samples (not exhaustive); labels geometrically-infeasible orientations negative. **Delta:** near-exhaustive *local* sampling (~10k positives) avoids our structural 10%-coverage sparsity and our explicit **tried-failed (negative) vs untried-reachable (unknown, mask)** distinction. **THREAT: HIGH** (closest *supervision* analog).
- VPG (IROS'18), HACMan++ (2024), Transporter Nets (CoRL'21) — dense map + single-cell update, but online RL sidesteps false-negatives — MED. Spatial Action Maps, VAT-MART — LOW.
- **Stop-Regressing** — Farebrother et al., ICML 2024 (2403.03950). Classification value head (our HL-Gauss head). **Cite positively** — head-design support, LOW threat.

**Verdict:** The specific problem — distinguish **untried-reachable (mask) vs tried-failed (negative)** over a fixed push-edge grid from offline planner data — is not directly addressed. Nearest = Where2Act (but near-exhaustive sampling, no false-negative framing). HACMan = lineage to acknowledge, not a threat.

---

## Synthesis — novelty verdict

**The COMBINATION is novel.** Each axis has one near-neighbor, but no prior work combines: (a) remaining- **push**-budget conditioning, (b) dead-end supervision with a **measured ranking gain**, (c) masked/sampled partial-label supervision of a dense **push-edge** critic, (d) **reactive↔search** amortization in an expensive-sim push-manipulation domain. Several specific findings appear **unreported**: the **H=2-dilutes- H=1 non-subsumption**, the **dead-end +3.2pp ranking gain**, and the **false-negative-avoidance over reachable-untried cells**.

**TOP-3 THREATS (must out-position):**
1. **C-Learning** (Eysenbach ICLR'21) — horizon-conditioned single-net value. *Delta:* push-count vs timesteps, manipulation vs navigation, H2-dilutes finding new.
2. **Where2Act** (Mo ICCV'21) — masked sampled affordance supervision. *Delta:* structural 10%-sparsity + tried-failed-vs-untried distinction; they sample near-exhaustively locally.
3. **SAVE / Bejjani** — amortized learned value for (push-)search. *Delta:* no budget-Q, no reactive baseline, no contact-edge catalog; games/clutter-rearrange vs single-object region-opening.

**TOP-3 SUPPORT (cite as foundation):**
1. **HACMan** (CoRL'23) — the per-edge contact critic architecture we build on.
2. **DeepCubeA** (Nature MI'19) — precedent for a learned (inadmissible) heuristic in weighted-A*/GBFS.
3. **Stop-Regressing** (ICML'24) — classification value head (our HL-Gauss).

**Positioning to-do for the paper:** (1) a paragraph distinguishing C-Learning (horizon = push-count, not time; the non-subsumption result). (2) a paragraph on Where2Act vs our partial-label/false-negative regime. (3) frame against SAVE/Bejjani as "budget-conditioned value enabling a *tunable* reactive↔search frontier in expensive-sim manipulation," not just "learned value for search."
