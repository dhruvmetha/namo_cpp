# Hard-instance generation + search-distillation: lit scan & novelty verdict

> Engine: codex GPT-5.5 (xhigh), 2026-07-12. Raw outputs: [Q1](gpt55_rawQ1_hardgen.md), [Q2](gpt55_rawQ2_searchdistill.md).
> Decision question: is our loop — constructive/solution-preserving hard-instance mining + search-distillation of a push-RANKER, under a PERFECT verifier, for discrete-primitive non-prehensile sequencing, validated on a NATURAL held-out set — novel?

**One-line answer: the full combination looks novel. Every individual piece has a strong precedent; nobody assembles them for physics-based non-prehensile push sequencing with a perfect verifier and a natural-held-out guard.**

---

## 1. Hard-instance generation / curriculum

The key axis is **solvable-by-construction**: does the generator hand you a solution witness (Yes), or only give a solvable instance after a solver/agent filters it (Filter), or neither (No)? Our loop wants **Yes** — carry the solution in, forward-verify with the exact sim.

| Work | Venue/yr | Core idea | Solvable-by-construction | Connection to us |
|---|---|---|---|---|
| DeepCubeA (Agostinelli, McAleer, Shmakov, Baldi) | Nature Mach. Intell. 2019 | Scramble backward from goal → solvable states; learn cost-to-go; A*/GBFS. [code](https://github.com/forestagostinelli/DeepCubeA) | **Yes** | Cleanest analog of our loop shape: backward-generate solvable hard states, learn heuristic, measure fewer expansions. But test = random scramble, not natural held-out. |
| MetaGen (Wang, Deng) | NeurIPS 2020 | Learn to synthesize theorems + proof trees to train a prover. [code](https://github.com/princeton-vl/MetaGen) | **Yes** | Structurally very close: generated data useful only because each carries a proof; evaluated on held-out human tasks — a real synthetic→natural transfer test. |
| INT (Wu, Jiang, Ba, Grosse) | 2020 (venue unverified) | Morph trivial inequalities through axiom sequences; proofs known. [code](https://github.com/albertqjiang/INT) | **Yes** | Template for difficulty knobs (proof length, axiom combos, OOD splits). Transfer is synthetic-OOD, not natural. |
| Go-Explore (Ecoffet, Huizinga, Lehman, Stanley, Clune) | Nature 2021 | Archive promising states, return-then-explore. [code](https://github.com/uber-research/go-explore) | Yes (archived states) | Not generation, but the anti-random-search principle: keep the path witness to rare states instead of hoping rollouts rediscover them. |
| Asymmetric Self-Play (Sukhbaatar et al.) | ICLR 2018 | Alice sets a task by doing it; Bob reproduces it. | Mostly yes | Primitive form of "task generated around a known trajectory" — apt if our mutation preserves the carried push chain. |
| PAIRED (Dennis et al.) | NeurIPS 2020 | Regret-maximizing adversary generates hard-but-solvable levels. [code](https://github.com/ucl-dark/paired) | **No** | Regret is a *soft* solvability pressure vs our exact verifier. But they do evaluate zero-shot transfer to unseen hand-designed envs — a guard to emulate. |
| PLR (Jiang, Grefenstette, Rocktäschel) | ICML 2021 | Replay levels with high learning potential. [code](https://github.com/facebookresearch/level-replay) | **No** | Directly reusable "which mined episodes to re-train on?" component. |
| DCD / robust-PLR (Jiang et al.) | NeurIPS 2021 | Combine generation + replay; PLR⊥ avoids training on uncurated levels. [code](https://github.com/facebookresearch/dcd) | **No** | Strong "mine then distill" template — swap regret proxy for our exact verifier. |
| ACCEL (Parker-Holder et al.) | ICML 2022 | Mutate archived levels, keep frontier tasks. [code](https://github.com/facebookresearch/dcd) | Filter, not proof | Closest UED to our local-mutation idea; ours can mutate *while carrying a solution witness* and forward-verify — a strictly stronger guarantee. |
| CLUTR (Azad et al.) | 2022 (venue unverified) | Learn latent task manifold; sample regret-maximizing tasks. | **No** | Useful if NAMO mutations need a learned "valid/hard room" manifold. |
| POET (Wang, Lehman, Clune, Stanley) | GECCO 2019 | Co-evolve environments + agents, transfer across niches. [code](https://github.com/uber-research/poet) | Filter | Open-ended framing; weak eval guard — success measured *inside* generated worlds (the self-licking risk we want to avoid). |
| GoalGAN (Florensa, Held, Geng, Abbeel) | ICML 2018 | Propose goals at intermediate difficulty. | **No** | Targets the "barely solvable" band but carries no solution. |
| PCGRL (Khalifa, Bontrager, Earle, Togelius) | AIIDE 2020 | RL level designer, playability/path checks as reward. [code](https://github.com/amidos2006/gym-pcgrl) | Filter | Design pattern: put the solver/verifier *inside* the generator reward — clean for us given the exact sim. |
| G2SAT (You et al.) | 2019 (venue unverified) | Learn graph generator for realistic SAT formulas. [code](http://snap.stanford.edu/g2sat/) | **No** | Synthetic→real lesson: generated instances helped tune solvers on unseen real formulas; no solution witness. |
| MAPF empirical hardness (Ren, Ewing, Kumar, Koenig, Ayanian) | ICAPS 2024 | Hardness via graph connectivity; quality-diversity map generator. | No / filter | Closest recent ICAPS hardness-control result — hardness control, not constructive solution-preserving generation. |

**Top-5 for our constructive-mining loop (ranked):**
1. **DeepCubeA** — backward-generate solvable states, learn heuristic, fewer expansions. The blueprint.
2. **MetaGen** — synthetic hard data must carry a proof; tests synthetic→natural transfer.
3. **INT** — difficulty knobs with proof-by-construction.
4. **ACCEL / DCD** — mutation + replay UED template (needs our verifier to replace regret).
5. **Go-Explore** — preserve rare solution paths instead of rediscovering them by rollout.

---

## 2. Search-distillation / learning-to-search

Learn an ordering/heuristic so search makes fewer expensive calls — exactly our push-ranker goal.

| Paper | Venue/yr | Idea | Connection to us |
|---|---|---|---|
| ExIt (Anthony, Tian, Barber) | NeurIPS 2017 | Expert Iteration: tree search makes better targets; net distills them; net guides later search. | The **search→distill→better-search** skeleton itself. No hard-instance mining, no verifier-built dataset. |
| SAVE (Hamrick et al.) | ICLR 2020 | Learned Q prior guides MCTS; MCTS produces improved Q targets. | Very close amortized-search idea; their sim is cheap game reasoning, not expensive push verification. |
| DeepCubeA (Agostinelli et al.) | Nature MI 2019 | Learn cost-to-go from backward-scrambled states; weighted A*/GBFS. [code](https://github.com/forestagostinelli/DeepCubeA) | **Closest algorithmic cousin** — constructive solvable states + learned heuristic + search. |
| Neural A* (Yonetani et al.) | ICML 2021 | Differentiable A*; learns guidance map, fewer expansions matching expert paths. [page](https://omron-sinicx.github.io/neural-astar/) | Same "fewer expansions" goal; grid path planning, not manipulation ranking. |
| learn2branch (Gasse et al.) | NeurIPS 2019 | GNN imitates strong branching in B&B. [code](https://github.com/ds4dm/learn2branch) | Same "learn ordering to cut solver calls"; different solver/domain. |
| GCN + guided tree search (Li, Chen, Koltun) | NeurIPS 2018 | GCN predicts promising vertices; tree search explores them. | Ranker-guided tree search for NP-hard graph problems; no self-mined curriculum. |
| Hypergraph net heuristics (Shen, Trevizan, Thiébaux) | 2019/2020 | Learn domain-independent STRIPS heuristics from state/value pairs. | Direct learned-heuristic-for-GBFS precedent; symbolic, not physics pushes. |
| Pre-image heuristic learning (O'Toole, Ramirez, Lipovetzky, Pearce) | arXiv 2022 | Backward regression from goals samples states at known goal-distance; train neural heuristic. | Very relevant "backward proposes training states" — symbolic regression stands in for our constructive physics mutation. |
| Learned value for clutter pushing (Bejjani, Papallas, Leonetti, Dogar) | Humanoids 2018 | Planner data trains a value heuristic for clutter pushing; RL refines. | **Closest robotics/domain cousin**; not a discrete NAMO ranker, no hard-instance mining. |
| PIGINet (Yang et al.) | RSS 2023 | Rank symbolic task plans by predicted feasibility before motion refinement. [page](https://piginet.github.io/) | Same "rank candidates before expensive checks"; no self-climbing construction. |

**Top picks:** DeepCubeA (loop shape), ExIt (distillation skeleton), O'Toole et al. (backward-labeling for a learned search heuristic), Bejjani et al. (closest robotics push-planning value function).

---

## 3. The stall problem (collapse-to-easy)

**What's known:** self-training/self-play/procedural-generation loops waste effort on easy or unsolvable samples and plateau. Two named diagnoses:
- **Random generation lacks structure; minimax adversaries make impossible tasks** — PAIRED's motivation (Dennis et al., NeurIPS 2020). Regret targets the hard-but-solvable band.
- **Uniform procedural sampling wastes compute on unhelpful/easy levels** — PLR's motivation (Jiang et al., ICML 2021).

**Fixes with precedent:**
- **Frontier-targeting curricula** — GoalGAN (ICML 2018) and Reverse Curriculum Generation (Florensa et al., CoRL 2017): grow outward only through states the current policy can *sometimes* solve. Closest robotics-side constructive curriculum to us.
- **Prioritized replay of high-learning-potential instances** — PLR / robust-PLR / DCD.
- **Mutate to stay near the frontier** — ACCEL (ICML 2022).
- **Verifier-filtered iterated self-training** — STaR (Zelikman et al., NeurIPS 2022) and ReST-EM (Singh et al., 2023): generate → filter by binary feedback → fine-tune → repeat. Their known weakness is a *weak* verifier; **our exact simulator is a strictly stronger binary filter** — the cleanest analogy to our loop, just for embodied search actions instead of text.

**Our built-in guard against the stall / self-licking curriculum:** measure the climb on a NATURAL held-out set, not on mined data. POET is the cautionary counter-example (success measured inside generated worlds). MetaGen and PAIRED are the positive templates (both report transfer to unseen/hand-designed distributions).

---

## 4. The intersection + NOVELTY VERDICT

**The intersection (auto hard-instance generation ∩ learned search guidance, as a mine→solve→distill loop) is thin.** The four closest:

| Paper | Why close | Main miss vs us |
|---|---|---|
| **DeepCubeA** | Construct solvable states from solved states, learn cost-to-go, search with it. | Reversible puzzle, no physics, no non-prehensile pushing, no natural-held-out-vs-mined guard. |
| **O'Toole et al. (pre-image heuristic learning)** | Backward goal regression makes training states with distance labels for learned GBFS. | Symbolic pre-images, not local scene mutation + exact forward physics verification. |
| **PAIRED / ACCEL / PLR** | Automatic hard-instance curriculum; explicitly guards random-easy and impossible-adversarial. | Learns a *policy* over generated levels, not a *ranker distilled from exact solver traces*; feasibility is learned/game-based, not verifier-guaranteed. |
| **Reverse Curriculum Generation** | Robotics "grow outward from known success." | Continuous RL start states, not discrete primitive sequencing or best-first cost reduction. |

**VERDICT: likely NOVEL as a combination.** Strong precedent exists for each piece, but no prior work assembles: constructive solution-preserving hard-instance mining + exact forward verification + solve→distill into a push-ranker + non-prehensile discrete-primitive sequencing + validation on a natural held-out set.

**Closest 3, and the exact axis each differs on:**
1. **DeepCubeA** (Nature MI 2019) — closest *loop shape*. Differs on **domain**: reversible puzzles with a trivial inverse, not irreversible non-prehensile physics; and **no natural-held-out-vs-mined guard**.
2. **O'Toole et al. pre-image heuristics** (arXiv 2022) — closest *backward-labeling / learned-search-heuristic* precedent. Differs on **substrate**: symbolic STRIPS pre-images, not simulator-verified push outcomes.
3. **PAIRED / ACCEL** (NeurIPS 2020 / ICML 2022) — closest *hard-instance curriculum*. Differs on **learner + guarantee**: trains a policy, not a solve→distill ranker; solvability is a learned regret/game pressure, not a perfect verifier.

**The defensible novelty claim:** the *irreversibility* of non-prehensile pushing is what makes DeepCubeA's backward-scramble trick fail here (you cannot cheaply invert a push), so solution-preserving *forward-verified* construction is a genuinely different generator; and pairing it with a distilled ranker + a natural-held-out guard is, per this scan, unclaimed.

---

## 5. Honest gaps

- **Venue unverified** for several entries GPT flagged: INT, PLR (arXiv vs ICML), CLUTR, G2SAT. Confirm before citing in a paper.
- **Code "not found in this pass"** for ExIt, SAVE, several classical-planning heuristics, STaR, ReST-EM — absence in the scan is not absence in the world; re-check.
- **ICAPS depth is thin.** Only MAPF-hardness (Ren et al., 2024) surfaced. The learned-heuristics-for-classical-planning literature (e.g., the STRIPS-HGN line, GOOSE/WL-features work) is likely deeper than one pass caught — worth a targeted ICAPS-only follow-up.
- **No direct NAMO / non-prehensile hard-instance-generation paper surfaced** — consistent with the novelty claim, but this is the one place a miss would most hurt; a targeted search on "NAMO benchmark generation" / "clutter scene generation difficulty" is warranted before claiming novelty in print.
- Single engine (GPT-5.5), single pass each — treat as a strong first map, not an exhaustive survey.
