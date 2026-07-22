> source: codex gpt-5.5, reasoning_effort=xhigh, read-only
> date: 2026-07-12

Bottom line: **your full loop looks novel as a combination**. Pieces exist, but I did not find prior work that combines **solution-preserving hard NAMO construction + exact forward verification + solve→distill into a push ranker + natural held-out evaluation**.

**(a) Learned Search Guidance / Search Distillation**

| Paper | Idea | Code | Connection to your loop |
|---|---|---|---|
| Anthony, Tian, Barber, **“Thinking Fast and Slow with Deep Learning and Tree Search,” NeurIPS 2017** ([arxiv.org](https://arxiv.org/abs/1705.08439?utm_source=openai)) | Expert Iteration: tree search makes better targets; network distills them; network then guides later search. | Official code not found in this pass. | Same **search→distill→better search** skeleton; no hard-instance mining, no verifier-built dataset. |
| Hamrick et al., **“Combining Q-Learning and Search with Amortized Value Estimates,” ICLR 2020** ([arxiv.org](https://arxiv.org/abs/1912.02807?utm_source=openai)) | SAVE: learned Q prior guides MCTS; MCTS produces improved Q targets. | Official code not found in this pass. | Very close amortized-search idea; their simulator/search is cheap game/physics reasoning, not expensive push verification. |
| Agostinelli et al., **“Solving the Rubik’s Cube with Deep Reinforcement Learning and Search,” Nature Machine Intelligence 2019** ([github.com](https://github.com/forestagostinelli/DeepCubeA)) | DeepCubeA learns cost-to-go, then uses weighted A*/GBFS-style search. Training states come from scrambling backward from solved states. | [DeepCubeA](https://github.com/forestagostinelli/DeepCubeA) ([github.com](https://github.com/forestagostinelli/DeepCubeA)) | **Closest algorithmic cousin**: constructive solvable-state generation + learned heuristic + search. Difference: reversible puzzle, not physics NAMO; no mined natural-vs-constructed split. |
| Yonetani et al., **“Path Planning using Neural A* Search,” ICML 2021** ([arxiv.org](https://arxiv.org/abs/2009.07476?utm_source=openai)) | Differentiable A* learns a guidance map that reduces search while matching expert paths. | Project page: [neural-astar](https://omron-sinicx.github.io/neural-astar/) ([arxiv.org](https://arxiv.org/abs/2009.07476?utm_source=openai)) | Same goal of fewer expansions; but grid path planning, not manipulation primitive ranking. |
| Gasse et al., **“Exact Combinatorial Optimization with Graph Convolutional Neural Networks,” NeurIPS 2019** ([arxiv.org](https://arxiv.org/abs/1906.01629?utm_source=openai)) | GNN imitates strong branching inside branch-and-bound. | [learn2branch](https://github.com/ds4dm/learn2branch) ([arxiv.org](https://arxiv.org/abs/1906.01629?utm_source=openai)) | Same “learn ordering decisions to cut solver calls”; different solver/domain. |
| Li, Chen, Koltun, **“Combinatorial Optimization with Graph Convolutional Networks and Guided Tree Search,” NeurIPS 2018** ([arxiv.org](https://arxiv.org/abs/1810.10659?utm_source=openai)) | GCN predicts promising vertices; tree search explores guided choices. | Official code not verified. | Ranker-guided tree search for NP-hard graph problems; no self-mined hard curriculum. |
| Shen, Trevizan, Thiébaux, **“Learning Domain-Independent Planning Heuristics with Hypergraph Networks,” 2019/2020** ([arxiv.org](https://arxiv.org/abs/1911.13101)) | Learns STRIPS planning heuristics from state/value pairs using hypergraph networks. | Official code not verified. | Direct learned-heuristic-for-GBFS/A* precedent; symbolic planning, not physics pushes. |
| O’Toole, Ramirez, Lipovetzky, Pearce, **“Sampling from Pre-Images to Learn Heuristic Functions for Classical Planning,” arXiv 2022** ([arxiv.org](https://arxiv.org/abs/2207.03336)) | Backward regression from goals samples states at known/estimated goal distance, then trains a neural heuristic. | Official code not verified. | Very relevant “backward proposes training states” idea; symbolic regression replaces your constructive physics mutations. |
| Bejjani, Papallas, Leonetti, Dogar, **“Planning with a Receding Horizon for Manipulation in Clutter using a Learned Value Function,” Humanoids 2018** ([arxiv.org](https://arxiv.org/abs/1803.08100?utm_source=openai)) | Planner-generated data trains a value heuristic for clutter pushing; RL refines it. | Official code not found. | Closest robotics/domain cousin: learned value helps push planning. Difference: not discrete NAMO ranker with hard-instance mining. |
| Yang et al., **“Sequence-Based Plan Feasibility Prediction for Efficient Task and Motion Planning,” RSS 2023** ([arxiv.org](https://arxiv.org/abs/2211.01576?utm_source=openai)) | PIGINet ranks symbolic task plans by predicted feasibility before motion refinement. | Project page: [piginet.github.io](https://piginet.github.io/) ([arxiv.org](https://arxiv.org/abs/2211.01576?utm_source=openai)) | Same “rank candidates before expensive checks”; not self-climbing hard-instance construction. |

**(b) The Stall / Collapse-to-Easy Problem**

| Paper | Idea | Code | Connection |
|---|---|---|---|
| Florensa et al., **“Reverse Curriculum Generation for Reinforcement Learning,” CoRL 2017** ([arxiv.org](https://arxiv.org/abs/1707.05300)) | Start near goal, expand outward only through states the current policy can sometimes solve. | Official code not found. | Same easy→hard growth; closest robotics-side constructive curriculum. Not search distillation. |
| Florensa, Held, Geng, Abbeel, **“Automatic Goal Generation for RL Agents,” ICML 2018** ([arxiv.org](https://arxiv.org/abs/1705.06366)) | Goal generator proposes tasks at the agent’s current difficulty frontier. | Official code not found. | Direct support for “don’t sample random easy tasks; mine frontier tasks.” |
| Dennis et al., **“Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design,” NeurIPS 2020** ([arxiv.org](https://arxiv.org/abs/2012.02096)) | PAIRED fixes two failures: random generation lacks structure; minimax adversaries make impossible tasks. Uses regret to generate hard-but-solvable levels. | Google [social_rl](https://github.com/google-research/google-research/tree/master/social_rl) ([github.com](https://github.com/google-research/google-research/tree/master/social_rl)) | Strongest citation for your “hard but solvable” mining rule. Your exact sim is a stronger verifier than their antagonist. |
| Jiang, Grefenstette, Rocktäschel, **“Prioritized Level Replay,” ICML 2021** ([arxiv.org](https://arxiv.org/abs/2010.03934)) | Replay levels with high learning potential instead of uniform procedural sampling. | [facebookresearch/level-replay](https://github.com/facebookresearch/level-replay) ([github.com](https://github.com/facebookresearch/level-replay)) | Same diagnosis: random generated data wastes effort on unhelpful/easy cases. |
| Parker-Holder et al., **“Evolving Curricula with Regret-Based Environment Design,” ICML 2022** ([arxiv.org](https://arxiv.org/abs/2203.01302)) | ACCEL edits levels to keep complexity near the agent’s frontier. | Code not verified; related DCD repo exists. | Closest “mutate levels to climb difficulty” citation. Difference: RL environments, not certified push solutions. |
| Zelikman et al., **“STaR: Bootstrapping Reasoning With Reasoning,” NeurIPS 2022** ([arxiv.org](https://arxiv.org/abs/2203.14465?utm_source=openai)) | Generate reasoning, keep/correct successful traces, fine-tune, repeat. | Official code not found. | Same self-training pattern; weaker verifier than your simulator. |
| Singh et al., **“Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models,” arXiv 2023** ([arxiv.org](https://arxiv.org/abs/2312.06585?utm_source=openai)) | ReST-EM: generate samples, filter with binary feedback, fine-tune, repeat. | Official code not verified. | Very clean analogy: your simulator is the binary filter, but for embodied search actions instead of math/code text. |

**(c) The Intersection: Mine / Solve / Distill**

The real intersection is thin.

| Paper | Why it is close | Main miss relative to you |
|---|---|---|
| **DeepCubeA** ([github.com](https://github.com/forestagostinelli/DeepCubeA)) | Construct solvable states from solved states, learn cost-to-go, use learned heuristic in search. | Reversible puzzle, no physics, no natural held-out scene distribution, no non-prehensile pushing. |
| **O’Toole et al. pre-image heuristic learning** ([arxiv.org](https://arxiv.org/abs/2207.03336)) | Backward goal regression creates training states with distance labels for learned GBFS heuristics. | Symbolic planning pre-images, not local scene mutation plus exact forward physics verification. |
| **PAIRED / ACCEL / PLR** ([arxiv.org](https://arxiv.org/abs/2012.02096)) ([arxiv.org](https://arxiv.org/abs/2203.01302)) ([arxiv.org](https://arxiv.org/abs/2010.03934)) | Automatic hard-instance curriculum; explicitly guards against random-easy and impossible-adversarial tasks. | Learns policies over generated levels, not a search ranker distilled from exact solver traces. |
| **Reverse Curriculum Generation** ([arxiv.org](https://arxiv.org/abs/1707.05300)) | Robotics-friendly “grow outward from known success” curriculum. | Continuous RL start states, not discrete primitive sequencing or best-first search cost reduction. |

**Novelty Verdict**

**Yes: likely novel.** I found strong precedents for each piece, but not the exact combination: **constructive solution-preserving hard-instance mining, exact forward verification, learned push-ranker distillation, non-prehensile discrete push sequencing, and validation on a natural held-out set.**

Closest 3 papers:

1. **DeepCubeA**: closest loop shape. Differs on domain: reversible puzzles, not non-prehensile physics/NAMO; no natural held-out-vs-mined guard.
2. **O’Toole et al. / learned classical-planning heuristics**: closest backward-labeling/search-heuristic precedent. Differs on substrate: symbolic pre-images, not simulator-verified push outcomes.
3. **PAIRED/ACCEL**: closest hard-instance curriculum. Differs on learner: policy training, not solve→distill ranker; feasibility is learned/game-based, not guaranteed by a perfect verifier.
