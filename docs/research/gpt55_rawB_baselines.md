> Source: codex CLI, model gpt-5.5, reasoning xhigh, live web search. Date 2026-07-10.
> Query B — BASELINES / METHODS (learned-heuristic-guided rearrangement search). Raw, unedited codex stdout.

Best fit: compare against methods that **order or shortlist pushes**, then let your exact simulator verify them.

Legend: **Easy** = mostly scoring/search wrapper; **Medium** = train a model on your traces; **Hard** = major representation or planner port. “No verified code” means I did not confirm a public repo from primary pages.

**1. Classical NAMO / Rearrangement**
| Method | What it does | Code | Fit for you |
|---|---|---|---|
| Stilman & Kuffner, “Navigation Among Movable Obstacles,” IJHR 2005 ([ri.cmu.edu](https://www.ri.cmu.edu/pub_files/pub4/stilman_michael_2005_3/stilman_michael_2005_3.pdf)) | Decomposes NAMO by reachable free-space and heuristic search. | No verified code. | **Easy/Med:** rank pushes by reachable-area gain, path clearing, obstacle displacement. |
| Chen & Hwang, “Practical Path Planning among Movable Obstacles,” ICRA 1991, discussed/cited by Stilman and van den Berg ([ri.cmu.edu](https://www.ri.cmu.edu/pub_files/pub4/stilman_michael_2005_3/stilman_michael_2005_3.pdf)) ([link.springer.com](https://link.springer.com/chapter/10.1007/978-3-642-00312-7_37)) | “Shove aside / push forward” path-plowing heuristic. | No verified code. | **Easy:** corridor-based push-out-of-path ranker. |
| van den Berg, Stilman, Kuffner, Lin, Manocha, “Path Planning among Movable Obstacles,” WAFR VIII/Springer 2009 ([link.springer.com](https://link.springer.com/chapter/10.1007/978-3-642-00312-7_37)) | Probabilistically complete movable-obstacle planner. | No verified code. | **Hard:** cite/position; too much for your one-object push grid. |
| Saxena, Saleem, Likhachev, “Manipulation Planning Among Movable Obstacles Using Physics-Based Adaptive Motion Primitives,” ICRA 2021 ([arxiv.org](https://arxiv.org/abs/2102.04324)) | Reduces slow physics calls using adaptive primitives + multi-heuristic search. | No verified code. | **Medium:** multi-queue heuristic over your existing primitives. |
| Wang, Gao, Nakhimovich, Yu, Bekris, “Uniform Object Rearrangement,” ICRA 2021 ([arxiv.org](https://arxiv.org/abs/2101.12241)) | Monotone/non-monotone rearrangement via region graph + buffers. | No verified code. | **Med/Hard:** useful “setup/buffer” analogy; less contact-rich. |
| Ren et al., “Search-Based Path Planning in Interactive Environments among Movable Obstacles,” ICRA 2025 ([arxiv.org](https://arxiv.org/abs/2410.18333)) | PAMO* searches only relevant robot/object states with heuristics. | No verified code. | **Medium:** occupancy-grid heuristic baseline if you expand beyond one blocker. |

**2. Learned Search Guidance**
| Method | What it does | Code | Fit for you |
|---|---|---|---|
| Chitnis et al., “Guided Search for Task and Motion Plans Using Learned Heuristics,” ICRA 2016 ([people.eecs.berkeley.edu](https://people.eecs.berkeley.edu/~pabbeel/papers/2016-ICRA-tamp-learning.pdf)) | Learns high-level plan ranking and low-level sampling for TAMP. | No verified code. | **Medium:** train a priority score over push prefixes. |
| Kim & Shimanuki, “Learning Value Functions with Relational State Representations for Guiding TAMP,” CoRL 2019/PMLR 2020 ([proceedings.mlr.press](https://proceedings.mlr.press/v100/kim20a.html)) | GNN Q/value from planning experience over movable-object relations. | No verified code. | **Medium:** graph baseline: object, robot, goal, bottleneck edges. |
| Khodeir, Agro, Shkurti, “Learning to Search in TAMP with Streams,” RA-L 2023 ([arxiv.org](https://arxiv.org/abs/2111.13144)) | GNN chooses which stream facts/objects to expand first. | No verified code. | **Med/Hard:** adapt as “which push branch to expand.” |
| Shen, Trevizan, Thiébaux, “Learning Domain-Independent Planning Heuristics with Hypergraph Networks,” ICAPS 2020/arXiv 2019 ([arxiv.org](https://arxiv.org/abs/1911.13101)) | GNN heuristic for symbolic planning search. | No verified code. | **Hard:** needs symbolic encoding; good citation, weak direct baseline. |
| Agostinelli et al., “Solving the Rubik’s Cube with Deep RL and Search,” Nature MI 2019 ([nature.com](https://www.nature.com/articles/s42256-019-0070-z)) | Learned cost-to-go inside weighted A*/GBFS. | Code: DeepCubeA ([github.com](https://github.com/forestagostinelli/DeepCubeA)) | **Medium:** learned leaf value + best-first search. |
| Hamrick et al., “Combining Q-Learning and Search with Amortized Value Estimates,” ICLR 2020 ([arxiv.org](https://arxiv.org/abs/1912.02807?utm_source=openai)) | SAVE: learned Q guides MCTS; MCTS improves Q targets. | No verified code. | **Medium:** very close algorithmically; use your sim as model. |

**3. Learned Action Samplers / Generators**
| Method | What it does | Code | Fit for you |
|---|---|---|---|
| Ichter, Harrison, Pavone, “Learning Sampling Distributions for Robot Motion Planning,” ICRA 2018 ([arxiv.org](https://arxiv.org/abs/1709.05448)) | CVAE samples promising planner states from demos. | No verified code. | **Medium:** CVAE over successful `(edge, depth)` or 2-push prefixes. |
| Yang et al., “Diffusion-CCSP,” CoRL 2023 ([diffusion-ccsp.github.io](https://diffusion-ccsp.github.io/)) | Diffusion samples continuous constraint solutions for TAMP. | Code ([github.com](https://github.com/zt-yang/diffusion-ccsp)) | **Med/Hard:** stronger sampler baseline; overkill for 300 discrete pushes. |
| Mo et al., “Where2Act,” ICCV 2021 ([cs.stanford.edu](https://cs.stanford.edu/~kaichun/where2act/)) | Per-pixel actionability + action proposal for push/pull. | Code ([github.com](https://github.com/daerduoCarey/where2act)) | **Medium:** per-edge actionability sampler/ranker. |
| Chi et al., “Diffusion Policy,” RSS 2023/IJRR 2024 ([diffusion-policy.cs.columbia.edu](https://diffusion-policy.cs.columbia.edu/)) | Diffusion generates robot action sequences. | Code ([github.com](https://github.com/columbia-ai-robotics/diffusion_policy)) | **Hard:** policy baseline, not a clean sim-call minimizer. |
| Zhou et al., “HACMan,” CoRL 2023 ([hacman-2023.github.io](https://hacman-2023.github.io/)) | Per-contact actor/critic map for non-prehensile manipulation. | Code ([github.com](https://github.com/HACMan-2023/HACMan)) | **Medium:** closest architecture baseline: spatial Q-map over contacts. |
| Jiang et al., “HACMan++,” RSS 2024 ([sgmp-rss2024.github.io](https://sgmp-rss2024.github.io/)) | Scores primitive type, contact location, and parameters. | Code ([github.com](https://github.com/JiangBowen0008/HACManPP)) | **Med/Hard:** useful if you add more primitive types. |

**4. Sequence / Plan-Feasibility Classifiers**
| Method | What it does | Code | Fit for you |
|---|---|---|---|
| Yang, Garrett, Lozano-Pérez, Kaelbling, Fox, “PIGINet,” RSS 2023 ([piginet.github.io](https://piginet.github.io/)) | Transformer predicts whether a whole task plan is refinable. | Code in `kitchen-worlds` ([github.com](https://github.com/Learning-and-Intelligent-Systems/kitchen-worlds)) | **Medium:** score `(push1, push2)` prefixes before sim. |
| Driess, Ha, Toussaint, “Deep Visual Reasoning,” RSS 2020/arXiv 2020 ([arxiv.org](https://arxiv.org/abs/2006.05398?utm_source=openai)) | Predicts promising action sequences from an initial scene image. | No verified code. | **Med/Hard:** direct sequence-proposal baseline. |
| Zhou, Schubert, Toussaint, Oguz, “Spatial Reasoning via Deep Vision Models,” arXiv 2023 ([arxiv.org](https://arxiv.org/abs/2306.17053?utm_source=openai)) | Predicts task-relevant objects to shrink TAMP search. | No verified code. | **Medium:** object/edge relevance filter before ranking. |

**5. RL / Self-Imitation / MCTS / ExIt**
| Method | What it does | Code | Fit for you |
|---|---|---|---|
| Zeng et al., “Learning Synergies between Pushing and Grasping,” IROS 2018 ([vpg.cs.princeton.edu](https://vpg.cs.princeton.edu/)) | Pixel Q-learning; pushes get value by enabling later grasps. | Code linked on project page ([vpg.cs.princeton.edu](https://vpg.cs.princeton.edu/)) | **Medium:** myopic-vs-setup Q baseline. |
| Bejjani, Papallas, Leonetti, Dogar, “Receding Horizon... Learned Value Function,” Humanoids 2018/arXiv ([arxiv.org](https://arxiv.org/abs/1803.08100?utm_source=openai)) | Learned value guides short-horizon clutter pushing. | No verified code. | **Medium:** train V(s) from solved searches; use as leaf score. |
| Labbé et al., “Monte-Carlo Tree Search for Efficient Visually Guided Rearrangement Planning,” RA-L 2020 ([ylabbe.github.io](https://ylabbe.github.io/rearrangement-planning/)) | MCTS over rearrangement actions. | Planner + perception code linked ([ylabbe.github.io](https://ylabbe.github.io/rearrangement-planning/)) | **Medium:** replace actions with your push grid. |
| Huang et al., “Visual Foresight Trees,” RA-L 2022 ([github.com](https://github.com/arc-l/vft)) | Learned push forward model + tree search. | Code ([github.com](https://github.com/arc-l/vft)) | **Medium/Hard:** use MuJoCo instead of learned model to keep it fair. |
| Huang, Guo, Boularias, Yu, “MORE,” ICRA 2022 ([github.com](https://github.com/arc-l/more)) | MCTS creates labels; DNN guides later MCTS. | Code ([github.com](https://github.com/arc-l/more)) | **Medium:** closest “search distills into ranker” robotics baseline. |
| Zhu, Meduri, Righetti, “Efficient Object Manipulation Planning with MCTS,” arXiv 2022; venue not verified here ([arxiv.org](https://arxiv.org/abs/2206.09023)) | Policy-value network guides contact-sequence MCTS. | No verified code. | **Hard:** useful concept, heavy trajectory optimization. |
| Anthony, Tian, Barber, “Expert Iteration,” NeurIPS 2017 ([arxiv.org](https://arxiv.org/abs/1705.08439?utm_source=openai)) | Tree search makes policy targets; network improves search. | No verified code. | **Medium:** exact skeleton for iterative push-ranker training. |
| DeepMind `mctx` / Gumbel MuZero family ([github.com](https://github.com/google-deepmind/mctx)) | Few-simulation MCTS with policy/value priors. | Code: `google-deepmind/mctx` ([github.com](https://github.com/google-deepmind/mctx)) | **Med/Hard:** good if you want formal few-sim MCTS machinery. |

**Top 5 To Implement First**
1. **Stilman/CH geometric ranker.**  
No learning, no training ambiguity, reviewer-friendly. Score “moves blocker out of current corridor / increases reachable goal-side space.”

2. **Exact myopic reachability-gain lookahead.**  
Sim each first push until it opens the goal, and count those sims. It should dominate easy 1-push and expose the 2-push setup gap cleanly.

3. **PIGINet-style 2-push prefix classifier.**  
This directly tests whether “score the whole candidate sequence” beats “score first pushes.” Enumerate top candidate `(a1,a2)` pairs cheaply, train binary solves/does-not-solve labels from your search traces.

4. **CVAE action sampler.**  
Clean sampler-vs-ranker comparison: generate top-K likely setup pushes, then MuJoCo verifies. Start CVAE before diffusion because your action space is small and discrete.

5. **MORE/SAVE-style guided search distillation.**  
Closest full-loop comparison: unguided search finds solutions, model learns to guide the next search, compare sim-call curves generation by generation.
