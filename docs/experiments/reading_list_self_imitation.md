---
status: ref
tags: [reading-list, rl, self-imitation]
updated: 2026-07-07
---
# Reading list — iterated self-imitation (our RL loop) and its neighbors

> Web-verified 2026-07-07 (Opus lit-check agent; all links resolve, titles/venues exact). Ordered by priority for OUR implementation — Tier 1 is the loop we are literally running. One-liners say why it matters for us.

## Tier 1 — the loop itself (read these first)

1. **GCSL — Learning to Reach Goals via Iterated Supervised Learning** — Ghosh et al., arXiv 2019 / ICLR 2021 — [arXiv:1912.06088](https://arxiv.org/abs/1912.06088). The formal version of our exact loop (collect own rollouts → imitate successes → iterate) with the convergence guarantee; the theory anchor for "iterating filtered BC is principled."
2. **ReST^EM — Beyond Human Data** — Singh et al., arXiv 2023 / TMLR 2024 — [arXiv:2312.06585](https://arxiv.org/abs/2312.06585). EM framing of generate→filter→fine-tune, and **the published precedent for our gen-1 flatline**: "most of the gains come from the first iteration," train rises while test doesn't.
3. **Self-Imitation Learning** — Oh et al., ICML 2018 — [arXiv:1806.05635](https://arxiv.org/abs/1806.05635). The name-giver: replay-buffer imitation of your own high-return trajectories; sparse-reward exploration effects.
4. **STaR: Bootstrapping Reasoning With Reasoning** — Zelikman et al., NeurIPS 2022 — [arXiv:2203.14465](https://arxiv.org/abs/2203.14465). The famous LLM instance of the same loop (filter own correct rationales, fine-tune, repeat).
5. **RL as Probabilistic Inference: Tutorial and Review** — Levine, 2018 — [arXiv:1805.00909](https://arxiv.org/abs/1805.00909). Why filtered maximum-likelihood IS the M-step of RL — the cleanest theory frame for our surrogate objective.
6. **Reward-Weighted Regression (for Operational Space Control)** — Peters & Schaal, ICML 2007 — [dblp](https://dblp.org/rec/conf/icml/PetersS07.html). The original EM policy improvement with the monotonic-improvement argument; generalizes Dayan & Hinton 1997.
7. **AWR — Advantage-Weighted Regression** — Peng et al., arXiv 2019 (preprint) — [arXiv:1910.00177](https://arxiv.org/abs/1910.00177). The advantage-weighted upgrade of the same family — the thing we keep OFF behind an identifiability trigger (GPT-5.5 consult).

## Tier 2 — our domain's neighbors (search + puzzle worlds)

8. **Expert Iteration — Thinking Fast and Slow with Deep Learning and Tree Search** — Anthony, Tian & Barber, NeurIPS 2017 — [arXiv:1705.08439](https://arxiv.org/abs/1705.08439). Search as the improvement operator — where we go if RL-only stays falsified.
9. **AlphaZero** — Silver et al., arXiv 2017 — [arXiv:1712.01815](https://arxiv.org/abs/1712.01815) (peer-reviewed version: *Science* 2018, different title). The reference point for π+V+search.
10. **LevinTS — Single-Agent Policy Tree Search With Guarantees** — Orseau et al., NeurIPS 2018 — [arXiv:1811.10928](https://arxiv.org/abs/1811.10928). The d(n)/π(n) expansion bound — the principled link between "train π on solution paths" and "minimize sims-to-solve."
11. **PHS — Policy-Guided Heuristic Search with Guarantees** — Orseau & Lelis, AAAI 2021 — [arXiv:2103.11505](https://arxiv.org/abs/2103.11505). π + heuristic (our π + V decomposition) with guarantees.
12. **DeepCube — Solving the Rubik's Cube Without Human Knowledge** — McAleer et al., 2018 — [arXiv:1805.07470](https://arxiv.org/abs/1805.07470). Origin of the backward-from-goal curriculum (our reverse-generation patch idea).
13. **DeepCubeA** — Agostinelli et al., *Nature MI* 2019 — [paper](https://www.nature.com/articles/s42256-019-0070-z). Cost-to-go value + weighted A*; value trained purely on solvable states (our censoring-free V precedent).
14. **Go-Explore** — Ecoffet et al., 2019 / *Nature* 2021 ("First return, then explore") — [arXiv:1901.10995](https://arxiv.org/abs/1901.10995). Detachment/derailment — why plain rollouts miss rare states; the justification for our forced first-push sweeps.
15. **HER — Hindsight Experience Replay** — Andrychowicz et al., NeurIPS 2017 — [arXiv:1707.01495](https://arxiv.org/abs/1707.01495). Relabel failures as successes for the goal they DID achieve — on our patch list for de-censoring.
16. **An Investigation of Model-Free Planning (DRC)** — Guez et al., ICML 2019 — [arXiv:1901.03559](https://arxiv.org/abs/1901.03559). Model-free Sokoban at 1e8–1e9 steps — the sample-scale reference for why pure model-free was out of budget. (DRC = Deep Repeated ConvLSTM.)

## Tier 3 — cousins and history (skim / reference)

17. **ReST — Reinforced Self-Training for Language Modeling** — Gulcehre et al., 2023 — [arXiv:2308.08998](https://arxiv.org/abs/2308.08998). The Grow/Improve loop, MT-focused; ReST^EM's predecessor.
18. **RAFT — Reward rAnked FineTuning** — Dong et al., 2023 — [arXiv:2304.06767](https://arxiv.org/abs/2304.06767). Same recipe for alignment.
19. **Scaling Relationship on Learning Mathematical Reasoning (RFT)** — Yuan et al., 2023 — [arXiv:2308.01825](https://arxiv.org/abs/2308.01825). Rejection-sampling fine-tuning; log-linear data scaling; distinct-solutions-per-problem matters (our buffer-diversity rule's cousin).
20. **Decision Transformer** — Chen et al., NeurIPS 2021 — [arXiv:2106.01345](https://arxiv.org/abs/2106.01345). Reward-conditioned supervised RL — the other way to make SL do RL.
21. **Upside-Down RL** — Schmidhuber 2019 [arXiv:1912.02875](https://arxiv.org/abs/1912.02875) (position) + Srivastava et al. [arXiv:1912.02877](https://arxiv.org/abs/1912.02877) (empirical).
22. **Using EM for Reinforcement Learning** — Dayan & Hinton, *Neural Computation* 1997 — [MIT Press](https://direct.mit.edu/neco/article/9/2/271/6034). The 1997 origin of the whole EM-RL lineage.
