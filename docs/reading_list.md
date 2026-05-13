# Reading List: Structure-Aware Grounding for Contact-Rich TAMP

## Part I — Foundations (Read First)

### Theses

| # | Thesis | Author | Where | Link |
|---|--------|--------|-------|------|
| 1 | **Sampling-Based Robot Task and Motion Planning in the Real World** | Caelan Garrett | MIT, 2021 | [DSpace](https://dspace.mit.edu/handle/1721.1/139990) |
| 2 | **Neuro-Symbolic Learning for Bilevel Robot Planning** | Tom Silver | MIT, 2024 | [PDF](https://dspace.mit.edu/bitstream/handle/1721.1/156646/silver-tslvr-phd-eecs-2024-thesis.pdf) |
| 3 | **Factored Task and Motion Planning with Combined Optimization, Sampling and Learning** | Joaquim Ortiz-Haro | TU Berlin, 2024 | [arXiv](https://arxiv.org/abs/2404.03567) |

### Surveys

| # | Title | Venue | Link |
|---|-------|-------|------|
| 4 | Integrated Task and Motion Planning | Annual Reviews, 2021 | [arXiv](https://arxiv.org/abs/2010.01083) |
| 5 | A Survey of Optimization-based TAMP: Classical to Learning | arXiv, 2024 | [arXiv](https://arxiv.org/abs/2404.02817) |
| 6 | Understanding World or Predicting Future? Survey of World Models | ACM CSUR, 2025 | [arXiv](https://arxiv.org/abs/2411.14499) |
| 7 | A Step Toward World Models: Survey on Robotic Manipulation | arXiv, 2025 | [arXiv](https://arxiv.org/abs/2511.02097) |
| 8 | Diffusion Models for Robotic Manipulation: A Survey | Frontiers, 2025 | [arXiv](https://arxiv.org/abs/2504.08438) |

---

## Part II — Thread 1: Feasible Set Structure & Learned Samplers

### The Sampler Evolution Chain (read in order)

| # | Title | Authors | Venue | Link |
|---|-------|---------|-------|------|
| 9 | Guided Search for Task and Motion Plans Using Learned Heuristics | Chitnis et al. | ICRA 2016 | [PDF](https://people.eecs.berkeley.edu/~pabbeel/papers/2016-ICRA-tamp-learning.pdf) |
| 10 | Learning Sampling Distributions for Robot Motion Planning | Ichter, Harrison, Pavone | ICRA 2018 | [arXiv](https://arxiv.org/abs/1709.05448) |
| 11 | Learning Compositional Models of Robot Skills for TAMP | Wang, Garrett, Kaelbling, Lozano-Perez | IJRR 2021 | [arXiv](https://arxiv.org/abs/2006.06444) |
| 12 | Learning Neuro-Symbolic Skills for Bilevel Planning | Silver et al. | CoRL 2022 | [arXiv](https://arxiv.org/abs/2206.10680) |
| 13 | Compositional Diffusion-Based Continuous Constraint Solvers (Diffusion-CCSP) | Yang et al. | CoRL 2023 | [arXiv](https://arxiv.org/abs/2309.00966) |
| 14 | DiMSam: Diffusion Models as Samplers for TAMP under Partial Observability | Fang et al. | IROS 2024 | [arXiv](https://arxiv.org/abs/2306.13196) |
| 15 | Learning Long-Horizon Action Dependencies in Sampling-Based Bilevel Planning | Cieslar et al. | CoRL 2024 | [OpenReview](https://openreview.net/forum?id=DsFQg0G4Xu) |

### Constraint Structure & Feasibility

| # | Title | Authors | Venue | Link |
|---|-------|---------|-------|------|
| 16 | Structured Deep Generative Models for Sampling on Constraint Manifolds | Ortiz-Haro et al. | CoRL 2022 | [PMLR](https://proceedings.mlr.press/v164/ortiz-haro22a.html) |
| 17 | Logic-Geometric Programming: Optimization-Based Combined TAMP | Toussaint | IJCAI 2015 | [PDF](https://www.ijcai.org/Proceedings/15/Papers/274.pdf) |
| 18 | Learning Feasibility and Cost to Guide TAMP | Bradley & Roy | ISER 2023 | [PDF](https://groups.csail.mit.edu/rrg/papers/cbradley_iser_2023.pdf) |

### TAMP Search Guidance

| # | Title | Authors | Venue | Link |
|---|-------|---------|-------|------|
| 19 | Sequence-Based Plan Feasibility Prediction (PIGINet) | Yang et al. | RSS 2023 | [arXiv](https://arxiv.org/abs/2211.01576) |
| 20 | Learning to Search in TAMP with Streams | Khodeir et al. | RA-L 2023 | [arXiv](https://arxiv.org/abs/2111.13144) |
| 21 | Learning Value Functions with Relational State Representations for TAMP | Kim & Shimanuki | CoRL 2019 | [PMLR](http://proceedings.mlr.press/v100/kim20a.html) |
| 22 | Embodied Lifelong Learning for TAMP | Mendez-Mendez et al. | CoRL 2023 | [arXiv](https://arxiv.org/abs/2307.06870) |

### TAMP Frameworks

| # | Title | Authors | Venue | Link |
|---|-------|---------|-------|------|
| 23 | PDDLStream: Integrating Symbolic Planners and Blackbox Samplers | Garrett, Lozano-Perez, Kaelbling | ICAPS 2020 | [arXiv](https://arxiv.org/abs/1802.08705) |
| 24 | Hierarchical Task and Motion Planning in the Now | Kaelbling & Lozano-Perez | ICRA 2011 | [PDF](https://people.csail.mit.edu/lpk/papers/hpnICRA11Final.pdf) |

### Differentiable / Parallel TAMP

| # | Title | Authors | Venue | Link |
|---|-------|---------|-------|------|
| 25 | STAMP: Differentiable TAMP via Stein Variational Gradient Descent | Lee et al. | RA-L 2025 | [arXiv](https://arxiv.org/abs/2310.01775) |
| 26 | cuTAMP: Differentiable GPU-Parallelized TAMP | Shen, Garrett, Kumar et al. | RSS 2025 | [arXiv](https://arxiv.org/abs/2411.11833) |

---

## Part III — Thread 2: World Models & Cheap Grounding

### World Models for Robotics

| # | Title | Authors | Venue | Link |
|---|-------|---------|-------|------|
| 27 | Mastering Diverse Domains through World Models (DreamerV3) | Hafner et al. | Nature, 2025 | [arXiv](https://arxiv.org/abs/2301.04104) |
| 28 | TD-MPC2: Scalable, Robust World Models for Continuous Control | Hansen, Su, Wang | ICLR 2024 | [arXiv](https://arxiv.org/abs/2310.16828) |
| 29 | DayDreamer: World Models for Physical Robot Learning | Wu, Escontrela, Hafner et al. | CoRL 2023 | [arXiv](https://arxiv.org/abs/2206.14176) |
| 30 | Robotic World Model: Neural Network Simulator for Robust Policy Optimization | Li, Krause, Hutter | arXiv, 2025 | [arXiv](https://arxiv.org/abs/2501.10100) |
| 31 | H-WM: Robotic TAMP Guided by Hierarchical World Model | Huang et al. | arXiv, 2026 | [arXiv](https://arxiv.org/abs/2602.11291) |
| 32 | Act2Goal: From World Model To General Goal-conditioned Policy | Zhou et al. | arXiv, 2025 | [arXiv](https://arxiv.org/abs/2512.23541) |

### Push / Contact Prediction

| # | Title | Authors | Venue | Link |
|---|-------|---------|-------|------|
| 33 | A Probabilistic Data-Driven Model for Planar Pushing | Bauza & Rodriguez | ICRA 2017 | [arXiv](https://arxiv.org/abs/1704.03033) |
| 34 | A Data-Efficient Approach to Precise and Controlled Pushing | Bauza, Hogan, Rodriguez | CoRL 2018 | [arXiv](https://arxiv.org/abs/1807.09904) |
| 35 | Push-Net: Deep Planar Pushing | Li, Hsu, Lee | RSS 2018 | [RSS](https://www.roboticsproceedings.org/rss14/p24.html) |
| 36 | DIPN: Deep Interaction Prediction Network | Huang et al. | ICRA 2021 | [arXiv](https://arxiv.org/abs/2011.04692) |
| 37 | Visual Foresight Trees for Object Retrieval from Clutter | Huang et al. | RA-L 2022 | [arXiv](https://arxiv.org/abs/2105.02857) |
| 38 | PIN-WM: Physics-Informed World Models for Non-Prehensile Manipulation | Li et al. | RSS 2025 | [arXiv](https://arxiv.org/abs/2504.16693) |
| 39 | Deep Visual Reasoning: Learning to Predict Action Sequences for TAMP | Driess, Ha, Toussaint | RSS 2020 | [arXiv](https://arxiv.org/abs/2006.05398) |

### Diffusion / Generative Models for Robotics

| # | Title | Authors | Venue | Link |
|---|-------|---------|-------|------|
| 40 | Planning with Diffusion for Flexible Behavior Synthesis (Diffuser) | Janner et al. | ICML 2022 | [arXiv](https://arxiv.org/abs/2205.09991) |
| 41 | Diffusion Policy: Visuomotor Policy Learning via Action Diffusion | Chi et al. | RSS 2023 | [arXiv](https://arxiv.org/abs/2303.04137) |
| 42 | SE(3)-DiffusionFields: Joint Grasp and Motion Optimization | Urain et al. | ICRA 2023 | [arXiv](https://arxiv.org/abs/2209.03855) |
| 43 | FlowMP: Motion Fields for Robot Planning with Flow Matching | Nguyen et al. | arXiv, 2025 | [arXiv](https://arxiv.org/abs/2503.06135) |

---

## Part IV — LLM Planners + Grounding Gap

| # | Title | Authors | Venue | Link |
|---|-------|---------|-------|------|
| 44 | Do As I Can, Not As I Say: Grounding Language in Robotic Affordances (SayCan) | Ahn et al. | arXiv, 2022 | [arXiv](https://arxiv.org/abs/2204.01691) |
| 45 | Inner Monologue: Embodied Reasoning through Planning with Language Models | Huang et al. | CoRL 2023 | [arXiv](https://arxiv.org/abs/2207.05608) |
| 46 | Trust the PRoC3S: Solving Long-Horizon Robotics Problems with LLMs and Constraint Satisfaction | Curtis et al. | CoRL 2024 | [arXiv](https://arxiv.org/abs/2406.05572) |
| 47 | LLM-GROP: Visually Grounded Robot Task and Motion Planning with LLMs | Zhang et al. | IJRR 2025 | [arXiv](https://arxiv.org/abs/2511.07727) |
| 48 | LLM3: Large Language Model-based Task and Motion Planning | Wang et al. | IROS 2024 | [arXiv](https://arxiv.org/abs/2403.11552) |
| 49 | NAMO-LLM: Navigation Among Movable Obstacles with LLM Guidance | Zhang & Kantaros | RA-L 2025 | [arXiv](https://arxiv.org/abs/2505.04141) |

---

## Part V — NAMO Domain

| # | Title | Authors | Venue | Link |
|---|-------|---------|-------|------|
| 50 | Navigation Among Movable Obstacles: Real-Time Reasoning in Complex Environments | Stilman & Kuffner | IJHR 2005 | [PDF](https://www.ri.cmu.edu/pub_files/pub4/stilman_michael_2005_3/stilman_michael_2005_3.pdf) |
| 51 | Path Planning among Movable Obstacles: A Probabilistically Complete Approach | Van den Berg et al. | WAFR 2008/Springer 2009 | [Springer](https://link.springer.com/chapter/10.1007/978-3-642-00312-7_37) |
| 52 | Navigation Among Movable Obstacles with Learned Dynamic Constraints | Scholz et al. | IROS 2016 | [IEEE](https://ieeexplore.ieee.org/document/7759546) |
| 53 | Locally Optimal NAMO in Unknown Environments | Levihn & Stilman | Humanoids 2014 | [PDF](http://www.martinlevihn.com/LevihnHUMANOIDS2014.pdf) |
| 54 | Hierarchical RL for NAMO with Mobile Manipulator | Yang et al. | IROS 2025 | [arXiv](https://arxiv.org/abs/2506.15380) |
| 55 | NAMOUnc: NAMO with Decision Making on Uncertainty Interval | Zhang et al. | ICINCO 2025 | [arXiv](https://arxiv.org/abs/2509.12723) |

---

## Part VI — Grasp & Rearrangement (secondary domain for F = T ∩ A generality)

| # | Title | Authors | Venue | Link |
|---|-------|---------|-------|------|
| 56 | 6-DOF GraspNet: Variational Grasp Generation | Mousavian et al. | ICCV 2019 | [arXiv](https://arxiv.org/abs/1905.10520) |
| 57 | Contact-GraspNet: Efficient 6-DoF Grasp Generation in Cluttered Scenes | Sundermeyer et al. | ICRA 2021 | [arXiv](https://arxiv.org/abs/2103.14127) |
| 58 | AnyGrasp: Robust and Efficient Grasp Perception | Fang et al. | T-RO 2023 | [arXiv](https://arxiv.org/abs/2212.08333) |
| 59 | Dex-Net 2.0: Deep Learning to Plan Robust Grasps | Mahler et al. | RSS 2017 | [arXiv](https://arxiv.org/abs/1703.09312) |
| 60 | Uniform Object Rearrangement: Monotone to Non-Monotone Search | Wang et al. | ICRA 2021 | [arXiv](https://arxiv.org/abs/2101.12241) |
| 61 | Monte-Carlo Tree Search for Visually Guided Rearrangement | Labbe et al. | RA-L 2020 | [arXiv](https://arxiv.org/abs/1904.10348) |

---

## Key Groups to Follow

| Group | PI(s) | Where | Why |
|-------|-------|-------|-----|
| **LIS Lab** | Kaelbling & Lozano-Perez | MIT | PDDLStream, Diffusion-CCSP, DiMSam, PIGINet, cuTAMP |
| **PRPL Lab** | Tom Silver | Princeton | Bilevel planning, neuro-symbolic skills, predicate invention |
| **Machine Learning for Robotics** | Marc Toussaint | TU Berlin | Logic-Geometric Programming, constraint manifold sampling |
| **Robot Vision & Learning** | Florian Shkurti | U Toronto | STAMP, GNN-guided TAMP search |
| **MCube Lab** | Alberto Rodriguez | MIT/NVIDIA | Push prediction, contact-rich manipulation |
| **NVIDIA Research** | Dieter Fox, Fabio Ramos, Caelan Garrett | NVIDIA | cuTAMP, DiMSam, Contact-GraspNet |
| **Hafner / Dreamer** | Danijar Hafner | Google DeepMind | DreamerV3, world models for control |
| **Energy-Based Models** | Yilun Du, Josh Tenenbaum | MIT | Compositional diffusion, energy composition |
| **Stanford ASL** | Marco Pavone, Brian Ichter | Stanford/Google | Learned sampling distributions, SayCan |

---

## Suggested Reading Order

**Week 1-2**: Theses #1-3 (skim, focus on related work chapters + problem formulations)

**Week 3**: NAMO foundations #50-52, then TAMP frameworks #23-24

**Week 4**: Sampler chain #9 -> #10 -> #11 -> #13 (IRL -> CVAE -> GP -> diffusion)

**Week 5**: Constraint structure #16-17, feasibility guidance #18-20

**Week 6**: World models #27-29, push prediction #33-34, #37-38

**Week 7**: Diffusion for robotics #40-42, differentiable TAMP #25-26

**Week 8**: LLM planners #44-46, the grounding gap #47-49
