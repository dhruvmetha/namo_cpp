# Reading List: Empirical Characterization of Feasible Sets in Contact-Rich Manipulation

A focused reading list for the F-characterization research direction. Complements the broader `reading_list.md` (which covers TAMP / sampler evolution / world models). This one is specifically about the intellectual neighborhood of "characterize the success set of a controller, then design models around its structure."

Organized by purpose, not chronology. Each entry has a one-line *why* — read this when, and what to take from it. Strong recommendations marked with **[★]**.

---

## Part I — The Core Lineage (Read These First)

These are the works your framing directly inherits from. If you can defend your work in conversation with the ideas in this section, you are fluent in the vocabulary the structural-manipulation community uses.

### Region of Attraction & Funnel Composition (the analytical roots)

| # | Paper / Book | Authors | Venue | Why read it |
|---|---|---|---|---|
| 1 | **[★] Sequential Composition of Dynamically Dexterous Robot Behaviors** | Burridge, Rizzi, Koditschek | IJRR 1999 | The original "funnels" paper. Defines composing controllers by their basins of attraction. Your F is the action-space analogue. |
| 2 | **[★] LQR-Trees: Feedback Motion Planning via Sums-of-Squares Verification** | Tedrake, Manchester, Tobenkin, Roberts | IJRR 2010 | Computes RoA explicitly for nonlinear systems via SOS. Your work is the empirical version where SOS doesn't apply. |
| 3 | Underactuated Robotics (textbook) — Chapter on Lyapunov Analysis & Funnel Composition | Tedrake | MIT, ongoing | Vocabulary source. Read the funnel chapter only. |
| 4 | **[★] Global Planning for Contact-Rich Manipulation via Local Smoothing of Quasi-dynamic Contact Models** | H.J. Terry Suh, Pang, Tedrake | IJRR 2023 | Contact Trust Region — the closest analytical relative of your F. RoA in action space for *contact* systems, but for tractable few-contact cases. You characterize where their analytical method can't reach. |
| 5 | **[★] Suh PhD Thesis (the one in your repo)** | H.J.T. Suh | MIT 2025 | Already in your reading. The bridge between analytical RoA and learned manipulation. Your work is empirical-side complement to his analytical-side contribution. |

### Empirical Characterization of Manipulation Success Sets (the methodological roots)

| # | Paper | Authors | Venue | Why read it |
|---|---|---|---|---|
| 6 | **[★] Dex-Net 2.0: Deep Learning to Plan Robust Grasps with Synthetic Point Clouds and Analytic Grasp Metrics** | Mahler, Liang, Niyaz, Laskey, Doan, Liu, Aparicio, Goldberg | RSS 2017 | The methodological precedent. Goldberg characterized grasp success regions across millions of synthetic instances and built a learned predictor on top. *Your work is Dex-Net for push primitives in clutter.* Cite this. |
| 7 | Dex-Net 3.0 / 4.0 follow-ups | Mahler, Goldberg et al. | ICRA 2018 / Sci. Robotics 2019 | Same methodology, scaled. Skim for evaluation conventions. |
| 8 | Learning Ambidextrous Robot Grasping Policies | Mahler et al. | Sci. Robotics 2019 | Goldberg's "characterize, then learn" applied at scale. Read the evaluation section. |
| 9 | **[★] Mechanics of Manipulation (textbook)** — Chapter on Pushing | Matt Mason | MIT, 2001 | Mason's voting theorem for planar pushing. The closed-form result for *single-contact* pushing that your contact-rich, multi-object setting breaks. Required cultural context. |
| 10 | A Probabilistic Framework for Object Search with 6-DoF Pose Estimation | Wong et al. | IJRR 2013 | Earlier example of empirical characterization of a manipulation primitive's success set. |

### Push Mechanics — Where the Physics Comes From

| # | Paper | Authors | Venue | Why read it |
|---|---|---|---|---|
| 11 | Mechanics of Pushing | Mason | IJRR 1986 | The original. Single-contact pushing with friction. |
| 12 | Stable Pushing: Mechanics, Controllability, and Planning | Lynch & Mason | IJRR 1996 | Push planning with mechanics-aware controller design. The lineage of "structured planning" your work descends from. |
| 13 | Reactive Planar Manipulation with Convex Hybrid MPC | Hogan & Rodriguez | ICRA 2016 | Modern model-based pushing controller. Useful baseline for what analytical methods can do when the contact model is known. |
| 14 | A Convex Polynomial Force-Motion Model for Planar Sliding | Zhou, Bauza, Walker, Mason | IJRR 2018 | Recent analytical work on push outcomes. Useful to cite as "where analytical methods are; here is where they stop." |

---

## Part II — Learned Samplers Over Action Spaces

The neighborhood of your evaluation. These papers train models that sample actions; almost none of them characterize the action-space success set first.

### Diffusion Policies / Generative Action Models

| # | Paper | Authors | Venue | Why read it |
|---|---|---|---|---|
| 15 | **[★] Diffusion Policy: Visuomotor Policy Learning via Action Diffusion** | Chi, Feng, Du, Xu, Burchfiel, Tedrake, Song | RSS 2023 | The reference diffusion-policy paper. The PushT benchmark comes from here. Read carefully — your F characterization of PushT would directly engage this paper. |
| 16 | Implicit Behavioral Cloning | Florence, Lynch, Zeng, Ramirez et al. | CoRL 2021 | Energy-based policies as an alternative framing. The "policy as energy landscape" view is closer to "policy as RoA approximator" than diffusion is. |
| 17 | Goal-Conditioned Imitation Learning via Action-Quantized Discretization | Lee et al. | CoRL 2023 | Discretized action spaces with learned policies — closer to your discrete primitive setting than continuous diffusion. |
| 18 | Conditional Behavior Cloning with Diffusion Models for Manipulation | Pearce et al. | NeurIPS 2023 workshop | Recent comparison of diffusion vs simpler approaches on manipulation. Useful baseline reference. |
| 19 | 3D Diffusion Policy | Ze et al. | RSS 2024 | Scene-conditioned diffusion in 3D. Close cousin to what your sage_learning model does. |

### Learned Samplers for Planning

| # | Paper | Authors | Venue | Why read it |
|---|---|---|---|---|
| 20 | Learning Sampling Distributions for Robot Motion Planning | Ichter, Harrison, Pavone | ICRA 2018 | Already in your main reading list. Listed here too because it is the sampler-learning paper your work has the cleanest dialogue with. |
| 21 | **[★] Learning Compositional Models of Robot Skills for TAMP** | Z. Wang, Garrett, Kaelbling, Lozano-Pérez | IJRR 2021 | Skill samplers learned from data, with explicit feasibility criteria. The intellectual closest cousin from the LIS group. |
| 22 | Generative Skill Chaining: Long-Horizon Skill Planning with Diffusion Models | Mishra, Chen, Park, Garg | CoRL 2023 | Diffusion samplers chained for multi-step problems. Relevant when you extend to N-push. |
| 23 | Diffusion Forcing | Chen, Monsó, Du, Simchowitz, Tedrake, Sitzmann | NeurIPS 2024 | New diffusion training technique that affects how you'd condition on goal. Worth knowing. |

### Calibration & Evaluation of Learned Policies

| # | Paper | Authors | Venue | Why read it |
|---|---|---|---|---|
| 24 | **[★] Are We Really Making Much Progress in Manipulation? A Study on the Reproducibility of Diffusion Policy** | Various | arXiv 2024 | A field self-criticism. Argues diffusion-policy results are inflated by evaluation choices. Aligns with your structural-evaluation framing. |
| 25 | Evaluating Real-World Robot Manipulation Policies in Simulation | Pumacay, Singh, Garg et al. | arXiv 2024 | Evaluation methodology in manipulation. Useful for your "structural alignment" metric design. |
| 26 | Diffusion Policy Policy Optimization | Ren et al. | arXiv 2024 | Where diffusion policies fail on harder tasks; motivates your "what does the model actually learn" framing. |

---

## Part III — TAMP & Structural Decomposition

The community whose ideas underpin the C ∩ R decomposition. Your work uses their decomposition machinery without their full TAMP apparatus.

| # | Paper / Thesis | Authors | Venue | Why read it |
|---|---|---|---|---|
| 27 | Hierarchical Task and Motion Planning in the Now | Kaelbling & Lozano-Pérez | ICRA 2011 | Foundational TAMP. The C ∩ R decomposition you use is implicit here. |
| 28 | **[★] Sampling-Based Methods for Factored Task and Motion Planning** | Garrett, Lozano-Pérez, Kaelbling | IJRR 2018 | PDDLStream. Operationalizes "what is feasible" as a sampler problem. Your F is the per-skill version. |
| 29 | Integrated Task and Motion Planning (Annual Reviews) | Garrett, Chitnis, Holladay, Kim, Silver et al. | Annual Reviews 2021 | Already in main list. The standard TAMP survey. |
| 30 | Online Replanning in Belief Space for Partially Observable TAMP | Garrett et al. | RSS 2020 | Where uncertainty enters TAMP. Relevant if you ever frame your characterization probabilistically. |

---

## Part IV — Methodology & Philosophy of Empirical-First Research

Less about manipulation, more about how to *do* this kind of work and write it up. Read these when your motivation flags.

| # | Source | Author | Why read it |
|---|---|---|---|
| 31 | **[★] Tom Silver's blog posts on robotics research methodology** | Tom Silver | Princeton | Already in your memory. "Be dogmatic about problems, not approaches." The methodological twin of your work. Re-read whenever you're tempted by method-first thinking. |
| 32 | **[★] The Bitter Lesson** | Rich Sutton | 2019 | The argument *against* your framing. Required reading because you need to be able to articulate why your work matters in a world where Sutton's argument is taken seriously. The honest answer: contact-rich manipulation is where Sutton's argument has the weakest empirical support, and you should say so. |
| 33 | A Few Useful Things to Know About Machine Learning | Pedro Domingos | CACM 2012 | Old but durable. The "dimensions of learning" framing applies cleanly to your "what does the model actually need to capture" framing. |
| 34 | Why Most Published Research Findings Are False | John Ioannidis | PLOS Med 2005 | Required reading for anyone making empirical claims. Helps calibrate how strongly to phrase your hypothesis-confirmation language. |
| 35 | Patterns, Predictions, and Actions: Foundations of Machine Learning (textbook), Chapter on Generalization | Hardt & Recht | 2022 | Useful background when you frame "structural alignment" as a generalization claim. |

---

## Part V — Adjacent Work You Should Cite Carefully

You don't need to read all of these in depth — knowing they exist and engaging with the central claim of each is enough. They are the works reviewers will ask "did you cite X?"

### NAMO Specifically

| # | Paper | Authors | Venue | Why know it |
|---|---|---|---|---|
| 36 | Navigation Among Movable Obstacles: Real-Time Reasoning in Complex Environments | Stilman & Kuffner | Humanoids 2004 | The original NAMO paper. Cite to establish the problem lineage. |
| 37 | Planning Among Movable Obstacles with Artificial Constraints | Stilman | IJRR 2008 | Stilman's full treatment. Your work modernizes the problem; cite the original. |
| 38 | Multi-Heuristic A* for Real-Time NAMO | various follow-ups | various | Skim. Establish that you know the search-based NAMO line of work. |
| 39 | Recent NAMO with learning (Wang, Driess, Ren, etc.) | various | recent | Spot-check. Position your work against any recent NAMO with learning. |

### Contact-Rich Manipulation Learning

| # | Paper | Why know it |
|---|---|---|
| 40 | Learning Contact-Rich Manipulation Skills with Guided Policy Search (Levine et al., ICRA 2015) | Early example of learning over contact dynamics. |
| 41 | Solving Rubik's Cube with a Robot Hand (OpenAI, 2019) | The "scale-and-train wins on contact-rich" canonical example. Engage with it — argue why pushing in clutter is different. |
| 42 | RT-2 / Open X-Embodiment / RT-X papers | The current scale-and-train wave. Be able to articulate why they don't address what you address. |
| 43 | π_0 / π_0.5 (Physical Intelligence) | The most ambitious recent VLA effort. Same — know it, argue why it doesn't replace structural understanding for contact-rich. |
| 44 | HACMan: Hybrid Actor-Critic Maps for Manipulation Learning (Zhou, Held, Fazeli) | Recent contact-rich pushing with learning. Useful contrast point. |

### Foundation / Vision-Language for Manipulation

| # | Paper | Why know it |
|---|---|---|
| 45 | Code as Policies (Liang et al., ICRA 2023) | LLMs writing manipulation programs. Relevant only as the alternative paradigm. |
| 46 | VoxPoser (Huang et al., CoRL 2023) | LLM-VLM for manipulation. Same — knowledge cost is low, citation expected. |
| 47 | RoboFlamingo / OpenVLA / Octo | Cite if reviewer asks; otherwise skip in your work. |

---

## Part VI — Specifically for the 1-Push Diffusion Evaluation Paper

If you are writing the 1-push diffusion-vs-classifier evaluation against ground-truth F as your next paper, these are the most directly relevant references. Read these before writing the related work section.

1. Diffusion Policy (Chi et al., RSS 2023) — the architecture you're evaluating against ground truth.
2. Dex-Net 2.0 (Mahler et al., RSS 2017) — the methodological precedent. Cite as "we apply Dex-Net's empirical characterization methodology to contact-rich pushing in clutter."
3. Suh's contact trust region work (IJRR 2023 + thesis) — the analytical contrast.
4. PushT benchmark papers — for replication study.
5. Ichter et al. on learned samplers (ICRA 2018) — to position your evaluation in the sampler-learning literature.
6. Wang/Garrett/Kaelbling/Lozano-Pérez learned compositional skills (IJRR 2021) — the LIS group's version.
7. Tom Silver's neuro-symbolic skills paper (CoRL 2022) — methodology twin.
8. Stilman's original NAMO papers — problem-lineage citations.

If you read only these eight before writing, the related-work section will be defensible.

---

## How to Use This List

- **Don't try to read all of it.** This is a reference, not a curriculum. Pick papers from Part I and Part VI for active reading; the rest is for citation and conversation.
- **Read the "why" column first**, then decide whether to read the paper itself.
- **For each paper you read, write 3 lines:** (1) central claim, (2) what your work does that this paper does not, (3) one sentence you could use to cite it. This compounds. After 30 papers you have a fluent citation vocabulary.
- **Annotate this file.** Add a column "read on [date], notes: [link to your notes]" as you go. The list is a living document.
- **The papers marked [★] are the high-leverage ones.** If you only have a week, read just those (10 papers).

---

## Honest meta-note

A reading list is not a substitute for thinking. It is a tool to ensure your thinking is informed by what others have already established. The danger is reading list as procrastination — accumulating papers as a way to feel productive without doing the actual work of running experiments and writing.

Suggested rule: **for every paper you read, run one experiment or write one paragraph of the paper.** Reading without producing is a trap. Producing without reading is shallow. The interleave is what makes the work real.
