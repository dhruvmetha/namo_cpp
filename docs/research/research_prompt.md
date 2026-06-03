# Research Prompt: Feasible Sub-Goal Sampling in Contact-Rich TAMP

## The Core Idea

In Task and Motion Planning (TAMP), every skill (push, grasp, place, insert, pour) takes continuous goal parameters. The robot must find parameters that are both **desirable** (achieve the task goal) and **achievable** (physically executable given the current state). We call this intersection the **feasible set F = T ∩ A**, where T is the set of task-satisfying parameters and A is the set of physically achievable parameters.

Finding F is a needle-in-a-needle problem: T is already sparse (few parameters achieve the goal), and grounding T in A filters further (not all good goals are achievable). This cascading sparsity is what makes sub-goal sampling the bottleneck in TAMP.

Despite dozens of papers on learned samplers for TAMP, nobody has defined F explicitly, measured its properties, or decomposed sampler failures into T vs A. Every sampler — CVAEs, GPs, diffusion models, CEM, differentiable optimization — searches for F without looking at it.

We propose to define F, measure it, and show that the T ∩ A decomposition is a diagnostic and improvement tool for any learned sampler in contact-rich TAMP.

---

## What to Search For

### 1. Learned samplers for TAMP continuous parameters

These are papers that learn to generate/score continuous parameters (goal poses, grasp poses, placement poses, push parameters, trajectories) to accelerate TAMP search. They are all implicitly searching for F = T ∩ A without naming it.

**Key papers to find and understand:**
- Chitnis et al. (ICRA 2016) — first learned TAMP sampler via IRL
- Ichter, Harrison, Pavone (ICRA 2018) — CVAE for sampling distributions in motion planning
- Wang, Garrett, Kaelbling, Lozano-Pérez (IJRR 2021) — GP-based samplers with active learning for TAMP
- Diffusion-CCSP (Yang et al., CoRL 2023) — compositional diffusion for constraint satisfaction, trains separate models per constraint type and composes at inference
- DiMSam (CoRL 2023) — diffusion as TAMP sampler under partial observability
- Silver et al. (CoRL 2022) — neuro-symbolic skills with learned samplers
- Ortiz-Haro et al. (CoRL 2022) — structured deep generative models for constraint manifold sampling
- Mendez-Mendez et al. (CoRL 2023) — embodied lifelong learning for TAMP samplers

**Questions to answer for each:**
- What does it sample? (goal poses, grasps, waypoints, trajectories?)
- How does it sample? (CVAE, GP, diffusion, GAN, neural network?)
- Does it define or analyze the feasible set?
- Does it decompose failures into task constraint vs physical constraint?
- What benchmarks does it evaluate on?

### 2. TAMP search guidance and heuristics

These operate at or above the sub-goal level — guiding which actions to try, which plans to refine, which parameters to explore.

**Key papers:**
- PIGINet (Yang et al., RSS 2023) — Transformer predicting plan skeleton feasibility
- Khodeir et al. (RA-L 2023) — GNN-guided BFS expansion in PDDLStream
- Kim & Shimanuki (CoRL 2020) — GNN action-value for geometric TAMP
- Bradley et al. (ISER 2023) — learning feasibility and cost to guide TAMP
- Driess et al. (ICRA 2020, RSS 2020) — deep visual heuristics for manipulation planning

**Questions to answer:**
- Do they address continuous parameter sampling or only discrete search guidance?
- Could their approach benefit from T vs A failure decomposition?

### 3. Differentiable/parallel TAMP (recent, 2025-2026)

New approaches that find feasible parameters through optimization rather than learning.

**Key papers:**
- STAMP (Lee et al., RA-L 2025) — Stein Variational Gradient Descent for TAMP, treats parameter finding as variational inference using differentiable simulation
- cuTAMP (RSS 2025, NVIDIA) — GPU-parallelized TAMP, samples thousands of candidates simultaneously and refines via differentiable optimization. Reported 0.3% feasible candidates on Tetris packing.
- Hybrid Diffusion (RSS 2025 workshop) — simultaneous symbolic and continuous planning

**Questions to answer:**
- Do they characterize the feasible set or just search through it?
- cuTAMP reports |F|/P ≈ 0.3% — do they analyze why, or just solve it with brute force?
- Could T vs A decomposition help these approaches target their optimization?

### 4. Grasp sampling and feasibility

Grasping is the most universal T ∩ A problem — "stable grasp that the arm can reach." Existing grasp samplers predict T (grasp quality) but often don't account for A (approach feasibility in clutter).

**Key papers:**
- 6-DOF GraspNet (Mousavian et al., ICCV 2019) — VAE grasp sampler + evaluator
- Contact-GraspNet (Sundermeyer et al., ICRA 2021) — end-to-end 6DOF grasps from point clouds
- AnyGrasp (Fang et al., T-RO 2023) — graspness-based sampling
- Dex-Net 2.0 (Mahler et al., RSS 2017) — GQ-CNN for grasp quality + cross-entropy sampling
- SE(3)-DiffusionFields (Urain et al., ICRA 2023) — diffusion for joint grasp and motion optimization

**Questions to answer:**
- Do they predict T (grasp quality) independently from A (approach feasibility)?
- How do they handle the case where a high-quality grasp is unreachable?
- Do they measure or report the cascading sparsity (many good grasps exist, few are reachable in clutter)?
- Is there any work that explicitly decomposes grasp failures into T vs A?

### 5. Non-prehensile manipulation and contact-rich planning

Tasks where contact physics during execution couples T and A — the same physics determines both what's achievable and what achieves the goal.

**Key papers:**
- Bauza & Rodriguez (ICRA 2017) — probabilistic push outcome prediction
- VFT / Visual Foresight Trees (Huang et al., RA-L 2022) — tree search with learned push prediction
- DIPN (Huang et al., ICRA 2021) — deep interaction prediction for clutter
- Push-Net (Li et al., RSS 2018) — deep planar pushing
- Any work on insertion/assembly planning with contact

**Questions to answer:**
- Do these forward models implicitly learn F, or just A (achievable outcomes)?
- Is the task constraint T ever explicitly separated from achievability A?
- In insertion tasks, how is the feasibility of the insertion separated from the goal of assembly?

### 6. NAMO (Navigation Among Movable Obstacles)

Our primary domain. Need to understand all prior formulations and how they handle the "where to push" question.

**Key papers:**
- Stilman & Kuffner (2005, 2007 thesis) — original NAMO, "keyhole" concept
- Van den Berg et al. (2009) — probabilistically complete NAMO, random sampling
- Levihn & Stilman (2014) — NAMO in unknown environments
- Scholz et al. (IROS 2016) — learned dynamics for NAMO
- Wang et al. (2022) — uniform object rearrangement, non-monotone search
- Yao et al. (2023) — deep RL for local NAMO
- Yang et al. (IROS 2025) — hierarchical RL for NAMO with mobile manipulator
- NAMO-LLM (Zhang & Kantaros 2025) — LLM-guided NAMO

**Questions to answer:**
- Does any NAMO paper define the feasible set of push goals explicitly?
- How does each paper decide WHERE to push the blocking object?
- Do any decompose the problem into clearance (T) and reachability (A)?

### 7. Constraint satisfaction and composition in robotics

How constraints are represented, learned, and composed — relevant to the T ∩ A decomposition.

**Key papers:**
- Diffusion-CCSP (Yang et al., CoRL 2023) — compositional diffusion per constraint type
- StructDiffusion (Liu et al., RSS 2023) — diffusion for language-guided placement + discriminator for physical validity
- RPDiff (Simeonov et al., CoRL 2023) — relational pose diffusion for rearrangement
- Any work on compositional energy-based models for robotics

**Questions to answer:**
- Diffusion-CCSP trains separate models per constraint and composes — does it analyze when composition works vs fails?
- Is there any work that measures the coupling/independence between different constraints?
- Does any compositional approach diagnose failures as "which constraint was violated"?

### 8. Difficulty characterization in planning

Any work that measures what makes planning problems hard — related to our |F|/P and cascading sparsity analysis.

**Key papers:**
- cuTAMP (RSS 2025) — reports 0.3% feasible candidates as a difficulty observation
- Any work on heavy-tailed search time distributions in TAMP
- Any work on problem difficulty prediction for motion planning / TAMP
- Neural A* (Yonetani et al., ICML 2021) — learned heuristic, implicitly about directing search toward sparse feasible paths

**Questions to answer:**
- Has anyone measured |F|/P (fraction of feasible candidates) as a difficulty metric?
- Has anyone decomposed difficulty into separate constraint contributions?
- Is there any work on predicting TAMP problem difficulty before solving?

### 9. Active learning and curriculum for robotics

Relevant to our F-guided training loop (measure F → identify hard scenes → generate more → retrain).

**Key papers:**
- Wang et al. (IJRR 2021) — active learning with straddle acquisition for TAMP skill models
- Any work on curriculum learning for manipulation skills
- Any work on hard example mining for robot learning
- Any work on procedural environment generation for training difficulty

**Questions to answer:**
- Does any work use feasible set analysis to guide data collection?
- How do existing curriculum approaches decide what's "hard"?
- Is there a principled way to generate training environments targeting specific difficulty properties?

### 10. Diffusion and generative models for manipulation

The broader landscape of generative models applied to robotics tasks.

**Key papers:**
- Diffuser (Janner et al., ICML 2022) — diffusion for trajectory planning
- Diffusion Policy (Chi et al., RSS 2023) — visuomotor policy via action diffusion
- Decision Diffuser (Ajay et al., ICLR 2023) — classifier-free guidance for decision-making
- Motion Planning Diffusion (Carvalho et al., IROS 2023) — diffusion for trajectory priors
- FlowMP (2025) — flow matching for robot motion planning

**Questions to answer:**
- Which of these could benefit from T vs A failure decomposition?
- Do any of them analyze what the model has learned (what region of parameter space it covers)?

---

## The Framework We're Proposing

### Definition
For any TAMP skill with goal parameters θ:
- **T(θ)** = 1 if executing skill(θ) achieves the task goal (desirability)
- **A(θ)** = 1 if skill(θ) is physically executable from the current state (achievability)
- **F = {θ : T(θ) = 1 AND A(θ) = 1}** — the feasible set

### Cascading sparsity
- |T|/P: how sparse is the task constraint (needle in haystack)
- |F|/|T|: how much does achievability grounding filter (needle in needle)
- |F|/P: final sparsity the sampler faces

### Diagnostic
When a sampler predicts θ̂ that fails:
- θ̂ satisfies T but not A → model learned the task, not the physics
- θ̂ satisfies A but not T → model learned the physics, not the task
- θ̂ satisfies neither → model is off

### The improvement loop
Measure F → Diagnose failures (T vs A) → Identify scene features producing hard F → Generate targeted training scenes → Retrain → Verify improvement on |F|/P curves

### Multi-step bootstrapping
Train single-step F model → Use it to evaluate feasibility at future states → Collect multi-step data efficiently → Scale to deeper planning

---

## What We're Trying to Show

1. **F = T ∩ A is the right abstraction** for understanding sub-goal sampling in TAMP
2. **Cascading sparsity** (T then A) explains difficulty in a measurable way
3. **T vs A diagnosis** reveals what learned samplers have and haven't learned — unavailable to black-box approaches
4. **The improvement loop** (diagnose → target → retrain) systematically improves any sampler
5. **The framework transfers** across skills (pushing, grasping) and coupling regimes

## What We're NOT Trying to Show

- That any particular model (diffusion, CVAE, GP) is best
- That F has deep mathematical structure (it might or might not)
- That this framework replaces existing TAMP systems

---

## Domains for Experiments

### Primary: NAMO region opening (pushing skill)
- F = C (clearance) ∩ R (reachability)
- High T×A coupling (contact physics)
- 600 discrete primitives → F exhaustively computable
- 1-push and 2-push (multi-step bootstrapping)
- Full pipeline: characterize F, train sampler, diagnose, improve, bootstrap

### Secondary: Grasping in clutter (grasp skill)
- F = grasp quality ∩ approach feasibility
- Coupling varies with clutter density
- Existing grasp samplers as baselines (analyze, don't retrain)
- Show the T vs A diagnostic reveals failure modes in existing systems

### Baselines for NAMO
- EPS (exhaustive) — uninformed
- Geometric heuristic — hand-crafted
- CEM — adaptive, no learning
- Ichter CVAE (ICRA 2018) — black-box learned sampler
- Diffusion-CCSP (CoRL 2023) — F-decomposed sampler (separate T and A models, composed)
- SAGE — F-aware joint sampler (this work)

### Key evaluation
- Sampler efficiency as a function of |F|/P (sparsity-controlled comparison)
- T vs A failure decomposition per sampler
- Improvement from F-guided training loop vs random data collection

---

## Target Venue

**CoRL** — learning for robotics, values principled methodology with empirical validation across domains. Framework papers with clear diagnostic tools and takeaways for the community.

**RA-L** as fallback — more space for thorough analysis, accepts methodological contributions.
