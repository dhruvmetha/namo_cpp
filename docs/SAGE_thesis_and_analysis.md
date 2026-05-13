# Feasible Sub-Goal Sampling for Contact-Rich TAMP

## The Paper Story

In contact-rich TAMP, the sub-goal sampling problem has hidden structure that nobody has looked at.

We define the feasible set F = C ∩ R — the intersection of what achieves the task and what's physically achievable. We compute F exhaustively on thousands of NAMO instances and, for the first time, characterize what it looks like: how sparse it is, whether clearance or reachability is the bottleneck, how the two constraints interact.

This characterization reveals something specific about the problem. Whatever we find, it's the first time anyone has measured this.

We then test whether this structure is exploitable. We train separate models for C and R and compose them (Diffusion-CCSP), and compare against a joint predictor (SAGE). The comparison directly tests whether the decomposition we defined is useful in practice or whether the coupling between constraints demands joint prediction.

**One sentence**: We define, measure, and exploit the structure of feasible sub-goal sets in contact-rich TAMP.

**The paper delivers:**
1. A new way to look at the problem (F = T ∩ A)
2. The first empirical characterization of F in a contact-rich domain
3. A finding about the structure of F (the discovery)
4. Evidence for whether that structure is exploitable by decomposed vs. joint learning

---

## The Problem

Every TAMP operator has continuous parameters — where to place, where to grasp, where to push. Finding feasible values for these parameters is THE bottleneck in TAMP. Task planning (symbolic level) is tractable. Motion planning (given start and goal) is mature. The hard part is the middle: **finding sub-goals that satisfy both the task constraint and the physical constraint simultaneously.**

This is the sub-goal level of TAMP — below task planning (which objects, which actions) and above motion planning (trajectories). It's where Chitnis et al. (2016), Diffusion-CCSP (Yang et al. 2023), and learned samplers operate. PIGINet (Yang et al. 2023) operates above this — it ranks plan skeletons but never finds continuous parameters.

Current approaches at this level:
- **Random sampling + rejection** (Van den Berg 2009, PDDLStream): Exponentially slow because the feasible region is sparse
- **Black-box learned samplers** (Chitnis et al. 2016, Ichter et al. 2018): Learn to imitate expert parameter choices, but no understanding of what makes parameters feasible or when the sampler will fail
- **Diffusion-based samplers** (Diffusion-CCSP, DiMSam): Generative models for constraint satisfaction, but no empirical study of the feasible set's structure

Nobody looks at the **structure** of the feasible parameter space itself.

---

## The Thesis

**The feasible sub-goal set in contact-rich TAMP has a decomposable structure — the intersection of a task constraint and a physical achievability constraint. Defining, measuring, and exploiting this structure enables better samplers and reveals why existing ones fail.**

In general form:

**F = T(task) ∩ A(physics)**

- **T** (task constraint): sub-goal parameters that achieve the desired task effect
- **A** (achievability constraint): sub-goal parameters the robot can physically realize
- **F = T ∩ A**: the feasible sub-goals — must satisfy BOTH

Neither T nor A is trivial in contact-rich domains. Both require forward simulation to evaluate because collisions between movable-static and movable-movable objects shape the outcomes. You can't geometrically compute either — you have to find them through physics. This coupling makes F sparse, multimodal, and hard to characterize analytically.

---

## Why This Matters

Sampling feasible sub-goals IS the TAMP problem. If you can sample well, planning is fast. If you can't, you're doing exponential search. This is the bottleneck that keeps TAMP from being practical for robots in unstructured environments — homes, warehouses, construction sites.

The F = T ∩ A framework matters because it turns sub-goal sampling from a black box into a diagnosable, improvable process:
- When a sampler fails, decompose: did it miss T or A?
- When problems are hard, measure: is F sparse because T is small, A is small, or the coupling is tight?
- When training data is insufficient, target: generate environments that stress the specific constraint the model struggles with

This is the difference between "retrain with more data and hope" and "diagnose, target, fix."

The framework applies broadly — any TAMP operator with continuous parameters has F = T ∩ A. NAMO (pushing), grasping in clutter, placement in tight spaces, assembly, buffer placement in rearrangement — all share this structure. NAMO is the testbed because F is exhaustively computable (600 primitives), giving us ground truth.

---

## NAMO: F = C ∩ R

### The decomposition

- **C(e,o)** — Clearance: object poses that merge two adjacent regions (open the passage). The task constraint.
- **R(o,q,q_r)** — Reachability: object poses physically achievable via pushing from the robot's current configuration, accounting for contact physics. The achievability constraint.
- **F = C(e,o) ∩ R(o,q,q_r)** — the feasible set.

Both C and R require simulation to evaluate — contact physics (movable-static and movable-movable collisions) during pushing determines where the object lands, which determines both whether the passage opens (C) and whether the push is achievable (R). Neither is geometrically computable.

### Novelty

| Concept | Prior art? |
|---------|------------|
| C(e,o) as an explicit set of region-merging poses | No — Stilman's "keyhole" is implicit, never a set |
| R(o,q,q_r) as a push-physics-aware set | No — prior work checks reachability per-sample, not as a set |
| F = C ∩ R as the central object to predict | No — nobody names, defines, or studies this intersection |
| Empirical characterization of F | No — not done in any prior NAMO or TAMP work |

### Prior NAMO work

- **Stilman & Kuffner (2005)**: Identifies blocking objects. Delegates placement to a motion planner. No explicit sets.
- **Van den Berg et al. (2009)**: Samples randomly, checks validity. No set characterization.
- **NAMO-LLM (Zhang 2025)**: Binary per-sample checks. Closest, but no explicit sets, no intersection, no analysis.

### Why NAMO is the right testbed

1. **F is exhaustively computable.** 600 primitives (60 edges × 10 depths) — run all of them, get exact F for every instance.
2. **F has non-trivial structure.** Contact physics couples C and R.
3. **Difficulty is measurable.** |F|/P directly measures sparsity.
4. **Multi-step extends naturally.** 1-push / 2-push / 3-push tests scaling.
5. **Data already exists.** Exhaustive search data from SAGE provides ground truth F for thousands of instances.

---

## The Discovery: Characterizing F

This is the core empirical contribution. Using exhaustive search data, analyze F for the first time.

### Questions to answer

**Is the bottleneck C or R?**
For every primitive (all 600 per instance), classify:
- Not reachable (robot can't reach contact point) → filtered by R before execution
- Reachable, push executed, object moved, but passage didn't open → passed R, failed C
- Reachable, push executed, passage opened → in F (passed both)

Compute the ratio across instances. Does R filter out 80% and C filters the rest? Or is C the dominant filter? **Does this ratio shift between easy and hard problems?**

**What shape is F?**
Map successful primitives in (edge, depth) space. Is F:
- A single connected cluster (one good push direction)?
- Multiple separated clusters (push left OR push right)?
- Scattered isolated points (no spatial structure)?

**Does F's shape predict difficulty better than F's size?**
Two problems with |F|/P = 5% — one has a single cluster, the other has scattered points. Which is harder to find? If shape matters more than size, samplers need mode coverage, not just accuracy.

**Is there a phase transition?**
As passage width narrows or clutter increases, does F shrink gradually or collapse suddenly? A phase transition would identify critical geometric thresholds.

**How coupled are C and R?**
Compute |C|, |R|, and |C ∩ R| independently. If C and R were independent, |C ∩ R| ≈ |C| × |R| / P. Measure the actual ratio. If |C ∩ R| is much smaller than expected → C and R are positively coupled (knowing something satisfies C tells you it's LESS likely to satisfy R). If larger → negatively coupled. This directly predicts whether Diffusion-CCSP (decomposed) will work.

### Why this analysis matters

These findings would be the **first empirical characterization of feasible sets in contact-rich TAMP.** The results directly inform:
- Whether decomposed learning (CCSP) can work (depends on coupling)
- What makes problems hard (sparsity? fragmentation? constraint ratio?)
- Where to focus model capacity (on whichever constraint is the bottleneck)
- How to generate hard training environments (target the features that make F sparse)

---

## Baselines

### The comparison ladder

| Method | Learned? | F-aware? | What it tests |
|---|---|---|---|
| Uniform / EPS | No | No | Lower bound |
| Geometric heuristic | No (hand-crafted) | No | Domain heuristic |
| CEM | No | No | Adaptive optimization vs. learning |
| Ichter CVAE | Yes | No | Black-box learned sampler |
| Diffusion-CCSP | Yes | Yes (decomposed) | Separate C and R models, composed |
| **SAGE** | Yes | Yes (joint) | Joint F prediction |

### Essential baselines

**1. EPS + Geometric heuristic** — already implemented. The uninformed baselines.

**2. CEM (Cross-Entropy Method)** — standard adaptive sampling (Rubinstein 1999). Maintains Gaussian over (edge_angle, push_depth), iteratively refits toward successful pushes. 64 samples × 5 iterations = 320 evaluations. Needs graded fitness to get signal when F is sparse. Tests: does learning help at all, or is adaptive optimization enough?

**3. Ichter CVAE** — Ichter, Harrison, Pavone. "Learning Sampling Distributions for Robot Motion Planning." ICRA 2018. CVAE conditioned on scene context, outputs SE(2) goal poses. Same data as SAGE, same primitive alignment. Single forward pass — fast inference. Known weakness: mode collapse on multimodal F. Tests: does F-awareness help beyond black-box learning?

**4. Diffusion-CCSP** — Yang, Mao, Du, Wu, Tenenbaum, Lozano-Pérez, Kaelbling. "Compositional Diffusion-Based Continuous Constraint Solvers." CoRL 2023. Train separate diffusion models for C (clearance) and R (reachability). Compose at inference by summing/averaging noise predictions during denoising. **The key baseline** — directly tests whether the F = C ∩ R decomposition is exploitable. Requires separate C and R training data.

**5. SAGE** — joint diffusion prediction of F. This work.

### The key showdown: CCSP vs SAGE

Both are F-aware. CCSP decomposes F into separate constraint models; SAGE predicts F jointly.
- **If CCSP wins**: the decomposition is valid, constraints are learnable separately, composition works. The framework is directly actionable as an architecture.
- **If SAGE wins**: the coupling between C and R matters — contact physics entangles them in ways separate models miss. Joint prediction is necessary, but the framework still provides the diagnostic.
- **Either outcome is a finding.**

### CCSP for NAMO: how it works

**Two separate diffusion models:**

Clearance model — learns "where should the object be to open the passage?"
- Conditioned on: scene geometry, passage/edge, robot goal
- Trained on: object poses that open the passage (satisfy C), regardless of reachability
- Training data: for each scene, record all push outcomes where passage opened. Or: hypothetically place object at grid poses, check via wavefront BFS.

Reachability model — learns "where can the robot actually push the object?"
- Conditioned on: object current pose, robot position, nearby obstacles
- Trained on: all landed poses from executed pushes (satisfy R), regardless of whether passage opened
- Training data: every push outcome from exhaustive search — most do NOT open the passage.

**Inference:**
```
q_target = random noise
for each denoising step:
    ε_C = clearance_model(scene, passage, goal, q_target, t)
    ε_R = reachability_model(obj_pose, robot_pos, scene, q_target, t)
    ε = (ε_C + ε_R) / 2
    q_target = denoise_step(q_target, ε, t)
# q_target should land in C ∩ R
```

Clearance model pulls toward passage-opening poses. Reachability model pulls toward achievable poses. Composition finds the intersection.

---

## Evaluation

### Primary metric: sampler efficiency as a function of |F|/P

- **X-axis**: |F|/P — sparsity of the feasible set (from exhaustive search)
- **Y-axis**: number of simulation checks to first valid solution

Curves per sampler. Random follows samples_needed ≈ P/|F|. Perfect oracle = 1. The improvement over random at each difficulty level is the measure.

This metric:
- Is grounded in the formulation (|F| comes from F = C ∩ R)
- Controls for problem difficulty
- Shows WHERE each sampler provides value (at what sparsity level)
- The F-aware sampler should dominate at small |F| (hard problems)

### Supporting metrics

- **Hit rate on F (top-K)**: fraction of top-K predictions in F. Pure sample quality.
- **Success@N**: fraction of problems solved within N simulation checks. Isolates sample quality from inference speed.
- **Success@T**: fraction solved within T seconds. The practical metric (includes inference time).

### Diagnostic: C vs R failure decomposition

For each failed prediction, classify:
- Misses C, hits R: achievable push, wrong direction → model hasn't learned task geometry
- Hits C, misses R: right direction, but can't push there → model hasn't learned reachability/physics
- Misses both: prediction is off entirely

This diagnostic is **unique to the F = T ∩ A framework** and unavailable to any black-box approach. It directly informs what to fix.

---

## Positioning in the Literature

### Three levels of TAMP

| Level | What's decided | Who does it |
|-------|---------------|-------------|
| **Task level** | Which objects, which actions, what order | PIGINet (Yang 2023), symbolic planners |
| **Sub-goal level** | Continuous parameters — WHERE to place/push/grasp | **This work**, Chitnis (2016), Ichter (2018), Diffusion-CCSP (2023) |
| **Motion level** | HOW to execute — trajectories | RRT, trajectory optimization (mature) |

### Key related work at the sub-goal level

**Chitnis et al. (2016)** — "Guided Search for Task and Motion Plans Using Learned Heuristics." ICRA. IRL-based samplers for continuous TAMP parameters. Black-box — no feasible set definition, no structural analysis. Shows you CAN learn sub-goal samplers. Important for positioning, not as a baseline.

**Ichter, Harrison, Pavone (2018)** — "Learning Sampling Distributions for Robot Motion Planning." ICRA. CVAE for non-uniform sampling. Conditioned on environment + start/goal. Black-box learned sampler. Baseline.

**Wang, Garrett, Kaelbling, Lozano-Pérez (2021)** — "Learning Compositional Models of Robot Skills for TAMP." IJRR. GP-based samplers with uncertainty, active learning, diversity. Most principled prior sampler work. Operates at same level. Not a baseline (scaling issues with our data volume) but important related work.

**Diffusion-CCSP (Yang et al. 2023)** — "Compositional Diffusion-Based Continuous Constraint Solvers." CoRL. Diffusion models per constraint type, composed at inference. Directly comparable — baseline.

**PIGINet (Yang et al. 2023)** — "Sequence-Based Plan Feasibility Prediction for TAMP." RSS. Plan skeleton ranking. Operates ABOVE our level — doesn't find continuous parameters. Related work, not comparable.

**Diffuser (Janner et al. 2022)** — Diffusion for full trajectory generation. Replaces the planner entirely. We guide search, not replace it. Different approach.

### NAMO literature

- **Stilman & Kuffner (2005)**: Original NAMO. No feasible set formulation.
- **Van den Berg et al. (2009)**: Probabilistically complete NAMO. Random sampling.
- **Scholz et al. (2016)**: Learned dynamics for NAMO via RL.
- **Yao et al. (2023), Yang et al. (2025)**: End-to-end RL for NAMO. Replace the planner.

### The gap

Everyone either learns black-box samplers (Chitnis, Ichter) or composes constraint models without studying the feasible set (CCSP). Nobody defines F, measures its structure, and uses that to diagnose and improve samplers.

---

## The Framework Applied to Other Tasks

F = T ∩ A applies to any contact-rich TAMP operator. NAMO is the primary testbed, but the structure is general.

**Grasping in clutter**: T = stable grasp, A = collision-free approach. F = grasps that are stable AND approachable. Sparse, multimodal, clutter-dependent.

**Placement in tight spaces**: T = object at target, A = collision-free insertion. F = placements that fit AND can be inserted. Contact-dependent.

**Single-object pushing to goal**: T = object at target pose, A = achievable by pushing. Simplest contact-rich task. Same structure as NAMO region opening.

**Buffer placement in rearrangement**: T = doesn't block future actions, A = executable placement. Same recursive structure as multi-push NAMO.

---

## Multi-Push: Recursive F

For multi-push, the feasible set at each level depends on subsequent levels:

- **F_k** (terminal): standard C ∩ R
- **F₁** (first push): { q₁' : F₂(state_after(q₁')) ≠ ∅ } ∩ R₁

This recursion is the mechanical consequence of multi-step planning. It identifies:
- Why multi-push is O(Pᵏ)
- Why learned models struggle at 2-push (trained on single-level F, asked to predict recursive F₁)
- What a world model would do (approximate state transitions to cheaply evaluate the recursion)

The recursion itself isn't novel — it's obvious. But it connects to the evaluation: test on 1-push / 2-push / 3-push to see how the framework scales with depth.

---

## Current SAGE Paper: What's There and What's Missing

### What's there (strengths)
- F = C ∩ R formulation (novel, verified against literature)
- Primitive-based BFS search structure
- Hybrid approach preserving completeness
- Meaningful speedup (3.7× on hard 1-push, 4.5× on hard 2-push)
- Clean problem decomposition and data pipeline

### What's missing (weaknesses)

1. **Framed as a NAMO/diffusion paper, not a framework paper.** The formulation is buried as setup. The diffusion model is presented as the contribution. Inverts the actual novelty.

2. **No analysis of F.** Never visualized, measured, or studied. Properties asserted but not demonstrated.

3. **No sub-goal sampling baselines.** Compared only against exhaustive search and geometric heuristic. No learned sampler baselines (CVAE, CCSP, CEM).

4. **Missing TAMP literature positioning.** Doesn't discuss Chitnis, PIGINet, Khodeir, Wang, Diffusion-CCSP, or the learned TAMP heuristic literature.

5. **The regression ablation tests the wrong alternative.** Single-point regression (0.54%) used to argue generative > discriminative. Doesn't test discriminative scoring over the fixed 600-candidate set.

6. **2-push limitations unresolved.** Diffusion-only degrades to 81.1%. Multi-Horizon makes it worse (66.8%).

---

## Future Directions (Beyond This Paper)

### World model for multi-step
For k-push problems, a learned dynamics model enables cheap evaluation of the recursive F structure without simulation. This bridges world models (TD-MPC, Dreamer) with TAMP heuristics.

### Discriminative scoring
When candidates are enumerable (600 primitives), a classifier scoring all candidates may beat generative sampling. An ablation-level question.

### Multiple task domains
Demonstrating F = T ∩ A on grasping, placement, or tabletop pushing would prove generality.

### Prescriptive training improvement
Use the C vs R diagnostic to identify failure modes → generate targeted training data → retrain → show improvement. The full prescriptive loop.

### Real robot deployment
Demonstrate that the speedup enables real-time replanning where exhaustive search cannot.
