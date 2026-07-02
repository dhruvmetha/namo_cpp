# Research Notes: Characterizing F in Primitive Action Space

## The Feasible Set as a Success Region

Given a scene and robot position, the feasible set F is the set of push primitives that successfully open a blocked passage. Each primitive is parameterized by a contact point on the object boundary (60 points, 15 per face) and a push duration (10 levels), yielding 600 candidates. The robot's reachability — computed exactly via wavefront BFS — filters this to a reachable subset R. Of those, the subset that actually opens the passage is F = C ∩ R, where C denotes clearance. F is the success set of the push controller conditioned on the scene and robot position: any primitive in F transitions the world to a state where the passage is open.

This is analogous to a region of attraction (RoA) for a controller, but defined in action parameter space rather than state space, and conditioned on scene context. Classical RoA analysis (Lyapunov, SOS) applies to smooth dynamics and computes these regions analytically. Contact-rich pushing — where objects slide along walls, collide with other movables, and land in poses shaped by friction and geometry — resists analytical characterization. The success boundary in action space is discontinuous and scene-dependent. We propose to characterize F empirically: exhaustively evaluate all reachable primitives, label each as success or failure, and study the resulting set.

## Discrete Characterization (600 Primitives)

In the discrete setting, F is a binary labeling over a 60 x 10 grid (contact point x push depth). For each (scene, robot position, blocking object) instance, we can visualize F as a heatmap: green for feasible, gray for reachable but failed, black for unreachable. This visualization directly reveals the structure of the success set.

The central hypothesis is that difficulty — measured by the number of simulation calls an exhaustive search requires — is governed by |F|/|R|, the fraction of reachable primitives that succeed. This ratio captures the sparsity of feasible solutions within the searchable space. Two instances with the same |F| can have very different difficulty if their |R| differs: 5 solutions among 50 reachable primitives (10%) is easy; 5 solutions among 300 (1.7%) is hard.

But |F|/|R| alone does not capture learnability. The spatial structure of F in the (contact point, depth) grid matters: are the feasible primitives clustered in a contiguous band (push roughly in this direction, at roughly this distance) or scattered as isolated points? Clustered F means a model that predicts the right direction will hit multiple solutions. Scattered F means the model needs precise, combinatorial prediction. We hypothesize that hard problems are characterized not just by sparse F but by fragmented F — multiple disconnected feasible regions, each small, requiring the model to represent multimodal predictions.

Additional structural questions the discrete analysis can answer: Is the bottleneck direction (only certain contact points work) or depth (only specific push distances work)? Do wall collisions create feasibility by redirecting objects into valid poses, or destroy it? Does F shift smoothly with robot position, or change abruptly? Each finding directly informs what the learned model must capture.

## Transition to Continuous Action Space

The 600 primitives are a discretization of a continuous 2D action space: position along the object boundary (push direction) and push duration (push distance). At 60 x 10 resolution, F appears as scattered binary labels. But the underlying truth is a continuous function — P(success | direction, distance, scene) — that varies smoothly between adjacent primitives. Nearby contact points produce similar push outcomes. Adjacent depths produce incrementally different displacements.

Increasing the resolution (e.g., 100 x 50 = 5,000 primitives) does not change the problem — it reveals the continuous structure that the coarse discretization aliases. What appeared as 5 scattered feasible points at 600 resolution becomes visible as 2 connected blobs at higher resolution. The ratio |F|/|R| remains roughly constant; the structure becomes clearer.

In the fully continuous limit, F becomes a region in (direction, distance) space — potentially visualizable as a polar plot with the object at the center, where the angular axis represents push direction around the perimeter and the radial axis represents push distance. Feasible regions appear as wedges, arcs, or islands. For an easy instance, F might be a broad wedge ("push roughly toward open space"). For a hard instance, F might be two small islands ("slide along this specific wall into this pocket, OR push into that narrow gap from the other side").

This continuous view connects to what a learned model must ultimately represent. A classifier over 600 primitives treats each primitive independently, ignoring the correlation between neighbors. A model that instead predicts P(success | direction, distance, scene) as a continuous function naturally captures this structure — and can be evaluated at any resolution without retraining. The discrete analysis at 600 points is the empirical foundation; the continuous density is the generalization target. If the discrete F has clear spatial structure (clusters, smooth boundaries), a model trained on 600 labels will interpolate correctly to finer resolutions. If F is genuinely discontinuous at fine scales (e.g., a thin wall creates a sharp feasibility boundary), higher resolution data is needed.

## Learning to Predict F

The learned model's task is to predict F for a new (scene, robot position) without evaluating all primitives in simulation. Given thousands of instances with exhaustively computed F, the model learns the mapping from scene context to the success set. In the discrete setting, this is multi-label classification: given scene masks, output a score per primitive. In the continuous setting, this is a conditional density: given scene masks and a candidate (direction, distance), output P(success).

Either way, the model is implicitly learning a scene-conditioned region of attraction for the push controller. To predict whether a primitive succeeds, it must internally represent something about contact physics — where the object will land given the push, and whether that landing pose opens the passage. A classifier that predicts F well has learned a compressed world model within its weights: not explicitly predicting trajectories, but capturing enough about push outcomes to classify their success.

This framing connects to both control theory (RoA characterization) and learned planning (scene-conditioned samplers). The empirical F characterization bridges them: it provides the ground truth that analytical methods cannot compute for contact-rich systems, and the structural understanding that black-box learned samplers lack. The discrete analysis comes first — measure F, understand its structure, confirm the hypotheses about sparsity, clustering, and difficulty. The continuous generalization and the learned predictor follow from what the analysis reveals.

## Research Questions: What We Need to Understand About F

Before building any solution, the following questions must be answered empirically. These are the questions that drive the data collection and analysis — the answers determine everything downstream.

### 1. What does F look like?
Visualize F in (contact point, depth) space for many instances. Is it a contiguous wedge, a scattered set of points, multiple disconnected islands, a thin arc along one face? The shape of F is the most basic unknown. Nobody has seen it.

### 2. Why does it look that way?
Connect the shape of F to the physics. Suh (MIT, 2025) showed analytically that contact feasible sets are asymmetric due to unilateral contact — you can push along a wall but not through it. Do our empirical F plots confirm this? Are the wedges aligned with wall directions? Do contact interactions (object sliding along wall into a pocket) create feasible regions that wouldn't exist in free space?

### 3. What makes F change between scenes?
Same blocking object, different surrounding geometry — how does F change? If F is primarily determined by the passage width and wall angles, it should be predictable from local geometry. If F depends on distant objects (another movable that the pushed object might hit), the model needs global scene understanding. How local vs global is F?

### 4. What makes F shrink on hard problems?
Easy problems have large F (many primitives work). Hard problems have small F (few primitives work). But why does F shrink? Is it because the passage is narrow (tight clearance constraint)? Because the robot is in a bad position (tight reachability)? Because the object is wedged between walls (contact interactions eliminate most push directions)? The answer tells you what makes problems hard and what a model must be sensitive to.

### 5. What geometric or physical features predict F's structure?
Can you predict |F|, the number of clusters in F, or the dominant push direction from measurable scene features — passage width, object size relative to passage, distance from object to walls, robot position relative to object? If yes, you have an interpretable difficulty metric. If no, the relationship between scene geometry and F is too complex for simple features and requires a learned model.

### 6. How does F chain for multi-step?
For 2-push problems, define F₁' = {push1 : F₂(state_after(push1)) ≠ ∅}. This is the set of first pushes that enable a feasible second push. How does F₁' compare to the full reachable set? Is it a small fraction (tight bottleneck — most first pushes are dead ends) or a large fraction (generous — many first pushes enable second-push solutions)? Does the structure of F₁' relate to the structure of F₂ in predictable ways?

This is the same question as 1-push, applied recursively. The "success" criterion changes from "opens the passage" to "enables a state where opening is possible." The analysis is the same heatmap visualization — but now coloring push-1 primitives by "does a feasible push-2 exist from the resulting state."

---

These six questions define the empirical research program. The answers come from data — exhaustive evaluation of all reachable primitives on hundreds of instances, with every outcome logged. No models needed. Just run, record, plot, and look.

Everything else — classifiers, diffusion models, world models, baselines — follows from what we find.

---

## Hypotheses

### Hypothesis 1: Difficulty is |F|/|R|, not |F|

**Statement:** The difficulty of a region opening problem — measured by the number of simulation calls an exhaustive search requires — is primarily governed by |F|/|R| (the sparsity of feasible primitives within the reachable set), not |F| alone.

**Why it matters:** If true, difficulty is about density of solutions in the search space, not the absolute number of solutions. Two instances with |F|=5 can have very different difficulty depending on |R|.

**What we discussed:** This hypothesis opens up the entire characterization program. |F|/|R| tells you how hard the problem is for random search, but not how hard it is for a learned model. For learnability, the spatial structure of F matters — clustering, fragmentation, smoothness in (contact point, depth) space. This connects to:

- **Discrete structure:** F as a 60x10 heatmap. Clustered F (contiguous band) is easy to learn — predict the right direction and you hit multiple solutions. Scattered F (isolated points) requires precise combinatorial prediction.
- **Continuous structure:** The 600 primitives sample a continuous 2D action space (push direction, push distance). F at fine resolution reveals connected blobs, wedges, arcs — shaped by unilateral contact physics (Suh, MIT 2025). The discrete heatmap is an aliased view of this continuous landscape.
- **Connection to RoA:** F is an empirical region of attraction of the push controller in action parameter space, conditioned on scene and robot position. Analytical RoA methods (Lyapunov, LQR Trees, Suh's Contact Trust Region) compute this for smooth or few-contact systems. Contact-rich cluttered scenes require empirical characterization — which is what we provide.
- **Two solution paths:** An implicit predictor (classifier: scene → F directly) learns the success set as a black box. An explicit predictor (world model: scene + action → outcome, then check clearance) learns the dynamics and derives F. Implicit is sufficient for 1-push. Explicit is needed for multi-push chaining because you must predict F at future states.

**How to test:**
1. Run exhaustive search with no pruning — all reachable primitives evaluated, every outcome logged as pass/fail
2. Compute |R|, |F|, |F|/|R| per instance
3. Plot |F|/|R| vs difficulty (EPS median pushes). Measure correlation.
4. Visualize F in (contact point, depth) heatmaps for easy/medium/hard instances side by side
5. Compare against |F| alone as a difficulty predictor

**Action items:**
- [ ] Modify data collection pipeline: disable pruning, log per-primitive outcomes (edge_idx, depth, success/fail, collision type)
- [ ] Run exhaustive collection on test set (all reachable primitives per instance)
- [ ] Build analysis script: compute |R|, |F|, |F|/|R| per instance from logged data
- [ ] Plot |F|/|R| vs difficulty stratification
- [ ] Generate (contact point, depth) heatmaps for representative easy/medium/hard instances
- [ ] Read Suh thesis Chapters 4.2-4.3 (Contact Trust Region, motion set M^u) and Chapter 5.1-5.3 (global planning via RRT stitching)

---

### Hypothesis 2: F is spatially clustered, not scattered

**Statement:** Feasible primitives in the (contact point, depth) grid are spatially correlated — they form contiguous clusters (push roughly in this direction, at roughly this distance) rather than being randomly scattered.

**Why it matters:** If F is clustered, a model that predicts the right push direction wins — it will hit multiple feasible primitives. If F is scattered, direction alone isn't enough and the model needs precise per-primitive prediction. Clustering also means the discrete-to-continuous transition is smooth: clusters at 600 resolution become connected blobs at higher resolution.

**How to test:**
- Count connected components in the (contact point, depth) heatmap per instance
- Measure cluster sizes (number of feasible primitives per cluster)
- Compare clustering between easy/medium/hard instances
- Check if clusters align with box faces (all feasible primitives on one face = "push this direction")

---

### Hypothesis 3: Hard problems have fragmented F

**Statement:** Easy problems have one large cluster in F (push anywhere to the left). Hard problems have multiple small, disconnected clusters (push into this specific corner OR slide along that specific wall). Fragmentation, not just sparsity, is what makes problems hard for learned models.

**Why it matters:** If true, multimodality is the core challenge for hard instances. A unimodal predictor (regression, single-mode classifier) will average between clusters and miss both. A multimodal predictor (diffusion, mixture model) is necessary specifically for hard problems. This would retroactively justify the generative model for hard instances — but also show it's unnecessary for easy/medium ones.

**How to test:**
- Measure number of connected components in F vs difficulty
- Measure inter-cluster distance vs difficulty
- Test whether a unimodal predictor degrades specifically on multi-cluster instances
- Check if SAGE's diffusion predictions align with multiple clusters or collapse to one

---

### Hypothesis 4: Depth matters more than direction for hard problems

**Statement:** On hard problems, the feasible primitives share contact points (correct direction) but only specific depths work — too shallow doesn't clear the passage, too deep hits a wall. F is narrow in the depth dimension but spans multiple contact points.

**Why it matters:** If true, the model needs to learn precise push distance, not just push direction. This shifts the learning challenge from "which face of the object to push" to "how hard to push." It would also explain why SAGE's diffusion model struggles on hard cases — mask-based prediction is good at direction but imprecise on distance.

**How to test:**
- For each cluster in F, measure its extent in contact-point dimension vs depth dimension
- Compare aspect ratio of clusters between easy/medium/hard
- Check if hard instances have F that spans multiple edges but only 1-2 depths

---

### Hypothesis 5: Wall collisions create F, not destroy it

**Statement:** The feasible primitives in hard instances are disproportionately those where the object slides along a wall or bounces off another movable during the push. Contact interactions redirect the object into poses that open the passage — poses that a free-space push would never reach. Walls create feasibility rather than destroying it.

**Why it matters:** If true, the model must learn wall interaction physics to predict F correctly. A model that avoids collisions (like a geometric heuristic that prefers "clean" pushes) will systematically miss the feasible set on hard problems. This also means the contact-rich setting is fundamentally different from pick-and-place — contact doesn't just constrain, it enables.

**How to test:**
- From logged data, check collision flags (wall_collision, movable_collision) for feasible vs infeasible primitives
- Compute: what fraction of F involves wall collisions? Does this fraction increase with difficulty?
- Compare landing poses of feasible primitives: are they poses that require wall contact to reach?
- Visualize push trajectories for feasible primitives on hard instances — do they show sliding/bouncing?

---

## Empirical Results (Full Test Set: 1,767 environments, 3,622 instances with F > 0)

*293 instances with F=0 (multi-push problems) excluded from analysis.*

### Dataset Overview

| Category | Count | % | Avg |F| | Avg |R| | Avg |F|/|R| |
|---|---|---|---|---|---|
| Very Hard (<5%) | 39 | 1.1% | 3.4 | 115 | 3.0% |
| Hard (5-15%) | 165 | 4.6% | 11.5 | 112 | 10.2% |
| Medium (15-40%) | 558 | 15.4% | 40.2 | 138 | 28.7% |
| Easy (40-70%) | 1,186 | 32.7% | 105.7 | 185 | 56.4% |
| Very Easy (>70%) | 1,674 | 46.2% | 160.9 | 188 | 86.8% |

### Bottleneck Hierarchy: Face → Contact Point → Depth

Each push primitive is parameterized by three levels: which of 4 object faces to push on, which of 15 contact points on that face, and which of 10 push depths. For each difficulty level, we measure what fraction of each level is "active" (contains at least one feasible primitive).

| Difficulty | n | Faces active (of 4) | Contact points active (of 15, within active face) | Depths active (of 10, within active contact point) |
|---|---|---|---|---|
| Very Hard | 39 | 26% → **1 face** | 9% → **1.4 points** | 26% → **2.4 depths** |
| Hard | 165 | 29% → 1.2 faces | 18% → 2.6 points | 44% → 3.8 depths |
| Medium | 558 | 35% → 1.5 faces | 35% → 5.1 points | 59% → 5.6 depths |
| Easy | 1,186 | 45% → 1.8 faces | 54% → 7.8 points | 75% → 7.5 depths |
| Very Easy | 1,674 | 44% → 1.9 faces | 69% → 10.3 points | 89% → 8.9 depths |

**Contact point has the steepest drop**: 69% → 9% (8× narrowing). Face drops 2×. Depth drops 3.5×. Contact point precision is what separates easy from hard.

### Face Activity

| Difficulty | n | 1 face only | 2 faces | 3 faces | 4 faces |
|---|---|---|---|---|---|
| Very Hard | 39 | 97% | 3% | 0% | 0% |
| Hard | 165 | 84% | 16% | 0% | 0% |
| Medium | 558 | 62% | 35% | 3% | 0% |
| Easy | 1,186 | 37% | 47% | 16% | 1% |
| Very Easy | 1,674 | 40% | 45% | 14% | 1% |

### Multimodality

| Difficulty | n | Unimodal (1 face, 1 band) | Within-face multi | Cross-face multi | Both |
|---|---|---|---|---|---|
| Very Hard | 39 | **95%** | 3% | 3% | 0% |
| Hard | 165 | **72%** | 12% | 14% | 2% |
| Medium | 558 | 54% | 8% | 32% | 6% |
| Easy | 1,186 | 34% | 3% | **50%** | 13% |
| Very Easy | 1,674 | 40% | 0% | **53%** | 7% |

Hard problems are overwhelmingly unimodal. Cross-face multimodality (2+ faces work, each with one contiguous band) dominates easy problems.

### Depth Analysis

#### Success rate by depth level

What fraction of reachable primitives succeed at each push depth (0=shallowest, 9=deepest)?

| Difficulty | d=0 | d=1 | d=2 | d=3 | d=4 | d=5 | d=6 | d=7 | d=8 | d=9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Very Hard | 0.0% | 0.1% | 1.0% | 1.6% | 3.8% | 3.4% | 5.6% | 5.5% | 9.9% | 13.0% |
| Hard | 1.0% | 2.3% | 5.6% | 12.1% | 17.8% | 17.9% | 18.6% | 18.6% | 17.7% | 18.1% |
| Medium | 6.0% | 12.5% | 24.2% | 34.7% | 42.1% | 45.5% | 44.5% | 42.8% | 41.9% | 40.7% |
| Easy | 22.2% | 37.1% | 54.1% | 65.2% | 72.2% | 75.0% | 73.3% | 70.0% | 66.6% | 63.3% |
| Very Easy | 60.4% | 78.0% | 87.2% | 91.9% | 94.2% | 95.1% | 93.7% | 90.7% | 87.0% | 82.3% |

**Key pattern**: Success rate increases with depth, peaks around d=4-5, then slightly decreases at maximum depths. Shallow pushes (d=0,1) almost never work on hard problems — the object doesn't move far enough. On very hard problems, only the deepest pushes (d=7-9) have any chance of success.

#### Feasible depth window

For each active contact point, where does the feasible depth window sit?

| Difficulty | n contact pts | Avg min depth | Avg max depth | Avg span | Contiguous |
|---|---|---|---|---|---|
| Very Hard | 55 | **5.8** | 7.2 | 2.4/10 | 96% |
| Hard | 501 | **3.8** | 6.7 | 3.8/10 | 98% |
| Medium | 4,012 | 2.6 | 7.2 | 5.6/10 | 99% |
| Easy | 16,634 | 1.4 | 8.0 | 7.5/10 | 99% |
| Very Easy | 30,369 | 0.5 | 8.4 | 8.9/10 | 100% |

**Hard problems require deep pushes**: the feasible depth window starts at d=5.8 on average for very hard — shallow pushes don't move the object far enough. Easy problems work from d=0.5 — even minimal pushes succeed.

**Depth windows are contiguous**: 96-100% of active contact points have gap-free depth ranges. The feasible depths form a continuous band [min_d, max_d], not scattered points.

#### Depth threshold: minimum depth needed

What is the minimum depth that works, per active contact point?

| Difficulty | d=0 | d=1 | d=2 | d=3 | d=4 | d=5 | d=6 | d=7 | d=8 | d=9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Very Hard | 0% | 2% | 9% | 11% | 15% | 6% | 16% | 7% | 18% | 16% |
| Hard | 7% | 8% | 15% | 19% | 16% | 13% | 9% | 5% | 5% | 3% |
| Medium | 19% | 18% | 18% | 15% | 11% | 8% | 5% | 3% | 2% | 1% |
| Easy | 39% | 25% | 16% | 9% | 5% | 3% | 2% | 1% | 1% | 0% |
| Very Easy | 69% | 20% | 7% | 3% | 1% | 1% | 0% | 0% | 0% | 0% |

On very easy problems, 69% of contact points work starting from depth 0. On very hard, the minimum feasible depth is spread across d=2 to d=9 — no shallow push ever works. The object must be pushed deep enough to clear the passage, and on hard problems "deep enough" is situation-specific.

### Wall Collision Role

| Difficulty | Wall % in success | Wall % in failure |
|---|---|---|
| Very Hard | **75.6%** | 67.5% |
| Hard | 62.1% | 62.8% |
| Medium | 57.8% | 56.1% |
| Easy | 45.7% | 49.9% |
| Very Easy | 23.1% | 31.2% |

On very hard problems, 76% of successful pushes involve wall collisions — walls redirect the object into valid poses that free-space pushes cannot reach. On very easy problems, 77% of successes are clean pushes.

### Contact Point Contiguity

| Difficulty | Avg contact pts per active face | Gap fraction |
|---|---|---|
| Very Hard | 1.4 / 15 | 2.0% |
| Hard | 2.6 / 15 | 8.2% |
| Medium | 5.1 / 15 | 4.7% |
| Easy | 7.8 / 15 | 4.1% |
| Very Easy | 10.3 / 15 | 1.8% |

Feasible contact points form contiguous bands on each face (gap fraction 2-8%). Nearby contact points produce similar outcomes.

---

## 1-Push vs N-Push: Two Fundamentally Different Problems

### The 1-Push Problem (Single-State Dynamics)

Given a scene (walls, objects, robot position) and a blocking object, find a push primitive that opens the passage.

The question for each primitive is: **"does this push open the passage?"**

- The answer is immediate and directly observable — execute the push, check if the passage is open.
- The initial scene is fixed. You always evaluate pushes from the same starting state.
- F can be characterized by exhaustive evaluation: try every reachable primitive, record pass/fail.
- There is no chain of decisions — each push is evaluated independently.

But each push evaluation is NOT static. The push itself involves dynamics: the object moves through the scene, interacts with walls (sliding, bouncing), possibly collides with other movables, and lands in a pose determined by contact physics. The 1-push characterization showed this directly — on hard problems, 81% of successful pushes involve wall collisions. The dynamics during the push are what create feasibility. F is not determined by geometry alone; it is determined by geometry + the physics of what happens when you push.

F₁ = {push ∈ R : push opens the passage}

The "single-state" qualifier means: you simulate dynamics from one fixed initial state. There is no need to reason about a sequence of state changes. But the dynamics within that single push are essential — they determine which primitives succeed.

### The N-Push Problem (Multi-State Dynamics)

The same blocking object, but F₁ is empty — no single push opens the passage. The object must be pushed multiple times: push-1 repositions it, push-2 repositions it further (or finishes), and so on.

The question for push-1 is: **"does this push lead to a state where the problem is solvable in fewer pushes?"**

Both 1-push and n-push involve contact dynamics during each push. What changes in n-push is the coupling between pushes:

**1. Evaluation becomes recursive.**
In 1-push, you check one thing: did the passage open? In n-push, you check: does a (n-1)-push solution exist from the resulting state? That check itself requires solving a smaller instance of the same problem. The success criterion for push-k is defined in terms of push-(k+1).

F₁' = {push1 ∈ R : F₂(state_after(push1)) ≠ ∅}

Push-1 is feasible not because of what it does, but because of what it enables.

**2. The state transition between pushes becomes the bottleneck.**
In 1-push, the dynamics matter within the push (wall collisions, friction, object interactions determine the landing pose). But you only reason about one state → one push → one outcome. In n-push, you must reason about a chain: state₀ → push₁ → state₁ → push₂ → state₂ → ... Each state is the result of the previous push's dynamics. The dynamics within each push still matter, but now the dynamics BETWEEN pushes (how push-1's outcome shapes push-2's feasible set) become the central challenge.

**3. The scene changes between decisions.**
Every push creates a new scene. Push-2's feasible set F₂ lives in a scene that didn't exist before push-1 executed. Different push-1 choices produce different scenes, each with a different F₂. You can't precompute F₂ without committing to a push-1. In 1-push, you always evaluate from the same fixed initial state.

### What This Means

Both problems involve dynamics. The difference is in how many state transitions you must reason about:

- **1-push**: one initial state → dynamics during push → outcome. The initial state is fixed. You evaluate each push independently. A model that maps (scene, primitive) → feasible/infeasible is sufficient, but it must implicitly capture the contact physics that determine outcomes (wall interactions, friction, object collisions).

- **N-push**: a sequence of states connected by push dynamics. Each push reshapes the scene for the next. A model must either (a) predict how each push changes the scene (world model) so it can evaluate future feasibility cheaply, or (b) learn to evaluate pushes by their downstream value without explicit state prediction (policy/Q-function). Either way, the model must reason about dynamics across multiple state transitions, not just within a single push.

### The Chaining Structure

The n-push problem has recursive structure:

- F_n (depth n): pushes that open the passage directly (same as 1-push F)
- F_{n-1}': pushes that produce a state where F_n is non-empty
- F_{n-2}': pushes that produce a state where F_{n-1}' is non-empty
- ...
- F_1': pushes that produce a state where F_2' is non-empty

Each level's feasible set is defined by the next level's. This is funnel composition: the feasibility "flows backward" from the final success criterion (passage open) through the chain of state transitions.

The structure parallels LQR Trees (Tedrake) and Contact Trust Regions (Suh), but in action parameter space rather than state space, and for contact-rich dynamics that resist analytical characterization.

### Open Questions for Multi-Push

The 1-push characterization revealed that F is contiguous, directional, and wall-dependent. For n-push, the key unknowns are:

1. **Does F₁' have the same structure as F₁?** Is it contiguous bands on specific faces, or does the recursive criterion scatter it?
2. **Does multimodality emerge?** In 1-push, hard problems are unimodal (1 face, 1-2 contact points). In 2-push, push-1 might have multiple viable intermediate poses — each corresponding to a different F₂ — creating genuine multimodality.
3. **How sensitive is the chain to push-1 precision?** If F₂ is contiguous and the object's landing pose varies smoothly with push-1 parameters, small errors in push-1 still produce states where F₂ is non-empty (robust chaining). If F₂ is fragile, push-1 must be precise (brittle chaining).
4. **Does |F₁'|/|R| predict multi-push difficulty the same way |F₁|/|R| predicts 1-push difficulty?**

These questions require 2-push exhaustive data to answer empirically.

---

## Scaling Beyond NAMO: When Reachability Is Not Free

### The NAMO Advantage

In NAMO, the feasible set decomposes as F = C ∩ R, where:
- **R** (reachability) is computed exactly and for free by wavefront BFS. Given the robot's position and the occupancy grid, BFS floods the grid and determines which contact points on the object the robot can physically reach. This is O(grid_size), takes milliseconds, and is exact.
- **C** (clearance/task success) is the learning target. Given that the robot can execute a push, does it open the passage? This depends on contact physics — wall collisions, friction, object interactions — and is what the classifier learns.

The decomposition is a gift: the classifier only needs to learn C, not F. Reachability is handled by an exact, free oracle. The learning problem is strictly easier because of this separation.

### When Reachability Is Unknown

In a general manipulation problem — a robot arm pushing objects on a table, a mobile manipulator in a warehouse, a hand manipulating a tool — R is not free:
- **Robot arm**: reachability depends on inverse kinematics, joint limits, collision checking with the environment. Computing whether the arm can reach a specific grasp or contact point requires solving IK and checking for collisions — expensive and approximate.
- **Mobile manipulator**: reachability depends on both base placement and arm configuration. The joint space is high-dimensional.
- **Dexterous hand**: reachability is contact-point-specific and depends on finger configurations. Even harder to compute.

In these settings, the model must predict both "can the robot execute this action?" (R) and "if executed, does it achieve the goal?" (C).

### Two Approaches

**Option 1: Learn F directly.** One model predicts feasibility end-to-end: (scene, action) → success/fail. This conflates reachability and task success. Failure could mean "the robot can't reach" or "the push doesn't work" — the model doesn't distinguish. Harder to learn, harder to debug, harder to transfer.

**Option 2: Preserve the decomposition.** Learn R and C as separate models:
- A **reachability model** predicts: "can the robot execute this action given its current configuration and the scene?" This is a kinematic/geometric question. It depends on the robot body and obstacles, not on the task.
- A **success model** predicts: "if the robot executes this action, does it achieve the task goal?" This is a dynamics/physics question. It depends on contact interactions and the task criterion, not on the robot body.
- F = C ∩ R, with both sides learned.

Option 2 is better because the two components generalize independently:
- Change the robot (different arm, different base) → retrain R, keep C.
- Change the task (open passage vs clear table vs sort objects) → retrain C, keep R.
- Change the environment (different scenes) → both may need updating, but the separation still simplifies each learning problem.

### What Transfers to Any System

The specific findings from NAMO (face → contact point → depth hierarchy, wall collisions creating feasibility, contiguous bands) are specific to box pushing in mazes. They do not directly apply to other domains.

What transfers is the **methodology and the decomposition**:

1. **F = C ∩ R decomposition**: separate what the robot can do from what achieves the goal. Even when R is learned rather than exact, this separation reduces the complexity of each learning problem.

2. **Exhaustive characterization methodology**: evaluate all actions in simulation, label outcomes, study the structure of the feasible set. This applies to any manipulation primitive — grasps, placements, pivots, slides. The specific structure of F will differ, but the approach of measuring it empirically before building models is universal.

3. **Structure-informed model design**: the characterization reveals what the model must capture. In NAMO, it revealed that contact point specificity and wall interactions matter most. In another domain, characterization might reveal different bottlenecks — but the principle of "measure first, then design" transfers.

4. **Classifier as the base case for chaining**: a model that predicts "which actions work at this state" can be applied iteratively for multi-step problems, regardless of the domain. The chaining structure (F₁' defined by F₂) is domain-independent.

The claim is not "push feasibility in NAMO generalizes to other problems." The claim is: "the approach of decomposing, characterizing, and exploiting the structure of feasible sets applies to any manipulation system with a finite set of parameterized actions."

---

## Research Methodology: Classifier-First, World Model If Needed

### The Principle

Do not assume a world model is necessary. Build the simplest model (classifier), test it, and let the failures — if any — reveal whether a dynamics model is needed and what it must capture.

### The Sequence

1. **Collect exhaustive 1-push data** — full F characterization at initial scenes.
2. **Geometric feature analysis** — can simple measurable features (wall distances, passage width, open space direction) predict which face and contact point region works? This determines what the classifier needs to learn beyond obvious geometry.
3. **Train 1-push classifier** — scene masks → per-primitive feasibility scores. Measure top-k accuracy by difficulty.
4. **Evaluate failure modes:**
   - Fails on hard instances → masks don't capture fine geometric details. Need richer representation.
   - Works on 1-push across all difficulties → classifier is sufficient for single-step. Proceed to multi-push.
5. **Add 2-push intermediate state data** — from existing 2-push solutions, extract (intermediate_scene, push_2, outcome) pairs. Retrain classifier on 1-push + 2-push data combined.
6. **Evaluate on multi-push:**
   - Works at intermediate states → classifier generalizes to dynamics-generated scenes. No world model needed. The classifier applied iteratively handles n-push.
   - Fails at intermediate states → the model can't generalize to scenes it hasn't seen. Intermediate states are out-of-distribution. This is the evidence that a dynamics model is needed — to predict what intermediate states look like.

### What Each Failure Reveals

| Failure mode | What it means | What to build next |
|---|---|---|
| Can't predict active face | Scene representation is too coarse | Better features or higher-resolution masks |
| Gets face right, misses contact points | Fine geometric reasoning isn't learnable from masks | Richer local geometry features or explicit geometric computation |
| Works on 1-push, fails at intermediate states | Distribution shift — model hasn't seen dynamics-generated scenes | World model to predict intermediate states, or more diverse training data |
| Works on easy multi-push, fails on hard multi-push | Chaining precision exceeds what scene-level prediction can achieve | World model for precise state prediction in tight-tolerance chains |

The world model is not a design decision. It is a conclusion forced by experimental evidence — or not. If the classifier handles everything, the world model is unnecessary complexity. If it fails, the failure mode tells you exactly what the world model must capture.

---

## Summary: Research Sequence

1. **Collect data**: exhaustive 1-push search, no pruning, all outcomes logged
2. **Characterize F**: compute |R|, |F|, |F|/|R|, visualize heatmaps, analyze structure
3. **Geometric feature analysis**: what scene features predict F without learning?
4. **Train 1-push classifier**: scene → per-primitive scores, evaluate by difficulty
5. **Evaluate failure modes**: where does the classifier fail and why?
6. **Add multi-push data**: retrain on 1-push + 2-push intermediate states
7. **Evaluate on multi-push**: does the classifier generalize to chained problems?
8. **If needed**: build world model, informed by exactly which failures require it
9. **Comparison**: classifier vs SAGE diffusion vs baselines on end-to-end task success

---

## Empirical Round 1 (2026-05-16): The Biased-Teacher Discovery

This section documents the first end-to-end ML-vs-F evaluation and the
unexpected failure mode it surfaced. Full results, plots, and commands are in
[`ML_vs_GT_F_results_round1.md`](../evaluation/ML_vs_GT_F_results_round1.md); the evaluation
plan is in [`ML_vs_GT_F_evaluation.md`](../evaluation/ML_vs_GT_F_evaluation.md).

### Setup

Model: `outputs/cropped_diffusion_crossattn_2push/2025-12-16/05-36-44` — 5-channel local-mask DiT cross-attention, trained on `h5_files/dec2/aug9_envs/2_push_train_corrected_overlaps_2.h5`. Evaluated on the 300-env stratified held-out split (`manifest_2push_test_minus_1push_test_filtered_difficulty_100each.txt`) at the 1-push horizon — i.e. asking "of the model's 32 SE(2) samples, how many align to primitives that are in F₁?"

### Headline result

The model is **strictly worse than uniform-random-from-R** at Top-K hit rate against F₁, on every difficulty bucket. The gap is catastrophic on hard problems (random hit@32 = 96%, ML hit@32 = 11%) and persists even on very_easy problems where F density is ~89%.

### The smoking gun (and where I was wrong)

Initial hypothesis: model trained on 2-push intermediate states learned "push small" and is over-applying that prior. *Wrong*, but in an instructive way.

The actual cause, traced by opening the training h5:

1. **The training-target displacement distribution is shallow.** Median 0.30m, p90 0.67m, **97% under 1m**. The model is faithfully reproducing this distribution at inference (≥95% of decoded ML samples at displacement 0.2–0.5m, mapping to primitive depths 0–2).
2. **F₁ on hard problems lives at deep displacements** (d=7–9, ~1m+). On `very_hard`, 58.6% of F is at d=8–9 — where ML predictions have *zero* mass. Misses are mathematically necessary, not noise.
3. **The training data is the planner's first solutions, not F.** Every row in the training h5 has `algorithm = "Region Opening Planner"` and `solution_depth ∈ {1, 2}`. The planner's BFS expands depths in order 0, 1, 2, …, so the *shallowest depth that works* is recorded per problem and the rest of F is discarded. Subsequent smoothing further biases the recorded action toward minimal displacement.
4. **Architectural crop is *not* the bottleneck.** Local mask covers 5m × 5m; model's effective 32×32 output crop covers a 2.5m × 2.5m physical region centered on the object. Only 1.3% of training targets exceed the half-extent (1.25m). The model has plenty of room to represent deep pushes; it just was never *shown* any.
5. **There is also a selection bias in which envs reach training.** The 300-env held-out split is "minus 1-push test filtered" — exactly the hard envs the original (pre-ML) planner struggled on. Many were probably dropped from training because the planner timed out or failed. So both *what target* and *which env* are biased away from the hard regime.

This is the **biased-teacher problem**: distilling a generative model from a teacher's *first* solutions reproduces the teacher's bias, not the underlying feasible set. Diffusion architecture is doing what generative-from-demonstrations always does — modeling the data distribution — and the data distribution happens to be a narrow projection of F.

### How this updates the hypotheses

- **Hypothesis 1 (|F|/|R| predicts difficulty):** still consistent with everything observed. Random-from-R baseline hit-rate tracks |F|/|R| almost perfectly per bucket. Hypothesis confirmed for the *search* difficulty side; the ML side is decoupled because the model isn't sampling from R at all — it's sampling from a shifted distribution that intersects R poorly.
- **Hypothesis 2 (F is clustered):** orthogonal to the finding. Not testable from the current data alone since ML never reaches the depth where most clusters live.
- **Hypothesis 3 (hard problems have fragmented F → need multimodal predictors):** **prematurely tested.** Multimodality only matters once the marginal over depth is even approximately right. Right now the model collapses to a single mode (shallow) regardless of scene — the fragmentation question can't be answered until the depth marginal is fixed.
- **Hypothesis 4 (depth matters more than direction on hard):** strongly supported. The model gets direction "close enough" via the alignment tolerance (0.2 rad) but has 0% recall on the right depth band, and the resulting hit-rate is essentially zero.
- **Hypothesis 5 (wall collisions create F):** untestable until ML reaches deep F. Most wall-mediated successes are at deep depths in our F-char data.

### How this updates the methodology

The "Classifier-First, World Model If Needed" sequence in this doc starts from "train a 1-push classifier on the existing data and look at failure modes." Step 4 of the sequence got skipped — we went straight to evaluating SAGE's diffusion model without first re-asking "what is this model actually learning?"

The failure surfaced because of step 1 (exhaustive F characterization). Without F, the failure looks like "model is mediocre on hard problems" with no mechanism. With F, the failure is unambiguous and immediately traceable to data distribution. **F characterization is doing exactly what it was designed for** — making model failures legible.

Add a step 0 to the sequence: *open the training h5 and look at its target distribution*. If the target distribution doesn't match F's distribution, nothing else matters until that's fixed.

### Fixes (ranked by cost / impact)

**Fix 1 — Re-build the training h5 from F-char data.** Free with existing data.
- For each `(xml, region, object)` instance in `/common/users/dm1487/namo_data/f_characterization/1_push_exhaustive_full/` and `1_push_exhaustive_train/`, sample one or more (edge_idx, depth) uniformly from F per instance. Convert each to an SE(2) goal pose using the primitive lookup. That becomes the training target.
- Result: target distribution matches F's distribution by construction. Hard envs are included by construction (F-char doesn't filter on planner success).
- Risk: sample size shrinks. Current h5 has 168k rows; F-char has ~5,925 (train) + 1,767 (test) env pkls with ~2–5 instances each ≈ 20–40k instances. Sampling 4–8 targets per instance gets us back to ~100–300k rows.
- This is the immediate action.

**Fix 2 — Re-collect with shuffled depth ordering.** Half-day of compute.
- Modify `region_opening.py` to expose a `--shuffle-depths` flag (analogue of the existing `--shuffle-edges`). Re-run `modular_parallel_collection.py` on the same env pool with shuffled depth order, building a new h5.
- Result: the recorded "first solution" is now a uniform sample from F (not the BFS-minimum). Less clean than Fix 1 (still one solution per env, not the full F) but easier to integrate into existing pipelines that depend on the planner-output format.
- Use as a back-pocket if Fix 1's sample size proves insufficient even after augmentation.

**Fix 3 — Collect multiple solutions per env in one pass.** Higher cost.
- Re-run collection with `region_max_recorded_solutions_per_neighbor` cranked up (e.g. to 20 instead of 1), keeping all distinct (edge, depth) solutions per neighbor.
- Result: training distribution now reflects the *full multi-modal F* per env, not a single sample. Best for testing Hypothesis 3 (multimodality) downstream.
- Tradeoff: training set size inflates substantially; storage cost.

**Fix 4 — Architectural changes.** *Not needed yet.* Crop size, model dim, diffusion sampler, k_nearest — none of these matter until the training target distribution covers F. Defer until Fix 1 has been tried and evaluated.

### Open question: does the chain-2 horizon change the picture?

The chain_depth=2 GT collection is still running (background, ETA ~hours). When it lands, the same evaluation against F₁′ (push-1s that *enable* a successful push-2) will tell us:

- If ML hits F₁′ *much* better than F₁ → the model has correctly learned "set-up pushes" for chains, just not "opening pushes." The biased teacher gave it the right tool for the chain-only case. Use this model for chain inference, train a separate model on F-char data for 1-push.
- If ML also misses F₁′ → the shallow bias affects both horizons. The biased teacher is the root cause for everything, and Fix 1 is the only path forward.

Either way, the action items above are unchanged. Fix 1 first, re-evaluate, then decide whether to keep the existing model around for the chain-only use case.

### Round-1 update: the direction test

Following the round-1 finding, the obvious question was whether the model had learned *anything* scene-conditional, or whether it was emitting essentially scene-independent shallow predictions. I claimed informally that it had learned a "direction prior" — which face / contact-point to push — even if it failed at depth. That claim was tested empirically and only partially survived.

The test ([`test_direction_hypothesis.py`](f_characterization/test_direction_hypothesis.py)): project both ML predictions and F to three granularities and compare ML hit@K to random-from-R hit@K at each:
- **Face** — 4-way (which side of the object); `edge_idx // 15`.
- **Contact-point** — 60-way (where on the side, ignoring depth); `edge_idx`.
- **Joint** — full (edge, depth).

Results on the 300-env held-out split, K=1, reachable-filtered:

| level   | very_hard ML | very_hard rand | lift   | hard ML | hard rand | lift   |
|---------|-------------:|---------------:|-------:|--------:|----------:|-------:|
| face    | 0.556        | 0.401          | **+0.155** | 0.500 | 0.448  | **+0.052** |
| contact | 0.000        | 0.118          | -0.118 | 0.111   | 0.190     | -0.079 |
| joint   | 0.000        | 0.026          | -0.026 | 0.000   | 0.091     | -0.091 |

**Accept H_direction at face granularity (very_hard).** On the 12× rlab7 follow-up (n=30 for very_hard), the model picks the correct face 73.3% of the time vs random 47.3% — **lift +0.267 with 95% CI [+0.03, +0.51], statistically significant**. The original 300-env estimate of +15.5pp was an *underestimate*, not an overstatement. On hard (n=132), the lift is +6pp but ns. On easier buckets, both ML and random hit ~90%+ at face level (ceiling effect, signal saturates).

**Inconclusive at contact-point granularity.** rlab7 contact-prior on very_hard is +10pp, on hard is +8pp — both point estimates positive but CIs cross zero. The model may have learned coarse contact-point preference, but the current sample doesn't support a clean accept/reject.

**Reject H_direction at joint granularity.** The depth bias dominates: ML matches random at K=1 on hard/very_hard, loses on medium+. The model's failure to scale with K (predictions all cluster in one shallow region) confirms the marginal collapse on depth.

Updates to the partial-reasoning frame: the model has learned a **face prior** that is significant and substantial on the hardest problems (+27pp on very_hard, where it matters most). It has *not* learned push depth (rock-solid rejection). Contact-point selection within face is inconclusive — point estimates suggest a small positive signal but the data doesn't yet support a definitive claim. So the model has one confirmed capability (face selection), one suggestive but unproven capability (contact-point preference), and one definitively absent capability (depth selection). The biased-teacher frame still dominates the explanation; the face prior is a scene-conditional layer on top of the depth-bias floor, strongest where the depth bias hurts most.

Implication for the hybrid use case: a `ml_face_prior` strategy that uses the model to weight face-order in the planner's BFS but runs primitive search within each face would extract real value from this model without depending on its broken contact-or-depth predictions. Modest speedup (~2–4× on hard cases where the face pick is correct ~55% of the time), and cheap to add. Defer until Fix 1 is in flight; revisit if Fix 1 is delayed or partial.

Where this changes the methodology: when evaluating a learned model against F, decompose hit-rate across the natural granularity hierarchy of the action space *first*, not after the joint metric has already disappointed. The face/contact/depth decomposition surfaces partial reasoning that the joint metric hides. Add to the methodology as "step 4.5: per-granularity hit-rate decomposition before drawing conclusions about the model's reasoning content."
