# Problem Formulation: Empirical Characterization of the Feasible Set F

Paper-style definitions of every object, operation, and claim in the F characterization work. Intended as the spine for the "Preliminaries" / "Problem Formulation" section of the paper, and as the reference any later document or experiment can point back to.

---

## 1. Setting

We study **contact-rich manipulation with parameterized primitives**. The robot operates in a planar environment containing static obstacles (walls), movable obstacles (boxes), and a navigation goal. The robot executes parameterized push primitives on movable obstacles to change scene configuration and enable navigation. We instantiate this setting in **NAMO** (Navigation Among Movable Obstacles) and characterize the structure of primitive feasibility empirically.

---

## 2. Positioning: Controller-Aware Planning and the Action-Space Narrow Passage

This work occupies a specific position in the manipulation research landscape. We make it explicit before introducing formal machinery, because the framing determines what the contribution is — and what it is not.

### 2.1 Two camps: robust controller design vs. controller-aware planning

Manipulation research divides along a methodological axis with two poles.

**Pole 1 — Robust controller design.** The intellectual effort is concentrated in the controller. The goal is to engineer (or learn) a controller whose success region across scenes is so large that planning around it becomes unnecessary. Examples: classical compliant control (Mason, Whitney, Salisbury), large-scale visuomotor policies (RT-X, π, OpenVLA), domain-randomized end-to-end training. The bet is that controllers can be made robust enough that explicit reasoning about their failure regions is dispensable.

**Pole 2 — Controller-aware planning.** The intellectual effort is concentrated in the planner. The controller is taken as given; the planner reasons explicitly about where it succeeds and fails. Examples: funnel composition (Burridge–Rizzi–Koditschek 1999), LQR-Trees (Tedrake et al. 2010), contact trust regions (Suh et al. 2023), TAMP with feasibility samplers (Garrett, Kaelbling, Lozano-Pérez). The bet is that controllers always have non-trivial failure regions, that those regions have structure worth understanding, and that planning over a characterized success set produces system-level capability beyond what improving the controller alone can achieve.

**This work is in Pole 2.** We take a contact-rich push primitive as given, characterize its feasible set $F$ empirically, and study how the structure of $F$ should inform models that sample within it. We do not contest Pole 1's bet; we observe that controller-aware planning is an underexplored axis in contact-rich manipulation that retains value regardless of how robust controllers become.

### 2.2 The narrow-passage problem in action space

Sampling-based motion planning has a half-century-old lineage devoted to the **narrow-passage problem**: configuration spaces where success requires sampling within a thin region of $\mathcal{C}_{\text{free}}$, and uniform sampling fails because the passage's measure is small (Hsu, Latombe, Motwani 1997). Bridge sampling (Hsu et al.), Gaussian-near-obstacles sampling (Boor et al.), and learned samplers (Ichter et al. 2018) are the classical responses: characterize the passage geometry, then design samplers that put mass on it.

Our work is the **action-space analogue** of the narrow-passage problem. The correspondence is structural:

| Classical motion planning | This work |
|---|---|
| Configuration space $\mathcal{C}$ | Action-parameter space $\mathcal{A}$ |
| Free space $\mathcal{C}_{\text{free}}$ | Reachable set $R(s)$ |
| Goal-reaching subspace | Feasible set $F(s, g)$ |
| Narrow passage in $\mathcal{C}_{\text{free}}$ | Sparse $F$ within large $R$ — low $\rho = \lvert F\rvert/\lvert R\rvert$ |
| Sample complexity $\sim 1/\epsilon^d$ | Expected uniform-sampler trials $\sim 1/\rho$ |
| Bridge / Gaussian / learned samplers | Structure-informed classifiers and generative samplers |

The difficulty metric $\rho = |F|/|R|$ adopted in Section 9 is the discrete-action-space analogue of the $\epsilon$-good measure from the narrow-passage literature. The structure-informed model variants (Tier 3 baselines, Section 13) are the action-space analogues of bridge sampling — non-uniform sampling distributions designed to place mass on the success region.

Two differences with the classical setting are essential, and they delimit what is genuinely new:

1. **The passage is in action space, not configuration space.** The narrow region is somewhere the planner has to *aim*, not somewhere the robot has to *thread through*. Local geometry near $F$ is a metric on $\mathcal{A}$, governed by the dynamics $T$ and the success-state set $G(g)$, not by configuration-space obstacles.

2. **The passage is dynamics-induced, not geometry-induced.** $F$'s narrowness is determined by contact-rich dynamics: which pushes happen to land the object in the small set of poses that satisfy the goal. Classical narrow passages are facts about obstacle layout. Our narrow passages are facts about how $T$ maps actions to states and how that state distribution intersects $G$. The structural characterization presented here (face → contact → depth bottleneck, wall dependence, depth-window contiguity) is empirical knowledge about the dynamics-induced passage geometry that has no analytical analogue in the classical literature.

The recursive multi-push extension ($F_k'$, Section 16) has no clean classical analogue. The closest classical setting — sequential narrow passages in motion planning — has not been characterized empirically for either configuration-space or action-space cases. Our recursive characterization is therefore novel by analogy as well as in substance.

### 2.3 What this positioning implies

These two framings — controller-aware planning and the action-space narrow passage — together specify the contribution:

- The work belongs to the controller-aware planning lineage (funnel composition, LQR-Trees, contact trust regions), extended to contact-rich manipulation in clutter where analytical RoA computation is intractable.
- It transfers the methodological discipline of the narrow-passage literature (characterize the passage, then design the sampler) from configuration-space motion planning to action-space manipulation primitives.
- It contrasts cleanly with the dominant Pole 1 trajectory in current manipulation research — not as a competitor but as a complementary axis whose value is independent of how robust controllers become.

The baseline design (Section 13) operationalizes this positioning: geometric-heuristic baselines (Tier 1) are the action-space analogues of bridge sampling; learned baselines (Tier 2) are the analogues of Ichter-style learned samplers; structure-informed baselines (Tier 3) test whether the empirical characterization causally improves the sampler. Showing the layered lift across tiers is the action-space replication of the narrative the narrow-passage literature established for configuration-space motion planning.

---

## 3. State and conditioning

Let $s \in \mathcal{S}$ denote a **scene state**, comprising:
- Static obstacle geometry (walls).
- Movable obstacle poses $\{(x_i, y_i, \theta_i)\}_{i=1}^M$.
- Robot pose $q_r = (x_r, y_r, \theta_r)$.

Let $g \in \mathcal{G}$ denote a **task goal**, here a target robot pose $g = (x_g, y_g, \theta_g)$. The conditioning context for the feasibility analysis is the tuple $(s, g)$.

---

## 4. Action space

Let $\mathcal{A}$ denote the **primitive action space**: the parameter space of the push skill. In our NAMO instantiation,

$$\mathcal{A} = \mathcal{O} \times \mathcal{F} \times \mathcal{P} \times \mathcal{D}$$

where $\mathcal{O}$ is the chosen movable obstacle, $\mathcal{F} = \{1,2,3,4\}$ is the face index, $\mathcal{P} = \{1, \dots, 15\}$ is the contact-point index along the face, and $\mathcal{D} = \{1, \dots, 10\}$ is the discretized push depth. For a fixed object, $|\mathcal{A}| = 600$.

We treat $\mathcal{A}$ as a **discretization of a continuous parameter space** $\mathcal{A}_c$ in (push direction, push depth), and verify that key structural properties are stable across discretization resolutions.

---

## 5. Reachability

The **reachability set** $R(s) \subseteq \mathcal{A}$ is the subset of primitives the robot can physically execute from state $s$:

$$R(s) = \{a \in \mathcal{A} : \text{the robot can reach the contact point of } a \text{ in } s\}.$$

In NAMO, $R(s)$ is computed exactly and at low cost via wavefront BFS over an inflated occupancy grid. Reachability is **task-independent**: it depends on $s$ but not on $g$. In the narrow-passage analogy of Section 2, $R(s)$ plays the role of free space $\mathcal{C}_{\text{free}}$ — the ambient region within which the success region $F$ is embedded.

---

## 6. Push dynamics

Let $T : \mathcal{S} \times \mathcal{A} \rightharpoonup \mathcal{S}$ denote the **push transition operator**: $T(s, a)$ is the post-push scene state obtained by executing primitive $a$ from state $s$ in simulation. $T$ is partial (defined only for $a \in R(s)$), discontinuous (contact-mode switches), and not analytically tractable. The intractability of $T$ is what forces empirical characterization of $F$ — analytical preimage computation is unavailable.

---

## 7. Success criterion

Given a task goal $g$, the **success-state set** is

$$G(g) = \{s' \in \mathcal{S} : g \text{ is reachable for the robot from } s'\}.$$

Equivalently, $G(g)$ is the set of post-push states from which a navigation plan to $g$ exists. Membership in $G(g)$ is decided by a wavefront reachability query — exact, fast, no learning involved.

---

## 8. The feasible set F

The central object of study is the **feasible set**:

$$F(s, g) = \{a \in R(s) : T(s, a) \in G(g)\}.$$

Equivalently, $F$ is the **preimage** in action space of the success-state set $G(g)$ under the push dynamics $T(s, \cdot)$, restricted to reachable actions:

$$F(s, g) = R(s) \cap T(s, \cdot)^{-1}\big(G(g)\big).$$

We adopt the decomposition $F = R \cap C$, where $C(s, g) = \{a \in \mathcal{A} : T(s, a) \in G(g)\}$ is the **clearance set** (task success conditional on execution). $R$ is exact and free; $C$ is what a learned model must capture.

**Conceptual framing.** $F(s, g)$ is the *empirical region of attraction in action-parameter space*, conditioned on scene and task goal. It is the manipulation analogue of classical control-theoretic regions of attraction (Lyapunov, SOS, contact trust regions), but defined in action space rather than state space and characterized empirically rather than analytically — because contact-rich dynamics admit no closed-form preimage. Through the lens of Section 2, $F$ is the **narrow passage in action space** induced by the controller's contact-rich dynamics.

---

## 9. Difficulty

We define instance **difficulty** operationally as

$$\rho(s, g) = \frac{|F(s, g)|}{|R(s)|}.$$

$\rho$ measures the sparsity of feasible primitives within the reachable set. Its inverse $1/\rho$ is the expected number of uniform random primitive evaluations required to find a feasible action — the discrete-action-space analogue of the $\epsilon$-narrowness measure from Hsu–Latombe–Motwani (1997). We stratify instances into difficulty buckets by $\rho$ (Very Hard, Hard, Medium, Easy, Very Easy) and report all structural results per stratum.

---

## 10. Structural descriptors of F

For each instance $(s, g)$ with $|F(s, g)| > 0$, we compute structural descriptors over the discrete action grid:

- **Active face count** $n_F \in \{0,\dots,4\}$: number of faces containing any feasible primitive.
- **Contact-point span** $n_P$: number of active contact points within active faces.
- **Depth window** $[d_{\min}, d_{\max}]$ per active contact point and its **contiguity** (gap fraction).
- **Connected-component count** $K(s, g)$: number of disconnected feasible regions in the (face, contact, depth) grid under face-aware adjacency.
- **Wall-collision involvement** $w(s, g)$: fraction of $a \in F(s, g)$ for which simulator rollout reports object-wall contact during execution.

These descriptors operationalize the *structure* of $F$ — what we mean when we say "we characterize $F$" rather than "we record $F$." In the narrow-passage analogy, they are the action-space analogues of measurements one would make on a configuration-space passage: how thin, how connected, how aligned with which obstacle features.

---

## 11. Empirical characterization protocol

For each scene $s$ in a generated dataset $\mathcal{D}_{\text{scenes}}$ and corresponding task goal $g$:

1. Compute $R(s)$ via wavefront BFS.
2. For each $a \in R(s)$, execute $T(s, a)$ in simulation.
3. Evaluate $T(s, a) \in G(g)$ via post-push wavefront reachability.
4. Record per-primitive outcome (success / failure), collision flags, and post-push pose.
5. Compute structural descriptors of $F(s, g)$.

We refer to this as **exhaustive 1-push F characterization**. It produces ground-truth $F(s, g)$ for every test instance, which serves as the reference against which any learned predictor or sampler is evaluated.

---

## 12. The learned predictor

We are interested in models $p_\theta(a \mid s, g)$ that **predict** or **sample** primitives intended to lie in $F(s, g)$. We consider two families:

- **Classifier**: $h_\theta(s, g, a) \in [0, 1]$ approximating $\Pr[a \in F(s, g)]$, trained on per-primitive labels from $\mathcal{D}_{\text{scenes}}$. Sampling proceeds by ranking $a \in R(s)$ by $h_\theta$ and drawing top-$k$ or threshold-filtered samples.
- **Generative sampler** (diffusion): $p_\theta(a \mid s, g)$ trained on $(s, g, a)$ tuples with $a \in F(s, g)$. Sampling proceeds by reverse diffusion conditioned on $(s, g)$.

For both families, we say the model **implicitly characterizes $F$** if its predictions respect the structural descriptors of $F$ — that is, if the model's behavior tracks $F$'s contiguity, hierarchy, modal structure, and wall dependence rather than incidental scene features. Both families are instances of the controller-aware planning paradigm of Section 2: the underlying push controller is held fixed; the model learns where to invoke it.

---

## 13. Baselines

We organize baselines into tiers, each isolating a distinct claim. Every baseline tests one specific component of the hypothesis chain; baselines that do not isolate a component are excluded.

### Tier 0 — Bounds

- **B0a — Oracle**: exhaustive evaluation of $R(s)$, returning ground-truth $F(s,g)$. Upper bound on what any method can achieve.
- **B0b — Random over $\mathcal{A}$**: uniform sampling ignoring reachability. Trivial floor.
- **B0c — Random over $R(s)$**: uniform sampling restricted to reachable primitives. Achieves precision $= \rho$ in expectation. **The critical floor:** the gap between any method and B0c isolates what the method has learned about clearance $C$ beyond the free reachability oracle.

### Tier 1 — Geometric heuristics (action-space analogues of bridge sampling)

- **B1a — Push-toward-open-space**: bias samples by the wavefront-distance gradient at the contact point.
- **B1b — Wall-aware sampling**: bias samples toward primitives that produce object-wall contact. Motivated by the wall-collision finding on Very Hard instances.
- **B1c — Center-of-face heuristic**: push from the center of an object face at moderate depth — the unlearned default.
- **B1d — Hierarchy-respecting random**: sample face → contact-point → depth uniformly within active levels. Tests whether the bottleneck hierarchy alone captures most of $F$.

### Tier 2 — Simple learned baselines

- **B2a — Logistic regression** on hand-engineered scene features. Tests whether $F$ is predictable from a few engineered features.
- **B2b — Random forest** on the same features. Diagnostic for non-linearity vs. representation learning.
- **B2c — Independent-logits classifier**: deep network outputting 600 independent logits, structure-blind.
- **B2d — Independent-logits + rejection over $R$**: B2c with reachability filtering at inference. The cleanest classifier baseline for the generative comparison.

### Tier 3 — Structure-informed learned baselines (the contributions)

Each is an architectural variant of B2d that exploits one structural finding from Section 10:

- **B3a — Conv-over-grid head**: convolutions over the (contact-point, depth) grid per face. Tests the contiguity prior.
- **B3b — Hierarchical face → contact → depth head**: sequential prediction respecting the bottleneck hierarchy. Tests the hierarchy prior.
- **B3c — Wall-distance-augmented input**: scene encoder receives a wall-distance field around the object. Tests the wall-dependence prior.

### Tier 4 — The generative comparison

- **B4a — Diffusion model**: scene + goal conditioned, samples primitives, restricted to $R$ at inference. Tests whether generative modeling lifts over the best classifier baseline (B2d or B3a–c).
- **B4b — Diffusion ablation, scene-only**: diffusion conditioned only on scene, no goal. Diagnostic for the importance of goal conditioning.

### Reporting protocol

Every baseline reports precision@$k$, coverage@$k$, and structural-alignment metrics per difficulty stratum (Very Easy, Easy, Medium, Hard, Very Hard). For each baseline, we also state in advance — based on the characterization of $F$ — how it should perform; the agreement or disagreement between predicted and observed performance is itself a hypothesis test on the structural findings.

### Minimum viable baseline set

If time-constrained, the minimum baselines that license the paper's central claims are:

| Baseline | Claim it isolates |
|---|---|
| B0c (uniform over $R$) | Method has learned anything about $C$ beyond reachability |
| B1b (wall-aware) | Learning lifts over a strong geometric heuristic |
| B2d (deep classifier + rejection) | Classifier earns its keep over heuristics |
| B3b (hierarchical head) | Structural prior causally improves the classifier |
| B4a (diffusion) | Generative modeling does or does not add lift over the best classifier |

Five baselines, each licensed by a specific component of the hypothesis chain.

---

## 14. Evaluation against ground-truth F

Given ground-truth $F(s, g)$ for each test instance and $k$ samples $\{a_1, \dots, a_k\} \sim p_\theta(\cdot \mid s, g)$, we report:

- **Precision@k**: $\frac{1}{k} \sum_i \mathbb{1}[a_i \in F(s, g)]$.
- **Coverage@k**: fraction of connected components of $F(s, g)$ hit by at least one sample.
- **Distributional divergence**: JS divergence between the empirical sample distribution and the uniform distribution over $F(s, g)$.
- **Structural alignment**: per-difficulty-stratum precision and coverage; correlation between failure mode (which structural axis the model gets wrong) and instance descriptors.

The **lift over a uniform-over-$R$ baseline** (B0c) isolates whether the model has learned anything about $C$ beyond reachability.

---

## 15. Hypotheses

We pre-state hypotheses tested by the characterization (these were registered before data collection):

- **H1 (Difficulty)**: Instance difficulty is governed by $\rho = |F|/|R|$, not $|F|$ alone.
- **H2 (Contiguity)**: $F$ is spatially contiguous in $\mathcal{A}$, not scattered.
- **H3 (Fragmentation)**: Hard instances have higher fragmentation $K(s, g)$ than easy instances. *(Falsified by data: hard instances are predominantly unimodal.)*
- **H4 (Depth precision)**: On hard instances, $F$ requires precise depth but spans multiple contact points. *(Refined by data: contact-point precision is the dominant bottleneck; depth windows are tight but contiguous.)*
- **H5 (Wall dependence)**: On hard instances, a disproportionate fraction of $F$ involves wall collisions. *(Confirmed: 76% on Very Hard.)*

We report results per hypothesis, including the falsification of H3, which we treat as a substantive negative finding rather than as a discarded prediction.

---

## 16. The N-push extension (defined for completeness; deferred for the present work)

For $n$-push problems where $F(s, g) = \emptyset$, we define recursively:

$$F_k'(s, g) = \{a \in R(s) : F_{k-1}'(T(s, a), g) \neq \emptyset\}, \quad F_1'(s, g) := F(s, g).$$

$F_k'$ is the set of first-of-$k$ pushes that lead to a state from which a $(k-1)$-push solution exists. This is **recursive feasibility composition** — feasibility flows backward from the goal through the chain — and is the action-space analogue of funnel composition (Burridge–Rizzi–Koditschek) and LQR-Trees (Tedrake).

The structural questions about $F$ extend recursively: does $F_k'$ inherit contiguity, unimodality, and wall dependence from $F_1'$? The closest classical analogue — sequential narrow passages in motion planning — has not been characterized empirically for either configuration-space or action-space cases. This forms the basis of the multi-push study and is left to future work.

---

## 17. Scope and what we do *not* claim

- We do not claim $F$'s structure transfers to manipulation primitives outside push-in-clutter without further characterization.
- We do not claim sim-measured $F$ matches real-world $F$; sim-to-real validation is left to future work.
- We do not claim our learned models are state-of-the-art on any external benchmark; the contribution is structural, not numerical.
- We do not claim the methodology is novel in concept; empirical characterization of feasibility regions has precedent (Dex-Net for grasping, contact trust regions for few-contact systems). We claim novelty in *application* (contact-rich pushing in clutter), in *positioning* (controller-aware planning + action-space narrow-passage framing), and in *coupling* characterization to classifier-first methodology with falsifiable hypotheses.
- We do not contest the robust-controller-design (Pole 1) bet of Section 2. Controller-aware planning is complementary to controller improvement; both can be pursued in parallel.

---

## Notation summary (single-page reference)

| Symbol | Meaning |
|---|---|
| $s$ | Scene state (geometry + robot pose) |
| $g$ | Task goal (target robot pose) |
| $\mathcal{A}$ | Primitive action space (600 push primitives in NAMO) |
| $R(s)$ | Reachable subset of $\mathcal{A}$ — exact via wavefront BFS |
| $T(s, a)$ | Post-push scene state |
| $G(g)$ | Success-state set (states from which $g$ is reachable) |
| $C(s, g)$ | Clearance set $\{a : T(s, a) \in G(g)\}$ |
| $F(s, g)$ | Feasible set $R(s) \cap C(s, g)$ |
| $\rho(s, g)$ | Difficulty $|F| / |R|$ |
| $K(s, g)$ | Number of connected components of $F$ |
| $w(s, g)$ | Wall-collision fraction in $F$ |
| $p_\theta(a \mid s, g)$ | Learned sampler |
| $h_\theta(s, g, a)$ | Learned classifier |

---

## Why this document exists

A paper lives or dies on whether the reader can hold the central object in their head precisely by the end of section 2. With these definitions in place, every later claim ("F is unimodal on hard instances," "the model fails on wall-dependent F," "diffusion does not lift over classifier+rejection on hard cases") becomes a precise statement about a precisely-defined object, evaluated by a precisely-defined protocol. That precision is what separates a paper that gets engaged with from a paper that gets called "interesting but unclear."

Pin this. Refer back to it. When confused about what an experiment tests, reread the definitions. They are the spine.
