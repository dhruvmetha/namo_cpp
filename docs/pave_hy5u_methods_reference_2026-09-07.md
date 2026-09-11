# HY5U region-opening ranker: implementation methods reference

Prepared 2026-09-07 for the PAVE manuscript. This is a standalone technical source document, not a proposed conference section. It records code, configurations, checkpoints, data, and evaluation artifacts inspected in the NAMO checkout and adjacent Sage and robot-control repositories. No experiments were run and no implementation was changed to prepare it.

The governing problem statement is [problem_and_approach.md](problem_and_approach.md). The learned model is a single action ranker used as a search heuristic. It orders simulator trials so that a verified region opening is found with few simulator calls. It is not conditioned on search horizon or simulator budget, and its outputs should not be presented as calibrated opening probabilities. A one-push versus two-push episode describes the depth needed to merge the same two regions across the same blocking object; it does not describe the number of region boundaries between the robot and its destination.

## 1. Scope, versions, and findings that affect the methods claim

The principal evaluated method is HY5U: a five-channel, contact-token transformer with a categorical scalar-value head, trained on the hybrid root/child dataset and used to prioritize a global best-first queue. The canonical fixed-physics v3 evaluation used NAMO revision `d32ec…`; the corrected 4,000-call timing comparison records revision `c0413c3`. The latest checkout inspected during this audit was NAMO `f5335c43c53d9506dbef4ce304d6ae3f90151578`, with uncommitted edits including `python/namo/planners/opening/best_first_search.py`. Adjacent Sage was `ceff4bf49f1fb55f91be47eaacb038c5fefde687`; robot-control was `ceb1db9e1f1730289f4ec7221404b17705c2b904`. Current production behavior must therefore be identified separately from evaluated behavior. Registry entries and saved per-run arguments are the authority for a result's complete protocol.

Three implementation findings materially constrain the manuscript description:

- **Censored supervision is implemented but dormant in the actual HY5U training artifact.** A read-only scan of all 1,302,659 rows and 390,797,700 cells of `hybrid_train_v1.h5/ceiling_mask` found zero nonzero entries. HY5U therefore did not receive an active censored-likelihood term through this dataset.
- **Unreachable-action supervision is not regression-only in the executed loss path.** The dataset supplies a separate ranking mask, but the shared loss wrapper overwrites it with the regression mask. Unreachable actions consequently enter both listwise and family ranking comparisons as zero-valued competitors, with full Boolean membership. Their regression weight remains 0.1. This behavior is present in the source used around the evaluated revision; it is not merely a recent dirty-checkout change.
- **Evaluated goal conditioning and verification are not always the same input.** Canonical model evaluation renders the region containing the XML goal or fallback goal, while region-opening success uses a separately sampled fixed target-region point set. Current production passes the selected neighbor-region samples into both rendering and verification. The evaluated model's fifth channel is a connected-region mask, not isolated sample dots; the discrepancy concerns which goal seeds that region.

The document gives the executed objective and search procedure first, then identifies dormant or optional machinery. It does not infer a contribution from an option merely because the option exists in the repository.

## 2. Notation and local task

| Symbol | Meaning |
|---|---|
| `s` | Simulator scene state at rest: robot pose and movable-body poses, with geometry and physical parameters supplied by the scene/configuration |
| `o` | The supplied blocking object in a registered local region-opening episode |
| `R_r`, `R_g` | Robot and selected adjacent goal free-space regions |
| `G={g_i}_{i=1}^N` | Fixed target-region verifier samples drawn at the initial state |
| `a=(o,e,d)` | Push action on object `o`, contact index `e∈{0,…,59}`, duration index `d∈{0,…,4}` |
| `T(s,a)` | Physics/controller transition produced by one simulator `env.step` |
| `V_G(s)` | Region-opening verifier at state `s` with fixed samples `G` |
| `I(s,o,G_r)` | Five-channel 64×64 crop, where `G_r` denotes the goal information actually supplied to the renderer |
| `C(o,s)` | 60×2 contact coordinates in the crop's pixel coordinate system |
| `z_θ(s,a)` | 51 categorical logits for one action |
| `v_θ(s,a)` | Expected categorical-bin-center score used for ranking |
| `H` | Maximum number of pushes in a candidate solution chain; a search setting, not a model input |
| `B_sim` | Maximum number of simulator trials during one search invocation |
| `r`, `m_v`, `c` | Reachability mask, observed-value mask, and ceiling/censor mask for a training cell |
| `γ` | Training target assigned to a successful setup; 0.5 in the actual HY5U loader |
| `f` | A training family keyed by `(xml, object_id)` |

A local episode supplies a scene, one blocking object, and one adjacent goal region. The initial robot pose determines `R_r`; moving the supplied object should merge robot and goal regions. The robot may reposition to contact the object as part of the push skill, but the local search does not solve an arbitrary route through multiple intervening blockers. A direct opener achieves the local verifier after one push. A two-push solution applies a setup and a finish to the same object; the setup alone does not open the selected pair. A nonadjacent target requiring traversal of multiple region boundaries belongs to full NAMO, not to the local two-push category.

For the canonical region verifier, `eval_m3.sample_goal_points` obtains an initial region snapshot using up to 100 goals per region, `use_xml_goal=True`, and seed 42. It keeps the chosen goal-region points fixed while evaluating successor states. Given `N` retained points, the predicate is

```math
V_G(s) = 1\left[\sum_{i=1}^{N} 1[g_i\text{ is robot-reachable in }s]
                  \ge \max(1,\lceil 0.2N\rceil)\right],\quad N>0.
```

An empty point set fails this predicate. Thus “20 out of 100” is only the special case `N=100`. Reachability is the backend's free-space/wavefront criterion under the selected robot and inflation configuration. This test does not execute a navigation trajectory to every point. The simulator is the verifier in the learning/search formulation: verification needs no additional push simulation, although its computation has nonzero wall time. It is not a claim that simulated contact mechanics perfectly predict hardware.

The canonical fixed-physics simulator evaluation uses the explicit 5 mm margin configuration. Current generic wavefront defaults and real-robot configurations can use 1 mm. Both the renderer's wavefront and the simulator verifier must use the intended configuration; silently combining margins changes the task and observation. Source: [eval_m3.py](../scripts/sandbox/eval_m3.py), [eval_sets.yaml](../config/eval_sets.yaml), [wavefront_inflation.yaml](../config/wavefront_inflation.yaml), and [eval-set registry](experiments/eval_set_registry.md).

## 3. Executable push library and state restoration

### 3.1 Discrete actions and nominal goals

The car primitive library has 60 contact indices and five durations per contact, giving at most 300 actions for one object in one state. The four rectangular faces have 15 samples each, including endpoints. Contacts are interleaved by opposing face, not stored as four consecutive blocks of 15. The discrete field called `depth` in an action is a **duration index**, unrelated to search-chain depth. Index `d` selects `push_steps=d+1`.

Primitive files use the `1x_car_d5_` family and shape-specific square/wide/tall tables. Python loads records containing nominal local translation, nominal rotation, contact index, and push-step count; each packed record has three floats and two one-byte indices after the record-count header. Records are grouped by contact and sorted by push-step count. The shape selector distinguishes near-square geometry from wide or tall geometry, matching the backend's intended selection. Exact file selection should be retained from the scene's object dimensions rather than inferred from an image.

For local primitive displacement `(Δx,Δy,Δθ)` and current object pose `(x_o,y_o,θ_o)`, the returned nominal goal is

```math
\begin{bmatrix}x^*\\y^*\end{bmatrix}
=\begin{bmatrix}x_o\\y_o\end{bmatrix}
+\begin{bmatrix}\cos\theta_o&-\sin\theta_o\\\sin\theta_o&\cos\theta_o\end{bmatrix}
\begin{bmatrix}\Delta x\\\Delta y\end{bmatrix},\qquad
\theta^*=\theta_o+\Delta\theta.
```

These nominal endpoints identify and describe the primitive; the actual resulting pose is produced by the contact controller and physics. There is no assumption that the object reaches `(x*,y*,θ*)`, and HY5U does not regress that endpoint. `PrimitiveGoalStrategy.generate_goals` returns the available contact/duration grid; its `max_goals` argument does not impose a top-k candidate cap in this path. Source: [primitive_goal_strategy.py](../python/namo/strategies/primitive_goal_strategy.py), especially loading, `generate_goals`, and `_select_primitive_file`.

### 3.2 Reachability and physical execution

Search first queries reachable objects and contact edges in the current state, then retains only allowed contacts on the supplied target object. The backend checks contact reachability again when the action is executed. The push skill plans a robot approach with the wavefront planner, places the robot at the final approach pose through the teleport-navigation path, checks placement collision, zeroes velocities, and settles before pushing. The contact/duration fields select the direct primitive execution path.

The inspected car configuration uses 550 controller ticks per push step. Requested duration is therefore `(d+1)×550` ticks, applied as a continuous push rather than five mandatory stop/start segments. Inspected canonical source XMLs use a 0.002 s timestep, giving nominal push intervals of 1.1, 2.2, 3.3, 4.4, and 5.5 s for those scenes, plus settling and approach overhead. Pre-push settling is 100 ticks. The timestep-to-seconds conversion was checked on two canonical source XMLs, not exhaustively across every XML; ticks are the portable specification.

Execution can terminate early. Robot collisions and some failure paths restore the pre-push state; a stuck outcome after meaningful object motion can retain that moved state. Object-wall and object-object contacts are not all equivalent to immediate rollback. The search must inspect the resulting state and opening predicate rather than equate a low-level success flag with local task success. The learned score likewise is not a guarantee of controller completion. Source: `src/skills/namo_push_skill.cpp`, `src/planning/namo_push_controller.cpp`, `src/planning/push_primitive_executor.cpp`, and the selected car push-skill YAML; consult the exact checkout paths when reproducing backend behavior.

### 3.3 What a restored search state means

The standard pattern is `get_full_state → query/render/step → set_full_state`. Full-state snapshots contain positions and velocities, but restoring a state intentionally zeroes velocity rather than continuing the captured moving trajectory; backend restoration also resets control and warm-start acceleration state. Consequently the searched transition is a push from a restored, stationary configuration. This is appropriate to the implemented stop-and-push abstraction, but it should not be described as arbitrary continuous-state trajectory replay. Rendering and reachability queries may touch simulator state, so callers restore around them as well. Source: [rl_env.cpp](../python/namo/cpp_bindings/rl_env.cpp) and `src/environment/namo_environment.cpp`.

## 4. Five-channel observation and contact coordinates

### 4.1 Raster contents and geometry

`LiveScorer` uses channels in this exact order: `static`, `movable`, `target_object`, `robot_region`, `goal_sample_region`. A sample tensor has shape `5×64×64`. The crop is **0.5 m on each side**, centered on the target object's current center, with a 0.25 m half-width. It remains aligned with world axes; it is not rotated into the object's frame. Its nominal pixel pitch is 0.5/64 = 0.0078125 m. This pitch is a rendering scale, not a claim that all relevant geometric distinctions are resolved at 7.8125 mm.

The static and movable channels render physical object footprints. The movable channel includes the target object; the target channel marks it a second time so its identity is explicit. Rectangular footprints use current positions, orientations, and half-extents from object metadata. The robot-region and goal-region channels come from the configured wavefront geometry, which includes robot/inflation effects. There is no separate robot-disc channel in the five-channel model, no target-motion arrow, no history channel, and no independent image for each candidate action.

The visualizer initially constructs a common 1024×1024 canvas with isotropic scale `1024/max(world_width,world_height)` and the world center at the canvas center. Both x and y increase with their corresponding image coordinates: world y maps to increasing image row, without a vertical sign flip. Integer polygon rasterization and crop-center rounding occur before resizing. A world-axis-aligned object-centered crop is padded with zeros outside the canvas, resized to the local 224×224 representation, then reduced to 64×64 with area interpolation. Hence final pixels can be fractional occupancies rather than binary labels.

The wavefront region map is indexed `(x,y)` and is transposed into image `(row,column)` order before placement on the same canvas. It is resized with nearest-neighbor interpolation while preserving world aspect ratio and centered padding. `robot_region` is the component labeled `robot` or `robot_goal`; `goal_sample_region` is the component labeled `goal` or `robot_goal`. These are filled components. A separate sample-dot channel exists in the visualizer but is not among HY5U's five inputs. The live scorer rejects the legacy fallback path that can use an incorrect robot size. Source: [live_scorer.py](../scripts/sandbox/live_scorer.py), `../sage_learning/sage_learning/visualizer.py` around `generate_all_masks_highres`, including the successful unified-wavefront branch around lines 1335–1400 and crop construction around lines 1575–1662.

### 4.2 Exact contact-coordinate convention

Let `h_x,h_y` be object half-extents, `θ` its yaw, and `t_j=j/14`. For contact `e<30`, set `j=floor(e/2)`, `l_x=-h_x+2h_x t_j`, and `l_y=h_y` for even `e` or `-h_y` for odd `e`. For `e≥30`, set `j=floor((e-30)/2)`, `l_y=-h_y+2h_y t_j`, and `l_x=h_x` for even `e` or `-h_x` for odd `e`. Rotate `(l_x,l_y)` by `θ` to obtain offset `(w_x,w_y)`. Then

```math
c_e=\left(32+\frac{w_x}{0.5/64},\;32+\frac{w_y}{0.5/64}\right).
```

The resulting `contact_px` tensor has shape `60×2`, in floating-point crop pixels with the same non-inverted y convention as the raster. Contacts at a geometric corner can occur under different face indices and are distinct library actions. The current object orientation affects both the footprint and these coordinates. Source: [eval_scorer.py:45](../scripts/eval_scorer.py:45) and [live_scorer.py:169](../scripts/sandbox/live_scorer.py:169).

### 4.3 Training, evaluation, and production goal inputs

The training renderer reuses the live visualizer rather than defining a separate image pipeline. `build_train_h5._render_row` and `build_rung2_h5` render with a goal/XML input but without explicitly supplied verifier samples. The hybrid builder copies stored tensors; it does not rerender them. Stored context tensors are converted to float32 by the training dataset. No image augmentation is applied by the inspected Q2 loader.

In the canonical HY5U evaluation at `d32ec…`, `extract_goal_with_fallback` supplies the environment/rendering goal. Fixed goal-region samples are separately used by the success predicate, and `solve_scene` does not receive them for the model scorer. At `c0413c3` and in the current evaluator, the explicit `region_samples` argument is passed for geometric priors but remains absent for the model prior. Thus it is inaccurate to state without qualification that the evaluated model is conditioned on exactly the fixed verifier region. Current `BestFirstRegionOpeningPlanner` does pass its selected neighbor-region samples to the ranker and verifier; the first sample supplies the goal seed for the rendered connected component. Whether the historical XML region always equals the verifier region on the evaluated population has not been established by this audit.

## 5. HY5U architecture and scoring

The instantiated network has 4,397,055 parameters. It consumes a batch of context tensors `B×5×64×64` and contact coordinates `B×60×2`. A convolution with kernel and stride 4 embeds nonoverlapping 4×4 patches into dimension 192, producing 256 scene tokens on a 16×16 grid. Learned scene-position embeddings have shape `1×256×192`. Four scene transformer blocks each apply pre-normalized six-head self-attention and a pre-normalized MLP `192→768→192` with GELU, each with residual addition. Scene features then receive a final layer normalization.

Each of the 60 contacts receives a query token composed of three terms: a Fourier embedding of the contact coordinate, a learned contact-index embedding, and a local feature sampled from the encoded scene grid. Normalize pixel coordinates as `g=2c/64−1`. For each of the two coordinates, use sine and cosine at eight frequencies `π2^k`, `k=0,…,7`, giving 32 Fourier features. A `32→192→192` GELU MLP maps those features to the query dimension. The learned contact-index table has shape `60×192`. Bilinear `grid_sample` obtains a 192-dimensional scene feature at `g`, with `align_corners=False` and border padding, followed by a `192→192` projection.

Four contact blocks refine these tokens. Each has pre-normalized contact-to-scene cross-attention, pre-normalized self-attention among the 60 contacts, and a pre-normalized `192→768→192` GELU MLP, with residual connections. Attention has six heads; the inspected network uses zero dropout. A final contact normalization and shared `192→192→255` GELU head produce `B×60×255`, reshaped to `B×60×5×51`.

There are **60 contact tokens, not 300 action tokens**. The five durations are output slots in each contact's final head. They are not separate transformer queries, and the model is not conditioned on a duration scalar, a remaining horizon, or a simulator budget. Optional newer builder flags for motion, finer crops, additional depth representations, or reachability do not describe these checkpoints. The actual HY5U state dictionaries contain no such branches.

For action `a`, softmax converts its 51 logits to `p_{a,j}`. The scalar used by search is

```math
b_j=\frac{j+1/2}{51},\quad j=0,\ldots,50,\qquad
v_\theta(s,a)=\sum_{j=0}^{50}p_{a,j}b_j.
```

The bin centers span approximately 0.0098039 to 0.990196, so even an endpoint one-hot distribution has an expected score inside `(0,1)`. “Raw Q” in the evaluation flags means this expected score without an additional sigmoid; it does not mean raw categorical logits. The model is trained to place successful and useful actions ahead of lower-valued alternatives. Neither the categorical representation nor the expected score establishes calibration as a physical opening probability.

Source: [train_gen.py:29](../python/namo/rl_loop/train_gen.py:29), `../sage_learning/src/model/dit/edge_crossattn.py`, `../sage_learning/src/model/hl_gauss.py`, and [live_scorer.py:180](../scripts/sandbox/live_scorer.py:180).

## 6. Sampled data, labels, families, and splits

### 6.1 Collection and hybrid composition

Training uses sampled search experience at initial and successor states. It does not enumerate all actions or all two-push paths to obtain exhaustive ground truth. Root observations record attempted first pushes; successful continuations produce setup evidence. Child observations represent the resulting states of selected first pushes and the attempted finish actions from those states. A row is a state/target-object observation with a 60×5 label grid; one room XML can contribute many rows and many episodes.

The HY5U artifact is `/common/users/dm1487/scratch_namo/aquaman/round0/hybrid_train_v1.h5`, containing 1,302,659 rows. `build_hybrid_h5.py` combines all 257,409 rows of the older `arjuna0v2_train.h5` with 1,045,250 rows selected by `chain_depth==2` from `family0_train_v2.h5`. The older block is not exclusively roots. The experiment card reports 215,856 root rows; subtracting from the full artifact gives 1,086,803 child rows. The hybrid builder copies the shared datasets and harmonizes old setup targets from 0.5 to 0.9 before the training loader remaps them to the selected training gamma.

The inspected H5 contains `ctx`, `contact_px`, `xml`, `object_id`, `chain_depth`, `r_mask`, `value_mask`, `value_target`, and `ceiling_mask`. It has no goal-region identifier and no guess-mask column. The full ceiling scan found every ceiling cell zero. `chain_depth` records root/child provenance, not a conditioning variable supplied to the network.

The family-building path selects a bounded set of children: up to four randomly chosen children with an observed finish, up to four high-root-score children without an observed finish, and up to two random remaining children without an observed finish. Selection occurs before rendering. “Dead” in this collector means no finish was found in its sampled search, not a proof that no executable finish exists. A historical collection cap of 12 limited child finish sweeps; later notes correct an earlier misleading interpretation that this was a comparable cap on all root labeling. Source: [build_hybrid_h5.py](../scripts/pipeline/build_hybrid_h5.py), [build_rung2_h5.py](../scripts/pipeline/build_rung2_h5.py), and [cross-board ranking card](experiments/log/EXP-2026-08-09-crossboard-ranking.md).

### 6.2 Label semantics and limitations

| Cell status | Builder/loader behavior | What the evidence establishes |
|---|---|---|
| Direct observed opener | Target 1, observed and reachable | This trial passed the local verifier |
| Root action with an observed successful child finish | Stored setup value 0.9, remapped to 0.5 by HY5U loader | This action participates in an observed two-push opening |
| Attempted action without a found opening/continuation in hard-label artifacts | Target 0, observed | No success was found under the collection procedure; not necessarily impossible |
| Untried reachable action | `value_mask=0` | No regression/ranking target in the ordinary observed mask |
| Unreachable action | `r_mask=0`; target forced to 0 in grouped training | Current contact is infeasible under the reachability query; receives extra supervision |
| Censored unresolved action in optional historical builders | Ceiling value plus `ceiling_mask=1` | Intended upper-bound supervision; not active in the HY5U hybrid artifact |

`build_rung2_h5` ORs success over repeated recorded trials, labels direct successes 1 and observed winning root setups with gamma, and otherwise follows the selected hard-label or optional censored policy. Its dataset setup-motion gate uses object translation greater than 1 cm on either axis or wrapped angular displacement greater than 0.05 rad; this is distinct from the search's much tighter no-op check. Missing pose information is treated permissively by that helper and is not proof of motion.

`arjuna_build_v2.py` deliberately converts bounded sampled failures into zero targets and clears ceiling masks. It also converts setup 0.9 to 0.5. This creates hard negatives that can include undiscovered setups. Optional `colossus` logic in the rung builder masks some unresolved capped roots or encodes ceilings such as 0.81 for exhausted roots and 0.9 for failed children; those numbers belong to that builder's discount/bound interpretation. They must not be described as active HY5U supervision. A historical audit's false-dead estimate is evidence about that audited sample, not an exhaustive error rate for the 1.3-million-row hybrid dataset.

The loader converts stored targets approximately equal to 0.9 to `NAMO_GAMMA`, which is 0.5 for this recipe. Gamma is a target-spacing choice between immediate openers and useful setups. It is not a requested planning horizon and does not create separate horizon-conditioned heads. The exact label vocabulary observed in sampled root/child blocks is consistent with `{0,0.5,1}` after loading, but only the all-zero ceiling claim was exhaustively scanned in this audit.

### 6.3 Room holdout and grouped batches

The Q2 dataset groups by exact `xml` string, sorts unique keys, shuffles them with `random.Random(0)`, and assigns whole groups until approximately 90% of rows are in training. Actual logs report 198,267 room keys, 1,172,394 training rows, and 130,265 validation rows; validation has 20,081 room keys and training therefore has 178,186. This is a room-key holdout, not an independent random split of push cells. This audit did not establish that different XML strings cannot encode duplicate geometry.

Family batching groups training rows by `(xml, object_id)`. Logs report 234,307 families, including 117,865 with more than one row. Families are shuffled using `random.Random(1000+epoch)` and packed whole into batches with a nominal maximum of 256 rows. An oversized family is truncated to its first 256 rows by this sampler. `ROOT_FRAC=0` disables root-fraction resampling. The family key omits goal region, although the conceptual episode includes it; because the H5 omits a goal-region column, the assumption that `(xml,object_id)` uniquely identifies the intended goal within these data remains an author/reproducibility question. Source: [q2_dataset.py](../python/namo/rl_loop/sage_ext/q2_dataset.py) and [train_q2_round2.py](../scripts/rl_loop/train_q2_round2.py).

## 7. Actual training objective

### 7.1 Masks and categorical regression

For cell `a` in row `i`, let `m_{ia}=m^v_{ia}r_{ia}` be the observed reachable mask and `u_{ia}=1−r_{ia}`. Grouped training forces unreachable targets to zero and supplies regression weight

```math
w_{ia}=m_{ia}+0.1u_{ia}.
```

Other sample weights are one in the inspected dataset. With 52 equally spaced edges `e_j=j/51`, the intermediate-target distribution is a Gaussian integrated into each bin and renormalized over `[0,1]`, with `σ=0.75/51`:

```math
q_j(y)=\frac{\Phi((e_{j+1}-y)/\sigma)-\Phi((e_j-y)/\sigma)}
                  {\Phi((1-y)/\sigma)-\Phi(-y/\sigma)}.
```

The censored-capable helper used by this loss path overrides endpoint targets: target zero is one-hot in the first bin, and target one is one-hot in the last bin. HY5U's regression term is

```math
L_{\mathrm{HL}}=
\frac{\sum_{i,a}w_{ia}\left[-\sum_jq_j(y_{ia})\log p_{ia,j}\right]}
     {\max(1,\sum_{i,a}w_{ia})}.
```

This reduction is over all weighted cells in the minibatch, not a mean of independently normalized board losses and not a separate unreachable-group mean. Unreachable cells therefore affect both numerator and denominator with weight 0.1. The endpoint override changes the target distribution, not the expected-score bin centers. Source: [hl_gauss_censored.py](../python/namo/rl_loop/sage_ext/hl_gauss_censored.py), [weighted_module.py](../python/namo/rl_loop/sage_ext/weighted_module.py), and `../sage_learning/src/model/hl_gauss.py`.

### 7.2 Per-state listwise ordering

For the actual all-zero ceiling artifact, the set used by the ranking functions is `A_i={a:w_{ia}>0}`. For each distinct observed target level `ℓ`, define positive set `P_{iℓ}={a∈A_i:|y_{ia}−ℓ|≤10⁻⁵}` and lower set `D_{iℓ}={a∈A_i:y_{ia}<ℓ−10⁻⁵}`. Only levels with both sets nonempty contribute. The comparison list is their union, excluding higher-valued actions. At temperature `τ=0.15`, the level loss is

```math
L_{i\ell}^{\mathrm{list}}
=-\frac{1}{|P_{i\ell}|}\sum_{a\in P_{i\ell}}
\max\left(-30,\log\frac{\exp(v_{ia}/\tau)}
 {\sum_{b\in P_{i\ell}\cup D_{i\ell}}\exp(v_{ib}/\tau)}\right).
```

Levels at least 0.999 form the opener group; lower positive levels with available lower competitors form the setup group. Each group is averaged over its eligible levels within a row, then averaged over rows with a term in that group. A missing group contributes zero. For the usual `{0,0.5,1}` targets, this contrasts openers with setups and zeros, and setups with zeros. It does not force a complete permutation among equal targets. Source: [train_q2_rankaux.py:80](../scripts/rl_loop/train_q2_rankaux.py:80).

### 7.3 Family maximum-rival margin

For each family represented by at least two rows in the current batch, flatten its rows and actions into one comparison list. At each eligible target level, compare every positive with the highest-scoring lower-valued action:

```math
L_{f\ell}^{\mathrm{margin}}
=\frac{1}{|P_{f\ell}|}\sum_{a\in P_{f\ell}}
\max\left(0,\max_{b\in D_{f\ell}}v_b-v_a+0.2\right).
```

Within a family, opener and setup groups are separately averaged over eligible levels and added. The family loss averages that sum over all batch families having at least two rows, including such families that contribute zero because no valid level comparison exists. The term lets root and child scores compete on a common scale, consistent with a global queue holding actions from different states. It is not a Bellman residual and does not fit a transition model. Source: [train_q2_round2.py:80](../scripts/rl_loop/train_q2_round2.py:80), `_family_lists`, and `Round2Module._weighted_loss`.

### 7.4 Combined deployed loss and the mask overwrite

The reconstructed full HY5U recipe is

```math
L_{\mathrm{train}}=L_{\mathrm{HL}}
 +0.1L_{\mathrm{open}}^{\mathrm{list}}
 +0.05L_{\mathrm{setup}}^{\mathrm{list}}
 +0.1L_{\mathrm{family}}^{\mathrm{margin}}.
```

Other ranking switches are disabled: `MM_LAMBDA=0`, `EG_LAMBDA=0`, and `XB_LAMBDA=0`; `EGMM_LAMBDA=0.1`, margin 0.2, and `UNREACH_W=0.1` are printed in the original train logs. The 0.1/0.05 listwise coefficients and temperature are reconstructed from the full-loss source defaults and matching ablation recipe; the checkpoint hyperparameter dictionary does not preserve every environment variable from the original launch. This limits exact command provenance even though the relevant loss files were checked against the evaluated source revision.

The intended separate rank mask is assigned in `Round2Module.on_after_batch_transfer`. However, when `ceiling_mask` is present, `WeightedClassifierModule._split_loss` assigns `_rank_list_mask=loss_mask` before dispatching `_weighted_loss`. This happens even when every ceiling value is zero. The ranking functions then test `mask>0`, so an unreachable action with regression weight 0.1 participates as a full lower-valued competitor. The family term reads the same overwritten mask. Therefore “unreachable actions are regression-only negatives” is not an accurate description of the executed HY5U objective. It also means the no-unreachable ablation changes both regression supervision and ranking competitor membership; its effects cannot be attributed solely to the regression term. Relevant source: [weighted_module.py:42](../python/namo/rl_loop/sage_ext/weighted_module.py:42), [train_q2_round2.py:175](../scripts/rl_loop/train_q2_round2.py:175), and [train_q2_rankaux.py:159](../scripts/rl_loop/train_q2_rankaux.py:159).

### 7.5 Dormant censored loss and validation

For a genuinely censored cell with upper bound `c_a`, the implemented optional loss is `−log P(v≤c_a)`, where categorical mass below the bound is summed with fractional coverage of the intersected bin. Probability is clamped below at `10⁻⁸`. This group is separately normalized by its censored-mask sum and multiplied by `NAMO_CENS_WEIGHT`, default 1. Such cells are excluded from exact categorical regression. This is an upper-bound likelihood, not Gaussian regression to the bound. Because the HY5U hybrid ceiling mask is identically zero, this term is identically inactive for the evaluated training artifact.

Validation computes the masked categorical regression and any active censor term, without the listwise or family auxiliary terms; here censoring is absent. Unreachable regression weights remain included. The saved “best” checkpoint therefore minimizes validation regression loss rather than the complete training objective or a simulator-call metric. Headers or older code paths mentioning binary classification, focal loss, diffusion, or behavior cloning do not describe this active branch.

## 8. Training recipe and checkpoint identity

| Item | Verified HY5U setting or artifact |
|---|---|
| Training rows / validation rows | 1,172,394 / 130,265 |
| Seeds | 1, 2, 3 |
| Epochs | 12; best checkpoints all at zero-based epoch 11 |
| Saved global step | 54,948 for all three best checkpoints |
| Batch construction | Family-packed, nominal maximum 256 rows |
| Optimizer | AdamW, base learning rate 0.0003, weight decay 0.01 |
| Schedule | Linear warmup 200 steps; cosine schedule over 100,000 subsequent steps, floor learning rate 0.000001 |
| Precision | `16-mixed`, one selected device |
| Scalar head | 51 bins over `[0,1]`, Gaussian width 0.75 bin |
| Root resampling | `ROOT_FRAC=0` |
| Checkpoint selection | Minimum validation loss; top checkpoint plus last checkpoint |
| Early stopping | Disabled in this launch recipe |
| Data augmentation | None in the inspected Q2 loader |

The implemented learning-rate multiplier is linear during warmup and then the maximum of the floor ratio and the cosine value, with cosine progress capped at one. Training ends before 100,000 decay steps, so the learning rate has not traversed the entire configured cosine schedule. The original training command and a complete source commit are not embedded in the checkpoint; logs, hparams, source comparison, and exact checkpoint hashes together are the available provenance.

All best checkpoints are under `/common/users/dm1487/scratch_namo/aquaman/round0/models/`:

| Seed | Relative checkpoint path | SHA-256 |
|---|---|---|
| 1 | `HY5U_s1/checkpoints/epoch011-val_loss0.3213.ckpt` | `ac43f004e720cede58115175e585bfb4d9a7f61087e630116593035be474d20c` |
| 2 | `HY5U_s2/checkpoints/epoch011-val_loss0.3256.ckpt` | `3cf348cf7ba247f2cb143376371fc06771665793783d12e3b37bf596e0e5a854` |
| 3 | `HY5U_s3/checkpoints/epoch011-val_loss0.3240.ckpt` | `c596b09b254d849c8464e511cccdfcacec81ae588251eede37616d22ef695285` |

The H5's content hash was not computed in this audit; its path, schema, row count, split logs, and exhaustive zero-ceiling scan were checked. Source: [train_q2.py](../scripts/rl_loop/train_q2.py), [train_q2_round2.py](../scripts/rl_loop/train_q2_round2.py), and the `train.log`/hparams artifacts beside these model directories.

## 9. Global best-first search

### 9.1 Queue and priorities

The canonical learned search uses a single priority queue containing unexecuted actions attached to potentially different parent states. Each candidate stores a parent snapshot, the push-chain prefix reaching it, a board identity for pruning, and the proposed action. At a new state, the ranker scores the full contact/duration grid once for the supplied object; reachability and allowed-object filters determine which actions enter the queue. Candidates are ordered by their own `v_θ(s,a)` value. The canonical `combine=q` mode does not multiply scores along the path, add a path-length cost, or discount by depth.

The queue uses negative priority and an insertion counter for stable ordering. A geometric comparison option adds a preference for deeper chains on ties; that option should not be generalized to all learned/random results. State summaries such as a mean of the top five action scores are computed for some optional combination modes, but the canonical direct-q combination does not use that summary to change the action priority. There is no fixed-width beam cutoff in this implementation despite reuse of planner/container names elsewhere.

The random baseline assigns an independent random priority to each generated feasible candidate, including candidates generated at child states, and uses the same queue procedure. It is not merely a one-time permutation of the root grid. Its random generator seed is part of the evaluation protocol. It skips learned inference.

### 9.2 Pseudocode for the evaluated search family

```text
SEARCH(s0, object o, fixed verifier G, horizon H, simulator budget B):
    Q := empty global priority queue
    jam := empty per-board contact-prefix cache
    calls := 0

    EXPAND(s0, empty prefix, root board id)
    while Q is nonempty and calls < B:
        candidate := pop highest priority from Q
        (s_parent, prefix, board, a=(o,e,d)) := candidate
        if jam cache excludes this duration at this board/contact:
            continue
        restore s_parent                         # restore at rest
        result := env.step(a)
        calls := calls + 1
        if V_G(current state):
            return prefix + [a], calls, verified success
        if result has a nonempty failure_reason:
            record shortest failed duration for this board/contact
        if robot and pushed-object poses are unchanged within 1e-6:
            continue
        if len(prefix) + 1 < H:
            s_child := snapshot current state
            EXPAND(s_child, prefix + [a], fresh board id)
    return failure under this horizon/budget/pruning policy

EXPAND(s, prefix, board):
    restore s
    obtain reachable target-object contacts
    render current state and score all 60 x 5 actions
    restore s after queries/rendering
    for each available, reachable, allowed contact/duration action:
        priority := expected model value, or seeded random priority
        push (priority, tie key, insertion id, s, prefix, board, action) into Q
```

The pseudocode abstracts implementation bookkeeping and optional initial-state checks. Current full-NAMO callers can request `require_push=True`; the canonical local population is an initially blocked opening task. Success is checked after a simulator trial even if the low-level push reports failure. Generating and filtering candidates, rendering, restoring, checking the opening predicate, and skipping a queued jam-pruned candidate do not increment the push-simulation counter. A trial that fails or makes no motion still consumes one call.

### 9.3 Pruning details and limits

The no-op check compares only three pose components of the robot and three of the pushed object, using absolute tolerance `10⁻⁶`. It does not compare all other movable objects, velocities, controls, or wrapped angular distance. It is a cheap implementation predicate, not a proof that the full physical scene is unchanged. This differs substantially from the collection setup-motion threshold and from real-robot progress thresholds.

The duration-prefix jam cache is updated by **any nonempty `failure_reason`**, not only an object-stuck reason. At that board/contact, an equal or longer duration can subsequently be skipped. Evaluated single-object code keys this by `(board,edge)`; current multi-object-capable code includes the object identity `(board,object,edge)`. The single-object evaluated restriction is relevant to the safety of the older key. Neither version establishes that every failure type is monotone in requested duration.

There is no global visited-state table or general transposition deduplication. A finite action library and finite horizon define a finite unpruned search tree, but the actual budget and pruning rules prevent a completeness guarantee. The first verified solution is returned; it need not minimize push count, travel distance, work, or simulator calls. “Verified” means satisfying the configured sampled reachability criterion in simulation. It does not imply a collision-free hardware execution guarantee.

`H=2` bounds each proposed solution chain at two pushes. A budget of 900 or 4,000 is a cap on simulator trials exploring many alternatives, not a physical plan of that many pushes. Failure under the cap does not certify that the episode is unsolvable. Source: [best_first_search.py](../python/namo/planners/opening/best_first_search.py), especially `rank_first_pushes_h2`, `solve_scene`, priority combination, no-op checks, and jam bookkeeping; use the recorded evaluation revision for exact reproduction.

## 10. Direct policy, simulator rollout, and hardware feedback

The pure policy path calls the ranker once on the current observation and returns the highest-ranked feasible allowed action with `simulate=False`. It does not call `env.step`, test simulated motion, screen alternatives with the local opening predicate, or spend the local simulator budget. Its output is an **unverified argmax action**. A zero simulator-call count does not imply zero rendering, inference, or planning time.

This must be distinguished from `simulate=True` greedy commit, which can simulate candidates and skip no-motion outcomes before committing to a moving action, and from offline reactive-policy evaluation, which actually executes the selected action in simulation, checks success, and then rerenders/rescores the resulting state. The latter is a closed-loop policy rollout; its simulator calls are execution trials, not a tree search over restored alternatives. A rollout allowance of `K=30` is not a model horizon input, and beyond two actions it is not depth-matched to an `H=2` search.

Current production's `run_reactive` name should not be read as proof that it runs an internal verified rollout: the inspected path returns a policy decision without simulator stepping. The external controller owns physical execution, new perception, and replanning. `FullNAMOPlanner` selects pure policy via `exec_mode="greedy_policy"`; other greedy modes can invoke simulation. Source: [best_first_search.py](../python/namo/planners/opening/best_first_search.py), `run_greedy_commit` and `run_reactive`, and [best_first_region_opening.py](../python/namo/planners/opening/best_first_region_opening.py).

On hardware, the external robot-control system compares observed object pose before and after a push. Current defaults classify progress as inadequate when translation is below 2 cm and rotation below 15 degrees. Failed `(object,edge)` contacts are blacklisted across replans; successful motion clears entries for moved objects, and a task reset clears the history. Geometrically blocked contacts are recomputed and unioned with this failure blacklist. The bridge maps real object identities to simulator identities and passes the contact exclusions into NAMO. The local wrapper removes all five durations of an excluded contact. No history channel is added to HY5U's image; history changes candidate availability outside the network.

These are current robot-control behaviors, not a claim that every historical trial used identical thresholds. The inspected `easy020formal_v2/hmax2` configuration uses seed-2 HY5U in pure-policy mode, with a maximum of 20 replans and one planning retry. Five recorded policy trials and five uniform-search trials succeeded. Their median recorded planning times are 0.299698 s and 17.800397 s, respectively, approximately 59.4× apart; these exclude a separately recorded warmup around 3.933 s. This is a single hardware scene/configuration comparison, not a general speedup across platforms or difficulties. The hard019 hardware claim was not available for verification in the remote artifacts and remains unresolved. Hardware task success is the controller's eventual goal-reaching outcome, not merely the local 20%-of-samples predicate.

Robot-control evidence is in the adjacent repository's `src/robot_control/controller/config.py`, `controller/push.py`, `planner/namo_planner.py`, and `planner/namo_bridge.py`. Keep robot-control revision and saved trial configuration with any hardware methods claim.

## 11. Full NAMO composition

The full planner uses a free-space region graph to choose a local boundary to open, invokes the local primitive ranker/search on that boundary, commits the resulting action(s), and recomputes topology. Initial route proposals use shortest region paths, with ranking based on hop count and accumulated attempt counts plus deterministic tie fields. HY5U does not itself select the global region route or represent a global multi-object action sequence.

Current code around boundary selection pools eligible blocking objects on the chosen boundary. A local registered RO episode still supplies one blocker; the current pooled-boundary production path can score candidates for multiple eligible objects and compare them in the local queue. This newer composition should not be retroactively substituted for single-object canonical RO evaluation. Optional joint-blocker region-graph behavior and the September boundary-group pilot are separate extensions.

Budget scope can be `full_problem`, the default, or per `keyhole`. In the former, local solves share the remaining problem budget; in the latter, a local boundary can receive a renewed budget. A particular experiment's reset-to-900 policy is therefore a configuration choice, not an invariant of full NAMO. The maximum iteration count is another independent setting. On the final boundary, current full NAMO uses XML-goal reachability rather than only the generic local sampled-region predicate. Optional preservation of access to the next keyhole is disabled by default and must be named when enabled.

After committing an opening, the planner can select a new boundary and update attempt counts, but it does not maintain a global rollback tree over previously committed openings. A locally good opening can therefore make a later boundary difficult or inaccessible. The ranker is a local heuristic within this composition, not a complete global NAMO solver with backtracking guarantees. Source: [full_namo_planner.py](../python/namo/planners/full_namo/full_namo_planner.py), including graph path construction, boundary cost/attempt ordering, budget scope, current blocker pooling around line 694, and final-boundary predicate around line 754.

The September 6 controlled full-NAMO set has 40 scenes in four source-overlap groups, 10 each. Those group names are not interchangeable with local easy/medium/hard tiers. Under the reported tight call criterion, seed-2 HY5U solved 10/8/9/7 across the groups compared with random means 3.7/5/1/2; final counts were 10/8/10/10 versus 9/10/7/7. These artifacts support the value of the composed local ordering on that set, while retaining failures due to global commitment. They do not establish exhaustive global planning or a new universal difficulty taxonomy.

## 12. Evaluation populations and result interpretation

### 12.1 Canonical populations and protocols

| Population/protocol | Size and split | Interpretation |
|---|---|---|
| Fixed-physics v3 one-push | 1,328 episodes, 997 rooms; easy/medium/hard 681/442/205 | Exhaustive-evaluation-defined direct-openers population |
| Fixed-physics v3 pure two-push | 992 episodes, 958 rooms; easy/medium/hard 387/487/118 | Same local pair/blocker requires setup then finish under the registered ground-truth protocol |
| Common nonempty-target one-push subset | 1,310; tiers 675/437/198 | Excludes 18 empty-target episodes for matched comparison |
| Common nonempty-target two-push subset | 973; tiers 381/475/117 | Excludes 19 empty-target episodes for matched comparison |
| Main HY5U v3 evaluation | `H=2`, 900 calls, direct expected-score priority, no discount, no-op and jam pruning | Three model seeds and matched random seeds |
| Corrected timing comparison | 4,000 calls, common 1,310/973 population, recorded tie rules and same CPU class | Separate protocol; cannot silently replace 900-call figures |

Canonical definitions are in [eval_sets.yaml](../config/eval_sets.yaml) and [eval_set_registry.md](experiments/eval_set_registry.md). The underlying v3 source has 2,541 episodes across 1,829 rooms; the pure-two-push source had 995 episodes before three with zero ground-truth setups were excluded. The exhaustive test construction is an evaluation luxury, not the training acquisition strategy. Earlier v1/v2 ground-truth collections had truncated-sweep issues and are not interchangeable with v3.

The evaluated-artifact registry is [horizon_q_model_registry.md](experiments/horizon_q_model_registry.md), section “Canonical evaluated-model artifacts.” Despite its historical filename, the current model is not horizon-conditioned. Canonical HY5U aggregates are under `/common/users/dm1487/scratch_namo/eval/fixed_physics_v3_20260821/full/HY5U_s{1,2,3}/aggregate.json`, with per-episode outputs in the `1push_hmax2` and `2push` subdirectories and analogous random arms. The corrected timing reduction is `/common/users/dm1487/scratch_namo/aquaman/round0/eval_walltime4k/geometric_region_corrected_v1/comparison_common.json`. Saved arguments and common-population membership matter more than a stale directory label.

### 12.2 Representative checks and what ablations support

The main v3 results report one-push solve@1 of 97.1/79.8/40.2% for HY5U versus 61.1/14.1/2.9% for random on easy/medium/hard. On pure-two-push episodes, solve@5 is 80.6/59.3/35.9% versus 22.8/7.2/2.0%. These are tiered simulator-call metrics, not standalone classification accuracy. Difficulty and one-/two-push splits should remain visible whenever these results are summarized.

| Ablation | One-push solve@1, easy/medium/hard (%) | Two-push solve@5, easy/medium/hard (%) |
|---|---|---|
| Full HY5U | 97.1 / 79.8 / 40.2 | 80.6 / 59.3 / 35.9 |
| No family term | 96.7 / 78.7 / 41.3 | 80.5 / 59.8 / 36.4 |
| Regression only | 97.6 / 80.1 / 32.4 | 57.8 / 42.0 / 24.3 |
| No unreachable supervision | 96.7 / 78.9 / 38.5 | 63.4 / 47.8 / 25.7 |
| Independent contact scoring | 96.2 / 76.3 / 35.4 | 76.7 / 52.8 / 30.5 |
| No local sampled feature | 97.2 / 80.3 / 38.7 | 79.5 / 58.3 / 33.6 |
| Global representation ablation | 76.5 / 50.2 / 14.8 | 57.8 / 36.6 / 17.0 |
| No learned edge embedding | 96.7 / 77.6 / 33.5 | 79.1 / 58.0 / 31.1 |

These comparisons support the utility of ranking supervision and the unreachable-action treatment for the tested objective/protocol. They do not show that every component is indispensable: removing the family term gives similar or slightly higher entries in some tiers, and removing local feature sampling has a modest effect. The mask-overwrite finding changes the causal reading of the no-unreachable arm. There is no basis here for claiming that active censored learning explains HY5U's results, or that every listed architecture difference isolates a single theoretical mechanism.

In the corrected common-population timing reduction, “3 versus 32 simulations” for two-push cases denotes the median across episodes of each episode's median across three seeds. Pooling all solved seed-episode records instead gives a different random median (30); averaging per-seed medians gives another value. The reduction and handling of failures must accompany the reported number. Wall times from different machines must not share an axis; simulator calls are the cross-machine comparison unit. The timing runs used the recorded common Cascade Lake CPU setting.

The August policy comparison uses a 30-action allowance on the common 1,310/973 population. Pure-two-push solve@30 for policy/search is 90.2/94.8% easy, 81.0/89.1% medium, and 67.0/75.5% hard. At small budgets policy can be competitive, while longer search preserves alternatives through restoration. After two physical actions, the policy rollout is not depth-matched to `H=2` search, so this is an operational budget comparison with different execution semantics.

The September 7 policy/group ablations were **completed**, not pending, by the latest inspected NAMO HEAD. The card records 21 new checkpoint arms across both policy legs on the exact cached 1,310/973 controls; reduction excludes newly eligible extras introduced by an additive adjacency change. Aggregates are under `/common/users/dm1487/scratch_namo/eval/policy_group_ablations_20260907/aggregate.{json,md}`. The 24-scene boundary-group pilot comprises 16 alternative-blocker and eight joint-marker cases with 288 tasks; it is exploratory and does not by itself prove that all marked scenes require coordinated pushes on multiple objects. Source: [September 7 card](experiments/log/EXP-2026-09-07-policy-and-boundary-group-ablations.md), [RESULTS.md](experiments/RESULTS.md), and the evaluated-artifact registry.

## 13. Reproducibility decisions and unresolved author questions

Before turning this reference into manuscript prose, the author should make the following choices explicit:

1. **Name the evaluated implementation.** State the checkpoint family, evaluation revision, target population, margin configuration, horizon, call budget, pruning rules, priority combination, and seed reduction. Do not describe a dirty production checkout as the implementation of an older result.
2. **Describe the actual loss.** The defensible HY5U methods equation is weighted categorical regression plus state-listwise and family-margin terms. Censored likelihood can be described as unused infrastructure or historical development, not an active contribution of these trained checkpoints.
3. **Acknowledge the unreachable-mask behavior.** Decide whether the manuscript reports the executed behavior as part of the method or labels it an implementation discrepancy. Correcting it would define a different training experiment; this audit has made no such change.
4. **Resolve goal-conditioning equivalence.** The canonical evaluator supplies XML/fallback goal conditioning to the model while verifying fixed sampled regions. Establishing equality on all evaluated episodes would require a separate audit. Until then, explain the actual two inputs rather than asserting they coincide.
5. **Resolve family identity.** The conceptual episode includes goal region; the H5 and grouped sampler use only `(xml,object_id)`. Confirm that this key does not join distinct goal episodes in the relevant source data, or state the limitation.
6. **Distinguish hard zeros from exhaustive negatives.** Training zeros can arise from bounded unsuccessful exploration. Do not imply exhaustive ground truth or impossible-action certificates at training scale.
7. **Keep policy and search accounting separate.** Pure policy has zero internal simulator trials; a simulated rollout or physical push is an execution. Search counts restored hypothetical trials. Specify which clock/counter each figure measures.
8. **Bound architecture claims.** The current observation is 64×64 over 0.5 m and uses 60 contact tokens with five output durations. Crop sufficiency, discretization saturation, and an isolated advantage of the categorical head over a matched scalar-loss alternative are not established by the inspected evidence.
9. **Bound full-NAMO and hardware claims.** Global boundary selection is heuristic, committed openings are not globally backtracked, and current pooled-blocker behavior differs from canonical local RO. Preserve real-controller revision/configuration and do not use the unverified hard019 claim.
10. **Complete command provenance if available.** Exact checkpoint hashes are recorded, but original launch environment variables and a full training-source SHA are not completely preserved in checkpoint metadata. A reconstruction should identify inferred defaults rather than present them as a recovered original command.

No external algorithm was reimplemented or benchmarked during this audit. The geometric prior is a repository baseline with its own documented formula and tie policy, not an automatically faithful reproduction of a named classical planner. Historical diffusion and horizon-Q documents should not determine the description of this single ranker. The strongest supported methods claim is that a learned, state-conditioned ordering over executable contact-duration pushes guides a simulator-verified local search, with measured savings over a matched random ranker across the registered difficulty and push-depth splits.

## 14. Source map for manuscript verification

| Topic | Primary source |
|---|---|
| Task and scope | [problem_and_approach.md](problem_and_approach.md) |
| Training architecture factory | [train_gen.py](../python/namo/rl_loop/train_gen.py) |
| Transformer / histogram head | Adjacent Sage `src/model/dit/edge_crossattn.py`, `src/model/hl_gauss.py` |
| Live crop and contact input | [live_scorer.py](../scripts/sandbox/live_scorer.py), [eval_scorer.py](../scripts/eval_scorer.py) |
| Raster implementation | Adjacent Sage `sage_learning/visualizer.py`, `generate_all_masks_highres` |
| Primitive action library | [primitive_goal_strategy.py](../python/namo/strategies/primitive_goal_strategy.py) and backend push skill/controller |
| Hybrid artifact construction | [build_hybrid_h5.py](../scripts/pipeline/build_hybrid_h5.py) |
| Sampled labels / child selection | [build_rung2_h5.py](../scripts/pipeline/build_rung2_h5.py), [arjuna_build_v2.py](../scripts/rl_loop/arjuna_build_v2.py) |
| Dataset split and masks | [q2_dataset.py](../python/namo/rl_loop/sage_ext/q2_dataset.py) |
| Shared regression/censor dispatch | [weighted_module.py](../python/namo/rl_loop/sage_ext/weighted_module.py), [hl_gauss_censored.py](../python/namo/rl_loop/sage_ext/hl_gauss_censored.py) |
| State ranking | [train_q2_rankaux.py](../scripts/rl_loop/train_q2_rankaux.py) |
| Family ranking / batch sampler | [train_q2_round2.py](../scripts/rl_loop/train_q2_round2.py) |
| Best-first / pure policy | [best_first_search.py](../python/namo/planners/opening/best_first_search.py) |
| Production RO wrapper | [best_first_region_opening.py](../python/namo/planners/opening/best_first_region_opening.py) |
| Evaluator and fixed-point predicate | [eval_bestfirst.py](../scripts/sandbox/eval_bestfirst.py), [eval_m3.py](../scripts/sandbox/eval_m3.py) |
| Full-NAMO composition | [full_namo_planner.py](../python/namo/planners/full_namo/full_namo_planner.py) |
| Canonical protocols/artifacts | [eval_set_registry.md](experiments/eval_set_registry.md), [horizon_q_model_registry.md](experiments/horizon_q_model_registry.md) |
| Results and completed cards | [RESULTS.md](experiments/RESULTS.md), [September 7 policy/group card](experiments/log/EXP-2026-09-07-policy-and-boundary-group-ablations.md) |

Source line numbers refer to the checkout inspected during preparation and can move as other tasks edit the repository. For historical results, inspect the registered commit plus saved argument/artifact files rather than treating current line numbers as immutable provenance.
