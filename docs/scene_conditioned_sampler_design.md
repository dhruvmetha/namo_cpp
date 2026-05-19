# Scene-Conditioned Action Sampler for NAMO

**Status:** design + findings synthesis. Active research doc, May 2026.
**Predecessors:**
- `docs/ML_vs_GT_F_evaluation.md` — original evaluation plan
- `docs/ML_vs_GT_F_results_round1.md` — first findings on 2-push diffusion
- `docs/research_notes_F_characterization.md` — F characterization research thread

This doc supersedes and integrates them.

---

## 1. Problem framing

We want **one scene-conditioned model** that, given a scene (5-channel local masks), produces a useful push primitive — for 1-push problems, 2-push problems, and (eventually) N-push problems.

"Useful" = the primitive either opens the passage directly (∈ F₁) or sets up a successful subsequent push within ≤N steps (∈ F₁' ∪ F₁'' ∪ ...).

The model is queried recursively by the planner: at each state encountered during planning, the model proposes primitives; the planner tries them in priority order. For chains, the planner queries the model again at the post-push state to pick the next push.

This is **action-value/policy learning over a discrete primitive grid**, trained from exhaustive simulator labels. Not RL (we have an oracle). Not classifier-only (we want sampling). Not diffusion-only (action space is discrete).

---

## 2. Data inventory

### What we have

| dataset | scope | format | use |
|---|---|---|---|
| Chain-1 F-char train | 5,925 envs, 11,617 instances | exhaustive per-cell F₁ labels | V1b training |
| Chain-1 F-char test | 1,767 envs, 3,605 instances | same | V1b test set |
| Chain-2 F-char (300-env) | 300 envs, 342 instances | full F₁ + F₁' via parent pointers | 2-push test set |
| Diffusion training h5 | 8,716 envs, 167,965 rows | (state, action) pairs from planner solutions | existing diffusion model training |
| ML preds (diffusion on rlab7) | 3,474 instances | aligned (edge, depth) per scene | reusable for any test |
| ML preds (diffusion on 300-env) | 284 instances | same | reusable for 2-push test |

### What we don't have

- Chain-2 F-char on **training** envs (only on the 300-env test split).
- Chain-3+ F-char anywhere.
- Cross-domain data (different robot, different env distribution).

### Trial-log patch (key dependency)

`region_opening.py` was patched to record `chain_depth`, `parent_edge_idx`, `parent_depth` per trial log row. This makes F₁' recoverable from chain-2 GT collection: scan chain_depth=2 success rows, collect their parent (edge, depth) tuples → that's F₁'.

Without the patch, chain-2 data would be unrecoverable for F₁' computation. Patch is in `car-baseline` branch.

---

## 3. Metric definitions

Two axes of choice in how we score a model's predictions.

### Axis 1: joint vs contact

- **Joint hit @ K**: top-K of (edge, depth) primitives intersected with F. Strict: exact primitive must be in F.
- **Contact hit @ K**: top-K of *edges* (ignoring depth) intersected with F's edge support. Forgiving: any depth at that contact counts. Matches how the planner operates — it can iterate depths cheaply at a chosen contact.

The contact metric is more operationally meaningful for 1-push, where the planner can cheaply scan depths at a chosen contact point. For chains, depth matters more (different depths produce different post-states), so joint becomes more relevant.

### Axis 2: oracle vs realistic

A primitive can fail two ways at runtime:
- **Contact-unreachable**: robot can't reach the edge (known *before* simulation via wavefront BFS).
- **Depth-stuck**: object hits a wall before completing the push (only known *during* simulation).

**Oracle metric**: pre-filter the model's top-K to primitives that are BOTH contact and depth reachable. Effectively gives the model a free pass on depth-precision errors.

**Realistic metric**: filter only by contact-reachability. Picks that get depth-stuck count as wasted attempts (one slot of the K-budget used, no F-hit). This is what the planner actually experiences.

The gap between oracle and realistic = cost of the model's depth-precision errors at deployment.

**Random baseline**: closed-form `1 - C(|R| - |F|, K) / C(|R|, K)` per instance, averaged per bucket. Where `|R|` and `|F|` are the appropriate counts (joint or contact-level) for each metric.

---

## 4. Difficulty bucketing

Per-instance difficulty defined by `|F₁| / |R₁|` ratio at chain depth 1:

| bucket | F/R range | typical avg |F| | typical avg |R| |
|---|---|---:|---:|
| very_hard | < 5% | 3.4 | 115 |
| hard | 5–15% | 11.5 | 112 |
| medium | 15–40% | 40.2 | 138 |
| easy | 40–70% | 105.7 | 185 |
| very_easy | > 70% | 160.9 | 188 |

Pure-chain instances (F₁=0, F₁'>0) by definition fall in `very_hard` (their ratio is 0).

---

## 5. Empirical findings

### 5.1 The biased-teacher problem (diffusion's 1-push failure)

The 2-push diffusion model (`cropped_diffusion_crossattn_2push/2025-12-16/05-36-44`) was trained on `2_push_train_corrected_overlaps_2.h5`. Direct measurement of that data:

- **97% of training-target object displacements are under 1m**
- Median displacement: 0.30m
- p90 displacement: 0.67m
- 69% of rows are 1-push solutions, 31% are 2-push solutions — all from planner's first/shortest successful primitives

Reason: data was collected by the BFS-shallowest planner. Each row records the *first* primitive the planner found that worked. Plus smoothing biases further toward minimal displacement.

Effect at inference:
- Model's predictions are 95%+ at depths 0-2.
- 0 predictions out of 5,000+ at depths 7-9 across all buckets.
- On hard problems where F₁ requires deep pushes (d ≥ 7-9), model can never hit F₁.

### 5.2 V1b classifier results (1-push F₁)

Architecture: DiT, 5-channel 64×64 input, per-cell sigmoid over 60×10. Training: F-char chain-1 labels with `pos_weight=5 + focal_loss + Dice`. Best ckpt: `epoch018-val_loss0.5517`.

**Realistic Top-1 (test on 3,605 instances):**

| difficulty | V1b | random | lift |
|---|---:|---:|---:|
| very_hard | 5.1% | 3.0% | +2.1pp |
| hard | 14.5% | 10.2% | +4.3pp |
| medium | 33.6% | 28.7% | +4.9pp |
| easy | 57.7% | 56.6% | +1.1pp |
| very_easy | 89.2% | 86.9% | +2.3pp |

Strictly positive everywhere. Small magnitudes (2-5pp). First learned model in the codebase to clear "above random on every bucket."

**Contact Top-1 (same test):**

| difficulty | V1b | random | lift |
|---|---:|---:|---:|
| very_hard | 28.2% | 7.7% | **+20.6pp** |
| hard | 47.9% | 16.0% | **+31.9pp** |
| medium | 75.8% | 33.2% | **+42.6pp** |
| easy | 92.0% | 58.5% | **+33.5pp** |
| very_easy | 98.9% | 90.6% | +8.3pp |

Much larger lifts at contact level. The model knows which side of the object to push; depth precision is the weakness.

### 5.3 Diffusion on 1-push (realistic-ish)

On the same rlab7 1-push test, diffusion's joint hit@1:

| difficulty | diffusion | random | lift |
|---|---:|---:|---:|
| very_hard | 3.3% | 3.5% | -0.2pp |
| hard | 9.1% | 11.0% | -1.9pp |
| medium | 20.8% | 28.9% | **-8.1pp** |
| easy | 46.2% | 56.7% | **-10.5pp** |
| very_easy | 77.8% | 86.7% | **-8.9pp** |

Diffusion is below random on every bucket at joint level on 1-push. Caused by the shallow-displacement training bias.

Contact level on 1-push: diffusion is positive everywhere (+15-19pp). The directional signal exists; it's the depth that fails.

### 5.4 Diffusion on 2-push F₁' (the surprise)

Same diffusion model, evaluated on chain-2 GT (300-env, 284 instances after dedup), scored against F₁' (chain-enabling push-1s):

**Pure-chain subset (n=92, F₁=0):**

| K | V1b (realistic joint) | Diffusion (realistic joint) | Random |
|---|---:|---:|---:|
| 1 | 10.9% | **41.2%** | 13.8% |
| 3 | 15.2% | **66.2%** | 30.2% |
| 5 | 17.4% | **71.2%** | 41.5% |
| 10 | 30.4% | **87.5%** | 59.2% |

Diffusion beats random by **+27-36pp on every K** at the realistic level. V1b is **below random** at every K on pure-chain.

The reason: diffusion's training distribution (shallow displacements) accidentally matches F₁' (chain-enabling setup pushes are typically shallow). What looked like the diffusion model's *fatal flaw* on 1-push is its *core strength* on 2-push.

### 5.5 The "right tool for the right horizon" finding

The 1-push and 2-push results point to a clean specialization:

| horizon | task | best model | why |
|---|---|---|---|
| 1-push | predict F₁ (direct opener) | **V1b** | trained on F₁ labels; contact prediction is strong |
| 2-push | predict F₁' (chain enabler) | **Diffusion** | training distribution matches F₁' depth profile |

Neither model is "better" universally. Each matches the distribution of its task.

### 5.6 Why depth matters differently across horizons

For **1-push at a correct contact point**: F's depths form a contiguous band (96-100% of cases per F-char analysis). The planner can iterate depths at a chosen contact via cost-first BFS — once the model picks the right contact, finding a working depth is cheap. Wrong-depth picks get stuck physically but the *next* depth iteration recovers.

For **2-push push-1 at a correct contact point**: different depths produce different *post-states*. The push-2 reachable set, the wall geometry around the displaced object — all change with depth. Wrong push-1 depth = wrong subtree = wasteful re-expansion of push-2 search from a fresh state.

So depth-imprecision is *cheap* for 1-push but *expensive* for chains. This is why V1b (sloppy on depth, sharp on contact) works for 1-push and fails for chains.

---

## 6. Oracle vs realistic: why we report both

| metric | what it shows | when it matters |
|---|---|---|
| Oracle | model's best-case prediction quality, ignoring depth-stuck cost | model diagnosis ("does the model have the right idea?") |
| Realistic | what the planner experiences with depth-stuck slot waste | deployment decision ("is the model operationally useful?") |
| Contact | edge-only ranking, depth ignored | most relevant for 1-push where depth iteration is cheap |

The earlier analysis over-emphasized oracle joint hit. Realistic joint and contact are the deployment-relevant numbers.

---

## 7. Architecture choice: classifier vs diffusion vs categorical policy

Three architectures considered for the unified sampler:

### Option A: Per-cell sigmoid classifier (V1b's architecture)
- Output: 60×10 sigmoid scores
- Training: BCE per cell, masked by reachability
- Sampling at inference: softmax with temperature over scores
- **Strength**: handles both positive AND negative training labels. Sample-efficient when labels are dense.
- **Weakness**: per-cell predictions are independent, no inherent multimodal mass competition.

### Option B: Continuous diffusion (current 2-push model)
- Output: SE(2) goal pose, aligned to nearest primitive slot
- Training: noise prediction on (state, goal pose) pairs from positive examples
- Sampling: native (denoise from noise)
- **Strength**: natively generative, multimodal in continuous space
- **Weakness**: precision loss in continuous → discrete alignment; only uses positive samples (wastes negative info); inference cost (multiple denoising steps)

### Option C: Categorical policy
- Output: softmax over 600 primitives
- Training: cross-entropy on (state, positive primitive) pairs
- Sampling: native multinomial with temperature
- **Strength**: discrete output matches action space; native sampling; competing-mass softmax forces commitment
- **Weakness**: cross-entropy on sparse positives over-confidently pushes down on unsampled cells (including potentially-positive ones); needs care with multi-label semantics

### Recommendation

For our setting (discrete primitives, V-guided sampling collection that yields both positive AND negative labels):

**Use V1b's architecture (per-cell sigmoid classifier) with masked BCE.** Reasons:

1. Action space is discrete — both classifier and categorical match it; diffusion doesn't.
2. V-guided sampling produces both labels (each chain attempt records success/fail) — sigmoid uses both; categorical-on-positives wastes the negative signal.
3. Inference-time sampling works the same for both classifier and categorical (softmax + multinomial). The "native sampler" framing is cosmetic.
4. Existing V1b infrastructure already trained, tested, validated.
5. Generative architectures (diffusion) lose information through discrete alignment and only use positives.

**Treat V1b's scores as a sampling distribution at inference** via softmax + temperature. Same model, just different consumer interface.

### What's NOT recommended

- Switching to diffusion as the unified model — wrong fit for discrete action space, loses negative-label information.
- Switching to pure categorical policy — wastes the negative information from V-guided sampling.
- Pure top-K greedy (V1b's current usage) — leaves diversity on the table; same model can be used as a sampler with no architecture change.

---

## 8. Data collection strategy: iterative deepening with V-guided sampling

### The bootstrap loop

```
Iteration 1:  Have F-char chain-1 data → train V_1 (= V1b architecture)
Iteration 2:  Use V_1 to bias chain-2 sampling → collect (state, primitive, success) tuples
              Train V_2 on combined chain-1 + chain-2 data
Iteration 3:  Use V_2 to bias chain-3 sampling → collect chain-3 tuples
              Train V_3 on combined chain-1 + chain-2 + chain-3 data
Iteration N:  Use V_{N-1} for chain-N sampling, train V_N
```

Each iteration extends horizon by 1. The previous model's predictions focus sampling on plausibly-successful regions, avoiding the combinatorial blowup of exhaustive search at deep horizons.

### Why bootstrap instead of exhaustive at every depth

Exhaustive F-char compute cost vs depth:
- chain-1: ~30 sec/env (manageable, ~5h cluster for 6,000 envs)
- chain-2: ~5 min/env (overnight, ~17h cluster)
- chain-3: ~30-60 min/env (~1 week)
- chain-4+: infeasible

V-guided sampling cost vs depth (with K=5 candidates per level):
- chain-N: K^N attempts/env × N sims/attempt = bounded linearly in env count
- chain-3 V-guided: 125 × 3 = 375 sims/env × 6,000 envs = 2.25M sims ≈ 2 hours cluster

Bootstrap scales; exhaustive doesn't.

### V-guided sampling produces both labels

Each chain attempt is one simulation. Success or failure is observed deterministically. So for each `(state, primitive)` actually attempted:
- success → positive label
- failure → negative label
- not attempted → unlabeled (mask out of training)

V1b's masked BCE handles this directly. No need to throw away failures (which is what diffusion-only-on-positives does).

### Why V-guided beats uniform random sampling

For pure-chain very_hard envs:
- Uniform random: P(any chain hits F) ≈ (0.05)^N. For N=3, P ≈ 0.0001. Need millions of attempts.
- V-guided: V_N has +20-40pp lift on direction at each step. P(chain hits F) is multiplicatively larger.

V_N doesn't need to be perfect for V-guided sampling to work — it just needs to be useful (better than random) at each step. V1b at contact level (+30pp lift) is good enough to seed.

### Failure case: when bootstrap fails

If V_{N-1} is BAD at horizon N-1 (e.g., we hit a depth where the model collapses), then V-guided sampling at horizon N collapses with it. Failure modes:

1. **Compounding error**: each step's model has accuracy p, joint chain has p^N. For p=0.6, N=5 → joint p=0.08. Too small.
2. **Distribution shift**: model trained on shallow-horizon states sees deeper-horizon states at inference. Distributions differ. Predictions degrade.
3. **Limited mode coverage**: if V_{N-1} commits to one mode, V-guided sampling never explores other modes. Need temperature/diversity at each level.

Mitigations:
- Periodic exhaustive sampling on a small fraction of envs to catch missed modes.
- Temperature sweep at sampling time (don't always greedy-sample).
- Train V_N on data spanning multiple horizons (chain-1, 2, ..., N), not just chain-N.

---

## 9. Concrete plan: from here forward

### Step 0 (cheap test, ~30 min, no new data)
Test V1b deployed recursively on the existing 300-env chain-2 GT:
- For each instance, query V1b on initial state → execute top-K push-1s → for each post-state, query V1b → execute top-K push-2s → check success
- Measure: success rate on the 92 pure-chain instances
- Decision trigger:
  - If V1b recursion succeeds on >60% of pure-chain → V1b is enough; skip new collection
  - If 30-60% → V1b helps; chain-2 data will tighten
  - If <30% → V1b doesn't transfer; chain-2 data is needed

### Step 1: V-guided chain-2 data collection (~5h cluster)
For each training env (~5,925), use V1b to bias chain-2 sampling:
1. V1b picks top-5 push-1 contact points from initial state
2. For each contact: cost-first BFS over depths (try shallow first, ~5 depths max before giving up)
3. For each non-trivially-failing push-1: V1b picks top-5 push-2 contacts from post-state
4. For each: cost-first BFS over depths
5. Record (state, primitive, success) for every attempt at both levels

Storage: per-env ~25-50 (state, primitive, success) tuples at chain-2. Compress and store as NPZ extensions to the chain-1 classifier_train_npz format.

### Step 2: Train V_2 (the chain-aware classifier)
- Same architecture as V1b (DiT, 5-channel 64×64, sigmoid output)
- Training data: chain-1 F-char (existing) + chain-2 V-guided samples (new)
- Masked BCE: only loss on labeled cells; unlabeled cells unconstrained
- Output interpretation: per-primitive probability of "leads to opening within ≤2 steps"

Single model, single output head. The horizon-2 information is baked into the labels, not a separate output.

### Step 3: Evaluate V_2
- Re-run pure_chain_compare.py with V_2 instead of V1b
- Measure: does V_2 beat the current diffusion model on F₁' joint realistic?
- Measure: does V_2 still beat V1b on 1-push F₁?

If V_2 substantially improves over V1b on chain prediction without regressing on 1-push, the bootstrap step worked.

### Step 4: Iterate
- Use V_2 to bias chain-3 sampling
- Train V_3 on chain-1 + chain-2 + chain-3
- Eval, decide whether to continue to chain-4+

---

## 10. What's NOT being done (and why)

### Re-training diffusion on F-direct data
- Possible (Fix 1 from earlier docs). Would fix diffusion's 1-push failure.
- Not the priority because: V_2 classifier approach is more aligned with how we want to USE the model (planner queries scores at each state).
- Diffusion remains useful for current 2-push chain prediction; can be replaced once V_2 is validated.

### Action-conditioned classifier
- Architecture where model takes specific (scene, primitive) → yes/no.
- Real architecture lever but ~3h of code + training.
- Defer until V_2 evaluation shows whether per-primitive precision is the bottleneck.

### Cross-domain generalization
- All envs from `aug9/medium`. Cross-domain (different env distributions, different robots) unmeasured.
- Defer; not the current bottleneck.

### RL / value iteration / bootstrapping in the RL sense
- Not needed. We have a simulator oracle. Supervised learning + smart sampling is strictly better than RL when the oracle is fast.
- The "iterative deepening" pattern looks like RL but is just iterated supervised learning with model-guided data collection.

### Random uniform sampling for chain data collection
- Strictly worse than V-guided sampling on hard envs (where joint success probability is tiny).
- Use V-guidance from the start; uniform random is a worse baseline.

---

## 11. Open questions

1. **Will V_2 substantially beat the current diffusion model on F₁'?** Diffusion currently +27-36pp realistic lift on pure-chain. V_2 might match if the F-char + V-guided collection covers F₁' well; might not if diffusion's continuous output has some advantage we're missing.

2. **How does V_N performance degrade with N?** Compounding error multiplies per step. Need to measure empirically at N=2, N=3 before trusting deeper-horizon predictions.

3. **Does the same V_N work at multiple horizons or do we need horizon-conditional heads?** A model trained on F₁ ∪ F₁' could be optimal for either but might be suboptimal for both. Multi-head architecture is a backup.

4. **Will V-guided sampling cover modes that V_{N-1} doesn't know about?** Risk of mode collapse. Mitigated by temperature sampling, but needs validation.

5. **What's the floor for sample efficiency in chain-N collection?** K=5 per level was a guess. Could be too small (miss positives) or too large (waste sims). Empirical tuning needed.

---

## 12. Single-paragraph summary

We're building a scene-conditioned classifier (V1b's architecture: per-cell sigmoid over the 60×10 discrete primitive grid) trained iteratively across horizons via the bootstrap pattern: V_1 from chain-1 F-char (exhaustive); V_2 from V_1-guided chain-2 sampling; V_N from V_{N-1}-guided chain-N sampling. The model uses both positive and negative labels (V-guided sampling gives both), trains with masked BCE on observed cells, and is consumed at inference as a sampler via softmax with temperature. Diffusion isn't needed — it was useful as a stopgap for 2-push F₁' prediction (where its training-distribution bias accidentally matched) but a properly-trained classifier will subsume that role. The data-collection bottleneck (exhaustive doesn't scale past chain-2) is solved by V-guided sampling, which scales linearly in env count at any horizon depth. No RL machinery needed — we have a simulator oracle and use it for direct supervised labels.

---

## Files referenced

- `python/namo/planners/opening/region_opening.py` — patched trial_log captures chain_depth + parent pointers
- `sage_learning/src/data/classifier_data.py` — V1b training data loader
- `sage_learning/src/model/classifier_module.py` — V1b architecture
- `sage_learning/src/eval_classifier.py` — V1b evaluation harness
- `python/namo/data_collection/ml_prediction_offline.py` — diffusion inference offline
- `docs/f_characterization/analyze_ml_vs_F.py` — Top-K hit rate analysis
- `/common/users/dm1487/scratch_namo/classifier_experiments/` — all session artifacts
- `/common/users/dm1487/namo_data/f_characterization/classifier_{train,test}_npz/` — V1b training data
- `/common/users/dm1487/scratch_namo/f_char_2push_test_300_chain2/` — chain-2 GT
