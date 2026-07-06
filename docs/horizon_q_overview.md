# Horizon-Q → NoHz: amortizing NAMO search into a single push-value

> **⛔ CURRENT-STATE BANNER (2026-07-06): budget/horizon-conditioning is DROPPED.** The live model is a **single value/ranker over pushes** with **no budget input, no remaining-horizon conditioning, no per-horizon heads** — the model line is called **NoHz** ("no-horizon"). Its whole job is **ranking the FIRST push (the "setup")**: telling a first push that leads to a solvable state from a dead one. Anything below that talks about a budget-conditioned `Q(s,a,H)`, per-horizon heads, or "aim a tiny query budget by remaining-H" is **HISTORICAL**, kept for the record.
> **Why horizon was dropped (measured — keep this record):** horizon-conditioned ≈ no-horizon, NoHz ahead — reactive **40.7 (NoHz) vs 34.1** for the budget/depth-conditioned value variant, and NoHz ≥ the remaining-budget "Hz" head on both reactive and best-first (2026-06-27 ledger). At ≤2 pushes the budget input has nothing to do — the second/"finish" push is already near-oracle, only the first/"setup" push is hard — so budget-conditioning would only earn its keep at ≥3 pushes.
> **Canonical current framing → [problem_and_approach.md](problem_and_approach.md).** Live results → [experiments/RESULTS.md §4](experiments/RESULTS.md); live hypothesis → [experiments/policy_value_search_hypothesis.md](experiments/policy_value_search_hypothesis.md); the positive-only bind → [research/positive_only_value_learning_litmap.md](research/positive_only_value_learning_litmap.md). This file is now the **deeper map + historical record**.

> The plain-language map of WHAT we're solving and HOW. Operational state lives in
> [experiments/horizon_q_build_journal.md](experiments/horizon_q_build_journal.md) (§9 = resume point);
> the 37-decision design spec with citations lives in
> [experiments/multipush_horizonQ_journal.md](experiments/multipush_horizonQ_journal.md).
> Status snapshot in this file: **2026-06-12** (framing since superseded — see banner).
> **All model checkpoints/numbers/eval dirs: [experiments/horizon_q_model_registry.md](experiments/horizon_q_model_registry.md).**
> **How we run experiments (the loop): [experiments/WORKFLOW.md](experiments/WORKFLOW.md)** · compiled results: [experiments/RESULTS.md](experiments/RESULTS.md).

## 1. The abstract problem

Strip away the robots and the rooms, and the problem is this:

> **A sequential decision problem with a large discrete action menu, a small move budget, and a
> ground-truth evaluator that is perfect but expensive.** Each state offers ~hundreds of candidate
> actions; only a state-dependent subset is legal; success is sparse and binary ("did it open or not");
> outcomes can only be known by querying an oracle (a simulator) that costs ~1 second per query; and at
> deployment the agent gets few or zero oracle queries per decision.

The objective: from a bounded offline budget of oracle interactions, learn a **single value / ranker over pushes**

```
V(s, a) = "how good is push a from state s" — does it open the region now, OR set up a state a later push can finish?
```

with **no budget input and no remaining-horizon conditioning** — one head, the **NoHz** ("no-horizon") line. One function supports BOTH deployment regimes — **act with zero queries** (rank by V, execute the top push) and **aim a tiny query budget** (use V as prior + leaf evaluator for shallow search). This is amortized search: the tree the oracle explored at training time, compressed into a forward pass.

> **Why not condition on remaining budget H (the old `Q(s,a,H)` framing)?** We built it and measured it: horizon-conditioned ≈ no-horizon, NoHz ahead (reactive 40.7 NoHz vs 34.1 for the budget/depth-conditioned value variant; NoHz ≥ the remaining-budget "Hz" head on both regimes). At ≤2 pushes the budget input has nothing to do — the finish push is already near-oracle, only the setup push is hard — so budget-conditioning would only earn its keep at ≥3 pushes. **Dropped;** see the top banner.

Three structural facts shape every design choice, and none is NAMO-specific:
1. **Legality is state-dependent and known** — a per-state candidate subset A(s) (here: reachability), so evaluation and pooling always happen over the candidate set. Reachability ("robot can't reach that contact") is a KNOWN negative — handled by a pre-filter + an input channel (the reachable-region map), **not learned**.
2. **Labels cost oracle queries** — so supervision is SAMPLED with known masks (never exhaustive beyond the cheapest tier), and the loss only scores what was actually tried.
3. **"Didn't find an opening" = UNKNOWN, not a certified dead-end** — a push we tried-or-could-try but never found a finish for is masked (no loss term), never labeled 0; forcing untried pushes to 0 is the measured "C15 poison" (false negatives suppress valid pushes). The exception: a *single* push's reachable options are ENUMERATED (cheap), so "no push opens this scene" IS a certified negative (the dead rows). The bottleneck is telling a good FIRST push (setup) from a dead one under exactly this positive-only bind.

## 2. The problem setting (the concrete instantiation)

**Region Opening (RO) for Navigation Among Movable Obstacles.** A 7 cm differential-drive car in a walled room must reach a goal region whose access is blocked by ONE movable object. An episode is the triple **(scene, target object, goal region)** — the same room hosts many episodes with different objects/regions, which is why every invariant in the pipeline keys on the episode, never the room.

- **State:** SE(2) poses of the robot and all movables in the room. The model sees a 5-channel object-centered crop (64×64, from a 0.5 m window): static walls · movables · the target object · the robot's wavefront-reachable region · the goal-region sample area.
- **Action space:** a = (contact point e, push depth d) with e ∈ 60 discrete points around the object perimeter (4 faces × 15) and d ∈ 5 depths → **300 cells**. Each cell names a SKILL invocation, not a motor command — see "Skill primitives" below.
- **Legality A(s):** the robot must be able to REACH the contact point — wavefront (BFS over the inflated obstacle map) marks ~40-75 of the 300 cells reachable in a typical state. Exact in sim, approximate from perception on a real robot; deployment always post-filters to A(s).
- **Oracle:** MuJoCo physics, ~1 s per push, deterministic given the state. Perfect ground truth, unaffordable at decision time, gone entirely on the real car.
- **Success ("opens"):** after the push, ≥20% of the goal region's sampled points are wavefront-reachable from the robot (frozen criterion).
- **Budget:** H ≤ 2 pushes for now (architecture supports more). Episodes can be dead at a given horizon: 1-push-dead but 2-push-solvable episodes are exactly the interesting middle (28.5% of our H1-dead episodes), and some episodes are dead at any budget we test.
- **Data:** offline only. The expert is a sampled tree search over pushes (k=30 per level, ≤3 restarts on failure); per tried cell we record the exact binary outcome; the tried set is the loss mask. Gamma-discounted targets (1.0 / 0.9 / 0) encode prefer-shorter; per-setup success FRACTIONS record robustness (8/24 robust vs 1/30 brittle).

### 2.1 Skill primitives (the action abstraction)

The Q-function's "action" is one invocation of a **push skill** (NAMOPushSkill) — a whole navigate-contact-push routine, not a torque:

1. **Navigate:** the robot is placed at the pre-push pose for contact point e (teleport-style in sim: set chassis SE(2), zero velocities, settle ~100 physics ticks — the navigation problem is considered solved by the wavefront; that is exactly why legality A(s) = "contact point reachable").
2. **Push:** the car tracks a precomputed straight-line push path through the object along the face normal (pure-pursuit + cross-track-error PD on the diff-drive wheels, 550 control steps), shoving the object a depth-dependent distance.
3. **Primitive library:** the (e,d) → expected object displacement map comes from a precomputed motion primitive database — 300 primitives per object shape class, with **shape-based selection** (square / wide / tall by side ratio, 5% tolerance). Primitives are regenerated whenever robot geometry or push duration changes (the car-geometry saga of §3 in the build journal); `se2_target` is object-local.

Two consequences for learning: (a) the action space is genuinely DISCRETE and small enough to score densely — no continuous actor needed; (b) each action is temporally extended (~seconds of physics), so H=2 means two long skill executions, not two timesteps — which is why even a 2-push tree is expensive and why budgets are tiny.

### 2.2 Similarity to HACMan

**HACMan's setting** (Zhou et al., CoRL 2023): a fixed ARM doing 6D object pose alignment on a tabletop — push/slide/FLIP one object in a bin until its full 6D pose (rotations included) matches a target. Observation: segmented point clouds; goal given as per-point flow toward the target pose. Action: hybrid — a DISCRETE contact point chosen on the object point cloud + a CONTINUOUS learned end-effector motion vector after contact; one short poke per step, greedy replanning. Trained with online off-policy RL on dense pose-distance rewards in sim, then sim-to-real on a Franka.

Ours swaps: arm → mobile robot in walled rooms (so reachability/navigation enters the action's legality); pose-matching → binary REGION OPENING (a connectivity objective about the SCENE, not the object's pose); continuous motion parameter → a discrete primitive library depth; dense reward + online RL → sparse success + offline search distillation; greedy replanning → an explicit push BUDGET.

**The action mapping that motivated the architecture** [USER's original design draw]:

| | HACMan | ours |
|---|---|---|
| WHERE to touch | contact point p on the segmented point cloud (thousands, perception-dependent) | contact point e, one of 60 FIXED perimeter anchors (4 faces x 15) |
| HOW to push | continuous motion vector m (learned per-point ACTOR) | depth d in {1..5}; direction baked into the primitive |
| action | (p, m) hybrid discrete x continuous | (e, d) fully discrete, 60x5 |
| critic | per-point score map | per-edge token -> 5 depth scores -> 60x5 map |
| policy | argmax over the map | top-k/argmax over the map (one-head) |

The inherited lesson: **the action's spatial anchor deserves its own representation** — score "push HERE" with a feature FOR here, not an index into a global vector. Our translation: each contact point is a token (Fourier-encoded pixel coord + learned edge-identity embedding) that cross-attends the scene and self-attends the other 59; its head emits the 5 depth values. That is the E2 architecture, and it beat the global-readout baseline (E0) for exactly the reason HACMan predicted. Where discretization CHANGED the design rather than copied it: (1) the primitive library collapses "how" to 5 enumerable depths, so HACMan's continuous actor disappears entirely — the critic map IS the policy; (2) the fixed 60-anchor topology enables stable per-edge identity embeddings and a fixed-size output map that search and eval index directly.

The architecture and action decomposition are deliberately HACMan-style (per-point critics for non-prehensile manipulation):

- **Shared idea:** score a dense map of contact-parameterized actions — WHERE to touch the object × HOW to push — with a per-contact-point critic that attends to scene context. Our per-edge tokens (Fourier-encoded contact pixels + edge embeddings, cross-attending the scene crop, self-attending each other) producing a 60×5 value map are exactly that pattern; a dense map makes argmax/top-k action selection trivial at decision time.

- **Where we deliberately diverge (each divergence is one of our measured gates):**
  1. **Training signal:** HACMan learns by online RL (TD on its own sim rollouts). We train on search-distilled MC targets — outcomes verified by the expert's tree, never bootstrapped from the model's own guesses at this horizon (online-TD baseline explicitly parked; TD-or-not-TD argues MC at short horizons; IQL-on-our-data is the planned head-to-head).
  2. **Budget conditioning [prototyped, then DROPPED]:** we tried an H input so one network could answer "within 1" vs "within 2", but measured it ≈ no-horizon (NoHz ahead) and dropped it — at H≤2 the budget input has nothing to do. The live model is single-head NoHz; foresight instead lives in the SETUP-value target (rank a first push by the finish it enables).
  3. **Negative knowledge:** we explicitly train on dead-ends (51% of rows) and measure dead→low-V; this turned out to also IMPROVE ranking (+3.2pp @1) — supervision HACMan's setup never sees.
  4. **Objective:** theirs is goal-conditioned object repositioning (continuous pose rewards); ours is binary region-opening under a push budget — value = P(open within H), which is also why classification value heads (HL-Gauss) fit naturally.
  5. **Legality:** our candidate set comes from wavefront reachability with deploy-time post-filtering (and reachability-as-signal is an active ablation, M2c/M2d).

So: HACMan supplies the *shape* of the critic; the contribution under test here is everything about the *training signal* — sampled masks, dead-ends, gamma/budget conditioning, search distillation, and the ExIt loop on top.

## 3. The problem in one paragraph (as deployed)

A robot (7 cm diff-drive car) needs to reach a region that is blocked by a movable object. It can fix this by pushing **that one object** — maybe one push, maybe a chain of two or three (Region Opening; the unit of work is always one (object, goal-region) episode). At every decision it faces **300 discrete push options** (60 contact points × 5 depths), and the only way to know what a push does is a **simulator that costs ~1 s per push** — a perfect but expensive oracle. Tree search over pushes solves the problem (our expert planner does exactly that) but costs hundreds of simulations per decision; a real robot has neither the time nor, eventually, the simulator.

**The problem: amortize that search into a network.** Learn a **single push-value / ranker**

```
V(state, push) = P(this push opens the region OR sets up a push that does, playing well afterward)
```

so the expensive tree the planner explores at data-collection time gets *distilled* into a single forward pass at decision time. Success is "opens" iff ≥20% of the region's sampled goal points become reachable (the frozen bar). Targets are gamma-discounted (1.0 opens-in-1 / γ=0.9 a later push opens / 0 certified-dead) so shorter solutions are preferred; a push we tried but never found a finish for is **masked (unknown), never 0**. No budget/horizon input — the **NoHz** single head.

**One function, two deployments** (every design decision is checked against BOTH):
- **Reactive (no search):** query V, execute the top push, re-query at the next state. Zero simulator calls — the real-robot mode.
- **Search:** use the same V as a *prior* (expand only its top-k first pushes in the sim) and as the *leaf evaluator* (V(s′) = top-k-mean) — replacing exhaustive sweeps with a handful of aimed simulations.

**Headline metric:** solve rate as a function of simulations spent — push the curve up at 0 sims, dominate the random ranker.

What the function must know — and where it's actually stuck:
1. which pushes **open the region now** (the "finish" — already near-oracle: its top pick opens ~58%),
2. when **nothing** works, so it stops wasting budget (dead-ends — enumerated 1-push negatives),
3. **⭐ the bottleneck: what a first push is worth as a SETUP** — a push that opens nothing *yet* but leads to a state a later push can finish. The model buries setups (they open nothing now), so the live research is ranking the first push by **setup value**, not by "does it open now." → [RESULTS.md §4](experiments/RESULTS.md).

## 4. The method — one change per rung, a number at every gate

> **⚠ Historical plan (2026-06-12).** This ladder is the budget-conditioned build as originally staged. The **M3 "H=2 head" foresight route was tested and dropped** (horizon-conditioned ≈ no-horizon, NoHz ahead); foresight now lives in the **setup-value target** (rank a first push by the finish it enables), not a horizon head — see the top banner and [RESULTS.md §4](experiments/RESULTS.md). Rows M1/M2a/M2b (data factory, arch, dead-ends) still stand as the NoHz build's real gates.

| stage | one change | gate (pre-registered) | result |
|---|---|---|---|
| 0. Certification | — | world/harness unchanged: car-geometry effect, byte-exact inference, curve-matched training | ✅ all clean; test set reusable |
| **M1** data factory | new data, frozen model (champion recipe) | reproduce champion hard@1 (23.27±1.38) | ✅ **29.40±1.50 (+6.1pp, all seeds)** |
| **M2a** architecture | + H-embedding + HL-Gauss value head | ≈ M1 (arch must be free) | ✅ **29.62±0.93** |
| **M2b** dead-ends | + 129,536 dead rows (51% of training) | ranking holds AND dead→low V | ✅ **32.86±2.38** (+3.2!); V_dead 0.065 vs control 0.313, AUC 0.987 |
| **M2c/M2d** reachability (side-quest) | +20 unreachable-cell labels / +reach bit input | [USER hypothesis] sharper scene understanding ⇒ better hard@k | 🟢 training |
| **M3** foresight | Q-full's H=2 head, zero sims, pure-2-push slice | beat 34.5%@1 (old champion's score WITH 49 sims) | ⏳ tonight/tomorrow |
| **M4** integration | one mixed-H network | H=1 ranking holds ≈M2b; end-to-end reactive 2-push solve | ⏳ |
| **M5** headline | deployment, both regimes | solve@k vs sims dominates the 49-sim beam | ⏳ |
| **M6** the loop | ExIt rounds 2-3: Q-guided re-collection | per-round hit@k climbs | ⏳ |

## 5. How the data is made (the part that took the discipline)

- **Scenes:** feb_car + aug9_car v3 pools; training composition matched to the test set at the EPISODE level (65:35), held out BY ROOM; episodes matched by object_center (multi_episode_rooms invariants).
- **H=1:** collect every reachable push per episode (~40-75 of 300), keep dead-ends (all-fail episodes — the old pipeline silently dropped them; fixed in three scripts). 123,269 solvable + 129,536 dead rows.
- **H=2 [USER design]: sampled, never exhaustive beyond depth 1.** k=30 uniformly sampled cells at EVERY chain level + ≤3 restarts with fresh draws only-while-unsolved; the tried set IS the training loss mask (masked loss, B30-validated). Labels stay exact per tried cell; dead-ends emerge statistically (all-low grids at scale). Empirically the sampled composition == the exhaustive remnant's (16/28.5/55 both) — sampling lost nothing at the population level. 110,824 scenes in ~7h (exhaustive was 24h+).
- **Mixed-H rows:** per H2 episode, an H=1 row and a gamma-valued H=2 row share one rendered state; a level-1-tried-but-never-expanded cell is MASKED at H=2 (unknown), never a false 0. Post-push states of solved chains render as free H=1 rows (the search regime's leaf states, in-distribution).
- **Robustness fractions:** per first push, n_succ/n_tried over its sampled children — the denominator is part of the label (1/30 brittle vs 8/24 robust).
- **Certified ground truth lives only in the test set** (namo_testset_v1: exhaustive at both depths, geometry-disjoint). Training data is judged by test performance, never by its own construction.

## 6. Measurement rules (locked)

- **Always evaluate/pool over the candidate set** (r_mask=1 / wavefront-reachable; edge-level). Scores on unreachable cells are a robustness diagnostic only — deploy post-filters reachable contact points.
- **Value = top-k-mean of the map, never raw max** (max is fluke-dominated, H0b).
- 3 seeds minimum, per-seed means over top-val ckpts, paired compare (hard@1 carries ±3-4 single-ckpt noise). Never compare single checkpoints. Pre-register predictions before numbers exist.
- Gates before trusting any eval: gt_in_valid ≈ 1.0, bad_match = 0, edge_align_err = 0, room leakage = 0.

## 7. Baselines (for the results write-up)

Internal ladder (banked, never retrain): random 11.8@1 · geometric oracle (~6% hard) · champion 23.27 · M1/M2a/M2b · old-champion+49sims 34.5 · M2b+49sims (in flight) · policy-distillation arms (=BC). Must-add cheap controls: Q-full's own H=1 head at the root (the H-conditioning ablation) · marginal-prior ranker (dataset-prior leakage check) · expert-at-budget curve (classical-planner anchor). One new training: IQL on the same data (TD vs distilled-MC). Adapted-protocol/positioning only: Bejjani-style value-RHP, HACMan (arch absorbed; deltas isolated by our own ladder), MORE.

## 8. Why this might matter beyond this robot

The recipe — sample-don't-enumerate with known masks, keep the failures, distill search outcomes as MC targets, verify-before-bootstrap, then let the model aim the next round's collection — is not NAMO-specific. (The one ingredient we tried and cut: conditioning the value on remaining budget — measured ≈ no-horizon, dropped.) It's a general pattern for problems with an expensive simulator-oracle and a discrete action menu. The lit map (journal §11) says nobody combines these pieces on continuous manipulation; the gates above are designed so that if the claim survives, every ingredient's contribution is separately measured.
