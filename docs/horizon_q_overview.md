# Horizon-Q: Amortizing NAMO Search into a Budget-Conditioned Value

> The plain-language map of WHAT we're solving and HOW. Operational state lives in
> [experiments/horizon_q_build_journal.md](experiments/horizon_q_build_journal.md) (§9 = resume point);
> the 37-decision design spec with citations lives in
> [experiments/multipush_horizonQ_journal.md](experiments/multipush_horizonQ_journal.md).
> Status snapshot in this file: **2026-06-12**.

## 1. The problem

A robot (7 cm diff-drive car) needs to reach a region that is blocked by a movable object. It can fix
this by pushing **that one object** — maybe one push, maybe a chain of two or three (Region Opening; the
unit of work is always one (object, goal-region) episode). At every decision it faces **300 discrete push
options** (60 contact points × 5 depths), and the only way to know what a push does is a **simulator that
costs ~1 s per push** — a perfect but expensive oracle. Tree search over pushes solves the problem (our
expert planner does exactly that) but costs hundreds of simulations per decision; a real robot has
neither the time nor, eventually, the simulator.

**The problem: amortize that search into a network.** Learn a budget-conditioned value

```
Q(state, push, H) = P(this push opens the region within H pushes, playing well afterward)
```

so the expensive tree the planner explores at data-collection time gets *distilled* into a single
forward pass at decision time. Success is "opens" iff ≥20% of the region's sampled goal points become
reachable (the frozen bar). Targets are gamma-discounted (1.0 opens-in-1 / γ=0.9 opens-in-2 / 0) so
shorter solutions are preferred. H_max = 2 for now, extensible.

**One function, two deployments** (every design decision is checked against BOTH):
- **Reactive (no search):** query Q at budget H, execute the top push, re-query at H−1. Zero simulator
  calls — the real-robot mode.
- **Search:** use the same Q twice — as a *prior* (expand only its top-k first pushes in the sim) and as
  the *leaf evaluator* (V(s′) = top-k-mean of Q(s′,·,H−1)) — replacing exhaustive sweeps with a handful
  of aimed simulations.

**Headline metric:** solve rate as a function of simulations spent — push the curve up at 0 sims, beat
the old 49-sim search with a few.

What the function must know (each is a gated milestone below):
1. which pushes **work** (ranking),
2. when **nothing** works, so it stops wasting budget (dead-ends),
3. what a push is worth **as a setup** for the next one (foresight).

## 2. The method — one change per rung, a number at every gate

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

## 3. How the data is made (the part that took the discipline)

- **Scenes:** feb_car + aug9_car v3 pools; training composition matched to the test set at the EPISODE
  level (65:35), held out BY ROOM; episodes matched by object_center (multi_episode_rooms invariants).
- **H=1:** collect every reachable push per episode (~40-75 of 300), keep dead-ends (all-fail episodes —
  the old pipeline silently dropped them; fixed in three scripts). 123,269 solvable + 129,536 dead rows.
- **H=2 [USER design]: sampled, never exhaustive beyond depth 1.** k=30 uniformly sampled cells at EVERY
  chain level + ≤3 restarts with fresh draws only-while-unsolved; the tried set IS the training loss mask
  (masked loss, B30-validated). Labels stay exact per tried cell; dead-ends emerge statistically (all-low
  grids at scale). Empirically the sampled composition == the exhaustive remnant's (16/28.5/55 both) —
  sampling lost nothing at the population level. 110,824 scenes in ~7h (exhaustive was 24h+).
- **Mixed-H rows:** per H2 episode, an H=1 row and a gamma-valued H=2 row share one rendered state; a
  level-1-tried-but-never-expanded cell is MASKED at H=2 (unknown), never a false 0. Post-push states of
  solved chains render as free H=1 rows (the search regime's leaf states, in-distribution).
- **Robustness fractions:** per first push, n_succ/n_tried over its sampled children — the denominator is
  part of the label (1/30 brittle vs 8/24 robust).
- **Certified ground truth lives only in the test set** (namo_testset_v1: exhaustive at both depths,
  geometry-disjoint). Training data is judged by test performance, never by its own construction.

## 4. Measurement rules (locked)

- **Always evaluate/pool over the candidate set** (r_mask=1 / wavefront-reachable; edge-level). Scores on
  unreachable cells are a robustness diagnostic only — deploy post-filters reachable contact points.
- **Value = top-k-mean of the map, never raw max** (max is fluke-dominated, H0b).
- 3 seeds minimum, per-seed means over top-val ckpts, paired compare (hard@1 carries ±3-4 single-ckpt
  noise). Never compare single checkpoints. Pre-register predictions before numbers exist.
- Gates before trusting any eval: gt_in_valid ≈ 1.0, bad_match = 0, edge_align_err = 0, room leakage = 0.

## 5. Baselines (for the results write-up)

Internal ladder (banked, never retrain): random 11.8@1 · geometric oracle (~6% hard) · champion 23.27 ·
M1/M2a/M2b · old-champion+49sims 34.5 · M2b+49sims (in flight) · policy-distillation arms (=BC).
Must-add cheap controls: Q-full's own H=1 head at the root (the H-conditioning ablation) ·
marginal-prior ranker (dataset-prior leakage check) · expert-at-budget curve (classical-planner anchor).
One new training: IQL on the same data (TD vs distilled-MC). Adapted-protocol/positioning only:
Bejjani-style value-RHP, HACMan (arch absorbed; deltas isolated by our own ladder), MORE.

## 6. Why this might matter beyond this robot

The recipe — sample-don't-enumerate with known masks, keep the failures, condition on remaining budget,
distill search outcomes as MC targets, verify-before-bootstrap, then let the model aim the next round's
collection — is not NAMO-specific. It's a general pattern for problems with an expensive simulator-oracle
and a discrete action menu. The lit map (journal §11) says nobody combines these pieces on continuous
manipulation; the gates above are designed so that if the claim survives, every ingredient's contribution
is separately measured.
