# Multi-push learning primer — models, training schemes, and what the field actually does

**Who this is for [USER request 2026-06-10]:** "I don't understand a lot of RL — give me intuitive explanations.
Help me get clarity on the different kinds of *models* and *training schemes* (offline or online, supervised or RL)
we can try for multi-push." Written plain-language-first; every concept ends with *"for us this means…"*.
Synthesized [CLAUDE] from 3 parallel literature sweeps (model families · training schemes · 10 case studies,
2015–2026); citations at the bottom of each part. Companion to [multipush_horizonQ_journal.md] (the parked design)
— this is the *map*, that is the *route we chose on it*.

---

# PART 1 — The cast of characters: what kinds of "model" can we even learn?

There are only ~6 kinds of function people learn for problems like ours. Everything in the literature is one of
these or a combination.

## 1. The per-action score map (a "Q-map") — what we already have
**Plain words:** paint a number on every possible action: "if you take THIS push, how well does it go?" Picking an
action = picking the brightest cell. Our scorer's (60×5) map IS this.
**Eats / outputs:** scene image → 300 numbers.
**Data it needs:** (scene, push, outcome) triples — one label per push you actually simulated.
**How it acts:** argmax / top-k. **Zero** sims at decision time.
**Where it shines in the lit:** this is *the* winning architecture for pushing — VPG (IROS'18), Spatial Action
Maps (RSS'20), Transporter (CoRL'20), **HACMan (CoRL'23: 83-89% success where the best non-spatial baseline got
22%** — and their ablation: remove the per-point spatial scoring and performance collapses to near zero).
**Failure mode:** myopic if the score only looks 1 step ahead (exactly our H0a hard-bucket gap).
**For us this means:** we hold the right architecture already; the multi-push question is only *what the number
means* (1-push success vs "success within budget" — the horizon-Q reframe).

## 2. The state value V(s) — "how good is this situation?"
**Plain words:** one number for the whole scene: "from here, how likely are we to win?" Doesn't say *which* action
— says whether you're standing in a good spot. A chess commentator who says "white is winning" without naming a move.
**Data:** (scene, eventual outcome) pairs — from finished episodes/searches.
**How it acts:** can't act alone. Used to *rank states a search reaches*: simulate a few candidate pushes, score
each aftermath with V, keep the best. (Exactly our H0b trick — except H0b recycled a 1-push scorer as a fake V,
and it saturated. A real V is *trained* on those aftermath states.)
**Modern detail that matters:** train V as **classification over outcome bins, not regression** ("Stop Regressing",
ICML'24 oral: +67% on manipulation value learning; HL-Gauss soft bins). We already follow this style.
**For us:** V = max (or pool) of the horizon-Q map — comes free with #1 if the labels are multi-step. A separate V
head only if that max turns out badly calibrated.

## 3. The policy π(a|s) — "just tell me what to do"
**Plain words:** a function that directly outputs the action (or a probability over actions). No scores, no
deliberation — reflex.
**Data:** (scene, action-an-expert-took) pairs — imitation; or RL gradients.
**Variants you'll hear about:** behavior cloning (BC = plain imitation), Diffusion Policy / ACT (generative,
multi-modal, for continuous arms — overkill for 300 discrete actions).
**Failure mode:** *compounding drift* — trained only on expert-visited states, one mistake leads somewhere it's
never seen, then errors snowball (the O(T²) imitation bound). And it can't say "all options are bad" — it must
always pick something.
**For us:** H1 just tested policy-style training (softmax-CE) head-to-head — it was a wash at @1 and soft targets
hurt it. Plus the masking asymmetry (policy targets corrupt under sampled data). The lit agrees: top-k of a Q-map
IS your policy. No separate policy network needed at our scale.

## 4. The world model — "learn a copy of the simulator"
**Plain words:** a network that predicts what the scene looks like *after* a push — then plan inside the network
instead of the simulator (MuZero, DreamerV3, Visual-Foresight-Trees for clutter).
**Why people do it:** their real environment is inaccessible or slow; a neural copy is queryable and fast.
**Failure mode:** prediction errors compound over steps; contact physics is the hardest case.
**For us: skip it.** We OWN a perfect simulator. Learning an approximate copy of something we have exactly adds
error for zero benefit. (One narrow exception in the lit — VFT trains a forward model because their sim was
100-1000× slower than the net; our sim is ~1s/push, annoying but usable as the *verifier*.)

## 5. The feasibility / cost-to-go classifier — "is this branch worth exploring?"
**Plain words:** a cheap yes/no (or how-many-steps) oracle that a *search* consults to skip dead branches —
A*'s heuristic, learned. (Wells'19 TAMP feasibility: order-of-magnitude planner speedups; Bejjani'18-21.)
**For us:** this is what V-as-max-of-the-map *is* when used for pruning ("max is low → abandon this state").
Same function as #2 wearing a different hat.

## 6. The proposer / affordance model — "look here, not everywhere"
**Plain words:** doesn't score every action precisely; just shortlists promising regions (Where2Act, goal-image
generators, our DiT mask model). A metal detector, not an appraiser.
**For us:** the Q-map's top-k already serves as the shortlist; a separate proposer earns a seat only if the action
space explodes (e.g., continuous placements — the n-push placement question someday).

### Part-1 bottom line
The field's repeated winner for push-like problems = **#1 (spatial per-action map) + #2 (a value to see past one
step) + a perfect sim as the verifier**. That's precisely the parked horizon-Q design: one map whose number means
"succeeds within budget" (= #1 and #2 in one head), verified by a few real sims.

---

# PART 2 — Training schemes: the two questions that define them all

Every training scheme is an answer to two questions:
- **WHERE does the data come from?** Collected once up front (**offline**) vs generated while training, by the
  current model (**online/iterated**).
- **WHAT do you fit to?** Fixed known answers (**supervised targets**) vs the model's own future predictions
  (**bootstrapped / TD "RL objectives"**).

## The schemes, plain
**Behavior cloning (offline + supervised).** Run your planner on many scenes; record what it did; train the net to
copy it. *One example = (scene, the action the planner took).* Cheap, stable; ceiling = the planner, plus the
compounding-drift problem. — *This is roughly how the DiT line was trained.*

**Supervised Q/score regression (offline + supervised) — OUR scheme.** Don't copy actions; record *outcomes* of
tried actions and fit "push → worked?". *One example = (scene, push, outcome).* No drift problem (each label is a
fact about the world, not about an expert). H5 just told us the labels can be SAMPLED (~30/scene) + masked.

**DAgger (online + supervised).** Fix BC's drift: run YOUR policy, let it wander where it wanders, ask the expert
"what should I have done HERE", retrain on those corrections, repeat. *The fix for "trained on the expert's road,
deployed on your own."* Needs an expert you can query anywhere — for us the sim+search is that expert.

**Expert Iteration / AlphaZero (online + supervised — yes, supervised!).** The loop:
1. SEARCH (slow, with sim): from each scene, search finds good action sequences + outcome values.
2. DISTILL: train the net to predict the search's conclusions (cross-entropy on its choices, classification on
   outcomes). These are *supervised* targets — just produced by a search instead of a human.
3. The smarter net makes the next search round cheaper/better. Repeat. (2–3 rounds suffice in robotics: MORE'22.)
**Gumbel MuZero** (ICLR'22) = the few-sim version: provable improvement with as few as 2–4 sims per decision.
**Reanalyze** = re-label old stored states with the newer net, free extra data without re-simulating.

**HER / hindsight relabeling (data trick, scheme-agnostic).** A failed attempt at goal A that accidentally achieved
B becomes a *success* example "for B". Works when goals are positional; awkward for our global binary "region open"
(what's the hindsight goal of a useless push?). Relevant only if we ever label per-region.

**Offline RL proper — CQL/IQL (offline + TD).** For when you have a fixed dataset of *trajectories* and must
stitch parts of them with Bellman backups, fighting overestimation with pessimism penalties. **The headaches it
manages — distribution shift, deadly triad, pessimism tuning — exist because those methods bootstrap. We don't.**

**Online model-free RL — SAC/PPO (online + TD).** Explore by trial, learn from sparse reward. For our case
(300 actions, binary success, 2-push success by chance ≈ 1/90,000, sim 1s/push) the lit's verdict matches common
sense: hopeless without shaping — and shaping would mislead (the best first push often moves the object *away*
from the goal). HACMan affords pure RL only because its sim steps are milliseconds; ours aren't.

## The one theory question, answered plainly: TD vs Monte-Carlo targets
**TD/bootstrapping:** train today's predictions to match *tomorrow's predictions* (chains of guesses; low variance,
adds bias; the source of RL's instability folklore — "deadly triad").
**Monte-Carlo / search-return:** train predictions to match *what actually happened* (ground truth; higher variance
on long horizons).
**At horizon 2–3 with binary outcomes, the variance MC pays is almost nothing and TD's bias buys nothing** —
"TD or not TD" (Fedus'19) says TD's benefit only appears at horizon ≥10; AlphaZero itself uses final outcomes,
not TD. → **Use search/sim returns as labels. Never bootstrap. We stay in supervised-learning land the whole way** —
which is also why the offline-RL headaches above simply don't apply to us.

### Part-2 bottom line
For our problem the map collapses to one path: **supervised Q/score training (what we do) → extended with
search-generated multi-step labels → optionally iterated 2–3 ExIt rounds if one round's coverage isn't enough**
(that's H5c's residual ~3pp question). Everything else is either our scheme wearing a costume, or a tool for
constraints we don't have.

---

# PART 3 — What complete systems actually did (10 case studies, distilled)

The patterns that repeat across VPG, Bejjani'18-21, MORE'22, VFT'22, HACMan/'++, Transporter, NAMO-ML'23,
Contact-MCTS'23, NAMO-HRL'25:

1. **Spatial action maps win, every time.** All strong pushing systems score actions pinned to scene geometry
   (pixel/point/contact = action). Global state-vector policies lose badly. *(We're aligned.)*
2. **Deployment search is SHALLOW everywhere: depth 2–4, never deeper.** A good terminal value substitutes for
   depth. 7 of 10 systems are even purely greedy.
3. **The learned function's job is to CONCENTRATE the search budget, not replace the sim.** MORE: same success as
   pure MCTS with 3× fewer rollouts. The sim stays as verifier. *(= our Q-orders/sim-verifies design.)*
4. **Bootstrap offline, then iterate briefly.** Purely-offline systems are brittle off-distribution; purely-online
   ones are slow. Winners: cheap offline start → 2–3 improvement rounds. *(Bejjani: planner→supervised→refine;
   MORE: MCTS→distill→guided-MCTS.)*
5. **Search-generated labels, zero human annotation.** "Did the search eventually succeed from here?" is the label.
   *(= our F1′/horizon-Q labels.)*
6. **Abstract scene encodings beat raw pixels.** Bejjani's colored-blob images; HACMan's flow features. *(Our
   5-channel mask crops are exactly this.)*
7. **Feasibility-filter before scoring.** Prune geometrically-impossible actions before the net sees them.
   *(= our reachability mask.)*
8. **Closest relative to the parked plan:** Contact-MCTS (Zhu/Righetti, IROS'23 best-paper finalist) = policy
   prior + value + feasibility classifier guiding shallow MCTS — AlphaZero's recipe on real contact planning.
   Ours is that, minus the separate policy net (the Q-map plays prior), at depth 2–3.

**One honest correction to a lit assumption [CLAUDE]:** several systems lean on sim steps costing *milliseconds*.
Our push costs ~1s (full controller rollout: approach, push, settle). That's exactly why our design pushes the sim
count per decision toward "verify top-3" instead of "roll out hundreds" — we sit closer to the Gumbel-MuZero
few-sim end of the spectrum than to MCTS-with-300-rollouts.

---

# PART 4 — The decision, given everything above + tonight's verdicts

What tonight's experiments pinned (1-push architecture journal):
- **Architecture:** sigmoid-sharp per-action map + edge self-attention — both challengers REJECTED (H1, H2).
- **Data recipe:** sampled labels at ~30/scene + masked loss MAINTAINS exhaustive quality; unmasked is catastrophic (H5).

What the literature adds (this primer):
- The model family is the right one; the missing ingredient for multi-push is only the **label horizon** (#1+#2 in
  one head) — not a new architecture, not RL machinery.
- The training scheme is **search-generated MC labels, supervised fit, optionally 2–3 ExIt rounds** — never TD.
- Deployment = **shallow: Q-map orders → sim verifies a handful → V-as-max prunes hopeless states.**

→ All three sweeps independently converge on the parked **horizon-Q** design ([multipush_horizonQ_journal.md]).
The plan that was derived from first principles in last night's conversation is the same plan the field's winners
use. When we un-park: collect at ~30 sampled pushes/state (H5's price), train the same champion architecture with
"succeeds within budget" labels, verify with top-3 sims, iterate only if round-1 coverage shows the H5c gap.

---

# Glossary (one-liners, no math)
- **Q(s,a)** — score of taking action a in scene s. Our (60×5) map is "a Q over 300 actions".
- **V(s)** — score of the scene itself, action-free. Here: max/pool of the map.
- **Policy π** — direct "do this" function. Here: top-k of the map.
- **Horizon** — how many steps ahead the score accounts for. Our scorer today: 1. The plan: 2–3.
- **Offline / online** — data collected once vs generated-while-training by the improving model.
- **Supervised target** — label is a known fact (sim outcome). **TD/bootstrap** — label is the model's own
  next-step prediction (we avoid this entirely).
- **BC** — imitate an expert's actions. **DAgger** — BC + ask the expert about states *you* reach.
- **ExIt / AlphaZero loop** — search makes labels → net learns them → better net makes search cheaper → repeat.
- **Gumbel MuZero** — the version of that loop proven to work with a handful of sims per decision.
- **Reanalyze** — relabel stored states with the newer net; free data, no new sims.
- **HER** — call a failure "a success for the goal it accidentally reached".
- **Compounding drift** — imitation's curse: one off-script step leads to unfamiliar states, errors snowball.
- **Deadly triad / pessimism / CQL-IQL** — the instability-and-fixes world of bootstrapped offline RL; irrelevant
  to us because we never bootstrap.
- **MC target** — label = what actually happened at the end. Our choice, everywhere.
- **HL-Gauss / "Stop Regressing"** — train value outputs as soft classification over bins, not regression to a number.
- **Masked loss** — only grade outputs where we actually know the answer (H5's validated recipe).
- **PU learning** — positives + *unlabeled* (not negatives): the formal name for why masking is right.

**Sources:** model families & schemes & case studies — 3 agent sweeps 2026-06-10; key refs: HACMan
(arXiv:2305.03942), HACMan++ (2407.08585), Stop Regressing (2403.03950), Gumbel MuZero (ICLR'22), MuZero Reanalyze
(2104.06294), ExIt (NeurIPS'17), Bejjani RHP series (1803.08100, RAS'21), MORE (2202.01426), VFT (2105.02857),
Contact-MCTS (2206.09023), NAMO-ML (IROS'23), TD-or-not-TD (1806.01175), CQL/IQL (2006.04779/2110.06169),
HER (1707.01495), V-GPS (2410.13816), Where2Act (2101.02692), Transporter (2010.14406), VPG (1803.09956).
