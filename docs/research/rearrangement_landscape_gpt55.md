# Rearrangement landscape for a learned push-ranker + perfect verifier — decision-oriented synthesis

> Source: codex CLI, gpt-5.5 @ xhigh, live web search, 2026-07-10. Raw per-query outputs: `gpt55_rawA_problems.md`, `gpt55_rawB_baselines.md`, `gpt55_rawC_hardgen.md`.
> Synthesis is bound to OUR setup, not a generic survey. Read the raws for the full un-pruned lists.

**Our setup, on 4 axes (the filter everything is judged against):**
1. The model is a learned **RANKER over a DISCRETE primitive grid** (contact-edge × depth), not a policy and not a continuous sampler.
2. We own a **fast EXACT simulator = a perfect free verifier** (try any push, get ground truth in ~1s). We never need a learned forward/dynamics model.
3. The objective is **minimize simulator CALLS** to a solution — a search-cost curve, not task success alone.
4. The interesting structure is **non-myopic SETUP→FINISH** (a first push that opens nothing, enabling a second that does). Myopic one-step-gain misses it.

**Two research forks we are choosing between (tag every item with which it serves):**
- **Fork 1 (EFFICIENCY paper):** scale the SAME problem — deeper chains / multi-object / robot must CHOOSE which object to move — until brute force is intractable, so the ranker becomes an *enabler*, not just a speedup.
- **Fork 2 (SCARCITY paper):** crack **hard-instance mining/generation** — the data bottleneck where forward random sampling collapses to easy and difficulty labeling requires solving.

**Fixed baselines (already decided, do NOT re-derive):** #1 hand geometric push-ranker; #2 myopic one-step opening heuristic. Everything in §2 must ADD to these.

---

## 1. Problems / benchmarks

Only items that plug into a discrete-push + verifier + minimize-sims frame are kept here. Pick-and-place-dominant benchmarks are moved to "Adjacent, not directly usable."

| Problem / benchmark | Repo | Contact-rich? | Combinatorial structure | Effort | Fork |
|---|---|---|---|---|---|
| **VFT — Visual Foresight Trees** (Huang, Han, Yu, Boularias, RA-L 2022) — push clutter so a target becomes graspable | `arc-l/vft` (has code) | Yes, nonprehensile push+grasp | Tree search over push sequences; setup pushes raise future graspability | Medium | 1 |
| **LAX-RAY / shelf mechanical search** (Huang et al., "Mechanical Search on Shelves using Lateral Access X-RAY," 2020) — laterally push occluders to reveal target | Repo unverified (reimplement 2D) | Yes, push-only | Cleanest setup/enabling moves: choose which occluder to push next | Easy-Medium | 1 |
| **Sokoban / Boxoban** (`mpSchrader/gym-sokoban`, `deepmind/boxoban-levels`; Guez et al. 2019) — grid push puzzle, boxes move only by pushes | Both repos exist | Abstract push (no physics) | Best-in-class: irreversible setup moves, deadlocks, non-monotone | Easy | 1+2 |
| **NAMO / VANAMO** (Muguira-Iturralde, Curtis, Du, Kaelbling, Lozano-Pérez, arXiv 2212.02671, 2022) — our own family; VANAMO adds visibility | No maintained repo | Yes, push+pick | Setup moves; non-monotone possible | Medium-Hard | 1 |
| **Ravens `sweeping-piles`** (Zeng et al., Transporter Nets, CoRL 2020) — PyBullet tabletop, includes a pushing task | `google-research/ravens` (archived, usable) | Partly (sweeping is push) | Multi-step, but push task is shallow | Easy-Medium | 1 |
| **VPG — Visual Pushing & Grasping** (Zeng et al., IROS 2018) — learn when a push makes grasping easier | `andyzeng/visual-pushing-grasping` | Yes, push+grasp | Short dependency (push→grasp); shallower than NAMO | Medium | 1 |
| **Nested Nonprehensile Rearrangement** (Song, Boularias, arXiv 2019) — push objects into a packing pattern | No repo | Yes, strong pushing | High combinatorics, mostly monotone packing | Hard | 1 |

**Adjacent, not directly usable** (pick-and-place-dominant or no clean push-grid/verifier map — cite, don't build on): Habitat 2.0 Rearrangement (`facebookresearch/habitat-lab`); AI2-THOR / ManipulaTHOR (`allenai/ai2thor`, `allenai/manipulathor`); OCRTOC; PuSHR (`prl-mushr/pushr`, multi-robot, heavy); PDDLStream / pybullet-planning / PDDLGym (`caelan/*`, `tomsilver/pddlgym` — great planner-baseline substrate, but not contact-rich unless you add pushing).

### TOP-3 problems to implement (ranked by how directly they advance OUR paper)

**1. Sokoban / Boxoban as the abstract push-search testbed. [Fork 1 + Fork 2] — Easy.**
Connection to us: it IS a discrete irreversible-push problem with a perfect verifier and genuine non-monotone SETUP→FINISH dependencies — the exact structure of our 2-push chain, minus physics. Huge ready-made level sets let us plot search-cost-vs-ranker curves in hours and stress deeper chains (Fork 1) AND prototype hard-instance mining (Fork 2, §3) where "solution known by construction" is native. Use it as the fast lab bench where the method's search-cost story is proven before porting to MuJoCo NAMO.

**2. LAX-RAY-style lateral shelf push search. [Fork 1] — Easy-Medium.**
Connection to us: push-only, SE(2), and the core decision is literally "which occluder do I push next to enable the reveal" — a setup/finish chain in a robotics-real skin. No maintained repo, but it's a 2D reimplement, and our simulator already does contact pushing. Best path to a *second contact-rich domain* that shows the ranker generalizes beyond navigation — directly supports scaling to "robot must choose which object" (Fork 1).

**3. VFT object retrieval from clutter. [Fork 1] — Medium.**
Connection to us: the only TOP-3 with real code (`arc-l/vft`) whose loop is exactly "rank pushes → search a tree → act." Swap their learned forward model for our exact simulator to make it a fair verifier-backed comparison, and it becomes both a richer testbed and a drop-in baseline family (see §2). Slightly heavier than Sokoban/shelf, so third.

---

## 2. Baselines (methods that ADD to our fixed geometric + myopic baselines)

Dropped from here: anything that just restates "hand geometric ranker" (our #1) or "one-step gain" (our #2). Kept: learned search guidance, sequence-feasibility classifiers, samplers, and search-distillation loops that fit discrete-primitive + perfect-verifier + minimize-sims.

| Method | Paper | Code | Bucket | Effort | Fork |
|---|---|---|---|---|---|
| **PIGINet** | Yang, Garrett, Lozano-Pérez, Kaelbling, Fox, RSS 2023 | `Learning-and-Intelligent-Systems/kitchen-worlds` | Plan-feasibility classifier | Medium | 1 |
| **Expert Iteration (ExIt)** | Anthony, Tian, Barber, NeurIPS 2017 | No verified code (skeleton only) | Search-distills-into-ranker | Medium | 1+2 |
| **MORE** | Huang, Guo, Boularias, Yu, ICRA 2022 | `arc-l/more` | MCTS labels → DNN guides next MCTS | Medium | 1+2 |
| **SAVE (Q + amortized value)** | Hamrick et al., ICLR 2020 | No verified code | Learned Q guides MCTS, sim as model | Medium | 1 |
| **DeepCubeA** | Agostinelli et al., Nature MI 2019 | `forestagostinelli/DeepCubeA` | Learned cost-to-go in weighted A*/GBFS | Medium | 1 |
| **CVAE action sampler** | Ichter, Harrison, Pavone, ICRA 2018 | No verified code | Learned sampler (shortlist top-K pushes) | Medium | 1 |
| **HACMan / HACMan++** | Zhou et al. CoRL 2023 / Jiang et al. RSS 2024 | `HACMan-2023/HACMan`, `JiangBowen0008/HACManPP` | Per-contact spatial Q-map | Medium | 1 |
| **Where2Act** | Mo et al., ICCV 2021 | `daerduoCarey/where2act` | Per-edge actionability ranker | Medium | 1 |
| **Kim & Shimanuki relational value (GNN)** | CoRL 2019 | No verified code | GNN value to guide TAMP | Medium | 1 |
| **Chitnis learned TAMP heuristic** | ICRA 2016 | No verified code | Learned plan ranking | Medium | 1 |
| **MCTS rearrangement** | Labbé et al., RA-L 2020 | Code on project page | MCTS over rearrangement actions | Medium | 1 |
| **`mctx` (Gumbel MuZero)** | DeepMind | `google-deepmind/mctx` | Few-simulation MCTS machinery | Med-Hard | 1 |

### TOP-5 baselines to implement first (ranked by advancing OUR paper)

**1. PIGINet-style 2-push PREFIX classifier. [Fork 1] — Medium.**
This is the cleanest test of our central claim: does *scoring the whole (a1,a2) sequence* beat *scoring the first push*? Enumerate top candidate pairs cheaply, label solve/no-solve straight from our verifier traces, train a binary transformer over prefixes. Directly separates myopic (#2) from non-myopic on the search-cost curve — the money plot.

**2. MORE / ExIt search-distillation loop. [Fork 1 + Fork 2] — Medium.**
The natural upper story for our whole method: unguided search finds solutions, the ranker distills them, the next search is cheaper, repeat — plot sim-calls per generation. `arc-l/more` gives a robotics-grounded reference implementation, and ExIt is the training skeleton. It also generates its own training data, so it bridges into Fork 2.

**3. Learned leaf-value best-first (DeepCubeA-style). [Fork 1] — Medium.**
Train V(state)=cost-to-solution from solved searches, drop it into weighted A*/GBFS over push states, count sims. `forestagostinelli/DeepCubeA` is real, runnable code for exactly "learned cost-to-go inside search." This is the value-side counterpart to our ranker and the standard learned-search-guidance comparator reviewers will expect.

**4. CVAE top-K push sampler. [Fork 1] — Medium.**
Sampler-vs-ranker is a distinct axis reviewers ask about: generate K likely setup pushes, let the verifier check them, compare against ranking all of them. Start with a CVAE (not diffusion) because our action space is small and discrete — diffusion (Diffusion-CCSP `zt-yang/diffusion-ccsp`) is overkill for ~300 discrete pushes and stays a "later, if asked" option.

**5. SAVE / few-sim MCTS with learned Q. [Fork 1] — Medium.**
Our simulator IS the MCTS model, so amortized-value MCTS is almost free to stand up and gives a principled "spend N sims wisely" baseline. Positions our ranker against the search-community default (policy/value-guided tree search) on the minimize-sims metric. Use `google-deepmind/mctx` if we want the formal few-simulation machinery.

---

## 3. Hard-instance generation (highest-value section for us — Fork 2)

**The bottleneck, precisely:** hard instances and their rare solutions are exponentially rare and unenumerable; forward random scene generation collapses to easy; random rollouts find only easy solutions; and labeling difficulty *requires solving* (chicken-and-egg). So we cannot sample-then-filter our way to a hard 2-push corpus at scale.

**The unifying insight from the literature: keep the solution, mutate the instance.** Every viable method here shares one move — start from something with a *known* solution and perturb locally, re-verifying with the real simulator, instead of sampling scenes blind and hoping they're hard-and-solvable.

| Approach | Paper | Code | Core idea | Fit for us |
|---|---|---|---|---|
| **ACCEL — evolve curricula by mutating hard levels** | Parker-Holder et al., arXiv 2022 | Project site (official code unverified) | Mutate previously-hard levels, keep the still-hard ones | **Best fit** |
| **PLR — Prioritized Level Replay** | Jiang, Grefenstette, Rocktäschel, ICML 2021 | `facebookresearch/level-replay` | Replay levels with high learning potential | Very high (simple) |
| **DCD — replay-guided adversarial design** | Jiang et al., arXiv 2021 | `facebookresearch/dcd` | Level replay AS environment design | Very high |
| **Backward/retrograde (Sokoban)** | Bento, Pereira, Lelis, arXiv 2019 | Unverified | Build starts backward from solved states | High (as proposer only) |
| **Reverse curriculum** | Florensa et al., 2017 | Unverified | Start near goal, expand outward | High for 2-push chains |
| **Hindsight relabeling (HER)** | Andrychowicz et al., NeurIPS 2017 | `openai/baselines` HER, `vitchyr/rlkit` | Treat what happened as the goal | High as data mining |
| **PCGRL — RL level generator** | Khalifa et al., AIIDE 2020 | `amidos2006/gym-pcgrl` | Train an agent to edit levels | Medium |
| **PAIRED / POET / asymmetric self-play** | Dennis 2020 / Wang 2019 / OpenAI 2021 | `uber-research/poet`, others | Adversarial generator vs solver | Low-medium (heavy) |

### Recommendation for constructing hard multi-push NAMO instances

**Primary: ACCEL-style solution-preserving mutation + a PLR/DCD hard buffer. [Fork 2] — this is the 3-day build.**
Recipe: (1) seed with the few solved 1-push and 2-push episodes we already have; (2) mutate locally — jitter object pose, goal region, wall/distractor placement, push-depth slots; (3) **forward-verify every mutant with our exact simulator** (does the intended chain still solve?); (4) score each survivor by simulator-call difficulty for random/current-ranker search; (5) keep the hard-and-solvable ones in a replay buffer, stratified by easy/med/hard × 1push/2push. This sidesteps the chicken-and-egg entirely: we never label difficulty on unsolved scenes, only on scenes whose solution we carried in by construction. `facebookresearch/level-replay` and `dcd` give the buffer/priority machinery to copy.

**Secondary: hindsight relabeling to mine failed rollouts. [Fork 2] — cheap add-on.**
Every failed search still yields labels: "this push achieved displacement X," "this state is a valid predecessor for the finish push," "this setup made contact-edge Y reachable." Free training signal and free predecessor-states that feed the mutation seed pool.

### Backward-generation viability verdict (the question you flagged as central)

**Can we construct hard 2-push NAMO instances backward from a chosen opened-state + chain? Verdict: viable ONLY constructively, never by reversing physics.**
Pushes are not physically reversible — you cannot "un-push" an object and trust the pre-image, because contact/friction dynamics are not invertible and many predecessor states map to the same successor. So "simulate the solution backward" is *unsound* and we should not build on it.
The sound version (concrete recipe): pick a setup→finish *skeleton* (which object, which contact-edges, which depths), hand-design a candidate PRE-push scene that ought to make that skeleton executable, then **run the real forward simulator** and keep the instance only if (a) the intended chain actually solves it and (b) random/greedy search burns many sim-calls (i.e. it's genuinely hard). Backward reasoning proposes; the forward verifier disposes. That is exactly the ACCEL loop above with a hand-authored skeleton as the seed instead of a mutation — so the concrete blocker (irreversibility) is dissolved by never trusting a backward state without forward re-verification.

---

## 4. Honest gaps — where our exact setup is unlike everything found

- **Nobody has our "perfect free verifier" as a first-class asset.** VFT/VPG/mechanical-search all fight a *learned, imperfect* forward model; that's their bottleneck and their novelty. Our exact simulator makes most of their machinery (learned dynamics, foresight nets) unnecessary — which is a genuine differentiator, but it also means their reported numbers/baselines don't transfer cleanly; we must re-run any borrowed baseline against our verifier to be fair.
- **Minimize-simulator-CALLS is not the standard metric.** The field optimizes task success, sample efficiency of RL, or wall-clock; the search-cost-vs-ranker curve is ours to define. No off-the-shelf benchmark reports it, so cross-paper number comparison is mostly unavailable — we generate our own curves.
- **The SETUP→FINISH non-myopic chain is rare in code.** Sokoban has it abstractly; VPG/VFT have a shallow push→grasp version; but a *navigation* setup push that opens nothing on its own is barely represented. This is both our contribution and the reason we can't just download a hard-instance set — hence §3 must be built, not borrowed.
- **UED/curriculum papers assume an RL agent + parametric generator, not a discrete-grid + verifier.** PAIRED/POET/ACCEL machinery is heavier than we need; we adopt only the *solution-preserving-mutation + regret-scored buffer* idea, not the full adversarial-RL apparatus.
- **"Regret" for us = (strong search solves cheap) − (our ranker/random burns many sims),** which is computable exactly because we have the verifier — a cleaner signal than the learned-value regret estimates PAIRED relies on. That's a small original methodological point worth claiming.
