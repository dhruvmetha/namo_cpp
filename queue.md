# NAMO — Reading list

*Compiled 2026-07-17. Found by gpt-5.6-sol (high) sweeps, citation-audited by Opus 4.8 (xhigh).
Every id, author and venue was checked against the source. Where the sweep and the audit
disagreed, the note says so — they disagreed more than once, and were both wrong once.*

**The problem these serve:** ~50–80 candidate pushes on one given object. The simulator executes
any candidate in ~1s and says exactly whether it opened the region — perfect, deterministic,
exact. So the model is a ranker: it only chooses what order to try things in. The objective is
literally `E[#simulator calls until a success]`.

The first two sections are that problem. The rest is domain and background.

## Tier 1 — Read first: decides the current fork (policy+value / V(s) vs label redesign)

### Search-guided value learning under a perfect verifier

- [ ] `htps` — **Lample et al. — HyperTree Proof Search for Neural Theorem Proving**
      (NeurIPS 2022, arXiv:2205.11491). Neural theorem proving with a critic trained on its own
      search. Has the same structure as a push search: the kernel confirms a proof exactly, but
      failing to find one doesn't prove there isn't one. §7.2.2 and Table 5 are about what to do
      with search nodes you never resolved.
- [ ] `pdvn` — **Liu et al. — Retrosynthetic Planning with Dual Value Networks** (ICML 2023,
      arXiv:2301.13755). Same shape in chemistry. Uses `0.8 × V_syn(s)` as the target for unsolved
      molecules, 0 only for confirmed dead ends.
- [ ] `eg-mcts` — **Hong et al. — Retrosynthetic Planning with Experience-Guided MCTS**
      (Communications Chemistry 2023, arXiv:2112.06028). Experience-guided MCTS; argues a score
      should reflect the actual decomposition situation rather than a penalty.
- [ ] `stahlberg-unsolvability` — **Ståhlberg, Francès, Seipp — Learning Generalized Unsolvability
      Heuristics for Classical Planning** (IJCAI 2021, distinguished paper). Labels training data
      by exhaustively exploring small instances, then generalises to large ones.
- [ ] `li-dantam-infeasibility` — **Li & Dantam — Learning Proofs of Motion Planning
      Infeasibility** (RSS 2021; journal version IJRR 2023). Constructs infeasibility proofs
      rather than inferring them from a timeout.
- [ ] `wells-feasibility` — **Wells, Dantam, Shrivastava, Kavraki — Learning Feasibility for Task
      and Motion Planning in Tabletop Environments** (RA-L 4(2):1255–1262, 2019). A learned
      feasibility classifier used to order a TAMP search, in tabletop. Handles "planner timeout ≠
      infeasible" explicitly and keeps probabilistic completeness when the classifier is wrong.

### Labels from search: censoring, and where heuristic-learning gets its targets

*Added 2026-07-18 (Opus, from the beast-0 post-mortem). The finding these serve: our depth-k labeling wrote "didn't find an opening within k" as a hard 0 — but that's a right-censored observation ("V ≤ γ^k"), not a value; 91% of the beast-0 loss was such false zeros, and the model regressed on depth-1 skill it already had. Two questions: (a) the principled loss for exact-labels-plus-ceilings on a categorical value head; (b) how the learning-for-search field generates labels at all — most of it appears to train on solved instances only and never proves negatives. **Citations below are from model memory, NOT yet PDF-audited** unless marked otherwise — the three research-agent sweeps have since landed and the marked entries were verified against primary sources; verify unmarked ids on fetch.*

*The combinatorial-search sweep's one-line verdict: NOBODY in this literature exhaustively proves negatives — the field trains on solved instances only (negatives come free from the loss), manufactures exact labels (backward generation, hindsight relabeling), or gives unresolved cases a soft finite proxy; and for minimize-expansions, ORDERING losses beat value regression (even true optimal cost-to-go fails the ordering property). The four entries below carry that verdict.*

- [ ] `garrett-rank` — **Garrett, Kaelbling, Lozano-Pérez — Learning to Rank for Synthesizing Planning Heuristics** (IJCAI 2016, arXiv:1608.01302). *PDF-audited 2026-07-24 (in `papers/1608.01302`).* The earlier precedent for `chrestien-ranking`'s thesis: train the heuristic with a ranking loss (RankSVM) because GBFS only consumes the ORDER; beats regression on IPC learning-track domains. *A web sweep attributed this to "Rosman & Ramamoorthy" — fabricated; title page says Garrett/Kaelbling/Lozano-Pérez.*
- [ ] `chrestien-ranking` — **Chrestien, Pevný, Edelkamp, Komenda — Optimize Planning Heuristics to Rank, not to Estimate Cost-to-Goal** (NeurIPS 2023, arXiv:2310.19463). *The sharpest result for us, audited:* minimal-expansion search is characterized by an ORDERING condition; even the TRUE h\* fails it (worked counterexample); ranking losses beat L2-to-h\* in 8/8 domains; the negatives are free — one-hop off-path siblings of solution-path states. Theory for the repo's own "we need the right ORDER, not calibrated probabilities" line.
- [ ] `minimo` — **Poesia et al. — MINIMO: Learning Formal Mathematics from Intrinsic Motivation** (NeurIPS 2024). *Audited.* The search-tree analog of HER: failed best-first proof searches are mined for subtrees that accidentally proved something else → exact positives from failures. Their ablation: solved-only WITHOUT hindsight relabeling starves the loop (~10-20% of attempts yield signal) — the fix was more exact positives from failures, not negatives. Maps directly to mining our failed sweeps for pushes that opened a *different* region pair. *arXiv:2407.00695 — the arXiv title is "Learning Formal Mathematics From Intrinsic Motivation", no "MINIMO:" prefix; MINIMO is the system name.*
- [ ] `ferber-boot` — **Ferber, Geißer, Trevizan, Helmert, Hoffmann — Neural Network Heuristic Functions for Classical Planning: Bootstrapping and Comparison to Other
      Methods** (ICAPS 2022, pp. 583–587). *Audited.* Short paper, 5 pages. The cleanest documented answer to censored labels in planning: timed-out states get a FINITE soft proxy (expansions-until-timeout), tested and preferred over both a big constant penalty and discarding; and self-bootstrap beats teacher imitation once the teacher hits its ceiling.
- [ ] `lts-cm` — **Orseau, Hutter, Lelis — Levin Tree Search with Context Models** (IJCAI 2023, arXiv:2305.16945). The expansions-bound Σd/π used ITSELF as a convex loss — flips the 5×5 sliding puzzle from 0.9% to 100% solved with the same objective where neural policy-only collapsed. The strongest version of "the training loss should literally be the deploy metric." NB: LevinTS is parked in this repo as "not properly tested," not falsified.

- [ ] `nnet-survival` — **Gensheimer & Narasimhan — A scalable discrete-time survival model for neural networks** (PeerJ 2019, arXiv:1805.00917). The standard ML treatment of right-censored targets with a discrete/categorical head: per-interval hazards trained with cross-entropy, censored cases contribute `-log P(survive past k)` — i.e. exactly "penalize only mass above the ceiling." The direct template for our censored value loss. *Audited 2026-07-18 against the PMC full text: censored obs "receives credit for surviving through the censoring interval but no likelihood component for an actual event" — the claimed form is verbatim there.*
- [ ] `deephit` — **Lee, Zame, Yoon, van der Schaar — DeepHit: A Deep Learning Approach to Survival Analysis with Competing Risks** (AAAI 2018). *The closest single precedent to our whole design:* parameterizes the discrete PMF over time-bins directly with a softmax (= our 51-bin head), trains with censored NLL (uncensored → `-log PMF@event`; censored → `-log CDF` past the censor point) PLUS a pairwise ranking loss restricted to pairs whose order is *provably known under censoring* — the exact "certain-pairs ranking aux" we sketched. Read §loss first.
- [ ] `elkan-noto-pu` — **Elkan & Noto — Learning Classifiers from Only Positive and Unlabeled Data** (KDD 2008). The other frame for "didn't find ≠ negative": our dead-within-k cells are unlabeled-not-negative. The classic PU correction; read with `nnpu`.
- [ ] `nnpu` — **Kiryo, Niu, du Plessis, Sugiyama — Positive-Unlabeled Learning with Non-Negative Risk Estimators** (NeurIPS 2017, arXiv:1703.00593). The modern deep-learning-safe PU risk estimator (the naive one goes negative and overfits).
- [ ] `time-limits-rl` — **Pardo, Tavakoli, Levdik, Kormushev — Time Limits in Reinforcement Learning** (ICML 2018, arXiv:1712.00378). Timeout-truncated returns are censored returns; bootstrapping-past-the-limit vs treating timeout as terminal. Same bug class as ours, RL dialect.
- [ ] `bootstrap-heuristics` — **Jabbari Arfaee, Zilles, Holte — Learning Heuristic Functions for Large State Spaces** (AIJ 175(16-17):2075–2098, 2011). The bootstrap procedure: solve what you can, train ONLY on solved instances, use the better heuristic to solve more. Trains on no negatives at all — the direct precedent for our ladder, minus our exhaustive dead-proving.
- [ ] `deepcubea` — **Agostinelli, McAleer, Shmakov, Baldi — Solving the Rubik's Cube with Deep Reinforcement Learning and Search** (Nature Machine Intelligence 1:356–363, 2019). Sidesteps censoring entirely: generate states BACKWARD from the goal so distance-to-goal is known by construction. Worth asking how far the trick carries when the generator owns scene construction (ours does).
      *No arXiv version. arXiv:1805.07470 is a **different, earlier** paper ("Solving the Rubik's
      Cube Without Human Knowledge", McAleer first author) — do not substitute it. The canonical
      `deepcube.igb.uci.edu` mirror is dead; copy here is from cse.sc.edu.*
- [ ] `phs-guarantees` — **Orseau & Lelis — Policy-Guided Heuristic Search with Guarantees** (AAAI 2021, arXiv:2103.11505). Learns a policy whose loss directly bounds EXPANSIONS-TO-SOLUTION — literally our success metric, and it needs only solution trajectories, no negatives. NB: LevinTS is parked in this repo as "not properly tested," not falsified.
- [ ] `exit` — **Anthony, Tian, Barber — Thinking Fast and Slow with Deep Learning and Tree Search** (NeurIPS 2017, arXiv:1705.08439). Expert Iteration — the search-generates-labels loop the whole curriculum is an instance of; read for what they feed back (search-improved targets, not proofs of deadness).
- [ ] `her` — **Andrychowicz et al. — Hindsight Experience Replay** (NeurIPS 2017, arXiv:1707.01495). Failed episodes relabeled as successes for the goals they DID reach — exact labels extracted from failures. The search-tree analog of mining our failed sweeps for free positives.
- [ ] `learn-to-branch` — **Gasse, Chételat, Ferroni, Charlin, Lodi — Exact Combinatorial Optimization with Graph Convolutional Neural Networks** (NeurIPS 2019, arXiv:1906.01629). Learning to rank branching candidates by imitating an expensive oracle (strong branching) — the OR field's version of sim-verified label generation, trained as ranking not regression.
- [ ] `stop-regressing` — **Farebrother et al. — Stop Regressing: Training Value Functions via Classification for Scalable Deep RL** (ICML 2024, arXiv:2403.03950). The HL-Gauss head we already use; re-read §method for how the categorical form composes with a censored (cumulative-mass) likelihood rather than a point target.

## Tier 2 — Read next: understand/extend what is already built and winning

### Adapting the search when the heuristic misleads (failure-discount lineage)

*Added 2026-07-24, motivated by the queue-suppression trace (EXP-2026-07-21 card): 63–77% of 2push search cost is depth-2 children of wrong roots flooding the static queue above the true setup. The proposed fix — per-board credibility demoted by verified failures — has ancestors in four separate communities. Citations web-verified 2026-07-24 (Haiku sweep, links checked), NOT PDF-audited. The sweep found NO published match for the exact mechanism (learned ranker + perfect verifier + sibling demotion on failure) — closest strands below; `adaptive-submodularity` above is the theory anchor for "failures are informative re-ranking evidence." Added PNS and df-pn 2026-07-26: these are the classical exact treatment of the same "proof/disproof credibility" quantity we soft-weight with verified failures; PNS/df-pn solve AND/OR trees with a perfect verifier, the exact shape of our problem.*

- [ ] `pns` — **Allis, van der Meulen, van den Herik — Proof-Number Search** (Artificial Intelligence 66(1):91–124, 1994, doi:10.1016/0004-3702(94)90004-3). The classical exact treatment of AND/OR tree search with a **perfect verifier**: proof and disproof numbers track minimum changes needed to prove/disprove a node, solving the same "binary feasibility + free verifier" problem our suppression trace measures. Lineage anchor for failure-driven heuristic demotion.
- [ ] `df-pn` — **Nagai — Df-pn Algorithm for Searching AND/OR Trees and Its Applications** (PhD thesis, University of Tokyo, 2002). Depth-first reformulation of PNS with transposition tables and iterative deepening: the practical, memory-efficient variant. Introduced λ-search for handling cycles; extended to Tsume-Go and Tsume-Shogi where contact/feasibility rules break straightforward minimax.
- [ ] `koopman-search` — **Koopman — Search and Screening: General Principles with Historical Applications** (OEG Report 56, 1946; republished MORS 1999). Bayesian search theory: failed looks in a box shift posterior mass away from it; allocate next look by posterior × detection probability. The exact math of our board demotion (boards = boxes, sims = looks); lineage runs through the USS Scorpion and MH370 searches.
- [ ] `gbfs-behaviour` — **Heusner, Keller, Helmert — Understanding the Search Behaviour of Greedy Best-First Search** (SoCS 2017). Formalizes how GBFS gets trapped in misleading heuristic regions (high-water-mark benches) — the named pathology our suppression trace measured with a verifier.
- [ ] `gbfs-exploration` — **Valenzano, Sturtevant, Schaeffer, Xie — A Comparison of Knowledge-Based GBFS Enhancements and Knowledge-Free Exploration** (ICAPS 2014). ε-greedy node selection as the cure for heuristic error in GBFS. *Sweep correction: venue is ICAPS, not SoCS.*
- [ ] `type-gbfs` — **Xie, Müller, Holte, Imai — Type-Based Exploration with Multiple Search Queues for Satisficing Planning** (AAAI 2014). Alternating typed queues so one misleading region can't monopolize expansion — the alternation flavor of anti-camping.
- [ ] `lds` — **Harvey & Ginsberg — Limited Discrepancy Search** (IJCAI 1995). Budget the number of times you disobey the heuristic; trust-but-bound. Closest classic in spirit to taxing (not obeying, not discarding) an unreliable ranking.
- [ ] `chaff-vsids` — **Moskewicz, Madigan, Zhao, Zhang, Malik — Chaff: Engineering an Efficient SAT Solver** (DAC 2001). VSIDS: the solver's own conflicts (failures) continuously re-weight its branching heuristic, with decay. Failure-driven heuristic adaptation at industrial scale.
- [ ] `dirt` — **Littlefield & Bekris — Efficient and Asymptotically Optimal Kinodynamic Motion Planning via Dominance-Informed Regions** (IROS 2018). Own lab. Heuristic-guided single-query planner that demotes a node after each selection so the heuristic can't camp — our regime, count-weighted where ours is verifier-evidence-weighted.
- [ ] `uct` — **Kocsis & Szepesvári — Bandit Based Monte-Carlo Planning** (ECML 2006). UCT: visit-count exploration + outcome backup. Our mechanism is the backup step in the deterministic perfect-verifier limit (no revisits, so the UCB term degenerates); PUCT (Silver et al., AlphaGo/AlphaZero) adds the learned prior, as we do.
- [ ] `rtaa-star` — **Koenig & Likhachev — Real-Time Adaptive A\*** (AAMAS 2006). The heuristic-learns-online family: exact admissibility-preserving h-updates from search experience. Contrast, not ancestor — our q has no cost algebra to correct, so our update is Bayesian trust, not arithmetic.

### Ordering candidates under a perfect, expensive verifier

*This is the objective, and it has a name and a 1979 solution. None of it was in the repo.*

- [ ] `weitzman` — **Weitzman — Optimal Search for the Best Alternative** (Econometrica
      47(3):641–654, 1979, doi:10.2307/1910412). *Pandora's Box.* N alternatives, each costs `c`
      to open, opening reveals the true value exactly, you stop when satisfied. Optimal rule is an
      index: a **reservation value** per box, open in that order. **With uniform costs — every push
      is ~1s — it collapses to: sort by success probability, descending.** This is the theory of
      the thing the ranker does, and it is 47 years old.
      *The Econometrica version is paywalled everywhere — Unpaywall, OpenAlex, Semantic Scholar
      and CORE all agree no published version is deposited anywhere. The copy here is the same
      paper as **MIT Energy Laboratory Report MIT-EL-78-008** (May 1978), written for the US
      Department of Energy under contract EX-76-A-01-2295. Pagination is `-1-` to `-21-`, not
      641–654, so cite the Econometrica version and read this one.*
      *Two traps if you re-fetch it: `scholar.harvard.edu/files/weitzman/...` ranks first in
      search and returns a 403 Akamai page; and MIT DSpace throttles with **HTTP 202 and zero
      bytes** rather than an error, so a naive fetch silently gets nothing.*
- [ ] `contextual-pandora` — **Atsidakou, Caramanis, Gergatsouli, Papadigenopoulos, Tzamos —
      Contextual Pandora's Box** (AAAI 2024, arXiv:2205.13114). Weitzman when the distributions
      are unknown and must be learned from context — i.e. when a network estimates the
      probabilities rather than an oracle handing them over.
- [ ] `pandora-correlations` — **Gergatsouli & Tzamos — Weitzman's Rule for Pandora's Box with
      Correlations** (NeurIPS 2023, arXiv:2301.13534). The index rule assumes independent boxes.
      Pushes on the same object are not independent. This is what breaks and what survives.
- [ ] `pandora-nonobligatory` — **Fu, Li, Liu — Pandora Box Problem with Nonobligatory
      Inspection** (STOC 2023, arXiv:2207.09545). When you may commit to a box without paying to
      inspect it. Hardness plus an approximation scheme.
- [ ] `adaptive-submodularity` — **Golovin & Krause — Adaptive Submodularity** (JAIR 2011,
      arXiv:1003.3967). The framework for "choose what to inspect next given what the last
      inspections revealed" — i.e. re-ranking after each failed push, if failures are informative.
- [ ] `online-mssc-pandora` — **Gergatsouli & Tzamos — Online Learning for Min Sum Set Cover and
      Pandora's Box** (ICML 2022, arXiv:2202.04870). Min-sum set cover is exactly "order things to
      minimise expected position of the first hit", learned online.
- [ ] `best-arm-fixed-confidence` — **Garivier & Kaufmann — Optimal Best Arm Identification with
      Fixed Confidence** (COLT 2016, arXiv:1602.04589). The bandit view: pure exploration, sample
      complexity lower bounds. Adjacent rather than exact — you stop at a *success*, not at
      confidence about the *best* arm — but it's the right neighbouring theory.

**The one thing this literature says that the repo does not:** sorting by `p̂/c` is Bayes-optimal
*only when the estimates are real conditional probabilities*. For 1-push that costs nothing —
order is invariant under any monotone transform, so "we need order, not calibration" is exactly
right. For 2-push it is not: comparing a direct push at `p=0.3` against a setup-then-finish at
`0.5 × 0.6 = 0.3` needs the numbers, not the ranking. The literature's form is a shallow
prerequisite tree with the finish conditioned on the realised setup state.

## Tier 3 — Domain context

### NAMO and multi-object manipulation

- [ ] `em4m` — **Saxena & Likhachev — Planning for Manipulation among Movable Objects: Deciding
      Which Objects Go Where, in What Order, and How** (ICAPS 2023, arXiv:2303.13385). 3D arm,
      nonprehensile pushing, rigid-body sim as the feasibility checker, 5–15 objects. Table 1:
      Easy 97/98, Medium 45/63, Hard 15/39 under a 5-minute timeout. The algorithm is E-M4M,
      *Enhanced*-M4M, extending M4M (Saxena, Saleem, Likhachev, ICRA 2021).
      *Three citation errors were caught here and only fetching the PDF found them: the sweep
      invented an author ("Zhu"), the audit "corrected" the venue to IROS 2023 and kept the
      invented author, and the sweep also called it "Contact-MCTS" — a repo name that appears
      nowhere in the paper. The paper is right; three attempts to cite it were not.*
- [ ] `stilman-kuffner` — **Stilman & Kuffner — Navigation Among Movable Obstacles** (IJHR 2005).
      The classic. Constraint/dependency-graph reasoning over which obstacle blocks what.
- [ ] `pamo-star` — **Ren, Suvonov, Chen, He, Liao, Fermüller, Zhang — Search-Based Path Planning
      in Interactive Environments among Movable Obstacles** (ICRA 2025, arXiv:2410.18333). PAMO* is
      complete and optimal, ~400 objects — but on a **2D occupancy grid**: objects are single cells,
      pushed exactly one cell, no chain pushes. The Box2D variant H-PAMO* adds physics and drops
      both guarantees.
- [ ] `krontiris-bekris` — **Krontiris & Bekris — Dealing with Difficult Instances of Object
      Rearrangement** (RSS 2015). Own lab. Non-monotone rearrangement: instances needing an object
      grasped more than once. Prehensile, so the contrast with pushing is the point.
- [ ] `namo-llm` — **NAMO-LLM** (arXiv:2505.04141). LLM-guided NAMO. Case Study V (n=50, Hmin=6)
      compares against two baselines: RandomTree (uniform sampling) and NAMO-SA (structured).
      *Worth reading the two baselines separately — the reported speedup differs by two orders of
      magnitude depending on which one you compare to.*
- [ ] `bench-push` — **Zhong et al. — Bench-Push** (CRV 2026, arXiv:2512.11736). A unified
      benchmark for pushing-based mobile robot navigation and manipulation, explicitly NAMO,
      open-source with baselines.

### Combinatorial search where contact dynamics decide feasibility

*Who else searches over discrete choices with physics deciding what's feasible. This is a small
field — seven papers, and two are the same group.*

- [ ] `m4m` — **Saxena, Saleem, Likhachev — Manipulation Planning Among Movable Obstacles Using
      Physics-Based Adaptive Motion Primitives** (ICRA 2021, arXiv:2102.04324). The predecessor of
      `em4m`. Physics-based primitives inside a discrete search.
- [ ] `saxena-interleaving` — **Saxena & Likhachev — Planning for Complex Non-prehensile
      Manipulation Among Movable Objects by Interleaving Multi-Agent Pathfinding and Physics-Based
      Simulation** (ICRA 2023, arXiv:2303.13352). Same group again: MAPF for the discrete
      structure, physics sim for feasibility. Closest thing to a template for interleaving the two.
- [ ] `vieira-bekris` — **Vieira, Gao, Nakhimovich, Bekris, Yu — Effective and Robust
      Non-Prehensile Manipulation via Persistent Homology Guided Monte Carlo Tree Search**
      (ISER 2023, arXiv:2210.01283). **Own lab** — and Vieira is the MORALS author. MCTS over
      non-prehensile pushes with a topological guide.
- [ ] `zhu-mcts` — **Zhu, Meduri, Righetti — Efficient Object Manipulation Planning with Monte
      Carlo Tree Search** (IROS 2023, arXiv:2206.09023). MCTS over contact modes with
      contact-implicit trajectory optimisation underneath.
      *Probably the source of the phantom "Zhu" the first sweep attached to `em4m` — there is a
      real Zhu doing MCTS for manipulation, just not that paper.*
- [ ] `ren-kinodynamic` — **Ren, Wang, Morgan, Kavraki, Hang — Object-Centric Kinodynamic Planning
      for Nonprehensile Robot Rearrangement Manipulation** (T-RO 2025, arXiv:2410.00261). The
      recent one, and Kavraki's group.
- [ ] `cheng-dexterity` — **Cheng, Patil, Temel, Kroemer, Mason — Enhancing Dexterity in Robotic
      Manipulation via Hierarchical Contact Exploration** (RA-L 2024, arXiv:2307.00383). Searching
      over contact modes. Mason's group — the funnel/mechanics lineage.
- [ ] `song-multiobject` — **Song, Haustein, Yuan, Hang, Wang, Kragic, Stork — Multi-Object
      Rearrangement with Monte Carlo Tree Search: A Case Study on Planar Nonprehensile Sorting**
      (IROS 2020, arXiv:1912.07024). MCTS, planar, nonprehensile.

### Search difficulty and curricula

- [ ] `feng-sokoban` — **Feng, Gomes, Selman — A Novel Automated Curriculum Strategy to Solve Hard
      Sokoban Planning Instances** (NeurIPS 2020, arXiv:2110.00898). Curriculum over instance
      difficulty for a hard combinatorial planning domain.
      *Sweep cited arXiv:2006.02689 — that's a different Feng/Gomes/Selman paper (IJCAI 2020).*
- [ ] `searchformer` — **Lehnert, Sukhbaatar, Su, Zheng, Mcvay, Rabbat, Tian (FAIR/Meta) — Beyond
      A\*: Better Planning with Transformers via Search Dynamics Bootstrapping** (arXiv:2402.14083).
      Transformer trained on A* search traces; optimally solves 93.7% of unseen Sokoban puzzles.
      *A separate ICLR 2024 workshop poster exists under a different title ("…Better LLM
      planning…"). "TMLR 2024" is often cited for this and could not be confirmed — use the id.*
- [ ] `phyrogen` — **Droß, Orthey, Toussaint — PhyRoGen** (arXiv:2606.06569). Generating physical
      robot manipulation puzzles with physics-based methods; 24 puzzles, 1–300s, KUKA validation.

## Tier 4 — Reference

### Methodology

- [ ] `instance-space` — **Smith-Miles & Muñoz — Instance Space Analysis for Algorithm Testing**
      (ACM CSUR 55(12):255, 2023). Footprints, benchmark bias and diversity, instance evolution.
- **Hooker — Testing heuristics: We have it all wrong** (J. Heuristics 1(1):33–42, 1995).
      *Not fetched: Springer paywalls it and no free PDF exists. Hooker's own site has PostScript
      only (`johnhooker.tepper.cmu.edu/heurist.ps`, convert with `ps2pdf`); CMU KiltHub has a PDF
      behind a signed URL that rejects HEAD requests.* On competitive vs. scientific testing of
      algorithms.
- [ ] `apple-tasting` — **Helmbold, Littlestone, Long — Apple Tasting** (Information and
      Computation 161(2):85–139, 2000). Online learning where you observe the label only when you
      predict positive. *The FOCS 1992 version ("…and nearly one-sided learning") is paywalled;
      this journal version is the expanded one and the better citation.*

### Notes on the sweep itself

The audit found no fabricated papers, but four defects worth knowing: HardSATGEN's "hardness
collapse" was a **quoted phrase that appears nowhere in that paper**; a HardCore runtime figure
could not be sourced; NeSIG's speedup was reported as 6.8× against a paper that says 15.5×; and
G2SAT was credited with preserving *relative* solver behaviour, which it does not claim. Two
arXiv ids were wrong (corrected above). Treat any number from a sweep as unverified until you've
seen it in the PDF.

`docs/papers/` holds 58 PDFs from the earlier `research_compass.md` line (TAMP, diffusion, world
models, grasping). Only `05_tamp_search_guidance/` overlaps with this list.
