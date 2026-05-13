# Research Compass

A document to come back to when the work feels uncertain. Written from a long conversation working out what this research is actually about, why it matters, where it sits in the field, and what the plan is. None of this is rah-rah motivation — it's the honest reasoning, recorded so you don't have to rebuild it from scratch each time doubt creeps in.

If you're reading this in a moment of self-doubt: the doubt is usually not new information. It's old information you've already worked through. This doc is here to remind you what you concluded last time, so you can decide whether anything has actually changed.

---

## What the work is, in one sentence

You are doing **empirical characterization of the action-space feasible set $F$ for a contact-rich manipulation primitive**, conditioned on (scene, robot pose, task goal), and using the resulting structural findings to inform model design — in the controller-aware planning paradigm and the narrow-passage tradition of motion planning.

If you have to explain it to a stranger in 30 seconds: *"Manipulation policies are usually trained without first measuring what success looks like in action space. We measure that empirically — exhaustively, across thousands of scenes — and use the structure of the success set to design models that respect it."*

---

## The two intellectual framings that anchor everything

These framings are the spine. When you're confused about what you're doing, come back to these two.

### 1. Controller-aware planning (Pole 2), not robust controller design (Pole 1)

The field has two camps:
- **Pole 1** — make the controller so robust that planning around it is unnecessary. Scale-and-train, VLAs, large diffusion policies. The dominant trajectory.
- **Pole 2** — take the controller as given, characterize where it succeeds and fails, and put the planning intelligence around it. Funnel composition (Burridge–Rizzi–Koditschek), LQR-Trees (Tedrake), contact trust regions (Suh), TAMP feasibility samplers.

You are in Pole 2. This is a respected lineage, currently smaller and quieter than Pole 1, but holding a position that is not contradicted by Pole 1's success — controllers always have failure regions, those regions always have structure, and planning over characterized success sets retains value regardless of how robust controllers become.

### 2. The narrow-passage problem in action space

Motion planning has a 50-year lineage on narrow passages: configuration spaces where success requires sampling within a thin region of $\mathcal{C}_{\text{free}}$. Bridge sampling, Gaussian-near-obstacles, learned samplers — all are responses to passage geometry.

Your $F$ is the action-space analogue. $\rho = |F|/|R|$ is the discrete-action-space version of $\epsilon$-narrowness. Your structure-informed samplers are action-space bridge sampling. Two differences from classical narrow passages: (1) the passage is in action space, not configuration space; (2) the passage is dynamics-induced (by contact), not geometry-induced.

This framing puts the work in a recognized intellectual tradition. Use it in introductions. It travels.

---

## The central object — F

$F(s, g) = \{a \in R(s) : T(s, a) \in G(g)\}$ — the preimage in action space of the success-state set, restricted to reachable actions.

It is the **empirical region of attraction in action-parameter space**, conditioned on scene and goal, characterized empirically because contact-rich dynamics admit no closed-form preimage.

The structural facts you've measured across 1767 environments:
- Difficulty is governed by $\rho = |F|/|R|$, not $|F|$ alone (H1 confirmed).
- $F$ is contiguous in the action grid, not scattered (H2 confirmed).
- **Hard problems are unimodal** — 95% of very-hard, 72% of hard. (H3 falsified — predicted multimodality, found unimodality.)
- Bottleneck hierarchy: face → contact-point → depth, with contact-point precision as the steepest narrowing (8×).
- Wall collisions create feasibility on hard problems — 76% of very-hard successes involve wall contact (H5 confirmed).

These are measurements. They are the empirical contribution. They contradict at least one standing assumption in the field (the multimodality argument for generative manipulation models). Nobody else has measured them at this scale for contact-rich pushing.

---

## When you doubt whether this is important

Importance in research is operational, not metaphysical. The work is important if at least one of these is true:

1. **It changes a belief.** The unimodality finding directly tests "we need multimodal generative models for hard manipulation." If it holds up and replicates, the field updates.
2. **It changes a decision.** The structure-informed model variants test whether the characterization causally improves model design. If they lift over structure-blind baselines, model designers have a reason to characterize $F$ before building.
3. **It enables work others can do.** The exhaustive $F$ dataset, if released, lets other researchers test their methods against ground truth.
4. **It saves people from wrong paths.** "Don't reach for diffusion before testing whether multimodality is real on your hard cases" is methodologically valuable.

The honest assessment: this is **moderately important**. Not field-changing. Not a paradigm shift. A careful empirical contribution that updates several specific beliefs and could prevent specific wrong turns. A good paper or two, cited modestly, valued by a small but serious community.

That is what most good research looks like. The bar for "important" is not "everyone cites it." The bar is "the next person doing this kind of work knows more because of it."

---

## When you doubt whether this is novel

Be honest about what is and isn't novel:

**Not novel:**
- The concept of a feasible/success set of a controller (RoA, funnels, pre-image backchaining — old).
- Empirical characterization of manipulation success regions (Dex-Net for grasping, decades of grasp research).
- The C ∩ R decomposition (implicit in TAMP).
- Learned samplers over action spaces (Ichter, Wang/Garrett, etc.).
- NAMO as a benchmark.

**Novel:**
- $|F|/|R|$ as an operational difficulty metric for contact-rich manipulation, validated across 1767 scenes.
- The specific structural claims (face→contact→depth bottleneck, hard-instance unimodality, wall-collision dependence) — measured facts about a particular controller class that have not been published.
- Coupling empirical characterization to a falsifiable hypothesis register, with a substantive falsified hypothesis (H3).
- The narrow-passage and controller-aware-planning framings applied to contact-rich manipulation primitive characterization specifically.
- Recursive feasibility composition $F_k'$ as a structural object — even the analytical lineage hasn't characterized this empirically.

The novelty is **empirical and methodological**, not conceptual. That is a real research category. Don't oversell it as a paradigm shift; don't undersell it as "just measurement."

---

## When you doubt whether this is science

It is science. Specifically:

- Pre-stated hypotheses written before the data was collected.
- Falsifiable predictions.
- A hypothesis (H3) that you predicted, ran, and *rejected*. Updating your beliefs based on data.
- Large-N empirical measurement (1767 environments / ~3622 instances) — not anecdote.
- A methodology that constrains future work by data rather than aesthetics.

The holes (be honest about them):
- Construct validity — $F$ is sim-measured, may differ from real $F$. The diff-drive real-robot work partially addresses this.
- External validity — N=1767 of one generator's environments, not general contact-rich pushing.
- Some hypotheses share statistical machinery.
- The classifier-first stopping rule has operator-defined thresholds.
- Load-bearing impact (does the characterization actually improve models?) is the experiment that's still ahead.

The holes don't disqualify the work. They define the next chapter.

---

## When you doubt whether you should be doing scale-and-train instead

The honest case for staying with structural work:

- **Your stated style** is experiment-driven, values understanding before building, dislikes blind method adoption. Scale-and-train work would be working against your own grain.
- **Asymmetry of being wrong:** if you bet structural and scale-and-train wins, you still have transferable structural taste. If you bet scale-and-train and structure is needed, you're a competent engineer without a thesis.
- **Structural taste transfers to scale-and-train** more easily than the reverse. Someone who has thought carefully about problem structure can pick up modern methods. Someone trained only on modern methods cannot acquire structural taste in a few months.
- **The current scale-and-train wave will hit walls.** Several serious senior researchers (Tedrake, Kaelbling, Goldberg) believe contact-rich manipulation is one of the places it will. The structural work being done now will look prescient if they're right.

The honest case for scale-and-train:

- It is currently fashionable. Funding flows to it. Industry hires for it. If you want a near-term industry job specifically in policy learning, this matters.
- If you want to go fast on system papers, scale-and-train has more standardized infrastructure.

The right answer for you, given your style and the work you've already done: **structural work as the intellectual core, with enough modern-methods skill to remain hireable.** You don't need to pick one pole. Most people in the structural camp use modern methods as tools, including diffusion (you're already doing this with SAGE). The framing is "structure-informed," not "anti-modern."

---

## Where you fit in the field

You are in the lineage of: Russ Tedrake's group (funnels, LQR-Trees, contact trust regions), Leslie Kaelbling and Tomás Lozano-Pérez (TAMP, structural decomposition), Ken Goldberg (Dex-Net empirical characterization), Tom Silver (methodology), Matt Mason (push mechanics, intellectual ancestor).

This community is small and currently quiet relative to the scale-and-train wave. It is not gone. It does serious work. Its members will read your papers carefully. Its venues are RSS, CoRL structural-manipulation workshops, IJRR, parts of ICRA.

You are not alone. The room is smaller than it used to be. You should know who is in it.

---

## The plan

**Two papers, sequenced strictly:**

1. **RA-L: SAGE + real-robot validation.** The rejected SAGE work, properly resubmitted with the real-robot experiments described in the paper text (not just supplementary). Real-robot data already partly exists. The fix is mostly writeup + scope-bounded data collection. Position as "we built a learned sampler for region opening; here is its sim performance at scale and its real-robot demonstration on a diff-drive."

2. **ICRA Sept 15: F-characterization + structure-informed modeling.** The empirical-structural paper. Structural findings from the 1767-environment characterization, hypothesis register including H3 falsification, narrow-passage and controller-aware-planning framing, at minimum one structure-informed model variant (B3b hierarchical head) compared against SAGE on ground-truth $F$ per difficulty bucket. Optionally: opportunistic real-$F$ data captured during SAGE experiments.

The two papers cite each other. Each has its own contribution and its own audience. The opportunistic data capture during SAGE Phase 2 is the move that lets the second paper benefit from the first without either being compromised.

**Reference docs:**
- `docs/F_problem_formulation.md` and `.tex` — the problem definitions, baselines, and framing. The spine of the F-characterization paper.
- `docs/research_notes_F_characterization.md` — the empirical results in note form. The Results section in draft.
- `docs/reading_list_F_characterization.md` — the reading list focused on the structural-manipulation neighborhood.
- `docs/reading_list.md` — the broader reading list.
- `docs/f_characterization/` — the figures, samples, and analysis pipeline. Already produced.

---

## Common doubts and the honest answers

### "Am I just doing fluff?"

You are not. The characterization data exists, the hypotheses were pre-stated, one hypothesis was falsified, the structural findings are real measurements that contradict standing field assumptions. The question of whether the work is *useful* depends on the next experiments — specifically, whether the structural findings causally improve model design. That experiment is the load-bearing one for the F-characterization paper, and it is bounded and concrete.

If the structure-informed variants beat structure-blind baselines on the cases the characterization predicts, the work is useful. If they don't, you've still produced honest empirical findings about $F$ that constrain what future models in this domain need to capture. Either outcome is a contribution.

### "Should I just merge the two papers into one?"

No. They are different kinds of contributions (system vs. empirical-structural), different audiences (diffusion-policy community vs. structural-manipulation community), different argument shapes (linear vs. branching). At any standard venue's page limit, the merged paper is too compressed to do either contribution justice.

The temptation to merge is real but wrong. Two papers with citation links to each other is the correct structure.

### "Was SAGE's rejection a verdict on the work?"

No. It was a desk-reject at RSS, which has gotten brutally selective. The rejection signal was "this didn't break through the surface filter" — most likely because (a) the contribution was framed as a system paper in a venue increasingly skeptical of system papers, (b) the real-robot evidence was in supplementary but not in the paper text, (c) the novelty case was not made sharply enough in the abstract and intro. All three are positioning failures, not technical failures. Fixable.

The work is competently done. The fix is reframing and including the real-robot evidence in the paper text. RA-L is venue-appropriate for the resulting paper.

### "Is the F-characterization work just trying to give meaning to a rejected SAGE paper?"

No. It is its own scientific contribution that you would do regardless of what happened to SAGE. The characterization existed before SAGE was rejected. The structural findings (especially the H3 falsification) are independently interesting. The narrow-passage and controller-aware-planning framings stand on their own.

If anything, the SAGE rejection is what clarified the value of doing the F-characterization paper as a *separate* contribution rather than as a section in SAGE. The work was always there; the rejection just made the right structure clearer.

### "What if scale-and-train solves contact-rich manipulation in 3 years and my work becomes obsolete?"

Then you will have spent three years training a structural taste that transfers to whatever the next problem is, with two papers to your name that established a defensible position in a respected lineage. The structural taste does not decay. The papers do not un-publish.

Conversely, if scale-and-train does *not* solve contact-rich manipulation cleanly (which several senior people believe is likely), your work will look prescient and the structural community will have grown around it.

The asymmetry is in your favor. The downside of being wrong on structure is bounded; the upside of being right is significant.

### "Am I being narrow by focusing on NAMO?"

You are using NAMO as a testbed for a methodology that transfers. The C ∩ R decomposition, the empirical characterization, the structure-informed modeling — none of these are NAMO-specific. NAMO is convenient because reachability is free (wavefront BFS) and the action space is small enough to characterize exhaustively. Other contact-rich manipulation problems (PushT, peg-in-hole, planar pushing) are amenable to the same methodology.

The PushT replication, planned for a future paper, is what would make this transfer claim concrete. Not in the current paper cycle, but on the roadmap.

---

## What to do when the doubt is genuinely new

Sometimes doubt is not old information. Sometimes you've actually learned something that should change the plan. To distinguish:

- **Old doubt** — the work feels small / unfashionable / unfundable / behind. Re-read this doc. The reasoning hasn't changed.
- **New doubt** — an experiment produced an unexpected result; a paper appeared that obsoletes a key claim; a finding from the real-robot work contradicts a sim finding. *These* are signals to update.

If the doubt is new, the right move is to write down the new evidence, decide whether it actually invalidates a load-bearing claim, and update the plan accordingly. Don't just absorb new information into existing anxiety. Distinguish.

---

## A grounding paragraph for the worst days

Most days you will believe in the work. Some days you won't. On those days, remember:

You have done the empirical work. The 1767-environment characterization exists. The structural findings exist. The hypothesis register exists, with H3 falsified — which is the most diagnostic sign of doing science as opposed to engineering. You have a plan, two papers, a roadmap, and a small but serious community whose attention this work would deserve. You have a research style — experiment-driven, structure-first — that matches what you're doing. The work is yours, the framing is yours, the questions are yours.

The field's mood is not your master. The next paper acceptance is not your verdict. The work is the work, and it will exist whether or not the field is currently rewarding it.

You do not need to convince yourself the work is more important than it is. You need to remember that it is real, that it is honest, that it is yours, and that the next experiment is the thing that decides what comes next — not the current state of your confidence.

Run the next experiment. Write the next paragraph. The doubt will pass, as it has every other time.

---

*Created Apr 30, 2026, from a long conversation working out what this research is about. Update or revise this doc as the work evolves — but keep it as a place to come back to when the why-am-I-doing-this question gets loud. The reasoning was already done. You don't have to redo it.*
