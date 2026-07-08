---
status: hub
thread: f-char
tags: [hub]
updated: 2026-07-08
---
# Thread — f-char

North-star: characterize F (the feasible push set, F = C ∩ R) directly and compare a learned diffusion sampler against ground-truth F, rather than only scoring proposed pushes.

**Status:** dormant — no active cards in `docs/experiments/`; work lives under `docs/f_characterization/` (scripts) and `docs/research/`.

**Where the work actually lives:**
- [`docs/f_characterization/`](../f_characterization/) — `analyze_F.py`, `analyze_ml_vs_F.py`, `test_direction_hypothesis.py`.
- [`docs/research/F_problem_formulation.md`](../research/F_problem_formulation.md) — formal F = C ∩ R definition, baselines B0–B4.
- [`docs/research/research_notes_F_characterization.md`](../research/research_notes_F_characterization.md) — empirical notes.
- [`docs/research/scene_conditioned_sampler_design.md`](../research/scene_conditioned_sampler_design.md) — unified sampler design.
- [`docs/research/reading_list_F_characterization.md`](../research/reading_list_F_characterization.md) — focused reading list.
- [`docs/evaluation/ML_vs_GT_F_evaluation.md`](../evaluation/ML_vs_GT_F_evaluation.md) + [`ML_vs_GT_F_results_round1.md`](../evaluation/ML_vs_GT_F_results_round1.md) — diffusion-vs-GT-F eval plan + round-1 results (point robot).

**Open items:** none scheduled — pick back up by reading the evaluation round-1 results and `research_notes_F_characterization.md` first.
