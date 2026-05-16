# ML vs GT-F — Round 1 Results (Point Robot, 1-Push Horizon)

**Date:** 2026-05-16. Generated during the autonomous AFK churn session.
**Model:** `cropped_diffusion_crossattn_2push/2025-12-16/05-36-44` (DiT cross-attn, 5-channel local masks, DDIM/5 steps, seed 42, 32 samples).
**Test split:** 300-env stratified held-out (`manifest_2push_test_minus_1push_test_filtered_difficulty_100each.txt`), point robot, `config/namo_config_complete_skill15.yaml`.
**Status:** 1-push horizon (F₁ only) is complete on 295/300 envs / 284 dedup (xml, region, object) instances. **2-push horizon (F₁′) is still collecting** — chain_depth=2 GT is at 10/300 envs and will take hours.

## Headline (and it is bad)

The 2-push-trained diffusion model is **strictly worse than uniformly random over the reachable primitive set** at putting a feasible 1-push primitive in its top-K, on **every** difficulty bucket of the held-out 300-env split.

| bucket    | n  | K=1 ml_hit | K=1 ml_hitR | K=1 rand | liftR (K=1) | K=32 ml_hitR | K=32 rand | liftR (K=32) |
|-----------|----|-----------:|------------:|---------:|------------:|-------------:|----------:|-------------:|
| very_hard | 9  | 0.000      | 0.000       | 0.029    | **-0.029**  | 0.222        | 0.651     | **-0.429**   |
| hard      | 18 | 0.000      | 0.000       | 0.093    | **-0.093**  | 0.111        | 0.964     | **-0.853**   |
| medium    | 35 | 0.171      | 0.171       | 0.293    | **-0.122**  | 0.686        | 1.000     | **-0.314**   |
| easy      | 40 | 0.350      | 0.450       | 0.546    | **-0.096**  | 0.875        | 1.000     | **-0.125**   |
| very_easy | 75 | 0.707      | 0.813       | 0.895    | **-0.081**  | 1.000        | 1.000     | +0.000       |

- `ml_hit` = ML top-K against F₁, **raw** (some top-K slots are not in R)
- `ml_hitR` = ML top-K against F₁, **reachable-filtered** (top-K picked only from slots ∈ R, which mirrors what the planner actually executes)
- `rand` = random-from-R Top-K baseline (50 draws/instance, averaged)
- `liftR` = `ml_hitR - rand` — **negative means ML is worse than random**

Even with the reachable filter (which I added specifically to give the model the most charitable comparison), the lift is negative on every bucket at every K. On **hard problems, the gap is catastrophic**: random gets 96% hit@32, ML gets 11%.

## What's happening

Three things, observable directly from the data:

### 1. The model votes for unreachable primitives.

`ml_cov` = fraction of ML top-K aligned slots that are even in R. On very_hard problems it's 0.56–0.70: nearly half of ML's top-K cannot be physically executed by the robot. That explains part of the gap — but the reachable-filtered metric still loses, so this isn't the whole story.

### 2. The model's precision is *below* the F density of R, not above it.

For very_easy where `|F|/|R| ≈ 0.89`, a model that picks uniformly from R would have precision 0.89. ML's reachable-filtered precision is **0.79**. The model is *actively avoiding* F somehow. Not random missing — biased missing.

### 3. Recall is tiny everywhere.

`ml_recR` stays under 0.10 across all buckets and Ks. The model surfaces a small set of slots, and that set rarely intersects F at all. The ML alignment yields ~20 unique slots, but most are not in F even when |F|/|R| is high.

## Hypothesis — *why* the 2-push model loses on 1-push F₁

**The model was trained to predict push-2 from a post-push-1 state, where the object has already been displaced and the next push is typically a small adjustment near the current pose.** When given a 1-push *initial* state, it predicts the same kind of small-adjustment goals — but F₁ on hard problems requires *deep* pushes (the F-characterization paper notes feasible depth windows start at d=5.8 for very_hard). The model trained on intermediate states samples shallow-depth primitives; F₁ is concentrated at deep-depth primitives; hence systematic miss.

**Direct check (TODO in round 2):** plot the depth-index distribution of ML aligned slots vs the depth-index distribution of F₁ per bucket. If ML predictions cluster at d=0–2 while hard-bucket F₁ clusters at d=5–9, the hypothesis is confirmed in one figure.

This would also predict the 2-push F₁′ comparison (when the chain_depth=2 GT finishes): for hard envs where F₁ = ∅ and a chain is needed, push-1 in F₁′ is often *also* a "small push" (set up the geometry for push-2 to finish), and **ML should do much better on F₁′ than on F₁**. The model is the right tool for the wrong question.

If that hypothesis bears out, the take-away is sharp:
- The model has *not* learned a generic "good push" prior — it learned "what does push-2 look like."
- A separate 1-push model (or a hybrid that switches based on whether F₁ is non-empty) would beat the current single-model approach.
- This is exactly the failure mode that motivates the world-model thread: if the model can only predict goals from *familiar* (intermediate) states, it can't generalize back to the initial state distribution.

## Cross-cutting observations

- **184/300 envs have F=0 at chain_depth=1 (multi-push required).** Gated out for this analysis. The 2-push GT collection (still running) will measure F₁′ on exactly those envs.
- **The "easy" bucket is large (40 instances) and the model still loses badly** (88% vs 100% at K=32). This isn't a "hard problem" pathology — it's a model pathology.
- **The "very_easy" bucket at K=20+ ties random.** That's a ceiling effect — at K=20 with `|F|/|R|=0.89`, random gets ≥99% hits, so there's no room to improve.

## What's still blocked

1. **Chain-depth-2 GT collection.** Running in background, ETA ~3h. Once done, re-run analysis with `--horizon 2push_chain_only` and `--horizon 2push_any` to test the "model is the right tool for the wrong question" hypothesis.
2. **rlab7 1-push ML preds.** 850/3474 done as of writing; will continue in background. Larger sample size will tighten the bucket-level error bars (especially on very_hard with only n=9).
3. **Depth-index distribution of ML predictions.** Needs a small new plot in `analyze_ml_vs_F.py`.

## Files / artifacts

- GT (chain_depth=1, 295 pkls): `/common/users/dm1487/scratch_namo/f_char_2push_test_300_chain1/modular_data_westeros/`
- GT (chain_depth=2, in-progress): `/common/users/dm1487/scratch_namo/f_char_2push_test_300_chain2/`
- ML preds (300 env): `/common/users/dm1487/scratch_namo/ml_preds_2push_test_300_chain1.pkl` (284 instances)
- ML preds (rlab7 1651-env clean subset, in-progress): `/common/users/dm1487/scratch_namo/ml_preds_rlab7_clean.pkl`
- Analysis: `/common/users/dm1487/scratch_namo/ml_vs_F_results_300/1push/` (`ml_vs_F_1push.csv`, `hit_at_K_1push.png`, `recall_at_K_1push.png`, `coverage_at_K_1push.png`)

Re-run cmd:

```bash
/common/users/dm1487/envs/mjxrl/bin/python docs/f_characterization/analyze_ml_vs_F.py \
  --gt-dir /common/users/dm1487/scratch_namo/f_char_2push_test_300_chain1/modular_data_westeros \
  --ml-preds /common/users/dm1487/scratch_namo/ml_preds_2push_test_300_chain1.pkl \
  --out-dir /common/users/dm1487/scratch_namo/ml_vs_F_results_300/1push \
  --horizon 1push
```

## Things I'd be cautious about before drawing conclusions

1. **Sample sizes on the hard tail are small** (n=9 very_hard, n=18 hard on this 300-env split). The pattern is consistent across buckets so I doubt it's noise, but a 1651-env rerun will confirm.
2. **The `manifest_2push_test_minus_1push_test_filtered_difficulty_100each.txt` split is biased toward 2-push envs.** That's by construction — these are the envs where the model was *supposed* to shine. The fact that |F₁| ≈ 0 on 184 envs reflects the manifest's design, not random sampling. The 2-push-horizon analysis (when ready) is the one this split is fair for.
3. **The alignment tolerances (0.2 m, 0.2 rad) were lifted from the existing ml_collection yaml.** Tightening them might change the precision picture (fewer alignments, but the ones that remain are more likely correct). Worth a sweep in round 2.
4. **Random baseline draws are uniform from R.** A "structured random" baseline (e.g. uniform-from-each-face) could be a better floor. The current floor is already enough to make the point.
