# ML vs GT-F — Round 1 Results (Point Robot, 1-Push Horizon)

**Date:** 2026-05-16. Generated during the autonomous AFK churn session, calibrated with the larger rlab7 follow-up. **Model:** `cropped_diffusion_crossattn_2push/2025-12-16/05-36-44` (DiT cross-attn, 5-channel local masks, DDIM/5 steps, seed 42, 32 samples). **Test splits used:**
- **Primary (held-out, planner-designed):** 300-env stratified split `manifest_2push_test_minus_1push_test_filtered_difficulty_100each.txt`. 284 dedup instances, 177 with `|F|>0`. **Small n on hard/very_hard buckets (18 / 9).**
- **Confirmation (rlab7 1-push test set):** `manifest_test.txt` filtered to envs disjoint from the 2-push training pool (1651 of 1767 envs). 3474 dedup instances, 3282 with `|F|>0`. **12× larger; what we use to calibrate per-bucket claims.**

**Status:** 1-push horizon (F₁ only) complete on both splits. **2-push horizon (F₁′) still collecting** — chain_depth=2 GT was at ~10–30/300 envs at last check and is the main outstanding piece.

> ⚠️ **Calibration note.** The original round-1 write-up (300-env only) made several per-bucket claims that turned out to be underpowered. The rlab7 follow-up either confirmed, refined, or revised each — see §"Calibrated findings" below.

## Calibrated findings (post-rlab7)

| claim | evidence | verdict |
|---|---|---|
| Model emits ≈0% predictions at depth ≥ 7 | 0 of 5000+ predictions across buckets, upper 95% CI < 2% | **rock solid** |
| Training-target displacement is shallow (97% < 1m, median 0.30m) | direct measurement of 167,965 training rows | **rock solid** |
| ML actively loses to random-from-R at K=1 on medium/easy/very_easy | rlab7 n=475 / 1079 / 1566; lifts -0.08, -0.11, -0.09; all SIG | **established** |
| ML **matches** random-from-R at K=1 on hard / very_hard | rlab7 n=132 / 30; lifts -0.02, +0.00; CIs cross 0 | **established — refines round 1** |
| ML loses badly to random at K=10–32 on every bucket (model fails to scale with K) | large lift magnitudes, narrow CIs even on small n | **established** |
| **Face-prior is real on very_hard** | rlab7 n=30; ML face hit 0.73 vs random 0.47; lift +0.27, SIG | **confirmed (stronger than round 1)** |
| Face-prior on hard | rlab7 n=132; lift +0.06; CI [-0.06, +0.18] | inconclusive (point estimate positive) |
| Contact-point prior within face | rlab7; lifts +0.08 to +0.10; ns | inconclusive |
| Coverage drops to ~0.55 on very_hard (ML votes for unreachable slots) | rlab7 average | established |
| **Multimodality / cluster coverage** | not directly measured yet | unknown |
| **2-push horizon (F₁′) performance** | chain2 GT still collecting | unknown |

The original round-1 claim "ML strictly worse than random on every bucket" was wrong on hard/very_hard — at the larger sample, ML *matches* random there (neither helping nor hurting). It still loses on the bigger buckets where random's floor is already high. The face-prior claim, which I nearly retracted as a small-n artifact, was actually *underestimated* by round 1 — on very_hard it's +27pp, not +15pp.

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

**Direct check:** done. Depth distribution per difficulty bucket:

```
% of predictions at each depth (0=shallowest push, 9=deepest)
 bucket      src    n    d=0   d=1   d=2   d=3   d=4   d=5   d=6   d=7   d=8   d=9
 very_hard   F      29   0.0   6.9   6.9   6.9   3.4   0.0   6.9  10.3  27.6  31.0
 very_hard   R    1307  17.8  14.3  11.3  10.1   9.4   8.6   7.7   7.3   7.0   6.6
 very_hard   ML    192  20.3  25.5  29.2  16.7   6.8   0.5   1.0   0.0   0.0   0.0
 hard        F     188   0.5   1.6   4.3   6.9  10.6  11.7  12.8  15.4  17.6  18.6
 hard        R    1987  19.5  17.0  11.8   9.8   8.3   7.7   7.2   6.9   6.1   5.7
 hard        ML    338  27.8  30.5  24.6  12.7   3.3   1.2   0.0   0.0   0.0   0.0
 medium      F    1465   2.3   4.6   6.8  10.2  11.2  11.6  12.6  13.4  13.5  13.7
 medium      R    4917  15.8  13.9  11.9  10.3   9.4   8.5   7.9   7.6   7.4   7.3
 medium      ML    687  34.4  28.2  25.9   7.7   2.8   1.0   0.0   0.0   0.0   0.0
 easy        F    4186   5.5   8.1   9.3  10.2  11.0  11.1  11.2  11.2  11.2  11.2
 easy        R    7339  13.1  12.8  11.0  10.2   9.4   9.2   8.8   8.6   8.5   8.4
 easy        ML    797  37.8  34.3  21.5   4.8   1.1   0.6   0.0   0.0   0.0   0.0
 very_easy   F   12330   7.7   9.3  10.0  10.2  10.3  10.4  10.5  10.5  10.5  10.5
 very_easy   R   13966  10.6  10.5  10.2  10.0  10.0   9.8   9.8   9.7   9.7   9.7
 very_easy   ML  1412  45.7  31.4  17.0   4.0   1.3   0.4   0.1   0.0   0.0   0.0
```

**Hypothesis confirmed — and stronger than expected.** Across all 5 difficulty buckets the ML model essentially never predicts depth ≥ 6 (0.0%–1.0% mass at those depths combined). On very_hard, **58.6% of F lives at depths 8–9 where ML has zero mass**. On hard, **36.2% of F lives at d=8–9, again where ML predicts nothing**. The model's depth distribution is concentrated at d=0–2 (~95% of predictions); F's depth distribution is the opposite for hard problems and roughly uniform for easy ones. So:

- On hard: ML predicts shallow → F is deep → systematic miss.
- On easy: ML predicts shallow → F has support everywhere including shallow → some hits, but precision is dragged down because F at deep is unreachable to ML.

The "model thinks all pushes should be small" pattern is unambiguous: 0% at d=7–9 on every bucket, including very_easy where R itself is roughly uniform in depth. This isn't the model adapting to scene context — it's a hard prior baked in by the 2-push training distribution.

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
