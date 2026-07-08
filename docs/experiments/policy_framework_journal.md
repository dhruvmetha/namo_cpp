---
status: frozen
tags: [experiment]
thread: scorer-search
updated: 2026-06-10
---

# 1-push ARCHITECTURE journal — what makes the scorer good, and what data it needs

**[USER] scope (2026-06-10):** this journal = **1-push, architecture + data-efficiency ONLY**. The questions: (1) **MASKING** — how does training on sampled+masked labels (vs exhaustive) impact the score? This prices data collection, the binding constraint. (2) **SELF-ATTN** — does inter-edge aggregation help or hurt the score map (the score is a pseudo-policy: it tells us what edge-info aggregation buys)? (3) **FRAMING** — sigmoid-value vs softmax-policy training of the same head. Everything 2-push / AlphaZero / horizon-Q is **parked** in [multipush_horizonQ_journal.md](multipush_horizonQ_journal.md) — do not let it drive experiments here.

**Rules:** hypothesis → pre-registered prediction → accept/reject with numbers; tag [USER]/[CLAUDE]; **paired matched-seed** comparisons (the E6 lesson: effects < seed-wobble ±1.7 → always pair; 3 seeds minimum). Eval = `eval_scorer.py` on `namo_testset_v1` (geometry-verified, [[project_canonical_testset]]); verdicts via `resolve_robust.sh`-style paired reads. All sampling-ablation runs use `+model.bce_reachable_only=true` (BCE-scope confound control), sampling on the TRAIN split only (val stays exhaustive).

**Ordering [USER 2026-06-10]: H5 (masking) FIRST → read + understand the verdict → then decide/launch H1 + H2.** (H1/H2 auto-chains were cancelled; implementations are ready and smoke-tested, launch is one sbatch each.)

---

## STATE — 2026-06-10 morning: ALL CLOSED ✅ (33 runs, 0 crashes, verdicts below)
- ✅ **H0a** — baseline on the clean test set: hard recall@10=75.8, misses 90% wrong-EDGE.
- ✅ **H5 — masking**: NECESSARY (unmasked bug catastrophic) and ~30 labels/scene MAINTAINS exhaustive quality.
- ✅ **H1 — framing**: wash @1; soft hurts BOTH framings → keep sigmoid-sharp.
- ✅ **H2 — self-attn**: helps in EVERY regime, most under sparsity → keep ON.

### RE-EVAL on the STRICTER 20% test set [2026-06-11] — verdicts ROBUST
All 33 H5/H1/H2 ckpts re-evaluated on `namo_testset_v1` regenerated under the wired 20% success bar (`labels/onepush_episodes.json`, 1323 1-push eps; see [[project_canonical_testset]]). Result table: `namo_testset_v1/stats/newbar_reeval_h5h1h2.json`. Format: hard@1 NEW(20%) vs OLD(≥1pt).

| | NEW hard@1 | OLD hard@1 | verdict |
|---|---|---|---|
| H5 staircase | Aexh 26.1 > B30 23.3 > B15 22.4 > B5 14.1 | 31.3/28.5/26.0/18.1 | **HOLDS** (monotone; ~30 maintains) |
| H5 masking-necessity | C15 12.4 < **even B5 14.1** | 13.4 < 18.1 | **HOLDS** (false-negatives poison) |
| H1 soft | C2 24.7<C1 26.1 AND C4 22.9<C3 24.0 | both < | **HOLDS** (soft hurts both → keep sigmoid-sharp) |
| H2 self-attn | ON>OFF every regime, +6.2 @k15 (>old +5.1) | +4.1/+5.1/+3.0 | **HOLDS** (keep ON) |

**Conclusion: every architecture/data verdict is ROBUST to the success-bar change.** Caveat [honest]: all absolute numbers drop ~5pp (Aexh 31.3→26.1) — a **train(old-bar)/test(new-bar) MISMATCH**, since these models were trained on the ≥1-point f_grid but graded on the ≥20% ground truth. RELATIVE orderings (what the verdicts are) are unchanged. The clean follow-up if ever needed: re-collect the TRAIN f_grid under the 20% bar so train/test match — separate larger run, NOT required to trust the H1/H2/H5 decisions. Eval key for all future runs is now the 20% one.

### NIGHT SUMMARY — the cross-cutting observations [2026-06-10]
1. **The champion architecture survived both challengers.** Sigmoid-sharp (H1) and self-attention (H2) were each put on trial with pre-registered alternatives; both alternatives lost. The architecture is now evidence-backed, not historical accident.
2. **The night's genuinely NEW knowledge is the DATA recipe:** sampled+masked training works. Diminishing-returns curve on hard@1: 5→15 labels = +7.9, 15→30 = +2.5, 30→75 = +2.8 (≈ seed noise ±3.6). **k≈30/scene is the knee** — 2.5× cheaper than exhaustive, indistinguishable quality. More never hurts; planning shouldn't require it.
3. **The PU lesson, quantified:** treating untried pushes as failures (C15) is worse than having 10 FEWER honest labels (B5). False negatives are ~not a degradation — they're poison. Masking is not optional.
4. **Self-attention surprise [USER hypothesis rejected, mechanism revised]:** the predicted failure mode ("unaccountable edges pollute via attention under sparse labels") inverted — the ON−OFF gap is LARGEST at k15 (+5.1). Revised mechanism: attention PROPAGATES sparse supervision (labeled edges teach unlabeled neighbors). The co-adaptation channel is a communication channel.
5. **Theories died properly:** H1b (action-smoothing is a policy thing — CLAUDE theory) and H2a (sparsity favors independence — USER theory) both rejected with numbers. Pre-registration kept us honest both ways.
6. **Process validations:** snapshot-feeler (ep~15) predicted every final ordering → cheap mid-training previews are trustworthy. Baseline registry saved 10 of 33 runs. Masked-loss bug audit (exact-zero grads outside mask, per-run config dumps) came back clean — the verdicts stand on audited code.
7. **Carry-forward to multi-push (parked journal + primer):** architecture unchanged; collect at ~30 sampled pushes/state, masked; search-generated MC labels; the residual ~3pp (B30→exh) is the "sample smarter (ExIt round)" question, deferred until something needs it.

---

## BASELINE REGISTRY [USER rule 2026-06-10: never retrain a baseline that already exists]
| asset (3 seeds each) | reusable as |
|---|---|
| `h5samp_Aexh_s{1,2,3}` (exhaustive, bce_reachable_only, self-attn ON) | H5 ceiling · H2 ON×exh arm · **H1 sigmoid-sharp cell C1** |
| `h5samp_B15_s{1,2,3}` (masked k=15) | H5 curve · H2 ON×masked arm |
| `h5samp_B30_s{1,2,3}` (masked k=30) | H5 curve · standing masked-operating-point baseline |
| champion `sharp_s1` ckpt | historical reference (re-eval only) |
Snapshot-feeler protocol (matched-epoch last.ckpt + CPU eval array) makes new conditions comparable to these without retraining. ⇒ H1 = 9 new runs (not 12; C1:=Aexh — also removes the BCE-scope footnote), H2 = 6.

---

## H5 — MASKING: sampled+masked labels vs exhaustive f_grid [DONE 2026-06-10 — the data-cost question]
**[USER] question:** collection is the expensive thing. If we train knowing only k pushes/scene (masked — unsampled cells excluded from the loss, treated as UNKNOWN not negative), how much score quality do we lose vs exhaustive? And is masking itself sound? If a lot of data is needed, so be it — but measure it. A validated masking verdict also means **any future signal can be trained the same masked way** (the recipe transfers). Lit grounding: positive-unlabeled / partial-label learning — unsampled ≠ negative; forcing unsampled-to-0 creates false negatives (the PU bug).

**Design (5 conditions × 3 matched seeds = 15 runs, champion sharp recipe; loss mask/labels the ONLY difference):**
| cond | labels | loss on |
|---|---|---|
| Aexh | exhaustive f_grid | all reachable (ceiling CONTROL, retrained with bce_reachable_only) |
| B5/B15/B30 | k sampled cells/scene (outcome known only there) | the k sampled cells (MASKED) |
| C15 | 15 sampled; unsampled FORCED to 0 (false negatives) | all reachable (the PU-BUG baseline) |
`sample_seed=seed` → seed variance includes sampling variance (honest for "is k enough"). Impl: `scorer_data.py` (`sample_k`/`unsampled_negative` → `loss_mask`) + `classifier_module.py` (training uses `loss_mask`); data-path unit-asserts PASS (row0: 36 positives → bug-mode creates 27 false negatives). Runner: `sage_learning/scripts/train_h5_sampling.slurm`.

**Pre-registered predictions:**
- **H5a [CLAUDE]:** masked-B15 ≈ Aexh on easy/med recall@10 but loses meaningfully on HARD (few positives per hard scene; k=15/75 rarely samples them → weaker hard signal).
- **H5b [CLAUDE, high confidence]:** C15 ≪ B15 — false negatives suppress valid pushes; masking is NECESSARY. If C15 ≈ B15, masking doesn't matter and the PU framing is moot.
- **H5c [CLAUDE]:** monotone in k (B5 < B15 < B30 ≤ Aexh). B30→Aexh gap = what more random sampling buys. Gap closed by B30 → "sample more" suffices. Persistent gap → smarter (on-distribution) sampling needed. **Decision rule:** B15 ≈ Aexh on hard → sampled collection at k≈15 is fine (just mask). Gap shrinking with k → budget more samples/scene. Gap not shrinking → where you sample matters more than how much.

**RESULT [2026-06-10 FINAL — best ckpt (ep≤21) per run, n=3 seeds, namo_testset_v1]:**
| cond | hard@1 | hard@5 | hard@10 | med@10 | easy@10 |
|---|---|---|---|---|---|
| Aexh | **31.3** ±3.6 | 64.3 ±2.0 | **78.6** ±0.8 | 98.2 | 100 |
| B30 | 28.5 ±0.9 | 61.1 ±1.4 | 75.3 ±3.8 | 97.5 | 99.9 |
| B15 | 26.0 ±3.7 | 56.5 ±0.9 | 72.3 ±1.0 | 96.9 | 100 |
| B5 | 18.1 ±0.2 | 50.2 ±0.6 | 66.1 ±0.7 | 96.2 | 100 |
| C15 (bug) | **13.4** ±2.0 | 41.9 ±5.8 | 58.5 ±5.6 | 92.1 | 99.6 |

**Verdicts:**
- **H5b ACCEPT (loud):** C15 vs B15 = −12.6 @1 / −14.6 @5 / −13.8 @10. The unsampled-as-negative bug is CATASTROPHIC — C15 is even worse than B5 (5 honest labels beat 15 labels + ~60 lies). **Masking is NECESSARY.**
- **H5c ACCEPT:** strictly monotone in k on every hard metric (B5 < B15 < B30 < Aexh).
- **H5a ACCEPT:** med/easy unaffected (96–100 across masked conditions); ALL damage concentrates on hard.
- **MAINTENANCE ([USER]'s question):** B30 = −2.8 @1 / −3.3 @10 vs exhaustive — at the seed-noise boundary (±3.6). **~30 labels/scene ≈ maintains; 15/scene costs ~5–6pp hard; 5/scene costs ~13pp.** THE PRICE LIST for future collection: ~30 sims/scene is the operating point (2.5× cheaper than exhaustive); the residual ~3pp gap is what smarter-than-random sampling could still recover. **Mid-training feelers (ep~15) predicted every ordering correctly** — snapshot-feeler protocol validated.

---

## H2 — EDGE SELF-ATTENTION: does inter-edge aggregation help the score map? [DONE 2026-06-10]
**[USER] question + hypothesis:** the score is a pseudo-policy — what does edge-info aggregation buy it? [USER] prediction: under sparse/masked labels, INDEPENDENT edges should hold up better (self-attn is a co-adaptation channel that sparse supervision can't discipline).

**Design (2×2; the self-attn=ON arms are H5's runs — same recipe/seeds — so only 6 new runs):**
| | exhaustive labels | masked k=15 |
|---|---|---|
| self-attn ON | = h5samp_Aexh_s{1,2,3} | = h5samp_B15_s{1,2,3} |
| self-attn OFF | h2_noattn_exh_s{1,2,3} | h2_noattn_k15_s{1,2,3} |
OFF = `+network.edge_self_attn=false` — the per-layer `slf` module is NOT constructed (3.75M vs 4.35M params); edges are scored truly independently given the scene (cross-attn only — the HACMan-faithful control the red-team asked for). Both arms CPU-smoke verified (shapes/finiteness). "Best kind of masking" for the masked arm = the PU-correct loss_mask masking H5 validates, at its recommended k (k=15 placeholder; re-run the arm if H5 says a different operating point).

**Pre-registered predictions:**
- **H2a [USER]:** with masked labels the OFF arm ≥ ON arm on hard recall@10 (sparsity favors independence).
- **H2b [CLAUDE]:** with exhaustive labels ON ≥ OFF by a small margin (the champion always had self-attn; if OFF matches it, self-attn was dead weight all along and the simpler arch wins).
- **Interaction is the real readout:** ON−OFF gap shrinking (or flipping) from exhaustive → masked = "aggregation is a luxury of dense supervision." **RESULT [2026-06-10 FINAL — n=3 each; OFF runs ep20-capped (peak ~ep15); +k30 arm added per [USER]]:**
| | hard@1 | hard@5 | hard@10 | med@10 |
|---|---|---|---|---|
| ON × exh (=Aexh) | **31.3** ±3.6 | 64.3 | 78.6 ±0.8 | 98.2 |
| OFF × exh | 27.2 ±2.4 | 59.6 | 75.0 ±0.6 | 97.1 |
| ON × k15 (=B15) | **26.0** ±3.7 | 56.5 | 72.3 ±1.0 | 96.9 |
| OFF × k15 | 20.9 ±1.7 | 51.9 | 68.4 ±0.5 | 95.8 |
| ON × k30 (=B30) | **28.5** ±0.9 | 61.1 | 75.3 ±3.8 | 97.5 |
| OFF × k30 | 25.5 ±2.0 | 58.0 | 72.5 ±0.4 | 96.8 |

**Verdicts:**
- **H2a [USER] REJECT:** sparsity does NOT favor independence — the ON advantage holds in every regime and is, if anything, LARGEST at k15 (−5.1 @1) where the hypothesis predicted it would flip. The "unaccountable whisperers" mechanism didn't materialize; inter-edge attention appears to PROPAGATE sparse supervision (a labeled edge's gradient shapes its neighbors through the attention weights), helping precisely when labels are few.
- **H2b ACCEPT (stronger than predicted):** ON > OFF by ~4pp @1 at exhaustive — self-attention was never dead weight; it earns its 0.6M params in all regimes.
- **Interaction readout:** ON−OFF gap @1 = 4.1 (exh) / 5.1 (k15) / 3.0 (k30) — no shrink under masking. **DECISION → keep edge self-attention ON, in every data regime.** (Runner: `train_h2_selfattn.slurm`; ON arms reused from H5 per the registry.)

---

## H1 — FRAMING: sigmoid-value vs softmax-policy training of the same head [DONE 2026-06-10]
Controlled 2×2 on the sharp backbone, E4 data, matched seeds {1,2,3}, paired:
```
                       sharp target            soft target (Gaussian σ=1.0 edge/depth)
 sigmoid-BCE (scorer)  C1 = champion replica   C2  (soft was REJECTED here historically)
 softmax-CE  (policy)  C3                       C4  (theory says soft is principled here)
```
softmax-CE = legal-move-masked CE over reachable cells, multimodal targets normalized (impl `head_mode=softmax_ce`, CPU-asserts pass: finite, masked-cells zero-grad, no-positive rows skipped, converges, sigmoid path regression-OK). NOTE: C1/C2 keep the champion loss package (BCE all-600 + Dice); C3/C4 are CE-over-reachable — compares PACKAGES. Red-team σ caveat stands: single σ conflates amount-vs-goodness of smoothing; sweep σ only if the interaction shows.

**Pre-registered predictions:**
- **H1a [CLAUDE]:** framing alone ≈ wash at sharp targets (C3 ≈ C1 within seed noise) — argmax is argmax.
- **H1b [CLAUDE, the key test]:** soft HELPS the policy but HURTS/neutral the sigmoid (C4 > C3 AND C2 ≤ C1) — action-smoothing is a *policy* thing. ACCEPT → policy+soft is a live champion candidate. REJECT if soft hurts both (smoothing just bad) or helps both (framing-independent). **RESULT [2026-06-10 FINAL — n=3, ep20-capped; all cells peaked ep12–19 (no cut-mid-climb; one pol_soft seed at 19)]:**
| cell | hard@1 | hard@5 | hard@10 | med@10 |
|---|---|---|---|---|
| C1 sig_sharp (=Aexh) | **31.3** ±3.6 | 64.3 ±2.0 | 78.6 ±0.8 | 98.2 |
| C2 sig_soft | 30.0 ±0.3 | 63.4 ±0.4 | 75.2 ±0.5 | 96.4 |
| C3 pol_sharp | 30.7 ±0.4 | **66.7** ±1.2 | **80.2** ±1.3 | 97.2 |
| C4 pol_soft | 27.7 ±1.5 | 63.6 ±2.5 | 76.4 ±1.6 | 95.8 |

**Verdicts:**
- **H1a ACCEPT:** sharp framing is a wash at @1 (30.7 vs 31.3) — argmax is argmax.
- **H1b REJECT (theory killed by data):** soft hurts the POLICY too (C4 < C3, −3.0 @1) and the sigmoid as always (C2 ≤ C1, −3.4 @10). At σ=1 smoothing is just bad — framing-INDEPENDENT. [USER skepticism vindicated.]
- **Noted (not pre-registered):** pol_sharp shows a small consistent recall bump @5/@10 (+2.4/+1.6, ~1–1.5σ) — CE ordering may buy some top-k recall on EXHAUSTIVE labels. Not actionable: CE targets are corrupted by sampled data (the masking asymmetry) and the sampled regime is where we're headed. Parked as an "ordering finetune" footnote. **DECISION → keep sigmoid-sharp:** simpler, per-cell calibrated, masking-compatible, not measurably worse. (Runner: array 3-11; C1 reused Aexh per the registry.)

---

## H0a — BASELINE: champion scorer on the clean test set [2026-06-09, DONE]
Champion `sharp` ckpt (`epoch017-val_loss0.2713`), `eval_scorer.py`, 1656 episodes, `namo_testset_v1`.

| bucket | n | success@1 | recall@5 | **recall@10** | recall@20 | rank-1st-valid median |
|---|---|---|---|---|---|---|
| hard (sr 2.8%) | 413 | 32.9 | 62.5 | **75.8** | 88.1 | 3.0 |
| med  (sr 16.8%)| 491 | 81.3 | 94.5 | **96.7** | 98.6 | 1.0 |
| easy (sr 65%)  | 752 | 99.6 | 99.9 | **100** | 100 | 1.0 |

- easy/med ≈ saturated (floor@10 is 98/81 — weak signal up there). **Hard is the battleground**: 24% of episodes lack a winner in the top-10.
- **Failure anatomy (actionable):** of hard fail@1, **90.3% wrong-EDGE**, 6.5% right-edge-wrong-depth (depth-acc|right-edge = 83.4%). The depth head is fine; **contact-edge selection on hard scenes is the gap** — exactly what H2 (edge aggregation) and H1 (ranking-shaped loss) poke at.
- Every arch variant below is judged on this table's metric, hard bucket first. Result JSON: `namo_testset_v1/stats/champion_1push_recall_gate.json`.

## Red-team notes retained for the arch line
- **σ confound (H1):** one σ conflates "amount" vs "goodness" of smoothing — sweep σ ∈ {0.5,1,2} before believing an interaction.
- **Independent-scoring control (H2 OFF arm):** scene-cross-attn-only is the obvious control for "does inter-action attention help or hurt" — now implemented as `edge_self_attn=false`.
- (2-push / deploy-metric / value-collection risks → parked journal.)
