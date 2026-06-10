# 1-push ARCHITECTURE journal — what makes the scorer good, and what data it needs

**[USER] scope (2026-06-10):** this journal = **1-push, architecture + data-efficiency ONLY**. The questions:
(1) **MASKING** — how does training on sampled+masked labels (vs exhaustive) impact the score? This prices data
collection, the binding constraint. (2) **SELF-ATTN** — does inter-edge aggregation help or hurt the score map
(the score is a pseudo-policy: it tells us what edge-info aggregation buys)? (3) **FRAMING** — sigmoid-value vs
softmax-policy training of the same head. Everything 2-push / AlphaZero / horizon-Q is **parked** in
[multipush_horizonQ_journal.md](multipush_horizonQ_journal.md) — do not let it drive experiments here.

**Rules:** hypothesis → pre-registered prediction → accept/reject with numbers; tag [USER]/[CLAUDE];
**paired matched-seed** comparisons (the E6 lesson: effects < seed-wobble ±1.7 → always pair; 3 seeds minimum).
Eval = `eval_scorer.py` on `namo_testset_v1` (geometry-verified, [[project_canonical_testset]]); verdicts via
`resolve_robust.sh`-style paired reads. All sampling-ablation runs use `+model.bce_reachable_only=true`
(BCE-scope confound control), sampling on the TRAIN split only (val stays exhaustive).

**Ordering [USER 2026-06-10]: H5 (masking) FIRST → read + understand the verdict → then decide/launch H1 + H2.**
(H1/H2 auto-chains were cancelled; implementations are ready and smoke-tested, launch is one sbatch each.)

---

## STATE — 2026-06-10 ~01:00
- ✅ **H0a — baseline**: champion scorer on the clean test set (table below). Hard recall@10=75.8, misses 90% wrong-EDGE.
- 🔄 **H5 — MASKING (the headline question)**: 15-run matrix training (job 55856898; 10 running, 5 queued).
  Eval + paired verdict when done (~morning).
- ⏸ **H1 — framing**: implemented (`head_mode=softmax_ce`, CPU-asserts pass), GATED on H5 verdict [USER].
- ⏸ **H2 — self-attn**: implemented (`edge_self_attn=false`, both arms CPU-smoke OK; ON-arms = H5's Aexh/B15 runs,
  only 6 OFF-runs needed), GATED on H5 verdict [USER].

---

## H5 — MASKING: sampled+masked labels vs exhaustive f_grid [RUNNING — the data-cost question]
**[USER] question:** collection is the expensive thing. If we train knowing only k pushes/scene (masked — unsampled
cells excluded from the loss, treated as UNKNOWN not negative), how much score quality do we lose vs exhaustive?
And is masking itself sound? If a lot of data is needed, so be it — but measure it. A validated masking verdict
also means **any future signal can be trained the same masked way** (the recipe transfers).
Lit grounding: positive-unlabeled / partial-label learning — unsampled ≠ negative; forcing unsampled-to-0 creates
false negatives (the PU bug).

**Design (5 conditions × 3 matched seeds = 15 runs, champion sharp recipe; loss mask/labels the ONLY difference):**
| cond | labels | loss on |
|---|---|---|
| Aexh | exhaustive f_grid | all reachable (ceiling CONTROL, retrained with bce_reachable_only) |
| B5/B15/B30 | k sampled cells/scene (outcome known only there) | the k sampled cells (MASKED) |
| C15 | 15 sampled; unsampled FORCED to 0 (false negatives) | all reachable (the PU-BUG baseline) |
`sample_seed=seed` → seed variance includes sampling variance (honest for "is k enough").
Impl: `scorer_data.py` (`sample_k`/`unsampled_negative` → `loss_mask`) + `classifier_module.py` (training uses
`loss_mask`); data-path unit-asserts PASS (row0: 36 positives → bug-mode creates 27 false negatives).
Runner: `sage_learning/scripts/train_h5_sampling.slurm`.

**Pre-registered predictions:**
- **H5a [CLAUDE]:** masked-B15 ≈ Aexh on easy/med recall@10 but loses meaningfully on HARD (few positives per hard
  scene; k=15/75 rarely samples them → weaker hard signal).
- **H5b [CLAUDE, high confidence]:** C15 ≪ B15 — false negatives suppress valid pushes; masking is NECESSARY.
  If C15 ≈ B15, masking doesn't matter and the PU framing is moot.
- **H5c [CLAUDE]:** monotone in k (B5 < B15 < B30 ≤ Aexh). B30→Aexh gap = what more random sampling buys.
  Gap closed by B30 → "sample more" suffices. Persistent gap → smarter (on-distribution) sampling needed.
**Decision rule:** B15 ≈ Aexh on hard → sampled collection at k≈15 is fine (just mask). Gap shrinking with k →
budget more samples/scene. Gap not shrinking → where you sample matters more than how much.

**Status:** 2-epoch smoke PASSED (B15 val_loss 0.884→0.749 healthy; C15 1.92→2.12 — bug baseline already diverging
on the exhaustive val, early corroboration of H5b). Full matrix = job 55856898. Next: eval all 15 ckpts on
namo_testset_v1 → paired verdict → [USER] reads it → then H1/H2.

---

## H2 — EDGE SELF-ATTENTION: does inter-edge aggregation help the score map? [READY, gated on H5]
**[USER] question + hypothesis:** the score is a pseudo-policy — what does edge-info aggregation buy it?
[USER] prediction: under sparse/masked labels, INDEPENDENT edges should hold up better (self-attn is a
co-adaptation channel that sparse supervision can't discipline).

**Design (2×2; the self-attn=ON arms are H5's runs — same recipe/seeds — so only 6 new runs):**
| | exhaustive labels | masked k=15 |
|---|---|---|
| self-attn ON | = h5samp_Aexh_s{1,2,3} | = h5samp_B15_s{1,2,3} |
| self-attn OFF | h2_noattn_exh_s{1,2,3} | h2_noattn_k15_s{1,2,3} |
OFF = `+network.edge_self_attn=false` — the per-layer `slf` module is NOT constructed (3.75M vs 4.35M params);
edges are scored truly independently given the scene (cross-attn only — the HACMan-faithful control the red-team
asked for). Both arms CPU-smoke verified (shapes/finiteness).
"Best kind of masking" for the masked arm = the PU-correct loss_mask masking H5 validates, at its recommended k
(k=15 placeholder; re-run the arm if H5 says a different operating point).

**Pre-registered predictions:**
- **H2a [USER]:** with masked labels the OFF arm ≥ ON arm on hard recall@10 (sparsity favors independence).
- **H2b [CLAUDE]:** with exhaustive labels ON ≥ OFF by a small margin (the champion always had self-attn; if OFF
  matches it, self-attn was dead weight all along and the simpler arch wins).
- **Interaction is the real readout:** ON−OFF gap shrinking (or flipping) from exhaustive → masked = "aggregation
  is a luxury of dense supervision."
Runner: `sage_learning/scripts/train_h2_selfattn.slurm` (smoke + 6 runs). LAUNCH AFTER H5 VERDICT [USER].

---

## H1 — FRAMING: sigmoid-value vs softmax-policy training of the same head [READY, gated on H5]
Controlled 2×2 on the sharp backbone, E4 data, matched seeds {1,2,3}, paired:
```
                       sharp target            soft target (Gaussian σ=1.0 edge/depth)
 sigmoid-BCE (scorer)  C1 = champion replica   C2  (soft was REJECTED here historically)
 softmax-CE  (policy)  C3                       C4  (theory says soft is principled here)
```
softmax-CE = legal-move-masked CE over reachable cells, multimodal targets normalized (impl `head_mode=softmax_ce`,
CPU-asserts pass: finite, masked-cells zero-grad, no-positive rows skipped, converges, sigmoid path regression-OK).
NOTE: C1/C2 keep the champion loss package (BCE all-600 + Dice); C3/C4 are CE-over-reachable — compares PACKAGES.
Red-team σ caveat stands: single σ conflates amount-vs-goodness of smoothing; sweep σ only if the interaction shows.

**Pre-registered predictions:**
- **H1a [CLAUDE]:** framing alone ≈ wash at sharp targets (C3 ≈ C1 within seed noise) — argmax is argmax.
- **H1b [CLAUDE, the key test]:** soft HELPS the policy but HURTS/neutral the sigmoid (C4 > C3 AND C2 ≤ C1) —
  action-smoothing is a *policy* thing. ACCEPT → policy+soft is a live champion candidate. REJECT if soft hurts
  both (smoothing just bad) or helps both (framing-independent).
Runner: `sage_learning/scripts/train_h1_framing.slurm` (smoke + 12 runs). LAUNCH AFTER H5 VERDICT [USER].

---

## H0a — BASELINE: champion scorer on the clean test set [2026-06-09, DONE]
Champion `sharp` ckpt (`epoch017-val_loss0.2713`), `eval_scorer.py`, 1656 episodes, `namo_testset_v1`.

| bucket | n | success@1 | recall@5 | **recall@10** | recall@20 | rank-1st-valid median |
|---|---|---|---|---|---|---|
| hard (sr 2.8%) | 413 | 32.9 | 62.5 | **75.8** | 88.1 | 3.0 |
| med  (sr 16.8%)| 491 | 81.3 | 94.5 | **96.7** | 98.6 | 1.0 |
| easy (sr 65%)  | 752 | 99.6 | 99.9 | **100** | 100 | 1.0 |

- easy/med ≈ saturated (floor@10 is 98/81 — weak signal up there). **Hard is the battleground**: 24% of episodes
  lack a winner in the top-10.
- **Failure anatomy (actionable):** of hard fail@1, **90.3% wrong-EDGE**, 6.5% right-edge-wrong-depth
  (depth-acc|right-edge = 83.4%). The depth head is fine; **contact-edge selection on hard scenes is the gap** —
  exactly what H2 (edge aggregation) and H1 (ranking-shaped loss) poke at.
- Every arch variant below is judged on this table's metric, hard bucket first.
Result JSON: `namo_testset_v1/stats/champion_1push_recall_gate.json`.

## Red-team notes retained for the arch line
- **σ confound (H1):** one σ conflates "amount" vs "goodness" of smoothing — sweep σ ∈ {0.5,1,2} before believing
  an interaction.
- **Independent-scoring control (H2 OFF arm):** scene-cross-attn-only is the obvious control for "does inter-action
  attention help or hurt" — now implemented as `edge_self_attn=false`.
- (2-push / deploy-metric / value-collection risks → parked journal.)
