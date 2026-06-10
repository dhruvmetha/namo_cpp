# Multi-push / horizon-Q journal — ⏸ PARKED [USER 2026-06-10]

**[USER] scope decision:** "Focus only on the architectural decisions on the 1-push problem. Don't worry about the
2-push problems yet." Everything 2-push / AlphaZero / policy+value / horizon-Q moved HERE from
[policy_framework_journal.md](policy_framework_journal.md) (now the 1-push ARCHITECTURE journal) for future use.
Nothing in this file drives current experiments. The arch journal's H5 (masking) and H2 (self-attn) verdicts feed
back into the collection design below when this line un-parks.

---

## Original thesis (now revised — see H3′ below)
From the 3-agent AlphaZero/MuZero sweep ([[project_policy_value_not_q]]): when you act via SEARCH, the net should
output a **policy prior** + a **value V(s)** — NOT a standalone Q (the search computes Q). Soft/Gaussian
action-smoothing is principled for a *policy* (cross-entropy over a distribution) but biased for a *Q* (independent
absolute values) — retro-explains the sharp-beat-soft result (the scorer is Q-like).
**Red-team demotion:** in the FEW-SIM regime the search visits k≪300 actions, so "search computes Q" is false for
the unvisited ones; the per-action scorer ranks ALL 300 and can't miss that way ⇒ policy-vs-Q was demoted from
decision to hypothesis. H0a/H0b then measured the free options (below).

## H3′ — HORIZON-Q: the converged design [2026-06-10 synthesis, USER+CLAUDE discussion]
**Decision trail (revises the thesis):** ONE function, not two — Q_H(s,a)="this push leads to success within the
remaining budget", same EdgeCrossAttn + per-cell sigmoid training. Policy = top-k(map) [USER: "treat the ranker as
a policy"]; Value = max(map) (calibration head only if max proves optimistic — H0b showed maxP is bias-prone,
mean_all more honest → pooling may be the patch). WHY Q-not-policy-CE for sampled data: per-cell labels are ABSOLUTE
facts that survive sampling+masking; softmax-CE targets are RELATIVE verdicts that sampling corrupts ("best of the
15 sampled" can be a lie). A policy also cannot express hopelessness (sums to 1) — the no-hopeless-scenes diagnosis
requires it. H1 retains one card: if CE ordering wins big on exhaustive data, consider CE-finetune for ordering.

**Target definition (precise):** Q_h(s,a) = 1[region opens within h pushes, starting with a from s]. Training label
= the collection search's empirical return: opened directly→1; ≥1 sampled follow-up opened→1 (graded variant:
fraction of sampled follow-ups that worked = route robustness); all sampled follow-ups failed→soft 0; untried→masked.
Monte-Carlo/search returns, NEVER the model's own predictions (no TD/deadly-triad); iterate = regenerate by fresh
search (Reanalyze pattern). Known wrinkle: s0 rows carry h=2 labels, s1 rows h=1 (state doesn't encode budget) —
fallback if miscalibrated: budget as input token.

**Data = the 5 species:** s0 direct openers, s0 ENABLERS (F1′-style), post-push s1 states with their 1-push labels,
HOPELESS s1 states (all-zeros — mandatory, diagnosis #2), soft negatives (budget-limited). Sampled k per state
(k = arch-journal H5 verdict) + masked loss; collection = the tagged depth-2 machinery pointed at TRAIN scenes.

**[USER] DECISION — robustness over optimality:** when a dense 2-push route and a rare 1-push needle coexist, PREFER
the easy 2-push. ⇒ single blended head; labels still recorded per-horizon (reversible). Eval = success within push
budget, NOT min-push. [CLAUDE honesty note: the binary label does NOT inherently encode route density — the
dense-route preference emerges only via sampling noise/recognizability/calibration side-effects; if wanted BY DESIGN,
use the graded (success-fraction) label. Verification-search makes the ranking choice non-critical either way.]

**Search compatibility:** Q orders, sim VERIFIES top-3 (vs the beam's blind ~49), V=max prunes; every verified push
= a new training sample (the ExIt loop, if H5c says random sampling has a ceiling).
**Target to beat (pre-registered):** H0b's 34.5% @1 at ~49 sims/scene — beat it at ZERO lookahead sims.

---

## H0b RESULT — training-free first-push baseline vs exhaustive F1′ [2026-06-10, FINAL]
**Setup:** 787 pure-2-push scenes; per scene, EVERY reachable first-push simulated (38,689 sims), the post-push state
scored by the champion, first-pushes ranked by training-free scalars, recall@k graded vs the exhaustive F1′
(`labels/pure2push.json`). Graded on the 391 episodes where ≥1 enabling first-push was inside the swept candidate set
(coverage filter — measures RANKING quality given coverage). Result: `namo_testset_v1/stats/fpv_step0_final.json`.

| ranker | @1 | @3 | @5 | @10 | @20 |
|---|---|---|---|---|---|
| mean_top5 | **34.5** | 52.9 | 63.4 | 72.6 | 90.3 |
| mean_all | 30.9 | 53.7 | 62.7 | **79.0** | 91.8 |
| maxP | 24.6 | 42.7 | 51.9 | 65.5 | 85.7 |
| random floor | 11.8 | 29.7 | 43.0 | 64.6 | 86.5 |

(95% CI ≈ ±4.7pp @1, ±4.3pp @10, n=391.)

**Verdicts:**
- **ACCEPT: real top-rank signal** (34.5 vs 11.8 @1 = ~9 SE; 3× random).
- **ACCEPT (operative): NOT sufficient for few-try selection** — ≈floor by @10-20. ⇒ learned first-push value
  justified by measurement. Costs ~49 sims/scene at deployment for a 34.5% pick — the baseline to crush at 0 sims.
- **Diagnosis #1:** champion SATURATES on post-push states (~0.99 on dead s1's — OOD; never trained there).
- **Diagnosis #2 [USER catch, verified]:** training data has **0 hopeless scenes** (all 98,387 rows have ≥1 valid
  push; mean 54% of reachable succeed, p10=12.5%) — the model never learned that "all-low" is a legal output.
  ⇒ HARD REQUIREMENT: value data must include dead-end/unsolvable states.
- **Surprise:** mean_all > mean_top5 at k≥10; maxP worst (single-cell flukes dominate the max).
- **Cost note [honest]:** verdict stable at ~300 episodes; full 787 bought CI tightness only. Next time: half.
**H0 pair CLOSED** (H0a in the arch journal + this). Both free options measured; both insufficient where it matters.

---

## Parked backlog (was items 3-5 of the arch journal's backlog)
3. **Value head** — present/absent; pool method; classification (Stop-Regressing) vs MSE. Needs value-target collection.
   (Largely subsumed by H3′: V = max/pool of the Q map; separate head only if max is miscalibrated.)
4. **Q vs policy+value in the search (deploy):** does Q-ordered + sim-verified search beat the current beam on
   solve@k / sims? Needs H3′ model + search integration.
5. **Targets from SEARCH (partial, masked) vs exhaustive** for depth ≥2 — superseded by the arch journal's H5
   (the 1-push oracle version of exactly this question).

## Parked red-team items (2-push / pipeline-ordering)
- **RISK-1:** 1-push arch verdicts may not transfer to POST-PUSH states (OOD). → H1.5 post-push probe before trusting
  any arch verdict for the multi-push build. (The depth-2 pkls already contain exhaustive F(s1) per expanded a1 —
  the probe's answer key needs only a parser, no new sims.)
- **RISK-3:** value labels are search-collected; search quality depends on the policy → early data biased.
  Ordering: arch verdicts → policy-only search → collect → iterate (DAgger-style, not one-shot).
- **Metric gap:** 1-push hard@1 is not predictive of solve@k/sims; use policy recall@k as the bridge metric; ablate
  policy-only vs +value separately at deploy.
- **Negatives are heterogeneous:** tag negative TYPE in collection (wrong-first-push / impossible-config /
  wrong-second-push) — don't collapse to one "unsolvable".

## Assets ready for un-parking
- `namo_testset_v1` 2-push tier: 808 pure-2-push episodes with exhaustive F1′ (`labels/pure2push.json`).
- The tagged depth-2 collection machinery (`region_opening.py` chain_depth/parent tags + `build_2push_validset.py`).
- H0b leaf dumps (38,689 scored post-push records) — seed data for a post-push probe (TEST scenes; do not train).
- `policy_value_v1` dataset home + collection-design rules (README).
