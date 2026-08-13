---
type: experiment
status: idea
created: 2026-08-02
commit:
metric:
thread: rl_loop
parent: EXP-2026-07-24-failure-discount-search
related: [EXP-2026-07-25-search-depth-horizon, EXP-2026-07-21-colossus-data-scaleup]
tags: [experiment, ranker, board-value, live-dead, cross-board, search]
---
# Board-live head — predict whether a post-push board has a finish now

**Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** The model remains a ranker for one region-opening search, and the simulator remains the perfect verifier; this head is meant to order post-setup boards, not replace simulation or predict an unbounded solution probability.

## Hypothesis

_(user, via chat 2026-08-02)_ Add a small board-level head that predicts whether the board reached after a push contains a useful finish, so search spends less time flooding dead post-setup boards and switches to another setup sooner.

_(Claude, falsifiable refinement)_ The missing signal is **one-step board liveness**: whether at least one reachable push from this board opens the region immediately. A dedicated liveness score used only as a child-board prior should reduce medium/hard 2push simulator calls without changing the root action ordering; merely adding more within-board ranking pressure is not the target.

## What the existing evidence says

The current `hmax=2` search gives every child board initial weight `w0=1`, then multiplies each candidate's action score by that board weight and lowers it after verified failed finishes. The earlier raw `w0=mean(top-5 action scores)` treatment failed because it moved nearly every child below the root list rather than separating live from dead boards; this motivates a supervised board score, not reuse of the current action-score aggregate.

The deployed training H5 is `round3/h5/d20_plus_setup_only.h5`: 257,409 rows, with 231,386 d20-base rows plus 26,023 Colossus setup roots. Labels below are row-local, so no XML-only join is used; train/validation remains grouped by room, and any future cross-artifact join must use the full episode identity `(room, target object, goal region)` and the parent action for child boards.

| deployed-H5 board role | root rows | moved child rows | meaning |
|---|---:|---:|---|
| finish only | 11,054 | 22,271 | at least one exact `1.0` opener, no exact `0.9` setup |
| both finish and setup | 150,717 | 0 | exact `1.0` opener and exact `0.9` setup both exist |
| setup only | 34,803 | 0 | no direct opener, but at least one exact `0.9` setup |
| dead for the collected horizon | 19,282 | 19,282 | neither exact opener nor exact setup; these are the controlled exhaustive dead dose |

The child split is unusually clean and balanced for the proposed target: **22,271 live versus 19,282 exact one-step-dead moved boards**. No-op child rows are absent because the base builder drops `depth2_noop`, and the setup-only Colossus addition contains roots only.

The canonical exhaustive test artifact `testset_gt_plus35.h5` supplies an evaluation-only board panel: among **51,036 moved child boards**, 18,953 contain a verified finish and 32,083 do not; the 16,205 `depth2_noop` rows must remain excluded because deployed search deduplicates no-op children. This test GT is never training data.

## Label options for discussion

### Option A — one-step-live binary head **(recommended)**

Define `board_live=1` when the current board contains any verified direct opener; on a post-setup board that opener is the useful finish. Define `board_live=0` only for a source-proven exhaustive moved child board with no opener. Mask capped, unswept, unknown, and no-op boards rather than converting absence of evidence into a negative.

For the deployed H5, train this loss on `is_root==0` only: positive if any reachable, supervised, non-ceiling cell has `value_target==1`; negative for the 19,282 source-proven exhaustive dead-finish rows; mask no row in this curated child subset. In future less-curated H5s, require the original node metadata (`n_win`, exhaustion/censor flag, `node_kind`, and setup motion) because `value_mask==0` alone cannot distinguish an unswept action from a controller-pruned jam depth.

Root rows can provide a secondary semantic check but should not dominate the head loss: direct-live roots are 161,771 rows, while 34,803 setup-only roots and 19,282 dead roots are immediate-live negatives. A setup-only root is useful to the two-push search but correctly negative for **opens now**; applying the head only to child-board priority prevents that label from suppressing root setups.

### Option B — horizon-aware solvability or value

Predict whether a board is solvable within one, two, or more remaining pushes, or regress a discounted board value. This would answer more questions, but the deployed data has no truth past the next push on child boards, censored failures are not dead, and the live model is deliberately not horizon-conditioned. This option would reintroduce the horizon semantics we removed and cannot be labeled cleanly from the chosen H5.

### Option C — categorical board role `{needs setup, finish-live, dead}`

At roots the deployed H5 can separate `finish only`, `both finish+setup`, `setup only`, and `dead` using exact `1.0` and exact non-ceiling `0.9` cells. At moved child boards it cannot tell “has another setup” from “dead,” because depth-2 failed cells are ceilings, not verified depth-3 setups. A three- or four-way role head would therefore learn a root-specific taxonomy that the current child-board search cannot use reliably.

## Recommendation

Choose Option A and name it `one_step_live`, not `board_value` or `solvable`. The name keeps the claim honest: it predicts whether the board offers a finish **now**, while simulation still verifies the chosen push and the existing failure discount still benches a board after misses.

Train one checkpoint with a small auxiliary binary loss on moved child rows only, balanced by class or by equal positive/negative sampling. Keep the existing 60×5 action-value and listwise losses unchanged, and use a low auxiliary weight so the shared representation does not sacrifice within-board ranking.

Evaluate the same checkpoint two ways under the primary no-discount search: **auxiliary-only**, where search ignores the new scalar and tests whether representation sharing changed action ranks; and **child-prior**, where a child board starts with a bounded monotone weight from the head and each candidate receives `action_score × board_weight`. Root board weight stays exactly one, so root setup ranking is untouched. A separate confidence-discount compatibility arm may test the deployed combination, but it cannot establish whether the head or the failure discount caused a switching change.

Do not prune a low-scored board outright. Floor its initial weight, as the current failure discount does, because a mistaken board score must delay a branch rather than make search incomplete.

## Planned evaluation if approved

Use `eval_auc.py` as the existing offline ranking tool, extended rather than duplicated, to report moved-child live/dead separation on the canonical exhaustive GT. The deploy verdict remains live best-first search: success-versus-simulator-calls and simulator-calls-to-solution for easy/medium/hard 2push, with the full registered easy/medium/hard 1push table as the shared-backbone guardrail.

Compare the deployed setup-only checkpoint, the head checkpoint with the head ignored, and the same head checkpoint used as the child prior under identical `hmax=2`, `combine=q`, discount-off, no-op-dedupe, jam-depth-pruning search. This separates training benefit from search-use benefit without training a second head model. Report the deployed confidence-discount combination only as a secondary compatibility result.

## Discussion — one decision needed

**[Claude 2026-08-02]** I recommend the narrow binary label: **does this post-push board contain a direct finish now?** The alternative categorical question—needs another setup, has both, only finish, or dead—cannot be answered for child boards with the present horizon-2 labels because “another setup” would require depth-3 truth. Please confirm whether to lock the binary child-only target, or whether you want to expand the data horizon before this model is trained.

## Cross-reference (2026-08-12) — still unimplemented, and NOT what HY5U did

Verified: no implementation exists (no `board_live`/`one_step_live` in `scripts/` or `python/`), no registry row, no results entry. `status: idea` is correct.

**It does not overlap `HY5U`** (2026-08-12, [EXP-2026-08-09-crossboard-ranking](EXP-2026-08-09-crossboard-ranking.md) § HY5U). This card proposes a NEW auxiliary binary head predicting "does this child board contain any opener", consumed at search time as a bounded multiplicative prior on child-board scores. HY5U instead put previously-masked UNREACHABLE cells back into the EXISTING scalar value head as exact zeros (`NAMO_UNREACH_W`, regression only, barred from ranking lists) — a label/masking change, no new head, no search-time change.

Relevance is that HY5U moved V6 (live-board max vs dead-board max) 0.749 → 0.803, i.e. board-liveness discrimination improved substantially as a side effect of geometry supervision. That RAISES the value of testing this card's dedicated head — the signal is evidently learnable — while also meaning its baseline has moved: any future run must compare against HY5U on the common episode set, not against the 2026-08-02 arms.
