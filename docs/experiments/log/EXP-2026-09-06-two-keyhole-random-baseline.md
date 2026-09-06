---
type: experiment
status: live
created: 2026-09-06
commit: 93d23d6d
metric: pending
tags: [experiment, full-namo, multihop, random-baseline, same-template, interaction, medium, hard]
---
# Three-seed random baseline on the controlled two-keyhole cohort

**Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** Random is the baseline ranker inside the same simulator-verified local search used for HY5U. Full NAMO still solves one local boundary at a time and carries the physical state into the next local search.

## Hypothesis

_(user, from chat)_ Compare Random against HY5U on the same 40 controlled two-keyhole scenes.

## Plan

Reuse the exact 40-scene population evaluated by HY5U: ten MM, ten MH, ten HM, and ten HH scenes, with seven clean and three K2-contact scenes in each cell. The XML-list SHA-256 is `55b0d2d13918bf349ff13b598a78ef3cfc7b3caba0566d72b83532bbf7920220`; the metadata-manifest SHA-256 is `d1f3550a84446b66ce9fe8ae129afd5f377068ec21e7723aa8c3d0041f43153e`.

Run uniform random ordering with the project's standard random ordering seeds 7000, 8000, and 9000 while pinning the region snapshot and 100 verifier points to seed 42, matching the existing HY5U arm exactly. Keep the rest of the HY5U protocol fixed: ordinary Full NAMO best-first search, `hmax=2` per local keyhole, `1x_car_d5_` primitives, raw `q`, discount off, no-op deduplication, jam pruning, and 900 simulator calls reset per keyhole. Keep the read-only next-keyhole audit enabled and the rejected strict preservation gate disabled. The random arm skips model scoring; the launcher receives the HY5U path only because its shared shell interface requires `CKPT`.

Smoke one complete scene with random seed 7000 on rlab7 using the production command. If selection, uniform ordering, Full-NAMO replanning, and artifact writing pass, run all 40 scenes for all three seeds on the same node. Report each random seed, the three-seed mean and range, paired solve outcomes against the existing HY5U seed-2 arm, simulator-call curves, and MM/MH/HM/HH by clean/contact. Join evaluation rows to the metadata manifest by `realpath`, never basename.

## Run

The one-scene production smoke ran as job `272304_0` at commit `0468af86`. It was initially submitted to rlab7, but remained pending because other jobs had allocated 496 of the node's 499 GiB. Before it started, the request was moved to rlab1, another CS-estate node with the same shared filesystem and verified-identical physics. It completed successfully in six seconds.

The smoke selected exactly MM contact scene `contact_0000.xml`, used `best_first_prior=uniform` with seed 7000, and solved the complete scene in 14 simulator calls split `[10,4]` across K1 and K2. HY5U solved the same scene in two calls split `[1,1]`. The read-only K1 audit reported unchanged K2 object identity, pose, and reachable edge set, with the path reduced from two hops to one. The random arm did not load or score the checkpoint.

The full run will use three independent one-shard jobs on rlab1, one for each seed, with ten workers per job and a two-hour safety limit. The smoke establishes the production path but is too easy to estimate the random tail; the existing 900-call ceiling and two-hour launcher limit remain the conservative bounds.

The first three full Random jobs, `272361_0`, `272362_0`, and `272363_0`, completed at commit `c77bddd7` with 30/40, 34/40, and 35/40 solves. Inspection before accepting the comparison found that the runner's single `--seed` controlled both uniform push ordering and `region_snapshot_seed`, so those three arms also changed the sampled verifier points away from HY5U's seed 42. Their artifacts remain under `$NAMO_SCRATCH/eval/keyhole_mixed_context_20260905/random40_v1/`, but they are superseded for the paired claim.

The runner now accepts a separate optional `--shuffle-seed`, and the shared SLURM launcher exposes it as `SHUFFLE_SEED`. If omitted, it defaults to `--seed`, so old invocations reproduce unchanged. The corrected arms will use verifier/snapshot seed 42 in every run and vary only the uniform ordering seed across 7000, 8000, and 9000. The runner, Full-NAMO budget/config, strict-BFS, and jam-guard tests pass 22/22 before the corrected smoke.

The corrected one-scene smoke ran as job `272544_0` at commit `93d23d6d`. The request was resized from 24 GiB to 4 GiB after the completed full arms measured about 1.1 GiB maximum RSS, which let it backfill onto rlab7. Its persisted config records verifier seed 42, shuffle seed 7000, and uniform prior. It solved the same MM contact scene in 14 calls split `[10,4]`, exactly reproducing the earlier seed-7000 smoke cost while now holding HY5U's verifier fixed.

## Result

Pending.

## Verdict

Pending.
