---
type: experiment
status: done
created: 2026-09-06
commit: c65f8031
metric: HY5U 38/40 versus Random 33.0±2.6/40; at 10 calls 34 versus 11.7±0.6
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

The corrected full arms ran as rlab7 jobs `272545_0`, `272546_0`, and `272547_0` at commit `c65f8031`. All three selected the exact 40 paths with zero selection errors and zero path-length mismatches. Their persisted configs record verifier seed 42 and shuffle seeds 7000, 8000, and 9000 respectively. The jobs completed in 0:30, 5:51, and 14:51, but these times are not compared with HY5U because rlab7 was shared and differently loaded. Maximum RSS was 1.10–1.11 GiB, validating the 4 GiB request.

Raw corrected rows are under `$NAMO_SCRATCH/eval/keyhole_mixed_context_20260905/random40_paired_v1/random_s{7000,8000,9000}/`. Standard aggregates are under the sibling `random_s{7000,8000,9000}_aggregate/` directories, and paired comparisons with HY5U are under `comparison_s{7000,8000,9000}/`.

## Result

HY5U is far ahead in the search regime that matters. It solves 34/40 scenes within ten total simulator calls, compared with a Random mean of 11.7/40. At the final observed ceiling, HY5U solves 38/40 and Random solves 30, 34, and 35, a mean of 33.0 with sample standard deviation 2.6.

| total scene-call cutoff | HY5U solved | Random s7000 | Random s8000 | Random s9000 | Random mean ± SD |
|---:|---:|---:|---:|---:|---:|
| 2 | **8** | 0 | 0 | 0 | 0.0 ± 0.0 |
| 5 | **19** | 3 | 7 | 7 | 5.7 ± 2.3 |
| 10 | **34** | 11 | 12 | 12 | 11.7 ± 0.6 |
| 30 | **37** | 22 | 23 | 26 | 23.7 ± 2.1 |
| 100 | **38** | 30 | 31 | 35 | 32.0 ± 2.6 |
| final | **38** | 30 | 34 | 35 | 33.0 ± 2.6 |

The corrected seed-level medians among solved scenes are 13.5, 21.5, and 16 calls for Random, versus 5.5 for HY5U. Random seed 8000 has one complete-scene solve at 1,061 total calls because the 900-call budget resets at each local keyhole invocation; HY5U's slowest solve uses 56 calls.

| ordered donor pair | HY5U solved by 10 | Random solved by 10, mean ± SD | HY5U final | Random final, mean ± SD (range) |
|---|---:|---:|---:|---:|
| MM | **10/10** | 3.7 ± 1.5 | **10/10** | 9.0 ± 0.0 (9–9) |
| MH | **8/10** | 5.0 ± 0.0 | 8/10 | **10.0 ± 0.0 (10–10)** |
| HM | **9/10** | 1.0 ± 1.0 | **10/10** | 7.0 ± 2.0 (5–9) |
| HH | **7/10** | 2.0 ± 1.0 | **10/10** | 7.0 ± 1.0 (6–8) |

HY5U wins the ten-call comparison in every ordered pair and wins final coverage in MM, HM, and HH. It does not dominate Random at the MH ceiling: every Random seed solves all ten MH scenes, while HY5U solves eight. These labels are the source donors' local tiers, not newly measured end-to-end scene difficulty.

| context | HY5U solved by 10 | Random solved by 10, mean ± SD | HY5U final | Random final, mean ± SD |
|---|---:|---:|---:|---:|
| clean, n=28 | **24** | 9.0 ± 2.6 | **27** | 24.0 ± 2.6 |
| K2 contact, n=12 | **10** | 2.7 ± 2.1 | **11** | 9.0 ± 0.0 |

The interaction subset keeps the same ordering result. HY5U solves ten of twelve K2-contact scenes within ten calls, while Random averages 2.7; final coverage is 11/12 versus 9/12 in every Random seed.

| Random seed | both solve | HY5U only | Random only | neither | median calls on both, HY5U / Random | HY5U faster / tied / slower | McNemar exact p |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7000 | 28 | 10 | 2 | 0 | 5 / 14 | 23 / 1 / 4 | 0.0386 |
| 8000 | 32 | 6 | 2 | 0 | 5.5 / 22 | 27 / 0 / 5 | 0.2891 |
| 9000 | 33 | 5 | 2 | 0 | 6 / 16 | 30 / 0 / 3 | 0.4531 |

Across the 93 seed-scene pairs solved by both arms, HY5U's repeated median is 5 calls against Random's 16, and HY5U is faster in 80, tied in one, and slower in 12. Only one per-seed final-outcome comparison reaches p<0.05, so the 40-scene cohort does not support a broad significance claim from final solve counts alone. The large separation in the early solve curve is the useful result.

Random solves both HY5U failures in every seed. The MH contact failure takes 9, 22, and 34 calls; the MH clean failure takes 13, 8, and 4. Conversely, HY5U solves every scene that Random misses in each seed. Thus every paired arm union covers all 40 scenes, and the same two HY5U misses are ranker-dependent greedy-commit failures rather than bad environments.

All 21 Random failures across the three seeds terminate as `region_path_exhausted`, not as the runner's terminal `simulation_budget_exhausted`. Several nevertheless spend one or more 900-call local budgets before the outer route is exhausted, including seed-9000 failures with 1,878–3,678 total calls. This is another reason to report the success-versus-total-calls curve rather than only final solve count.

The read-only exact K2-interface audit passes for 30/30, 25/34, and 34/35 Random solves, compared with 37/38 HY5U solves. The many successful Random paths that fail the exact-interface diagnostic reinforce the earlier rejection of that condition as a hard gate.

Separating verifier and ordering seeds was necessary even though the corrected final solve counts happened to remain 30/34/35. Five seed-8000 scene costs changed under the fixed verifier; one HM scene moved from 12 to 976 calls. A shared population is not enough for an exact paired comparison when the verifier points also change.

## Verdict

**ACCEPT the main ordering claim on this controlled cohort, with one important exception.** HY5U reaches 85% complete-scene success by ten calls while Random averages 29.2%, and HY5U's final 95% exceeds Random's 82.5±6.6%. The gain holds at ten calls in MM, MH, HM, HH, clean scenes, and K2-contact scenes. However, Random eventually beats HY5U in the MH cell, 10/10 versus 8/10, because every seed avoids the two locally valid but globally bad K1 commits that trap HY5U. The ranker is doing its intended job overall, but the outer planner still needs a way to reject or recover from a K1 opening that does not advance the global region path. This 40-scene approval cohort is a controlled pilot, not a canonical population estimate.
