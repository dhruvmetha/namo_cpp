# Dhruv-matching Amarel container implementation plan

> **For agentic workers:** Use executing-plans with independent read-only specification and quality reviews. The main orchestrator owns edits and commits, following the repository's delegation rules.

**Goal:** Build and validate an isolated Amarel runtime matching dhruv-linux's Ubuntu 20.04 compiler and native libraries, without changing either existing installation or launching a replacement campaign.

**Architecture:** Use an Apptainer Ubuntu 20.04 image with the exact recorded GCC 9.4 and glibc 2.31 package versions. Rebuild the pinned NAMO and MuJoCo source in fresh directories using dhruv's resolved native compiler flags, reuse the already frozen Python environment and validation artifacts read-only, and validate actual loaded libraries before replaying the existing fixed/search checks. Record differences rather than relaxing validation to hide them.

**Resolved Python layout:** The earlier Python archive was packed for the old Amarel prefix, so two native libraries already contain relocated path strings. Extract a separate copy, restore libX11.so.6.4.0 and libcrypto.so.3 from their hash-verified dhruv originals, and mount this copy read-only at the original dhruv Python prefix inside the container. PYTHON_ENV_SOURCE is the physical artifact directory; PYTHON_ENV is its container-visible original prefix. Both views are read-only. The previous Amarel environment and archive remain untouched.

**Tech stack:** Apptainer, Bash, GCC 9.4, CMake 3.16, existing Python 3.12.13 and pytest validation tools.

**Engineering standards:** Follow plan-coding-standards. Keep machine paths in an untracked environment file and reusable build/validation recipes under scripts/amarel/dhruv_container. Do not change C++/planner policy, the 1 mm margin, original outcomes, source environments, or existing jobs. New artifacts use new paths; fail on existing build/output targets. Pin package versions and collect hashes, actual library mappings, source trees and CPU flags. No credentials or whole-home-directory snapshots enter the image. Tests precede new behavior, and each coherent stage ends with a commit before its execution.

## Approved scope and choices

The user approved assembling the matching container after confirming that a rebuild is acceptable and Amarel permits user-managed containers. This is a refinement of authorized integration steps 1–7, not permission for steps 8–12. Rebuilding in an Ubuntu image is preferred to downgrading host libraries or manually mixing a second glibc into the RHEL process. The image shares the host kernel and CPU; equality of the software stack does not prove bitwise cross-machine physics in advance. Full-NAMO validation remains hmax=2, 20,000 total calls, CPU-only, single-threaded and 1 mm. Do not retrain or change the benchmark membership.

## Stage 1: Freeze reference and test the build contract

- [x] Read exact installed package versions, binary/linkage hashes and resolved native flags; verify the clean isolated worktree starts from integration commit 1fa32fd68d7dac75e5401293bb770f5c45eeeb34.
- [x] Run the nine existing focused test files from the integration validation command using the unchanged dhruv binding. Result: 73 tests passed in 25.12 seconds.
- [x] Add tests under scripts/amarel/dhruv_container/tests for the pinned Ubuntu definition, required environment variables, existing-output refusal, and source/runtime identity comparisons. Verified eight failures before recipes existed; further red assertions covered review findings before fixes; thirteen tests pass in the final revision.
- [x] Commit this execution plan and tests: `942a940e build: specify isolated dhruv-matching container contract`.

## Stage 2: Implement isolated container and native-build recipes

- [x] Add ubuntu20.def, build.sh, native-build.sh and validate.sh together under scripts/amarel/dhruv_container. The definition pins GCC 9.4.0-1ubuntu1~20.04.2, libc6/libc6-dev 2.31-0ubuntu9.18 and the recorded native dependencies; it must not infer the target CPU from an Amarel host.
- [x] Keep all absolute machine paths in container.env under the new artifact root. Build the image and native libraries only in new directories. Supply the frozen source/dependency archives and the existing environment/artifacts as read-only inputs.
- [x] Reuse scripts/pipeline/validate_full_namo_alignment.py instead of forking its planning or replay code. Added the small native_identity.py checker for compiler/glibc/loaded-library matching.
- [x] Run contract tests and bash syntax checks, then request read-only spec and quality reviews. Initial implementation commit: `35709cf6`. Compute-node launch corrections are `eb9013d3` and `9446c0af`; thirteen contract tests pass after all runtime-audit refinements.

## Stage 3: Build and validate on Amarel

- [x] Submit a bounded CPU build job using main on an eligible Piscataway node, as the machine card specifies. Pin math threads; use separate cache/temp/log directories. Verify rootless container support and package availability. No system packages altered. Job 61417886 built the image and native libraries on hal0289 (Icelake), eight CPU cores, 24 GiB, one-hour limit; its post-build audit initially stopped on duplicate extension basenames.
- [x] Require successful image and native builds, correct imported namo_rl path, MuJoCo source/version and header agreement, and matching compiler/glibc versions. Hash the image, libraries, source inputs, configuration and environment manifest. Image SHA-256 is e1bf7fab109927ec30bb95ea43511d03e59ca26bd97bbbed73c0fc7dde95e525. Rebuilt MuJoCo SHA-256 is b36148a1ddc9a8cf7620e69bf71e8a38ccbdb219a778eaf9f4b3c8012e632a3f, byte-for-byte identical to the dhruv reference.
- [x] Run the existing 73-test suite and all six fixed replays plus five search checks and the canonical timing runner. Compare original qpos/qvel, task outcomes, action sequences, simulator calls, route/reachability information and package/input identities; exclude cross-host timings. All checks pass; the final SLURM job completed with exit 0:0.
- [x] Investigate remaining mismatches with bounded checks. Corrected the audit's duplicate-basename handling and restored the two relocated libraries from dhruv in the separate Python copy. No library-audit mismatch or behavioral comparison mismatch remains in this validation suite; tolerances were not relaxed.

## Stage 4: Handoff

- [x] Review exact paths, commands, results and provenance. Preserve the old builds and all original results. The previously divergent pooled-doorway replay now matches bitwise at all saved states. This is validated on the bounded suite, not a claim about every possible future scene or CPU.
- [x] Commit verification findings to this plan: `build: record dhruv-container validation evidence`. Leave the worktree intact, clean and unmerged, with no benchmark arrays launched.

## Build review and exact matching boundary

The installed native OpenCV is custom 4.3.0 under /usr/local, while apt OpenCV is 4.2.0. The image therefore includes the captured custom headers, CMake metadata, and shared libraries with an input SHA-256 manifest. GCC9's full native expansion on dhruv is frozen as explicit Skylake feature/cache flags. The evaluator's actual loaded C++ runtime comes from its Conda environment, not the system library shown by standalone ldd; the container uses the same MuJoCo-only LD_LIBRARY_PATH convention so Python resolves its own frozen runtime dependencies.

Independent read-only specification and shell reviews approved bounded execution after adding an immutable Ubuntu amd64 digest, archive checksums, explicit read-only config binding, and symmetric loaded-library/duplicate-name checks. The required native packages and all loaded CPU/evaluation libraries must match the captured reference; the entire workstation apt package universe is not promised identical. Rebuilt NAMO/MuJoCo binary hashes are recorded rather than required byte-equal, with their source/build/header/version/input identities checked by the unchanged alignment validator and their behavior compared through replay/search. The workstation-only libcuda driver is a documented CPU-only exception; CUDA user-space libraries bundled in the Python environment are still compared. Wall times are recorded by the existing runner but excluded from cross-machine conclusions.

Job 61417731 exposed two launcher issues before compilation: Apptainer does not accept a Docker tag and digest together, and host git is absent on compute nodes. The definition now uses the digest alone and checks source cleanliness inside the image, where git is installed. Job 61417801 downloaded the pinned base but apt's initial sandbox-user switch failed because Amarel provides only a root-mapped user namespace. The initial apt bootstrap now explicitly uses its mapped root user; subsequent installation runs under Focal's own fakeroot. Both failed logs are preserved. Job 61417886 passed package installation and the image's GCC/glibc/OpenCV tests.

The comparison runfile reuses the existing strict equivalent()/maximum_delta() helpers. Its self-comparison returns exact zero state differences; its negative control against the old Amarel results still fails on the known six replay differences and model search cost 8 versus 11. Packaging-only git changes are allowed after checking the actual diff and identical C++ tree stamps; no runtime source changes are allowed. A read-only review confirmed the numerical gates are preserved and terminal qvel comparison is additionally required.

The native audit was corrected to preserve all files with duplicate Python-extension basenames (cydriver and cyruntime) and compare sorted hash multisets, including multiplicity. The reference has 131 loaded files under 129 basenames. Schema mismatches produce an explicit diagnostic. Commits b5dd94cd and 7b66f759 cover these changes. The initial relocated-environment audit and subsequent archive-only audit each identified exactly the same two native-library mismatches, X11 and OpenSSL; their embedded path strings confirmed the relocation. Both failed audit reports and both archived library copies remain preserved. Commit c23dd10a adds the original-prefix read-only bind contract. Thirteen packaging tests pass.

Validation job 61419250 completed on hal0320 (Icelake), exit 0:0, at 2026-09-11 19:33:58 EDT. The actual loaded-library audit passes after restoring the two exact libraries in the separate copy. The existing 73 tests plus 13 packaging tests pass (86 total). All six fixed replays have zero qpos/qvel difference from dhruv; an additional packed-IEEE-double comparison confirms bitwise equality at every recorded state, including pooled_doorway. All five search checks match outcomes, actions, iteration traces, score-call counts, simulator calls, and terminal qpos/qvel, with zero terminal state differences. Counts are pooled_search 31, optimistic_edge_zero_budget 0, model_multihop 8, goal_terminal_recovery 14, and saved_consistency_recovery 2344 on both machines. The canonical runner also matches. The model check therefore no longer exhibits the old Amarel build's 11-versus-8 difference. The expected saved-consistency planner_invariant_violation remains the same on both machines; this packaging work does not claim to fix that separate algorithmic diagnostic.

## Artifact handoff

The artifact root is selected by CONTAINER_ROOT in the untracked runfiles/container.env. Keep this root together with its frozen input root; the SIF contains the compiler/system stack, while NAMO, MuJoCo, and the frozen Python copy are mounted from the artifact directories. Source worktree branch: build/dhruv-runtime-container-20260911. The validated source checkout is c23dd10a64a2e9d61db908e63cc225750da9daff; the final subsequent commit changes only this evidence record.

- Image: dhruv-focal.sif; native outputs: code/build_python and mujoco/build/lib; Python copy: env-original, exposed through the original-prefix read-only bind.
- Validation: results/comparison.json (pass=true), results/bitwise-fixed.json, results/native-identity.json, results/focused-tests.log, results/fixed, and results/search.
- Provenance: provenance/image.sha256, native.sha256, restored-native-libraries.sha256, reference-v2.sha256, frozen-python-archive.sha256, BUILD_INFO, both CMake caches, package list, packaging-changed-files.txt, and job-61419250.exit.
- Launch record: runfiles/validate-container.slurm and logs/validate-61419250.log. The earlier failed launch/audit logs and relocated-library copies remain preserved. runfiles/restore-python-complete.slurm records the complete environment preparation procedure for a fresh destination.
- No existing host installation, original campaign result, or existing job was changed. No replacement benchmark array was submitted. Wall-time comparisons remain out of scope.
