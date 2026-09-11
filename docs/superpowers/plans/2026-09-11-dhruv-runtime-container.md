# Dhruv-matching Amarel container implementation plan

> **For agentic workers:** Use executing-plans with independent read-only specification and quality reviews. The main orchestrator owns edits and commits, following the repository's delegation rules.

**Goal:** Build and validate an isolated Amarel runtime matching dhruv-linux's Ubuntu 20.04 compiler and native libraries, without changing either existing installation or launching a replacement campaign.

**Architecture:** Use an Apptainer Ubuntu 20.04 image with the exact recorded GCC 9.4 and glibc 2.31 package versions. Rebuild the pinned NAMO and MuJoCo source in fresh directories using dhruv's resolved native compiler flags, reuse the already frozen Python environment and validation artifacts read-only, and validate actual loaded libraries before replaying the existing fixed/search checks. Record differences rather than relaxing validation to hide them.

**Tech stack:** Apptainer, Bash, GCC 9.4, CMake 3.16, existing Python 3.12.13 and pytest validation tools.

**Engineering standards:** Follow plan-coding-standards. Keep machine paths in an untracked environment file and reusable build/validation recipes under scripts/amarel/dhruv_container. Do not change C++/planner policy, the 1 mm margin, original outcomes, source environments, or existing jobs. New artifacts use new paths; fail on existing build/output targets. Pin package versions and collect hashes, actual library mappings, source trees and CPU flags. No credentials or whole-home-directory snapshots enter the image. Tests precede new behavior, and each coherent stage ends with a commit before its execution.

## Approved scope and choices

The user approved assembling the matching container after confirming that a rebuild is acceptable and Amarel permits user-managed containers. This is a refinement of authorized integration steps 1–7, not permission for steps 8–12. Rebuilding in an Ubuntu image is preferred to downgrading host libraries or manually mixing a second glibc into the RHEL process. The image shares the host kernel and CPU; equality of the software stack does not prove bitwise cross-machine physics in advance. Full-NAMO validation remains hmax=2, 20,000 total calls, CPU-only, single-threaded and 1 mm. Do not retrain or change the benchmark membership.

## Stage 1: Freeze reference and test the build contract

- [x] Read exact installed package versions, binary/linkage hashes and resolved native flags; verify the clean isolated worktree starts from integration commit 1fa32fd68d7dac75e5401293bb770f5c45eeeb34.
- [x] Run the nine existing focused test files from the integration validation command using the unchanged dhruv binding. Result: 73 tests passed in 25.12 seconds.
- [x] Add tests under scripts/amarel/dhruv_container/tests for the pinned Ubuntu definition, required environment variables, existing-output refusal, and source/runtime identity comparisons. Verified eight failures before recipes existed; three further red assertions covered review findings before fixes; ten tests now pass.
- [x] Commit this execution plan and tests: `942a940e build: specify isolated dhruv-matching container contract`.

## Stage 2: Implement isolated container and native-build recipes

- [ ] Add ubuntu20.def, build.sh, native-build.sh and validate.sh together under scripts/amarel/dhruv_container. The definition pins GCC 9.4.0-1ubuntu1~20.04.2, libc6/libc6-dev 2.31-0ubuntu9.18 and the recorded native dependencies; it must not infer the target CPU from an Amarel host.
- [ ] Keep all absolute machine paths in container.env under the new artifact root. Build the image and native libraries only in new directories. Supply the frozen source/dependency archives and the existing environment/artifacts as read-only inputs.
- [ ] Reuse scripts/pipeline/validate_full_namo_alignment.py instead of forking its planning or replay code. Add only a small standard-library runtime identity checker if needed to enforce compiler/glibc/loaded-library matching.
- [ ] Run contract tests and bash syntax checks, then request read-only spec and quality reviews. Commit: `build: add pinned Ubuntu runtime and isolated Amarel recipes`.

## Stage 3: Build and validate on Amarel

- [ ] Submit a bounded CPU build job using main on an eligible Piscataway node, as the machine card specifies. Pin math threads; use separate cache/temp/log directories. Verify rootless container support and package availability. Do not alter system packages.
- [ ] Require successful image and native builds, correct imported namo_rl path, MuJoCo source/version and header agreement, and matching compiler/glibc versions. Hash the image, libraries, source inputs, configuration and environment manifest.
- [ ] Run the existing 73-test suite and all six fixed replays plus five search checks and the canonical timing runner. Compare original qpos/qvel, task outcomes, action sequences, simulator calls, route/reachability information and package/input identities; exclude cross-host timings.
- [ ] Investigate any remaining mismatch with bounded checks. If exact software matching requires an unavailable artifact, report the precise gap rather than calling an approximate environment identical.

## Stage 4: Handoff

- [ ] Review exact paths, commands, results and provenance. Preserve the old builds and all original results. Record whether the previously divergent pooled-doorway replay now matches and distinguish that result from broader equivalence claims.
- [ ] Commit verification findings to this plan: `build: record dhruv-container validation evidence`. Leave the worktree intact, clean and unmerged, with no benchmark arrays launched.

## Build review and exact matching boundary

The installed native OpenCV is custom 4.3.0 under /usr/local, while apt OpenCV is 4.2.0. The image therefore includes the captured custom headers, CMake metadata, and shared libraries with an input SHA-256 manifest. GCC9's full native expansion on dhruv is frozen as explicit Skylake feature/cache flags. The evaluator's actual loaded C++ runtime comes from its Conda environment, not the system library shown by standalone ldd; the container uses the same MuJoCo-only LD_LIBRARY_PATH convention so Python resolves its own frozen runtime dependencies.

Independent read-only specification and shell reviews approved bounded execution after adding an immutable Ubuntu amd64 digest, archive checksums, explicit read-only config binding, and symmetric loaded-library/duplicate-name checks. The required native packages and all loaded CPU/evaluation libraries must match the captured reference; the entire workstation apt package universe is not promised identical. Rebuilt NAMO/MuJoCo binary hashes are recorded rather than required byte-equal, with their source/build/header/version/input identities checked by the unchanged alignment validator and their behavior compared through replay/search. The workstation-only libcuda driver is a documented CPU-only exception; CUDA user-space libraries bundled in the Python environment are still compared. Wall times are recorded by the existing runner but excluded from cross-machine conclusions.
