"""Passive, independently optional search measurements and stable run identities.

This module never constructs an environment, restores a state, or enumerates an
action. Callers supply facts from operations the canonical search already made.
"""

from collections.abc import Mapping
from collections import Counter
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter


SCHEMA_VERSION = 1
RESTORE_CONTRACT = "RLState_qpos_qvel; canonical_set_full_state_zeroes_qvel"


def measurement_options(options=None):
    """Validate YAML options once at the runner boundary, without bool coercion."""
    result = dict(schema_version=SCHEMA_VERSION, record_statistics=False, record_timing=False)
    if options is not None:
        if not isinstance(options, Mapping):
            raise ValueError("measurement must be a mapping")
        for key, value in options.items():
            if key not in result:
                raise ValueError(f"unknown measurement option: {key}")
            if key == "schema_version":
                if type(value) is not int or value != SCHEMA_VERSION:
                    raise ValueError(f"measurement.schema_version must be {SCHEMA_VERSION}")
            elif type(value) is not bool:
                raise ValueError(f"measurement.{key} must be a boolean")
            result[key] = value
    return result


def clock_start(timing):
    """Read the measurement clock only when a timing sink is enabled."""
    return perf_counter() if timing is not None else None


def clock_finish(timing, key, started):
    """Accumulate an existing operation's duration; disabled means no clock read."""
    if timing is not None:
        timing[key] = timing.get(key, 0.0) + perf_counter() - started


def elapsed_ms(timing, started):
    """Legacy PlannerResult duration; None is deliberately not a zero duration."""
    return (perf_counter() - started) * 1000.0 if timing is not None else None


def action_record(action):
    """Serialize a canonical action without quantizing its target pose."""
    return dict(object_id=str(action.object_id), edge_idx=int(action.edge_idx), depth=int(action.depth),
                target=[float(action.x), float(action.y), float(action.theta)])


def snapshot_record(snapshot):
    """Copy decision evidence already computed by the native snapshot call."""
    result = {key: snapshot[key] for key in (
        "robot_label", "goal_label", "goal_reachable", "goal_in_free_space", "source",
        "region_cells", "goal_clearance") if key in snapshot}
    for key in ("adjacency", "multi_object_edges"):
        result[key] = {label: sorted(values) for label, values in snapshot.get(key, {}).items()}
    result["edge_objects"] = {label: {other: sorted(objects) for other, objects in neighbors.items()}
                              for label, neighbors in snapshot["edge_objects"].items()}
    result["region_labels"] = {str(key): value for key, value in snapshot["region_labels"].items()}
    result["region_goals"] = {label: {"samples": [[float(g.x), float(g.y), float(g.theta)] for g in bundle.goals]}
                              for label, bundle in snapshot["region_goals"].items()}
    return result


def canonical_json(value):
    """Serialize identity/event data deterministically, without float rounding."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_digest(value):
    """Hash an ordered JSON value with deterministic mapping-key ordering."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_digest(path):
    """Hash file contents, never an installation-dependent absolute path."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def state_record(state):
    """Copy the exact canonical RLState arrays; this is not a MuJoCo tick state."""
    return {"qpos": list(state.qpos), "qvel": list(state.qvel)}


def run_identity(problem, protocol, method, checkpoint_hash, seeds):
    """Build the shared problem/protocol/run IDs from semantic input facts only."""
    identity = {"problem_id": content_digest(problem), "semantic_protocol_hash": content_digest(protocol)}
    identity["run_id"] = content_digest(dict(identity, method=method, checkpoint_hash=checkpoint_hash, seeds=seeds))
    return identity


@lru_cache(maxsize=16)
def runtime_fingerprints(config_path, primitive_dir, primitive_prefix, binding_path, checkpoint_path=None):
    """Fingerprint the pinned worker runtime outside any measured search.

    A worker must not mutate its code or inputs. Cache only this worker-wide
    evidence; scene contents and initialized states are hashed for every task.
    Absolute paths remain provenance, not semantic identity.
    """
    repo = Path(__file__).resolve().parents[3]
    sources = {str(p.relative_to(repo)): file_digest(p) for root in ("python/namo", "scripts")
               for p in sorted((repo / root).rglob("*.py"))}
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True)
    config = Path(config_path)
    sidecar = config.with_name("wavefront_inflation.yaml")
    tables = sorted(Path(primitive_dir).glob(f"{primitive_prefix}motion_primitives_15_*.dat"))
    if not tables:
        raise ValueError(f"no primitive tables for {primitive_prefix!r} in {primitive_dir}")
    # Record the libraries actually mapped by this process, including MuJoCo;
    # a pip distribution version is not the linked simulator's identity.
    libraries = {}
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split()
        if len(fields) >= 6 and fields[-1].startswith("/") and ".so" in fields[-1]:
            path = Path(fields[-1]).resolve()
            libraries[str(path)] = path
    by_name = {}
    for path in sorted(libraries.values()):
        by_name.setdefault(path.name, []).append(file_digest(path))
    sage = os.environ.get("SAGE_REPO")
    scorer_sources = ({str(p.relative_to(sage)): file_digest(p)
                       for p in sorted(Path(sage).rglob("*.py"))} if checkpoint_path and sage else {})
    return {
        "code_commit": git.stdout.strip() if git.returncode == 0 else None,
        "source_sha256": content_digest(sources),
        "binding_sha256": file_digest(binding_path),
        "linked_libraries": {name: sorted(hashes) for name, hashes in sorted(by_name.items())},
        "python_version": platform.python_version(),
        "config_sha256": file_digest(config),
        "wavefront_inflation_sha256": file_digest(sidecar) if sidecar.exists() else None,
        "primitive_sha256": {p.name: file_digest(p) for p in tables},
        "checkpoint_sha256": file_digest(checkpoint_path) if checkpoint_path else None,
        "scorer_source_sha256": content_digest(scorer_sources) if scorer_sources else None,
        "restore_contract": RESTORE_CONTRACT,
    }


class SearchMeasurements:
    """Small per-run holder: optional sinks plus an always-on streaming digest."""

    def __init__(self, *, record_statistics=False, record_timing=False, schema_version=SCHEMA_VERSION):
        self.options = measurement_options(dict(record_statistics=record_statistics,
                                               record_timing=record_timing, schema_version=schema_version))
        self.statistics = {"attempts": [], "decisions": [], "commits": [], "checkpoints": [],
                           "states": {}, "snapshots": []} if record_statistics else None
        self.timing = {"t_sim": 0.0, "t_score": 0.0, "n_score": 0} if record_timing else None
        self.attempt_count = 0
        self.commit_count = 0
        self.attempt_digests = []
        self.local_timing = [] if record_timing else None
        self.complete = False
        self._digest = hashlib.sha256()
        self._attempt_digest = None
        self.active_attempt = None
        self.local_timer = None
        self.total_sim_calls = 0
        self.decision_count = 0
        self.scene_version = 0
        self.physical_push_count = 0
        self._state_digest = None
        self._active_stats = None
        self._clock_origin = None

    def next_attempt(self):
        """Allocate a zero-based invocation ID, including zero-call attempts."""
        current = self.attempt_count
        self.attempt_count += 1
        return current

    def next_commit(self):
        """Allocate a commit-event ID independently of physical state changes."""
        current = self.commit_count
        self.commit_count += 1
        return current

    def event(self, kind, **fields):
        """Hash available canonical facts; never retain speculative state traces."""
        encoded = canonical_json(dict(kind=kind, **fields)).encode("utf-8") + b"\n"
        self._digest.update(encoded)
        if self._attempt_digest is not None:
            self._attempt_digest.update(encoded)

    def append(self, section, record):
        """Retain a behavioral record only when explicitly requested."""
        if self.statistics is not None:
            self.statistics[section].append(record)

    def start_clock(self):
        """Set a run-relative Python clock origin, only for requested timing."""
        self._clock_origin = clock_start(self.timing)

    def begin_attempt(self, **facts):
        """Start one opener invocation, even when it returns without a simulation."""
        if self.active_attempt is not None:
            raise RuntimeError("nested local search measurements are not supported")
        self.active_attempt = self.next_attempt()
        self._attempt_digest = hashlib.sha256()
        self._attempt_start_calls = self.total_sim_calls
        self._attempt_started = clock_start(self.timing)
        self.local_timer = ({"t_sim": 0.0, "t_score": 0.0, "n_score": 0,
                             "t_model_score": 0.0, "n_model_score": 0, "t_local_verify": 0.0}
                            if self.timing is not None else None)
        self.event("attempt_start", attempt_id=self.active_attempt, calls=self.total_sim_calls, **facts)
        self._active_stats = (dict(attempt_id=self.active_attempt, sim_call_start=self.total_sim_calls,
                                   scene_version_before=self.scene_version, **facts,
                                   root_candidates_by_object={}, simulations_by_object={},
                                   simulations_by_chain_depth={}, object_sequence_rle=[], boards=[])
                              if self.statistics is not None else None)
        if self.statistics is not None and self.statistics["decisions"]:
            self.statistics["decisions"][-1]["attempt_id"] = self.active_attempt
        return self.active_attempt

    def attempt_target(self, **facts):
        """Record the resolved fixed target and pool, without recomputing them."""
        self.event("attempt_target", **facts)
        if self._active_stats is not None:
            self._active_stats.update(facts)

    def board(self, board_id, parent_board_id, depth, pool):
        """Observe an existing candidate list; do not enumerate or rank again."""
        self.event("board", board_id=board_id, parent_board_id=parent_board_id,
                   chain_depth=depth, n_candidates=len(pool))
        if self._active_stats is not None:
            counts = dict(Counter(str(obj) for obj, _goal, _q in pool))
            row = dict(board_id=board_id, parent_board_id=parent_board_id, chain_depth=depth,
                       candidates_by_object=counts)
            self._active_stats["boards"].append(row)
            if depth == 0:
                self._active_stats["root_candidates_by_object"] = counts

    def simulated(self, *, board_id, parent_board_id, chain_depth, action, opened, info):
        """Account for exactly one completed env.step and its existing result."""
        self.total_sim_calls += 1
        record = dict(call=self.total_sim_calls, board_id=board_id, parent_board_id=parent_board_id,
                      chain_depth=chain_depth, action=action_record(action), opened=bool(opened), info=dict(info or {}))
        self.event("simulation", **record)
        if self._active_stats is not None:
            obj = str(action.object_id)
            for key, item in (("simulations_by_object", obj), ("simulations_by_chain_depth", str(chain_depth))):
                counts = self._active_stats[key]
                counts[item] = counts.get(item, 0) + 1
            sequence = self._active_stats["object_sequence_rle"]
            if sequence and sequence[-1][0] == obj:
                sequence[-1][1] += 1
            else:
                sequence.append([obj, 1])

    def end_attempt(self, *, end, success, calls, actions=(), failure_reason=None):
        """Finish local accounting and retain a small digest row in every mode."""
        if self.active_attempt is None:
            raise RuntimeError("local search was not started")
        actual_calls = self.total_sim_calls - self._attempt_start_calls
        if actual_calls != calls:
            raise ValueError(f"local simulation accounting mismatch: engine={actual_calls}, adapter={calls}")
        self.event("attempt_end", attempt_id=self.active_attempt, local_end=end, success=bool(success),
                   calls=calls, chain=[action_record(action) for action in actions], failure_reason=failure_reason)
        digest = dict(attempt_id=self.active_attempt, sim_call_start=self._attempt_start_calls,
                      sim_call_end=self.total_sim_calls, local_end=end, success=bool(success),
                      digest=self._attempt_digest.hexdigest())
        self.attempt_digests.append(digest)
        if self._active_stats is not None:
            self._active_stats.update(digest, calls=calls, failure_reason=failure_reason,
                                      chain_length=len(actions), scene_version_after=self.scene_version)
            self.append("attempts", self._active_stats)
        if self.local_timer is not None:
            ended = perf_counter()
            local = dict(self.local_timer)
            local.pop("t_wall", None)  # engine-only duration must never replace the full planner duration
            local["t_local_search"] = ended - self._attempt_started
            local["t_rank"], local["n_rank"] = local["t_score"], local["n_score"]
            for key, value in local.items():
                self.timing[key] = self.timing.get(key, 0) + value
            self.local_timing.append(dict(attempt_id=self.active_attempt,
                                          sim_call_start=self._attempt_start_calls, sim_call_end=self.total_sim_calls,
                                          start_offset=self._attempt_started - self._clock_origin,
                                          end_offset=ended - self._clock_origin, **local))
        self.active_attempt = self._attempt_digest = self._active_stats = self.local_timer = None

    def checkpoint(self, state, *, trigger, observation=None, snapshot=None, commit_id=None, attempt_id=None):
        """Save a committed decision state once; speculative states are never checkpoints."""
        state_data = state_record(state)
        digest = content_digest(state_data)
        if self._state_digest is None:
            self._state_digest = digest
        if self.statistics is not None:
            self.statistics["states"].setdefault(digest, state_data)
            record = dict(state_digest=digest, trigger=trigger, scene_version=self.scene_version,
                          commit_id=commit_id, attempt_id=attempt_id, action_prefix_end=self.physical_push_count,
                          sim_call=self.total_sim_calls)
            if observation is not None:
                record["poses"] = {key: [float(v) for v in value] for key, value in observation.items()
                                   if key.endswith("_pose")}
            if snapshot is not None:
                record["snapshot_id"] = len(self.statistics["snapshots"])
                self.statistics["snapshots"].append(snapshot_record(snapshot))
            self.append("checkpoints", record)
        return digest

    def committed(self, state, actions, *, attempt_id, opened, task_kind, observation=None):
        """Hash each commit event; scene versions advance only on exact physical change."""
        before = self._state_digest
        after = content_digest(state_record(state))
        commit_id = self.next_commit()
        changed = before != after
        self.scene_version += int(changed)
        start = self.physical_push_count
        self.physical_push_count += len(actions)
        facts = dict(commit_id=commit_id, attempt_id=attempt_id, state_before=before, state_after=after,
                     scene_version=self.scene_version, state_changed=changed, task_kind=task_kind,
                     opened=bool(opened), action_start=start, action_end=self.physical_push_count,
                     sim_call=self.total_sim_calls, actions=[action_record(action) for action in actions])
        self.event("commit", **facts)
        self.append("commits", facts)
        if self.statistics is not None and self.statistics["attempts"]:
            self.statistics["attempts"][-1].update(commit_id=commit_id, scene_version_after=self.scene_version,
                                                   action_start=start, action_end=self.physical_push_count)
        self._state_digest = after
        self.checkpoint(state, trigger="commit", observation=observation, commit_id=commit_id, attempt_id=attempt_id)
        return commit_id

    @property
    def execution_digest(self):
        """Current ordered execution fingerprint, independent of optional sinks."""
        return self._digest.hexdigest()
