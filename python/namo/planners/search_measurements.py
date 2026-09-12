"""Passive, independently optional search measurements and stable run identities.

This module never constructs an environment, restores a state, or enumerates an
action. Callers supply facts from operations the canonical search already made.
"""

from collections.abc import Mapping
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
    sources = {str(p.relative_to(repo)): file_digest(p)
               for p in sorted((repo / "python/namo").rglob("*.py"))}
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
        self.statistics = {"attempts": [], "decisions": [], "commits": [], "checkpoints": []} if record_statistics else None
        self.timing = {"t_sim": 0.0, "t_score": 0.0, "n_score": 0} if record_timing else None
        self.attempt_count = 0
        self.commit_count = 0
        self.attempt_digests = []
        self.local_timing = [] if record_timing else None
        self.complete = False
        self._digest = hashlib.sha256()

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
        self._digest.update(canonical_json(dict(kind=kind, **fields)).encode("utf-8") + b"\n")

    def append(self, section, record):
        """Retain a behavioral record only when explicitly requested."""
        if self.statistics is not None:
            self.statistics[section].append(record)

    @property
    def execution_digest(self):
        """Current ordered execution fingerprint, independent of optional sinks."""
        return self._digest.hexdigest()
