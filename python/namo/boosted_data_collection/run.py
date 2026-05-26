#!/usr/bin/env python3
"""Boosted deterministic cell-opening data collection pipeline."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
import re
import socket
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import namo_rl

from namo import __version__ as NAMO_PRODUCER_VERSION
from namo.boosted_data_collection.miner import (
    get_wavefront_snapshot,
    mine_environment_manifest,
    serialize_for_metadata,
)
from namo.boosted_data_collection.schema import (
    CELL_INDEXING_CONVENTION,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    ensure_schema_or_raise,
)


def _sanitize_run_name(name: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z._-]", "_", name.strip())
    return sanitized or "run"


def _short_hostname() -> str:
    return socket.gethostname().split(".")[0]


def _discover_environment_files(
    xml_dir: str,
    start_idx: int,
    end_idx: int,
    manifest_file: Optional[str],
) -> List[str]:
    if manifest_file:
        manifest_path = Path(manifest_file)
        if manifest_path.exists():
            entries: List[str] = []
            with manifest_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    xml_path = line.split("\t", 1)[0].strip()
                    entries.append(xml_path)
            return entries[start_idx:end_idx]

    all_xml_files = sorted(
        str(p)
        for p in Path(xml_dir).rglob("*.xml")
        if not str(p).endswith("_temp.xml")
    )
    return all_xml_files[start_idx:end_idx]


@dataclass
class WorkerTask:
    task_id: str
    xml_path: str
    config_file: str
    output_dir: str
    boosted_config: Dict[str, Any]
    config_checksum: str
    worker_slot: int


def _write_manifest(path: Path, manifest: Mapping[str, Any], compression: str) -> None:
    if compression == "gzip":
        with gzip.open(path, "wb") as f:
            pickle.dump(dict(manifest), f, protocol=pickle.HIGHEST_PROTOCOL)
        return

    with path.open("wb") as f:
        pickle.dump(dict(manifest), f, protocol=pickle.HIGHEST_PROTOCOL)


def _resolve_grid_metadata(
    env: Any,
    candidate_object_ids: Sequence[str],
    boosted_config: Mapping[str, Any],
) -> Dict[str, Any]:
    probe_object: Optional[str] = None
    if candidate_object_ids:
        probe_object = str(candidate_object_ids[0])
    else:
        reachable = [str(v) for v in env.get_reachable_objects()]
        if reachable:
            probe_object = reachable[0]

    if probe_object is not None:
        snap = get_wavefront_snapshot(
            env,
            probe_object,
            use_cpp_grid_fastpath=bool(boosted_config.get("boosted_use_cpp_grid_fastpath", True)),
        )
        return {
            "grid_shape": [int(snap.grid_shape[0]), int(snap.grid_shape[1])],
            "resolution": float(snap.resolution),
            "bounds": [float(v) for v in snap.bounds],
            "cell_indexing_convention": CELL_INDEXING_CONVENTION,
            "snapshot_source": snap.source,
        }

    raise RuntimeError(
        "Unable to resolve boosted grid metadata from the namo_cpp C++ wavefront path: "
        "no candidate or reachable probe object is available in this environment."
    )


def _run_single_task(task: WorkerTask) -> Dict[str, Any]:
    start_ts = time.time()
    result: Dict[str, Any] = {
        "task_id": task.task_id,
        "xml_path": task.xml_path,
        "success": False,
        "error": "",
        "duration_sec": 0.0,
        "output_path": "",
        "object_count": 0,
    }

    try:
        env = namo_rl.RLEnvironment(task.xml_path, task.config_file, visualize=False)
        env.reset()

        mined = mine_environment_manifest(env, task.boosted_config)
        candidate_object_ids = mined["candidate_object_ids"]
        grid_metadata = _resolve_grid_metadata(env, candidate_object_ids, task.boosted_config)

        config_snapshot = serialize_for_metadata(task.boosted_config)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "schema_name": SCHEMA_NAME,
            "producer_version": NAMO_PRODUCER_VERSION,
            "run_metadata": {
                "seed": int(task.boosted_config.get("seed", 42)),
                "worker_id": int(task.worker_slot),
                "hostname": _short_hostname(),
                "config_snapshot": config_snapshot,
                "config_checksum": task.config_checksum,
                "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            },
            "grid_metadata": grid_metadata,
            "environment": {
                "task_id": task.task_id,
                "xml_path": task.xml_path,
                "config_file": task.config_file,
                "static_object_info": mined.get("static_object_info", {}),
            },
            "objects": mined["objects"],
            "summary": {
                "baseline_reachable_objects": mined.get("baseline_reachable_objects", []),
                "candidate_object_ids": candidate_object_ids,
                "region_snapshot_source": mined["region_snapshot_source"],
                "region_snapshot_robot_label": mined["region_snapshot_robot_label"],
            },
        }

        ensure_schema_or_raise(manifest)

        compression = str(task.boosted_config.get("boosted_output_compression", "gzip")).lower()
        ext = ".pkl.gz" if compression == "gzip" else ".pkl"
        out_path = Path(task.output_dir) / f"{task.task_id}_boosted_manifest{ext}"
        _write_manifest(out_path, manifest, compression=compression)

        result["success"] = True
        result["output_path"] = str(out_path)
        result["object_count"] = len(mined["objects"])
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()

    result["duration_sec"] = time.time() - start_ts
    return result


def _build_boosted_config(namespace: argparse.Namespace, yaml_cfg: Mapping[str, Any], unknown_cli: Sequence[str]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = dict(yaml_cfg)

    def set_if_not_none(key: str, value: Any) -> None:
        if value is not None:
            cfg[key] = value

    # CLI overrides.
    set_if_not_none("output_dir", namespace.output_dir)
    set_if_not_none("start_idx", namespace.start_idx)
    set_if_not_none("end_idx", namespace.end_idx)
    set_if_not_none("workers", namespace.workers)
    set_if_not_none("episodes_per_env", namespace.episodes_per_env)
    set_if_not_none("xml_dir", namespace.xml_dir)
    set_if_not_none("config_file", namespace.config_file)
    set_if_not_none("manifest", namespace.manifest)
    set_if_not_none("seed", namespace.seed)
    set_if_not_none("verbose", namespace.verbose)
    set_if_not_none("run_name", namespace.run_name)
    set_if_not_none("unique_run_dir", namespace.unique_run_dir)

    # Boosted aliases / defaults.
    boosted_max_horizon = namespace.boosted_max_horizon
    if boosted_max_horizon is None:
        boosted_max_horizon = namespace.region_max_chain_depth
    if boosted_max_horizon is None:
        boosted_max_horizon = cfg.get("boosted_max_horizon", cfg.get("region_max_chain_depth", 1))

    boosted_output_compression = namespace.boosted_output_compression
    if boosted_output_compression is None:
        boosted_output_compression = namespace.output_compression
    if boosted_output_compression is None:
        boosted_output_compression = cfg.get("boosted_output_compression", cfg.get("output_compression", "gzip"))

    cfg["boosted_max_horizon"] = int(boosted_max_horizon)
    cfg["boosted_ignore_xml_goal"] = bool(
        namespace.boosted_ignore_xml_goal
        if namespace.boosted_ignore_xml_goal is not None
        else cfg.get("boosted_ignore_xml_goal", True)
    )
    cfg["boosted_cell_filter"] = str(
        namespace.boosted_cell_filter
        if namespace.boosted_cell_filter is not None
        else cfg.get("boosted_cell_filter", "newly_reachable")
    )
    cfg["boosted_same_object_only"] = bool(
        namespace.boosted_same_object_only
        if namespace.boosted_same_object_only is not None
        else cfg.get("boosted_same_object_only", True)
    )
    cfg["boosted_use_cpp_grid_fastpath"] = bool(
        namespace.boosted_use_cpp_grid_fastpath
        if namespace.boosted_use_cpp_grid_fastpath is not None
        else cfg.get("boosted_use_cpp_grid_fastpath", True)
    )
    cfg["boosted_output_compression"] = str(boosted_output_compression).lower()

    # Compatibility metadata for legacy/unknown fields.
    known_arg_dests = {
        "output_dir",
        "start_idx",
        "end_idx",
        "workers",
        "episodes_per_env",
        "xml_dir",
        "config_file",
        "manifest",
        "seed",
        "verbose",
        "run_name",
        "unique_run_dir",
        "output_compression",
        "boosted_output_compression",
        "boosted_max_horizon",
        "region_max_chain_depth",
        "boosted_ignore_xml_goal",
        "boosted_cell_filter",
        "boosted_same_object_only",
        "boosted_use_cpp_grid_fastpath",
    }

    cfg["legacy_unknown_yaml_keys"] = {
        str(k): v
        for k, v in yaml_cfg.items()
        if str(k) not in known_arg_dests
    }

    legacy_cli_overrides: Dict[str, Any] = {}
    i = 0
    unknown_cli_list = list(unknown_cli)
    while i < len(unknown_cli_list):
        token = unknown_cli_list[i]
        if not token.startswith("--"):
            i += 1
            continue

        key = token[2:].replace("-", "_")
        value: Any = True
        if i + 1 < len(unknown_cli_list) and not unknown_cli_list[i + 1].startswith("--"):
            value = unknown_cli_list[i + 1]
            i += 2
        else:
            i += 1

        legacy_cli_overrides[key] = value

    cfg.update(legacy_cli_overrides)
    cfg["legacy_unknown_cli_overrides"] = legacy_cli_overrides
    cfg["unknown_cli_tokens"] = unknown_cli_list

    # Final defaults for legacy-compatible fields.
    cfg.setdefault("workers", 1)
    cfg.setdefault("episodes_per_env", 1)
    cfg.setdefault("seed", 42)
    cfg.setdefault("verbose", False)
    cfg.setdefault("unique_run_dir", False)

    return cfg


def _build_arg_parser(pre_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Boosted deterministic cell-opening collection",
        parents=[pre_parser],
    )

    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (required via CLI or YAML)")
    parser.add_argument("--start-idx", type=int, default=None, help="Start index (inclusive)")
    parser.add_argument("--end-idx", type=int, default=None, help="End index (exclusive)")

    parser.add_argument("--workers", type=int, default=None, help="Worker process count")
    parser.add_argument("--episodes-per-env", type=int, default=None, help="Kept for compatibility; boosted uses one deterministic pass")

    parser.add_argument("--xml-dir", type=str, default=None, help="Directory containing XML environments")
    parser.add_argument("--config-file", type=str, default=None, help="NAMO config YAML")
    parser.add_argument("--manifest", type=str, default=None, help="Optional manifest path (first column must be xml path)")

    parser.add_argument("--seed", type=int, default=None, help="Deterministic seed")
    parser.add_argument("--verbose", action="store_true", default=None, help="Verbose logging")

    parser.add_argument("--run-name", type=str, default=None, help="Optional run suffix")
    parser.add_argument("--unique-run-dir", action="store_true", default=None, help="Create unique timestamped run directory")

    parser.add_argument("--output-compression", type=str, choices=["none", "gzip"], default=None)
    parser.add_argument("--boosted-output-compression", type=str, choices=["none", "gzip"], default=None)

    parser.add_argument("--boosted-max-horizon", type=int, default=None)
    parser.add_argument("--region-max-chain-depth", type=int, default=None)

    parser.add_argument("--boosted-ignore-xml-goal", dest="boosted_ignore_xml_goal", action="store_true")
    parser.add_argument("--no-boosted-ignore-xml-goal", dest="boosted_ignore_xml_goal", action="store_false")
    parser.set_defaults(boosted_ignore_xml_goal=None)

    parser.add_argument("--boosted-cell-filter", type=str, default=None)

    parser.add_argument("--boosted-same-object-only", dest="boosted_same_object_only", action="store_true")
    parser.add_argument("--no-boosted-same-object-only", dest="boosted_same_object_only", action="store_false")
    parser.set_defaults(boosted_same_object_only=None)

    parser.add_argument("--boosted-use-cpp-grid-fastpath", dest="boosted_use_cpp_grid_fastpath", action="store_true")
    parser.add_argument("--no-boosted-use-cpp-grid-fastpath", dest="boosted_use_cpp_grid_fastpath", action="store_false")
    parser.set_defaults(boosted_use_cpp_grid_fastpath=None)

    return parser


def _prepare_output_dir(cfg: Mapping[str, Any]) -> Path:
    base = Path(str(cfg["output_dir"]))
    base.mkdir(parents=True, exist_ok=True)

    host = _short_hostname()
    if cfg.get("run_name"):
        run_dir_name = f"{host}_{_sanitize_run_name(str(cfg['run_name']))}"
    elif bool(cfg.get("unique_run_dir", False)):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir_name = f"{host}_boosted_{int(cfg['start_idx'])}_{int(cfg['end_idx'])}_{timestamp}"
    else:
        run_dir_name = host

    output_dir = base / run_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _build_tasks(cfg: Mapping[str, Any], output_dir: Path, checksum: str) -> List[WorkerTask]:
    xml_files = _discover_environment_files(
        xml_dir=str(cfg["xml_dir"]),
        start_idx=int(cfg["start_idx"]),
        end_idx=int(cfg["end_idx"]),
        manifest_file=str(cfg["manifest"]) if cfg.get("manifest") else None,
    )

    tasks: List[WorkerTask] = []
    for idx, xml_path in enumerate(xml_files):
        task_id = f"env_{int(cfg['start_idx']) + idx:06d}"
        tasks.append(
            WorkerTask(
                task_id=task_id,
                xml_path=str(xml_path),
                config_file=str(cfg["config_file"]),
                output_dir=str(output_dir),
                boosted_config=dict(cfg),
                config_checksum=checksum,
                worker_slot=idx % max(1, int(cfg.get("workers", 1))),
            )
        )
    return tasks


def _save_run_metadata(output_dir: Path, cfg: Mapping[str, Any], checksum: str, task_count: int) -> None:
    metadata = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "producer_version": NAMO_PRODUCER_VERSION,
        "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "hostname": _short_hostname(),
        "task_count": int(task_count),
        "config_checksum": checksum,
        "config_snapshot": serialize_for_metadata(cfg),
    }
    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config-yaml", type=str, help="Path to YAML config for defaults")

    pre_args, remaining = pre_parser.parse_known_args(argv)
    parser = _build_arg_parser(pre_parser)

    yaml_cfg: Dict[str, Any] = {}
    if pre_args.config_yaml:
        try:
            import yaml

            with open(pre_args.config_yaml, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            if isinstance(loaded, dict):
                yaml_cfg = dict(loaded)
                parser.set_defaults(**yaml_cfg)
        except Exception as exc:
            print(f"Warning: failed to load YAML defaults from {pre_args.config_yaml}: {exc}", file=sys.stderr)

    args, unknown_cli = parser.parse_known_args(remaining)
    cfg = _build_boosted_config(args, yaml_cfg, unknown_cli)

    if cfg.get("output_dir") is None or cfg.get("start_idx") is None or cfg.get("end_idx") is None:
        print("Error: output_dir, start_idx, end_idx are required via CLI or YAML", file=sys.stderr)
        return 1
    if cfg.get("xml_dir") is None:
        print("Error: xml_dir is required via CLI or YAML", file=sys.stderr)
        return 1
    if cfg.get("config_file") is None:
        print("Error: config_file is required via CLI or YAML", file=sys.stderr)
        return 1

    if int(cfg["start_idx"]) < 0 or int(cfg["end_idx"]) <= int(cfg["start_idx"]):
        print("Error: require start_idx >= 0 and end_idx > start_idx", file=sys.stderr)
        return 1

    compression = str(cfg.get("boosted_output_compression", "gzip")).lower()
    if compression not in {"none", "gzip"}:
        print("Error: boosted_output_compression must be one of {none,gzip}", file=sys.stderr)
        return 1

    output_dir = _prepare_output_dir(cfg)

    cfg_json = json.dumps(serialize_for_metadata(cfg), sort_keys=True, separators=(",", ":"))
    checksum = hashlib.sha256(cfg_json.encode("utf-8")).hexdigest()

    tasks = _build_tasks(cfg, output_dir, checksum)
    _save_run_metadata(output_dir, cfg, checksum, len(tasks))

    if not tasks:
        print("No environments found for requested range.")
        return 0

    print(f"Boosted collection: {len(tasks)} environments -> {output_dir}")

    workers = max(1, int(cfg.get("workers", 1)))
    results: List[Dict[str, Any]] = []

    if workers == 1:
        for task in tasks:
            results.append(_run_single_task(task))
    else:
        with Pool(processes=workers) as pool:
            for out in pool.imap_unordered(_run_single_task, tasks):
                results.append(out)
                if cfg.get("verbose"):
                    status = "ok" if out.get("success") else "fail"
                    print(f"[{status}] {out.get('task_id')}: {out.get('duration_sec', 0.0):.2f}s")

    results_sorted = sorted(results, key=lambda x: str(x.get("task_id", "")))
    summary = {
        "total": len(results_sorted),
        "success": sum(1 for r in results_sorted if r.get("success")),
        "failed": sum(1 for r in results_sorted if not r.get("success")),
        "results": results_sorted,
    }

    summary_path = output_dir / "collection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        f"Completed boosted collection: success={summary['success']} failed={summary['failed']} "
        f"summary={summary_path}"
    )
    return 0 if summary["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
