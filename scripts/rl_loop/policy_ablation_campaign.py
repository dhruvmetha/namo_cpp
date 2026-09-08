#!/usr/bin/env python3
"""Frozen 5 mm policy ablations: smoke, bundled evaluation, and strict reporting."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "python"), str(REPO / "scripts"), str(REPO / "scripts/rl_loop")]
from namo import eval_sets
from namo.paths import SCRATCH, resolve
from agg_testset_reactive import load_divisions

FAMILIES = {
    "HY5U": "models", "HY5": "models",
    "HY5U_no_family": "ablations_20260831/models",
    "HY5U_regression": "ablations_20260831/models",
    "HY5U_independent": "ablations_20260831/models",
    "HY5U_global": "architecture_ablations_20260904/models",
    "HY5U_no_local": "architecture_ablations_20260904/models",
    "HY5U_no_edge": "architecture_edge_id_20260905/models",
}
LEGS = {"1push": eval_sets.ONEPUSH, "2push": eval_sets.PURE2PUSH}
CUTS = (1, 2, 5, 10)


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def verify_config():
    import yaml
    cfg = Path(os.environ["NAMO_CFG"])
    margin_file = cfg.parent / "wavefront_inflation.yaml"
    margin = yaml.safe_load(margin_file.read_text())["tier1"]["base_inflation_margin_m"]
    if margin != 0.005:
        raise ValueError(f"This campaign requires 5 mm, got {margin}")
    from namo.visualization.wavefront_snapshot import WavefrontSnapshotExporter
    if WavefrontSnapshotExporter._load_tier1_inflation_margin(cfg) != margin:
        raise ValueError("Ranker renderer and simulator disagree on the clearance margin")
    return {"config": str(cfg), "config_sha256": digest(cfg),
            "margin_m": margin, "margin_sha256": digest(margin_file)}


def prepare(root):
    arms = []
    for family, subdir in FAMILIES.items():
        for seed in (1, 2, 3):
            name = f"{family}_s{seed}"
            candidates = list((SCRATCH / "aquaman/round0" / subdir / name / "checkpoints").glob("epoch*.ckpt"))
            if len(candidates) != 1:
                raise ValueError(f"Expected one registered selected checkpoint: {name}: {candidates}")
            ckpt = candidates[0]
            arms.append({"name": name, "family": family, "seed": seed, "prior": "q",
                         "source_ckpt": str(ckpt), "sha256": digest(ckpt),
                         "new_full": family != "HY5U"})
    for seed in (7000, 8000, 9000):
        arms.append({**arms[0], "name": f"rand_s{seed}", "family": "Random",
                     "seed": seed, "prior": "uniform", "new_full": False})
    write_json(root / "plan.json", {"schema": 1, "arms": arms, "max_pushes": 10,
               "bundles_per_arm": 8, "workers_per_bundle": 40,
               "manifests": {leg: {"sha256": digest(path),
                                   "rooms": len(json.loads(path.read_text()))}
                             for leg, path in LEGS.items()},
               "control_root": "eval/policy_v3_jamguard_20260822",
               "sage_commit": "ceff4bf49f1fb55f91be47eaacb038c5fefde687",
               "bindings_sha256": "cb2977a8a13c287e6c6a9e042deb50b1174ea50363f5d380e2e076be9abb8de3"})


def run_leaf(root, arm, leg, start, end, shard, stage, groups=None, mode="policy"):
    outdir = root / stage / arm["name"] / f"{leg}_{mode}"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"shard_{shard}.json"
    cmd = [sys.executable, str(REPO / "scripts/rl_loop/eval_policy.py"),
           "--ckpt", str(root / "checkpoints" / (arm["name"] + ".ckpt")),
           "--key", str(LEGS.get(leg, eval_sets.ONEPUSH)), "--max-pushes", "10" if not groups else "2",
           "--prior", arm["prior"], "--seed", str(arm["seed"]),
           "--start", str(start), "--end", str(end), "--mode", mode,
           "--out", str(out), "--leaf-out", str(out.with_suffix(".jsonl"))]
    if groups:
        cmd += ["--groups", str(groups)]
    started = time.perf_counter()
    with out.with_suffix(".log").open("w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    return {"arm": arm["name"], "leg": leg, "start": start, "end": end,
            "elapsed_s": time.perf_counter() - started, "output": str(out),
            "summary": json.loads(out.read_text())}


def task(root, stage, index):
    plan = json.loads((root / "plan.json").read_text())
    provenance = verify_config()
    binding = list(Path(os.environ["NAMO_BINDINGS"]).glob("namo_rl*.so"))
    if len(binding) != 1 or digest(binding[0]) != plan["bindings_sha256"]:
        raise ValueError("Simulator binding differs from the registered 5 mm controls")
    sage_commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                    cwd=os.environ["SAGE_REPO"], text=True).strip()
    if sage_commit != plan["sage_commit"]:
        raise ValueError("Sage code differs from the checkpoint training revision")
    provenance.update({"bindings_sha256": digest(binding[0]), "sage_commit": sage_commit})
    for leg, path in LEGS.items():
        if digest(path) != plan["manifests"][leg]["sha256"]:
            raise ValueError(f"Manifest changed: {leg}")
    if stage == "smoke":
        arm = plan["arms"][index]
        commands = [(leg, 0, 8, 0) for leg in LEGS]
        workers = 2
    else:
        arms = [a for a in plan["arms"] if a["new_full"]]
        bundles = plan["bundles_per_arm"]
        workers = plan["workers_per_bundle"]
        arm, bundle = arms[index // bundles], index % bundles
        commands = []
        per_leg = workers // 2
        total = bundles * per_leg
        for leg in LEGS:
            count = plan["manifests"][leg]["rooms"]
            for offset in range(per_leg):
                shard = bundle * per_leg + offset
                commands.append((leg, shard * count // total, (shard + 1) * count // total, shard))
    ckpt = root / "checkpoints" / (arm["name"] + ".ckpt")
    if digest(ckpt) != arm["sha256"]:
        raise ValueError(f"Checkpoint hash mismatch: {arm['name']}")
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_leaf, root, arm, leg, start, end, shard, stage)
                   for leg, start, end, shard in commands]
        results = [f.result() for f in futures]
    write_json(root / "markers" / f"{stage}_{index}.json",
               {"elapsed_s": time.perf_counter() - started, "leaves": results,
                "provenance": provenance, "commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()})


def read_leaves(directory):
    rows = {}
    for file in sorted(directory.glob("shard_*.jsonl")):
        for line in file.read_text().splitlines():
            row = json.loads(line)
            key = (str(resolve(row["xml"])), row["object_id"], row.get("region"))
            if key in rows:
                raise ValueError(f"Duplicate episode: {key}")
            rows[key] = row
    return rows


def gate(root):
    plan = json.loads((root / "plan.json").read_text())
    markers = [json.loads((root / "markers" / f"smoke_{i}.json").read_text())
               for i in range(len(plan["arms"]))]
    checked = 0
    for arm in plan["arms"]:
        if arm["new_full"]:
            continue
        for leg in LEGS:
            old = read_leaves(SCRATCH / plan["control_root"] / arm["name"] / f"{leg}_policy")
            new = read_leaves(root / "smoke" / arm["name"] / f"{leg}_policy")
            if not new:
                raise ValueError("Empty control smoke")
            for key, row in new.items():
                if key not in old or any(row[field] != old[key][field]
                                         for field in ("opened_at", "n_push", "n_noop")):
                    raise ValueError(f"Cached control parity failed: {arm['name']} {key}")
                checked += 1
    write_json(root / "smoke_gate.json", {"passed": True, "control_episodes_checked": checked,
               "slowest_smoke_s": max(m["elapsed_s"] for m in markers),
               "arms_checked": len(markers)})


def aggregate(root):
    plan = json.loads((root / "plan.json").read_text())
    count = sum(a["new_full"] for a in plan["arms"]) * plan["bundles_per_arm"]
    for i in range(count):
        if not (root / "markers" / f"full_{i}.json").exists():
            raise ValueError(f"Missing completed bundle {i}")
    report = {"protocol": {"margin_m": 0.005, "max_pushes": 10, "search_lookahead": False}, "legs": {}}
    lines = ["# Policy ablations at 5 mm", "", "Mean ± sample SD across three seeds. Every attempted push is counted. Target object and initial target points stay fixed. No wall-time comparison.", ""]
    for leg in LEGS:
        divfile = eval_sets.ONEPUSH if leg == "1push" else eval_sets.DIVISIONS
        divisions = {(str(resolve(x)), o, r): t for (x, o, r), t in load_divisions(str(divfile)).items()}
        expected = set(read_leaves(SCRATCH / plan["control_root"] / "HY5U_s1" / f"{leg}_policy"))
        values = {}
        for arm in plan["arms"]:
            path = root / "full" if arm["new_full"] else SCRATCH / plan["control_root"]
            rows = read_leaves(path / arm["name"] / f"{leg}_policy")
            if set(rows) != expected:
                raise ValueError(f"{leg}/{arm['name']}: expected {len(expected)}, got {len(rows)}; keys differ")
            rates = {}
            for tier in ("easy", "medium", "hard", "all"):
                selected = [r for k, r in rows.items() if tier == "all" or divisions[k] == tier]
                rates[tier] = {"n": len(selected), **{f"open@{k}": 100 * sum(0 < r["opened_at"] <= k for r in selected) / len(selected) for k in CUTS}}
            values[arm["name"]] = rates
        summary = {}
        for family in [*FAMILIES, "Random"]:
            arms = [a for a in plan["arms"] if a["family"] == family]
            summary[family] = {tier: {f"open@{k}": {"mean": statistics.mean(values[a["name"]][tier][f"open@{k}"] for a in arms),
                                                       "sd": statistics.stdev(values[a["name"]][tier][f"open@{k}"] for a in arms)} for k in CUTS}
                               for tier in ("easy", "medium", "hard", "all")}
        report["legs"][leg] = {"n": len(expected), "per_seed": values, "summary": summary}
        for cut in CUTS:
            lines += [f"## {leg}: open@{cut}", "", "| Model | Easy | Medium | Hard | Overall |", "|---|---:|---:|---:|---:|"]
            for family, tiers in summary.items():
                cells = [f"{tiers[t][f'open@{cut}']['mean']:.1f}±{tiers[t][f'open@{cut}']['sd']:.1f}" for t in ("easy", "medium", "hard", "all")]
                lines.append(f"| {family} | " + " | ".join(cells) + " |")
            lines.append("")
    write_json(root / "aggregate.json", report)
    (root / "aggregate.md").write_text("\n".join(lines) + "\n")


def census(root, smoke=False):
    verify_config()
    out = root / ("census_smoke.jsonl" if smoke else "census.jsonl")
    cmd = [sys.executable, str(REPO / "scripts/pipeline/probe_static_topology.py"),
           "--episode-manifests", str(eval_sets.ONEPUSH), str(eval_sets.PURE2PUSH),
           "--out", str(out), "--config", os.environ["NAMO_CFG"],
           "--workers", "1" if smoke else os.environ.get("SLURM_CPUS_PER_TASK", "32")]
    if smoke:
        cmd += ["--end", "2"]
    started = time.perf_counter()
    subprocess.run(cmd, check=True)
    summary = json.loads(Path(str(out) + ".summary.json").read_text())
    if summary["exclusion_counts"].get("error", 0):
        raise ValueError("Census contains errors; inspect exclusions before selecting a cohort")
    write_json(root / ("census_smoke_done.json" if smoke else "census_done.json"),
               {"elapsed_s": time.perf_counter() - started, "summary": summary})


def group_task(root, index, smoke=False):
    verify_config()
    plan = json.loads((root / "plan.json").read_text())
    arms = [a for a in plan["arms"] if not a["new_full"]]
    groups = root / "census.jsonl.pilot.jsonl"
    n_groups = len(groups.read_text().splitlines())
    modes = ("policy", "search")
    if smoke:
        arm = next(a for a in arms if a["name"] == ("HY5U_s1" if index < 2 else "rand_s7000"))
        group_idx, mode = 0, modes[index % 2]
        stage = "group_smoke"
    else:
        arm = arms[index // (n_groups * 2)]
        group_idx, mode = (index % (n_groups * 2)) // 2, modes[index % 2]
        stage = "groups"
    result = run_leaf(root, arm, "group", group_idx, group_idx + 1,
                      group_idx, stage, groups=groups, mode=mode)
    write_json(root / "markers" / f"{stage}_{index}.json", result)


def aggregate_groups(root):
    plan = json.loads((root / "plan.json").read_text())
    groups = [json.loads(x) for x in (root / "census.jsonl.pilot.jsonl").read_text().splitlines()]
    expected = {g["group_id"] for g in groups}
    results = {}
    for arm in [a for a in plan["arms"] if not a["new_full"]]:
        for mode in ("policy", "search"):
            rows = list(read_leaves(root / "groups" / arm["name"] / f"group_{mode}").values())
            if {r["object_id"] for r in rows} != expected:
                raise ValueError(f"Incomplete group pilot {arm['name']}/{mode}")
            results[f"{arm['name']}/{mode}"] = rows
    strata = sorted({s for g in groups for s in g["source_strata"]})
    report = {"n_unique_groups": len(groups), "source_strata_overlap": True,
              "strata_are_source_labels_not_group_difficulty": True, "per_arm": {}}
    for name, rows in results.items():
        split = {}
        for stratum in ["all", *strata]:
            selected = [r for r in rows if stratum == "all" or stratum in r["group"]["source_strata"]]
            mode = name.split("/")[1]
            cuts = (1, 2) if mode == "policy" else (1, 2, 5, 10, 30, 100, 900)
            split[stratum] = {"n": len(selected), "success": {str(k): sum(
                (0 < r["opened_at"] <= k) if mode == "policy" else (r["solved"] and r["sims"] <= k)
                for r in selected) for k in cuts}, "switching_solutions": sum(
                    len({a["object_id"] for a in r["actions"]}) > 1
                    and (r.get("solved", False) if mode == "search" else r["opened_at"] > 0)
                    for r in selected)}
        report["per_arm"][name] = split
    write_json(root / "group_aggregate.json", report)


def submit(root, stage, count=1, cpus=1, minutes=20, dependency=None):
    queued = subprocess.check_output(["squeue", "-r", "-h", "-u", os.environ["USER"], "-o", "%P"], text=True)
    if sum(x == "main" for x in queued.splitlines()) + count > 490:
        raise RuntimeError("Amarel submission cap: wait for current tasks to finish")
    cmd = ["sbatch", "--parsable", f"--array=0-{count-1}", f"--cpus-per-task={cpus}",
           f"--mem={max(6, cpus * 2)}G", f"--time={minutes}",
           f"--job-name=pol_{stage}", f"--output={root}/logs/{stage}_%A_%a.out"]
    if dependency:
        cmd += [f"--dependency=afterok:{dependency}"]
    cmd += [str(REPO / "scripts/slurm/policy_ablation_task.slurm")]
    env = dict(os.environ, CAMPAIGN_ROOT=str(root), CAMPAIGN_STAGE=stage, NAMO_REPO=str(REPO))
    job = subprocess.check_output(cmd, env=env, cwd=REPO, text=True).strip().split(";")[0]
    return job


def monitor(root):
    """Lightweight detached controller. Poll artifacts every five minutes."""
    state_path = root / "controller_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "smoke_job": os.environ["SMOKE_JOB_ID"],
        "census_smoke_job": os.environ.get("CENSUS_SMOKE_JOB_ID"),
    }
    plan = json.loads((root / "plan.json").read_text())
    while True:
        try:
            if not state.get("census_smoke_job"):
                state["census_smoke_job"] = submit(root, "census-smoke", minutes=10)
                write_json(state_path, state)
            if not state.get("full_job") and all((root / "markers" / f"smoke_{i}.json").exists() for i in range(len(plan["arms"]))):
                gate(root)
                elapsed = json.loads((root / "smoke_gate.json").read_text())["slowest_smoke_s"]
                count = sum(a["new_full"] for a in plan["arms"]) * plan["bundles_per_arm"]
                minutes = max(15, min(90, int(elapsed * 3 / 60) + 5))
                state["full_job"] = submit(root, "full", count, plan["workers_per_bundle"], minutes)
                write_json(state_path, state)
                state["aggregate_job"] = submit(root, "aggregate", dependency=state["full_job"])
                state["calibrated_wall_minutes"] = minutes
            if not state.get("census_job") and (root / "census_smoke_done.json").exists():
                state["census_job"] = submit(root, "census", cpus=64, minutes=15)
            if not state.get("group_smoke_job") and (root / "census_done.json").exists():
                census_info = json.loads((root / "census_done.json").read_text())
                n = len(census_info["summary"]["pilot_group_ids"])
                if n:
                    state["group_smoke_job"] = submit(root, "group-smoke", 4, 1, 30)
                else:
                    state["group_smoke_job"] = "no_eligible_groups"
                    state["groups_done"] = True
            if state.get("group_smoke_job") not in (None, "no_eligible_groups") and not state.get("group_job"):
                if all((root / "markers" / f"group_smoke_{i}.json").exists() for i in range(4)):
                    n = len((root / "census.jsonl.pilot.jsonl").read_text().splitlines())
                    state["group_job"] = submit(root, "group", n * 12, 1, 30)
                    state["group_aggregate_job"] = submit(root, "group-aggregate", dependency=state["group_job"])
            state["policy_done"] = (root / "aggregate.json").exists()
            state["groups_done"] = state.get("groups_done", False) or (root / "group_aggregate.json").exists()
            state["last_checked"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            write_json(state_path, state)
            if state["policy_done"] and state["groups_done"]:
                state["status"] = "complete"
                write_json(state_path, state)
                return
            # Accounting is authoritative for failures, even when a worker died before logging.
            jobs = [v for k, v in state.items() if k.endswith("_job") and str(v).isdigit()]
            if jobs:
                accounting = subprocess.check_output(["sacct", "-nP", "-j", ",".join(jobs), "--format=JobID,State,ExitCode"], text=True)
                failed = [line for line in accounting.splitlines() if any(s in line for s in ("|FAILED|", "|TIMEOUT|", "|OUT_OF_MEMORY|", "|PREEMPTED|", "|CANCELLED"))]
                if failed:
                    raise RuntimeError("Campaign task failures: " + "; ".join(failed[:10]))
        except Exception as exc:
            state.update(status="needs_attention", error=str(exc))
            write_json(state_path, state)
            raise
        time.sleep(300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "smoke", "full", "gate", "aggregate", "census-smoke", "census", "group-smoke", "group", "group-aggregate", "monitor"])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--index", type=int, default=0)
    a = ap.parse_args()
    if a.stage in {"smoke", "full"}:
        task(a.root, a.stage, a.index)
    elif a.stage in {"census", "census-smoke"}:
        census(a.root, a.stage == "census-smoke")
    elif a.stage in {"group", "group-smoke"}:
        group_task(a.root, a.index, a.stage == "group-smoke")
    elif a.stage == "group-aggregate":
        aggregate_groups(a.root)
    else:
        globals()[a.stage](a.root)


if __name__ == "__main__":
    main()
