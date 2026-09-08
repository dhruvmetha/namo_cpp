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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "smoke", "full", "gate", "aggregate"])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--index", type=int, default=0)
    a = ap.parse_args()
    if a.stage in {"smoke", "full"}:
        task(a.root, a.stage, a.index)
    else:
        globals()[a.stage](a.root)


if __name__ == "__main__":
    main()
