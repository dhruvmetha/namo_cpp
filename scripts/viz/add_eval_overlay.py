#!/usr/bin/env python3
"""Join solvability_runner eval output onto the scene gallery: solved/sims per arm, per room.

Writes eval.json next to scenes.json. Two parts.

`rooms`: for every gallery room (key = scene, the same string scenes.json rows carry) that shows up
in any run, and for every run, the raw per-arm result (arm name, solved, sims) split into model and
random groups, plus each group's median sims and solve rate over the arms that solved, plus the
speedup (random median / model median, over arms that solved -- null if either group has no solve).

`aggregate`: per run, solve@k (k = 1,2,5,10,30,100,900) and median sims, model vs random, stratified
by (door_needs_both_blocks, tier, horizon). A room's arm-level results are room-level -- the eval
never asked which of a room's objects or horizons got pushed -- so a room's results are counted into
EVERY (tier, horizon) stratum its cards span; door_needs_both_blocks is constant per room.

An arm is a model arm if its directory name starts with HY5U or ends with _model, a random arm if it
starts with rand_ or ends with _uniform. Anything else is skipped with a note on stdout -- the arm
list itself is never hardcoded, so a later run with new seed names needs no edit here.

Reads scenes.json and cards/*.json for the join key and door_needs_both_blocks; writes neither.

    python scripts/viz/add_eval_overlay.py --out $NAMO_SCRATCH/viz/real_2mov \\
        --run gallery_0830=$NAMO_SCRATCH/eval/two_movable_1hop_20260830 \\
        --run doorway=$NAMO_SCRATCH/eval/group_doorway_bestfirst_fb2d8cc
"""
import argparse
import glob
import json
import os
import statistics as stats
from collections import defaultdict

K_LEVELS = [1, 2, 5, 10, 30, 100, 900]


def classify_arm(name):
    if name.startswith("HY5U") or name.endswith("_model"):
        return "model"
    if name.startswith("rand_") or name.endswith("_uniform"):
        return "random"
    return None


def scene_of(xml_path):
    # The gallery's scene key is pool/family/rb_xxxxx -- the three path parts before the filename --
    # regardless of which box or scratch root the eval ran from.
    parts = xml_path.rstrip("/").split("/")
    return "/".join(parts[-4:-1])


def load_arm(arm_dir):
    rows = []
    for fn, solved in ((os.path.join(arm_dir, "solved.jsonl"), True),
                        (os.path.join(arm_dir, "unsolved.jsonl"), False)):
        if not os.path.exists(fn):
            continue
        for line in open(fn):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            xml = r.get("xml_path")
            if not xml:
                continue
            rows.append({"scene": scene_of(xml), "solved": solved,
                         "sims": r.get("simulation_budget_used_total")})
    return rows


def median(vals):
    return round(stats.median(vals), 2) if vals else None


def summarize(rows):
    n = len(rows)
    solved_sims = [r["sims"] for r in rows if r["solved"] and r["sims"] is not None]
    solve_at = {}
    for k in K_LEVELS:
        c = sum(1 for r in rows if r["solved"] and r["sims"] is not None and r["sims"] <= k)
        solve_at[str(k)] = round(c / n, 3) if n else None
    return {"n": n, "solve_at": solve_at, "median_sims": median(solved_sims)}


def load_scene_index(out_dir):
    index = json.load(open(os.path.join(out_dir, "scenes.json")))
    scene_cards = defaultdict(list)
    for row in index["cards"]:
        scene_cards[row["scene"]].append(row)
    door = {}
    for scene, rows in scene_cards.items():
        meta = json.load(open(os.path.join(out_dir, "cards", rows[0]["file"])))["meta"]
        door[scene] = bool(meta.get("door_needs_both_blocks"))
    return scene_cards, door


def load_run(run_dir):
    """arm name -> (group, rows). Skips directories that are not a solvability_runner arm and arms
    whose name matches neither the model nor the random pattern."""
    out = {}
    for d in sorted(glob.glob(os.path.join(run_dir, "*"))):
        if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "summary.json")):
            continue
        arm = os.path.basename(d.rstrip("/"))
        grp = classify_arm(arm)
        if grp is None:
            print(f"  skip arm '{arm}': matches neither the model nor the random naming pattern")
            continue
        out[arm] = (grp, load_arm(d))
    return out


def build_run(run_name, run_dir, scene_cards, door):
    arms = load_run(run_dir)

    # scene -> arm -> row, joined to the gallery: a room that never got a card is not in this gallery.
    scene_arm = defaultdict(dict)
    for arm, (grp, rows) in arms.items():
        for r in rows:
            if r["scene"] in scene_cards:
                scene_arm[r["scene"]][arm] = r

    rooms = {}
    for scene, by_arm in scene_arm.items():
        model_list, random_list = [], []
        for arm, r in by_arm.items():
            entry = {"arm": arm, "solved": r["solved"], "sims": r["sims"]}
            (model_list if arms[arm][0] == "model" else random_list).append(entry)
        model_list.sort(key=lambda e: e["arm"])
        random_list.sort(key=lambda e: e["arm"])
        m_med = median([e["sims"] for e in model_list if e["solved"] and e["sims"] is not None])
        r_med = median([e["sims"] for e in random_list if e["solved"] and e["sims"] is not None])
        rooms[scene] = {
            "model": model_list,
            "random": random_list,
            "model_median_sims": m_med,
            "random_median_sims": r_med,
            "model_solve_rate": round(sum(e["solved"] for e in model_list) / len(model_list), 3)
                                 if model_list else None,
            "random_solve_rate": round(sum(e["solved"] for e in random_list) / len(random_list), 3)
                                  if random_list else None,
            "speedup": round(r_med / m_med, 2) if (m_med and r_med) else None,
        }

    # A room's result counts into every (tier, horizon) its own cards span; door is constant per room.
    buckets = defaultdict(lambda: {"model": [], "random": []})
    for arm, (grp, rows) in arms.items():
        for r in rows:
            cards = scene_cards.get(r["scene"])
            if not cards:
                continue
            strata = {(c["tier"], c["horizon"]) for c in cards}
            for tier, horizon in strata:
                buckets[(door[r["scene"]], tier, horizon)][grp].append(r)

    aggregate = [
        {"door_needs_both_blocks": d, "tier": tier, "horizon": horizon,
         "model": summarize(g["model"]), "random": summarize(g["random"])}
        for (d, tier, horizon), g in sorted(buckets.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2]))
    ]
    # Gallery-joined rows only, both pooled (one row per room-arm, whatever horizons it spans) and
    # split by horizon (a room with both horizons counts once in each -- same rule as `aggregate`).
    pooled = {"model": [], "random": []}
    by_horizon = defaultdict(lambda: {"model": [], "random": []})
    for arm, (grp, rows) in arms.items():
        for r in rows:
            cards = scene_cards.get(r["scene"])
            if not cards:
                continue
            pooled[grp].append(r)
            for h in {c["horizon"] for c in cards}:
                by_horizon[h][grp].append(r)

    return rooms, aggregate, pooled, by_horizon, arms


def print_table(run_name, rooms, arms, pooled, by_horizon):
    n_model_arms = sum(1 for grp, _ in arms.values() if grp == "model")
    n_random_arms = sum(1 for grp, _ in arms.values() if grp == "random")
    print(f"\n{run_name}  ({n_model_arms} model arm(s), {n_random_arms} random arm(s), "
          f"{len(rooms)} rooms joined to the gallery)")
    print(f"  {'':<8} {'model solved':>14} {'model sims':>11} {'random solved':>15} "
          f"{'random sims':>12} {'speedup':>9}")
    for label in ["overall"] + sorted(by_horizon):
        if label == "overall":
            m, rd = summarize(pooled["model"]), summarize(pooled["random"])
        else:
            m, rd = summarize(by_horizon[label]["model"]), summarize(by_horizon[label]["random"])
        up = round(rd["median_sims"] / m["median_sims"], 2) if (m["median_sims"] and rd["median_sims"]) else None
        m_solved = round(m["solve_at"]["900"] * m["n"]) if m["n"] else 0
        r_solved = round(rd["solve_at"]["900"] * rd["n"]) if rd["n"] else 0
        print(f"  {label:<8} {m_solved:>7}/{m['n']:<6} {str(m['median_sims']):>11} "
              f"{r_solved:>8}/{rd['n']:<6} {str(rd['median_sims']):>12} {str(up) + 'x' if up else '-':>9}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="gallery data root (holds scenes.json, cards/)")
    ap.add_argument("--run", action="append", default=[], metavar="NAME=DIR",
                     help="one solvability_runner output dir (holds arm subdirs); repeatable")
    a = ap.parse_args()
    if not a.run:
        raise SystemExit("give at least one --run NAME=DIR")

    runs = {}
    for spec in a.run:
        name, sep, d = spec.partition("=")
        if not sep or not name or not d:
            raise SystemExit(f"--run must be NAME=DIR, got '{spec}'")
        runs[name] = d

    scene_cards, door = load_scene_index(a.out)

    eval_rooms = defaultdict(dict)
    aggregate = {}
    for run_name, run_dir in runs.items():
        print(f"reading {run_name} <- {run_dir}")
        rooms, agg, pooled, by_horizon, arms = build_run(run_name, run_dir, scene_cards, door)
        for scene, rec in rooms.items():
            eval_rooms[scene][run_name] = rec
        aggregate[run_name] = agg
        print_table(run_name, rooms, arms, pooled, by_horizon)

    out_path = os.path.join(a.out, "eval.json")
    json.dump({"schema_version": 1, "runs": list(runs.keys()),
               "rooms": eval_rooms, "aggregate": aggregate}, open(out_path, "w"))
    print(f"\neval.json: {len(eval_rooms)} rooms across {len(runs)} run(s) -> {out_path}")


if __name__ == "__main__":
    main()
