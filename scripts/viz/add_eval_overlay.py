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
EVERY (tier, horizon) stratum its cards span; that makes `n` here a CARD-STRATUM count, not a room
count (a room with three cards spanning two strata is counted six times per arm, once per stratum).
Kept for tier/horizon breakdowns only; do not read `n` here as "rooms".

`room_aggregate`: the room-accurate version, per run, three rows (scope "overall" / "open" / "door",
matching door_needs_both_blocks False/True) with `n` = an actual room count. solve@k per room is the
mean over that room's arms in the group of "solved and sims <= k" (0/1 per arm), then meaned again
over rooms in the scope -- so a room with three seeds and a room with one seed count equally. Only
rooms joined to a gallery card are eligible (see `rooms` above); an eval run can cover rooms this
gallery has no card for, and those are invisible here, not folded into `n`.

`chain`: per card file, the label's own chain vocabulary -- see `load_card_chains`.

`model_chain`: per run, per card file, the model's chain vocabulary -- see `card_model_bucket`. Each
arm entry under `rooms[scene][run]["model"/"random"]` carries its raw "pushes" (that arm's executed
solution, object_id/edge_idx/depth per push, [] when unsolved); `model_chain` is the classification
of those pushes against ONE card's object_id, which `rooms` itself cannot do since a room can carry
several cards. Both vocabularies: "single push" (one push, whichever object), "card object only" /
"other object only" / "both objects" (two or more pushes, read against the card's object_id) --
`chain` never produces "other object only" (see `load_card_chains`). `model_chain` adds "seeds
disagree" (the run's solved model arms landed in different categories) and "unsolved" (no model arm
solved this room).

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
            sol = r.get("solution") or []
            pushes = [{"object_id": st.get("object_id"), "edge_idx": st.get("edge_idx"),
                       "depth": st.get("depth")} for st in sol]
            rows.append({"scene": scene_of(xml), "solved": solved,
                         "sims": r.get("simulation_budget_used_total"), "pushes": pushes})
    return rows


def median(vals):
    return round(stats.median(vals), 2) if vals else None


def summarize(rows):
    """Card-stratum summary: `rows` is arm-level results already multiplied across every
    (tier, horizon) stratum a room's cards span, so `n_cards` counts strata-hits, not rooms."""
    n = len(rows)
    solved_sims = [r["sims"] for r in rows if r["solved"] and r["sims"] is not None]
    solve_at = {}
    for k in K_LEVELS:
        c = sum(1 for r in rows if r["solved"] and r["sims"] is not None and r["sims"] <= k)
        solve_at[str(k)] = round(c / n, 3) if n else None
    return {"n_cards": n, "solve_at": solve_at, "median_sims": median(solved_sims)}


def room_solve_fraction(entries, k):
    """Fraction of one room's arms (in one group) that solved within k sims, or None if the room
    has no arms of that group -- so a room with no random arms does not silently score 0."""
    if not entries:
        return None
    hits = sum(1 for e in entries if e["solved"] and e["sims"] is not None and e["sims"] <= k)
    return hits / len(entries)


def summarize_rooms(room_recs, group):
    """Room-accurate summary over `room_recs` (a scope's rows from `rooms`, i.e. one entry per
    room regardless of how many cards or arms it has). solve@k = mean over ROOMS of that room's own
    arm-average -- a room contributes once no matter how many seeds it ran, so `n` is a room count."""
    n = len(room_recs)
    solve_at = {}
    for k in K_LEVELS:
        fracs = [f for rec in room_recs for f in [room_solve_fraction(rec[group], k)] if f is not None]
        solve_at[str(k)] = round(sum(fracs) / len(fracs), 3) if fracs else None
    medians = [rec[f"{group}_median_sims"] for rec in room_recs if rec.get(f"{group}_median_sims") is not None]
    return {"n": n, "solve_at": solve_at, "median_sims": median(medians)}


def load_scene_index(out_dir):
    index = json.load(open(os.path.join(out_dir, "scenes.json")))
    scene_cards = defaultdict(list)
    for row in index["cards"]:
        scene_cards[row["scene"]].append(row)
    door = {}
    for scene, rows in scene_cards.items():
        meta = json.load(open(os.path.join(out_dir, "cards", rows[0]["file"])))["meta"]
        door[scene] = bool(meta.get("door_needs_both_blocks"))
    return index["cards"], scene_cards, door


def load_card_chains(cards, out_dir):
    """Per-card label chain, run-independent: "single push" (1push), "card object only" (every step
    pushes the card's own object_id), "both objects" (a step lands on the other block), or null (a
    2push card with no replay). Same three names the model side uses, minus "other object only" --
    the label solution never skips the card's own object entirely, so that bucket cannot occur here.

    ⛔ UNDERCOUNTS "both objects". This solution comes from the exhaustive sweep
    (scripts/pipeline/exhaustive_hmax2.py), whose finish loop stops at the first push that opens and
    walks objects in list order (exhaustive_hmax2.py:129-135) -- a same-object finish always wins the
    race over a cross-object one when both exist. So a "card object only" label here can hide an
    untried cross-object finish; a "both objects" label cannot be wrong the other way. Fine for the
    setup-vs-dead-end call the sweep makes, wrong to read as "how often does the chain switch objects."
    """
    chains = {}
    for row in cards:
        f = row["file"]
        if f in chains:
            continue
        if row["horizon"] != "2push":
            chains[f] = "single push"
            continue
        rp = os.path.join(out_dir, "replay", f)
        if not os.path.exists(rp):
            chains[f] = None
            continue
        steps = json.load(open(rp)).get("steps", [])
        if len(steps) < 2:
            chains[f] = "single push"
            continue
        chains[f] = ("card object only" if all(s.get("object_id") == row["object_id"] for s in steps)
                      else "both objects")
    return chains


def categorize_pushes(pushes, card_obj):
    """One solved arm's push sequence, read against ONE card's own object_id (the reference the
    label side also uses) -- not a room-level classification, since a room's arms can be read against
    several cards, one per object it has a card for."""
    if len(pushes) == 1:
        return "single push"
    objs = {p["object_id"] for p in pushes}
    if objs == {card_obj}:
        return "card object only"
    if card_obj not in objs:
        return "other object only"
    return "both objects"


def card_model_bucket(rec, card_obj):
    """Room `rec` (rooms[scene], or None if the run never touched this room) read against one card's
    object_id: the shared category if every solved model arm agrees, "seeds disagree" if they split,
    "unsolved" if none did. A 2push room solved in a single push lands in "single push" here, not
    "unsolved" -- the old room-level version conflated the two."""
    if not rec:
        return "unsolved"
    cats = [categorize_pushes(e["pushes"], card_obj) for e in rec["model"] if e["solved"] and e["pushes"]]
    if not cats:
        return "unsolved"
    return cats[0] if all(c == cats[0] for c in cats) else "seeds disagree"


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
            # pushes: this arm's own executed solution, [] when unsolved or the run predates it.
            # Read against a card's own object_id downstream (card_model_bucket) -- classifying it
            # here would need a card, and one room can carry several.
            entry = {"arm": arm, "solved": r["solved"], "sims": r["sims"], "pushes": r.get("pushes") or []}
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

    # Room-accurate: one row per room, split only on door (constant per room, no tier/horizon
    # multiplication) so `n` here is an actual room count.
    room_aggregate = [
        {"scope": "overall", "n": len(rooms),
         "model": summarize_rooms(list(rooms.values()), "model"),
         "random": summarize_rooms(list(rooms.values()), "random")},
        {"scope": "open", "n": sum(1 for s in rooms if not door[s]),
         "model": summarize_rooms([r for s, r in rooms.items() if not door[s]], "model"),
         "random": summarize_rooms([r for s, r in rooms.items() if not door[s]], "random")},
        {"scope": "door", "n": sum(1 for s in rooms if door[s]),
         "model": summarize_rooms([r for s, r in rooms.items() if door[s]], "model"),
         "random": summarize_rooms([r for s, r in rooms.items() if door[s]], "random")},
    ]

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

    # Per-card model chain, read against each card's own object_id -- fixes the two bugs the
    # room-level version had: a room with two cards no longer shares one verdict, and a 2push room
    # solved in one push is "single push", not "unsolved".
    model_chain = {}
    for scene, cards in scene_cards.items():
        rec = rooms.get(scene)
        for c in cards:
            model_chain[c["file"]] = card_model_bucket(rec, c["object_id"])

    return rooms, aggregate, room_aggregate, pooled, by_horizon, arms, model_chain


def print_table(run_name, rooms, arms, pooled, by_horizon, room_aggregate):
    n_model_arms = sum(1 for grp, _ in arms.values() if grp == "model")
    n_random_arms = sum(1 for grp, _ in arms.values() if grp == "random")
    print(f"\n{run_name}  ({n_model_arms} model arm(s), {n_random_arms} random arm(s), "
          f"{len(rooms)} rooms joined to the gallery)")
    overall = next(s for s in room_aggregate if s["scope"] == "overall")
    print(f"  room-accurate solve@1 (this run's room_aggregate): "
          f"model {overall['model']['solve_at']['1']} random {overall['random']['solve_at']['1']} "
          f"(n={overall['n']} rooms)")
    print(f"  {'':<8} {'model solved':>14} {'model sims':>11} {'random solved':>15} "
          f"{'random sims':>12} {'speedup':>9}")
    for label in ["overall"] + sorted(by_horizon):
        if label == "overall":
            m, rd = summarize(pooled["model"]), summarize(pooled["random"])
        else:
            m, rd = summarize(by_horizon[label]["model"]), summarize(by_horizon[label]["random"])
        up = round(rd["median_sims"] / m["median_sims"], 2) if (m["median_sims"] and rd["median_sims"]) else None
        m_solved = round(m["solve_at"]["900"] * m["n_cards"]) if m["n_cards"] else 0
        r_solved = round(rd["solve_at"]["900"] * rd["n_cards"]) if rd["n_cards"] else 0
        print(f"  {label:<8} {m_solved:>7}/{m['n_cards']:<6} {str(m['median_sims']):>11} "
              f"{r_solved:>8}/{rd['n_cards']:<6} {str(rd['median_sims']):>12} {str(up) + 'x' if up else '-':>9}")


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

    all_cards, scene_cards, door = load_scene_index(a.out)
    chain = load_card_chains(all_cards, a.out)
    CARD_CATEGORIES = ["single push", "card object only", "other object only", "both objects"]
    label_counts = {c: sum(1 for v in chain.values() if v == c) for c in CARD_CATEGORIES}
    n_no_replay = sum(1 for v in chain.values() if v is None)
    print("label chains: " + ", ".join(f"{label_counts[c]} {c}" for c in CARD_CATEGORIES) +
          f", {n_no_replay} 2push cards with no replay")

    eval_rooms = defaultdict(dict)
    aggregate = {}
    room_aggregate = {}
    model_chain = {}
    two_push_cards = [row["file"] for row in all_cards if row["horizon"] == "2push"]
    for run_name, run_dir in runs.items():
        print(f"reading {run_name} <- {run_dir}")
        rooms, agg, room_agg, pooled, by_horizon, arms, run_chain = build_run(
            run_name, run_dir, scene_cards, door)
        for scene, rec in rooms.items():
            eval_rooms[scene][run_name] = rec
        aggregate[run_name] = agg
        room_aggregate[run_name] = room_agg
        model_chain[run_name] = run_chain
        print_table(run_name, rooms, arms, pooled, by_horizon, room_agg)
        by_cat = defaultdict(int)
        for f in two_push_cards:
            by_cat[run_chain.get(f, "unsolved")] += 1
        print("  model chain (2push cards): " +
              ", ".join(f"{by_cat[c]} {c}" for c in CARD_CATEGORIES + ["seeds disagree", "unsolved"]))

    out_path = os.path.join(a.out, "eval.json")
    json.dump({"schema_version": 3, "runs": list(runs.keys()),
               "rooms": eval_rooms, "aggregate": aggregate, "room_aggregate": room_aggregate,
               "chain": chain, "model_chain": model_chain}, open(out_path, "w"))
    print(f"\neval.json: {len(eval_rooms)} rooms across {len(runs)} run(s) -> {out_path}")


if __name__ == "__main__":
    main()
