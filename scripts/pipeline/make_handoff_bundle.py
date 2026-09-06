#!/usr/bin/env python3
"""Bundle a scene pool for the robot machine: env.xml plus everything needed to read its labels.

    python scripts/pipeline/make_handoff_bundle.py --pool v3_b2 --out $NAMO_SCRATCH/handoff/v3_b2_all \
        [--shard i --nshards n]
    python scripts/pipeline/make_handoff_bundle.py --rooms rooms.txt --no-cards --out ...
    python scripts/pipeline/make_handoff_bundle.py --pool v3_b2 --patch-sheets <delivered tree>

`--patch-sheets` adds `doorway` and `episodes` to sheets ALREADY delivered, in place, touching no
other key. It exists because the robot box re-captures poses off the camera when a scene shifts on
the table, so rewriting a delivered sheet would throw away a measurement we cannot reproduce. A
room with no gallery card gets its doorway from a fresh region snapshot and an empty `episodes`
list with `no_episode_reason`, since not qualifying is a fact about the room worth shipping.

`--rooms` takes one pool-relative scene dir per line (`v2/dense_solo0/rb_00001`) instead of a whole
pool, so a bundle can carry exactly the gallery rooms and not the pool's non-qualifying rest.
`--no-cards` drops gallery_card/gallery_replay: the robot box reads env.xml and the derived sheet
and nothing else, and the cards are 3x the bytes of the scenes they describe.

Layout, one directory per scene:

    <pool>/<group>/<scene>/env.xml                  the scene, exactly as the pool ships it
    <pool>/<group>/<scene>/build_sheet_derived.json poses, for their check_build.py
    <pool>/<group>/<scene>/sweep_record.json        the raw label, every enumerated cell
    <pool>/<group>/<scene>/gallery_card.json        per (object, goal) episode, if it has one
    <pool>/<group>/<scene>/gallery_replay.json      the recorded solution, if one replays

⛔ `build_sheet_derived.json` is NOT generator output and says so in its own `schema` field. These
pools ship env.xml and nothing else, so the poses are read back out of the XML. The robot start is
NOT in the file, the harness places the robot, so it comes from a simulator reset. That is the one
field you cannot get without loading the scene, and it is why this needs namo_rl rather than being
a text transform.

A scene can have several gallery cards, one per (pushed object, goal region) episode. They are
written as gallery_card__<object>.json so nothing silently overwrites, which is the failure mode
that made the two-movable pools confusing in the first place.
"""
import argparse
import collections
import glob
import json
import os
import re
import shutil

GEOM = re.compile(r'<geom name="(\w+)"[^>]*pos="([-\d.eE]+) ([-\d.eE]+) ([-\d.eE]+)"'
                  r'[^>]*euler="0 0 ([-\d.eE]+)"[^>]*size="([-\d.eE]+) ([-\d.eE]+) ([-\d.eE]+)"')
SITE = re.compile(r'<site name="goal"[^>]*pos="([-\d.eE]+) ([-\d.eE]+)')


def doorway_of(snap):
    """The two flags, same definition as scripts/viz/add_group_edge_flag.py:69-70.

    A doorway in `multi_object_edges` is one no SINGLE object opens; `adjacency` minus those pairs
    is the single-object graph, so a goal unreachable in it has no route avoiding a two-block door.
    """
    from viz.add_group_edge_flag import reachable
    adj = {k: set(v) for k, v in dict(snap.get("adjacency", {})).items()}
    grp = {k: set(v) for k, v in dict(snap.get("multi_object_edges", {})).items()}
    rl, gl = snap.get("robot_label"), snap.get("goal_label")
    if not rl or not gl:
        return None
    solo = {k: (v - grp.get(k, set())) for k, v in adj.items()}
    return {"needs_both_blocks": gl in grp.get(rl, set()),
            "has_route_around": reachable(solo, rl, gl)}


def margin_now():
    """The tier-1 inflation margin the C++ will actually use, in metres.

    Written into every sheet this script touches. A sheet without it is a sheet whose numbers you
    cannot interpret: the margin moved from 5 mm to 1 mm on 2026-09-05 and took 204 rooms out of the
    two-movable pool with it, and nothing on either side of the handoff recorded which value a given
    label was made at.
    """
    import yaml
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for c in (os.path.join(repo, "config", "wavefront_inflation.yaml"),):
        if os.path.exists(c):
            return float(yaml.safe_load(open(c))["tier1"]["base_inflation_margin_m"])
    return None


def episodes_from_record(rec):
    """Per (object, horizon) episodes straight off one exhaustive_hmax2 record.

    Same tier rule as everywhere else: 1push scores openers over that object's enumerated pushes,
    2push scores openers plus setups, cuts at 5% and 30%.

    ⛔ Returns [] when the goal is already open at the root, and the caller must NOT fall back to
    the cells. A sweep of an already-open room still files every push that has a follow-up as a
    "setup", so such a record can show hundreds of setups and read as a rich 2-push room. It is a
    dead room. rb_00150 with its shifted goal is the worked example: 218 setups, 130 with contact,
    no problem in it at all.
    """
    if rec.get("goal_open_at_start"):
        return []
    cells = rec.get("cells") or []
    per = collections.defaultdict(list)
    for c in cells:
        per[c["object_id"]].append(c)
    out = []
    for obj, cs in sorted(per.items()):
        n_op = sum(1 for c in cs if c["kind"] == "opener")
        n_su = sum(1 for c in cs if c["kind"] == "setup")
        for horizon, green, scoring in (("1push", n_op, n_op), ("2push", n_su, n_op + n_su)):
            if not green:
                continue
            pct = 100.0 * scoring / len(cs)
            hits = [c for c in cs if c["kind"] == ("opener" if horizon == "1push" else "setup")]
            touched = sum(1 for c in hits
                          if c.get("movable_collisions") or c.get("finish_movable_collisions"))
            out.append({"object_id": obj, "horizon": horizon,
                        "tier": "hard" if pct < 5 else ("medium" if pct < 30 else "easy"),
                        "n_green": green, "n_tried": len(cs), "n_green_contact": touched})
    return out


def why_no_episode(sweep):
    """A room with no gallery card: say which of the reasons it was, from its own sweep record."""
    if sweep is None:
        return "no sweep record"
    rec = json.load(open(sweep))
    if rec.get("goal_open_at_start"):
        return "goal region already reachable at the root"
    cells = rec.get("cells") or []
    solving = [c for c in cells if c["kind"] in ("opener", "setup")]
    if not solving:
        return "no push opens the region within two pushes"
    if not any(c.get("movable_collisions") or c.get("finish_movable_collisions") for c in solving):
        return "solves, but no solving push touches the other movable"
    # Qualifies on the labels and still has no card: its card carried no replay, so the gallery
    # index dropped it. 42 cards across the pool are in this state.
    return "qualifies, but its recorded solution did not replay from a clean start"


def derived_sheet(xml_path, env, gallery=None):
    text = open(xml_path).read()
    movable, static = [], []
    for name, x, y, _z, yaw, hx, hy, hz in GEOM.findall(text):
        entry = {"name": name, "x_m": float(x), "y_m": float(y), "yaw_deg": float(yaw),
                 "half_extent_m": [float(hx), float(hy)], "height_m": float(hz) * 2}
        (movable if "movable" in name else static).append(entry)
    goal = SITE.search(text)
    robot = env.get_observation().get("robot_pose")
    sheet = {
        "schema": "derived_from_env_xml",
        "note": "NOT a generator build sheet. This pool ships env.xml and nothing else, so poses "
                "are read back out of that file. robot_start_m comes from the simulator after "
                "reset, because the scene XML carries no robot body. Metres, yaw in degrees, "
                "half-extents as authored.",
        "robot_start_m": [round(float(robot[0]), 6), round(float(robot[1]), 6)] if robot else None,
        "robot_start_yaw_rad": round(float(robot[2]), 6) if robot else None,
        "goal_marker_m": [float(goal.group(1)), float(goal.group(2))] if goal else None,
        "movables": movable,
        "statics": static,
    }
    # Room-level, straight off the gallery card meta (scripts/viz/add_group_edge_flag.py). The
    # operator needs this BEFORE building: a both-blocks door with no way around is two blocks that
    # must both move, and it is where search fails 13.8% of the time against 2.9% elsewhere.
    # `episodes` is a LIST because tier is per (pushed object, horizon), never per room -- a room
    # can be medium for one block and easy for the other, and a single room-level tier would be a
    # lie. Same reason the gallery keys cards by object.
    if gallery:
        sheet["doorway"] = gallery["doorway"]
        sheet["episodes"] = gallery["episodes"]
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", help="pool directory name, e.g. v3_b2; omit when --rooms is given")
    ap.add_argument("--rooms", help="file of pool-relative scene dirs, one per line")
    ap.add_argument("--no-cards", action="store_true", help="skip gallery_card/gallery_replay")
    ap.add_argument("--patch-sheets", help="add doorway/episodes to delivered sheets under this tree")
    ap.add_argument("--from-relabel", help="read episodes from an exhaustive_hmax2 output dir instead "
                                           "of the gallery cards; use after a relabel at a new margin")
    ap.add_argument("--out", help="bundle destination; not used by --patch-sheets")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--config", default="config/namo_config_complete_skill15_car_1x.yaml")
    a = ap.parse_args()
    assert a.out or a.patch_sheets, "--out is required unless --patch-sheets is given"

    import namo_rl
    root = os.path.join(os.environ["NAMO_SCRATCH"], "real_buildable_2mov")
    key = "real_buildable_2mov/"
    norm = lambda p: p.split(key, 1)[1] if key in p else p

    sweeps = {}
    for d in sorted(glob.glob(f"{root}/*_exh2_pull")) + sorted(glob.glob(f"{root}/*/exh2")):
        for f in glob.glob(f"{d}/*.json"):
            try:
                sweeps[norm(json.load(open(f))["xml"])] = f
            except Exception:
                pass

    gal = os.path.join(os.environ["NAMO_SCRATCH"], "viz", "real_2mov")
    cards, gallery = {}, {}
    if os.path.exists(f"{gal}/scenes.json"):
        for row in json.load(open(f"{gal}/scenes.json"))["cards"]:
            meta = json.load(open(f"{gal}/cards/{row['file']}"))["meta"]
            k = norm(meta["xml"])
            cards.setdefault(k, []).append((row["file"], meta["object_id"]))
            g = gallery.setdefault(k, {"doorway": {}, "episodes": []})
            g["doorway"] = {"needs_both_blocks": bool(meta.get("door_needs_both_blocks")),
                            "has_route_around": bool(meta.get("has_route_around"))}
            g["episodes"].append({"object_id": meta["object_id"], "horizon": meta["horizon"],
                                  "tier": meta["tier"], "n_green": meta["n_green"],
                                  "n_tried": meta["n_tried"],
                                  "n_green_contact": meta.get("n_green_contact")})

    relabel = {}
    if a.from_relabel:
        for f in glob.glob(os.path.join(a.from_relabel, "*.json")):
            try:
                r = json.load(open(f))
            except Exception:
                continue
            relabel[norm(r["xml"])] = r
        print(f"  relabel source: {len(relabel)} rooms", flush=True)

    if a.patch_sheets:
        patched = missing_door = 0
        margin = margin_now()
        status = collections.Counter()
        for xml in sorted(glob.glob(f"{root}/{a.pool}/*/*/env.xml")):
            rel = os.path.dirname(norm(xml))
            sheet = os.path.join(a.patch_sheets, rel, "build_sheet_derived.json")
            if not os.path.exists(sheet):
                continue
            k = norm(xml)
            if a.from_relabel:
                rec = relabel.get(k)
                if rec is None:
                    eps, why = [], "no goal region to sample at this margin"
                elif rec.get("goal_open_at_start"):
                    eps, why = [], "goal region already reachable at the root at this margin"
                else:
                    eps, why = episodes_from_record(rec), None
                status["dead" if why else "live"] += 1
                env = namo_rl.RLEnvironment(xml, a.config, False)
                env.get_reachable_objects()
                door = doorway_of(env.get_region_snapshot(100, -1.0, False, 42, True))
                d = json.load(open(sheet))
                d["doorway"], d["episodes"] = door, eps
                d["inflation_margin_m"] = margin
                d.pop("no_episode_reason", None)
                if why:
                    d["no_episode_reason"] = why
                missing_door += door is None
                json.dump(d, open(sheet, "w"), indent=1)
                patched += 1
                continue
            g = gallery.get(norm(xml))
            if g:
                door, eps, why = g["doorway"], g["episodes"], None
            else:
                env = namo_rl.RLEnvironment(xml, a.config, False)
                env.get_reachable_objects()
                door = doorway_of(env.get_region_snapshot(100, -1.0, False, 42, True))
                eps, why = [], why_no_episode(sweeps.get(norm(xml)))
            d = json.load(open(sheet))
            d["doorway"], d["episodes"] = door, eps
            if why:
                d["no_episode_reason"] = why
            missing_door += door is None
            json.dump(d, open(sheet, "w"), indent=1)
            patched += 1
        print(f"patched {patched} sheet(s) at margin {margin} m; "
              f"{missing_door} with no robot/goal label; {dict(status)}", flush=True)
        return

    if a.rooms:
        want = [l.strip() for l in open(a.rooms) if l.strip()]
        scenes = [f"{root}/{r}/env.xml" for r in want]
        gone = [s for s in scenes if not os.path.exists(s)]
        assert not gone, f"{len(gone)} listed scene(s) have no env.xml, first: {gone[0]}"
    else:
        scenes = sorted(glob.glob(f"{root}/{a.pool}/*/*/env.xml"))
    made = 0
    for n, xml in enumerate(scenes):
        if n % a.nshards != a.shard:
            continue
        rel = os.path.dirname(norm(xml))
        dest = os.path.join(a.out, rel)
        os.makedirs(dest, exist_ok=True)
        shutil.copy2(xml, os.path.join(dest, "env.xml"))
        try:
            env = namo_rl.RLEnvironment(xml, a.config, False)
            env.reset()
            json.dump(derived_sheet(xml, env, gallery.get(norm(xml))),
                      open(f"{dest}/build_sheet_derived.json", "w"), indent=1)
        except Exception as exc:
            json.dump({"schema": "derived_from_env_xml", "error": str(exc)},
                      open(f"{dest}/build_sheet_derived.json", "w"), indent=1)
        k = norm(xml)
        if k in sweeps:
            shutil.copy2(sweeps[k], os.path.join(dest, "sweep_record.json"))
        for fn, obj in ([] if a.no_cards else cards.get(k, [])):
            shutil.copy2(f"{gal}/cards/{fn}", f"{dest}/gallery_card__{obj}.json")
            rp = f"{gal}/replay/{fn}"
            if os.path.exists(rp):
                shutil.copy2(rp, f"{dest}/gallery_replay__{obj}.json")
        made += 1
    print(f"shard {a.shard}: DONE {made} scenes", flush=True)


if __name__ == "__main__":
    main()
