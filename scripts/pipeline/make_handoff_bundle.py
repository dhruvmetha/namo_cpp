#!/usr/bin/env python3
"""Bundle a scene pool for the robot machine: env.xml plus everything needed to read its labels.

    python scripts/pipeline/make_handoff_bundle.py --pool v3_b2 --out $NAMO_SCRATCH/handoff/v3_b2_all \
        [--shard i --nshards n]

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
import glob
import json
import os
import re
import shutil

GEOM = re.compile(r'<geom name="(\w+)"[^>]*pos="([-\d.eE]+) ([-\d.eE]+) ([-\d.eE]+)"'
                  r'[^>]*euler="0 0 ([-\d.eE]+)"[^>]*size="([-\d.eE]+) ([-\d.eE]+) ([-\d.eE]+)"')
SITE = re.compile(r'<site name="goal"[^>]*pos="([-\d.eE]+) ([-\d.eE]+)')


def derived_sheet(xml_path, env):
    text = open(xml_path).read()
    movable, static = [], []
    for name, x, y, _z, yaw, hx, hy, hz in GEOM.findall(text):
        entry = {"name": name, "x_m": float(x), "y_m": float(y), "yaw_deg": float(yaw),
                 "half_extent_m": [float(hx), float(hy)], "height_m": float(hz) * 2}
        (movable if "movable" in name else static).append(entry)
    goal = SITE.search(text)
    robot = env.get_observation().get("robot_pose")
    return {
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="pool directory name, e.g. v3_b2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--config", default="config/namo_config_complete_skill15_car_1x.yaml")
    a = ap.parse_args()

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
    cards = {}
    if os.path.exists(f"{gal}/scenes.json"):
        for row in json.load(open(f"{gal}/scenes.json"))["cards"]:
            meta = json.load(open(f"{gal}/cards/{row['file']}"))["meta"]
            cards.setdefault(norm(meta["xml"]), []).append((row["file"], meta["object_id"]))

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
            json.dump(derived_sheet(xml, env), open(f"{dest}/build_sheet_derived.json", "w"), indent=1)
        except Exception as exc:
            json.dump({"schema": "derived_from_env_xml", "error": str(exc)},
                      open(f"{dest}/build_sheet_derived.json", "w"), indent=1)
        k = norm(xml)
        if k in sweeps:
            shutil.copy2(sweeps[k], os.path.join(dest, "sweep_record.json"))
        for fn, obj in cards.get(k, []):
            shutil.copy2(f"{gal}/cards/{fn}", f"{dest}/gallery_card__{obj}.json")
            rp = f"{gal}/replay/{fn}"
            if os.path.exists(rp):
                shutil.copy2(rp, f"{dest}/gallery_replay__{obj}.json")
        made += 1
    print(f"shard {a.shard}: DONE {made} scenes", flush=True)


if __name__ == "__main__":
    main()
