#!/usr/bin/env python3
"""Export one root-state scene card per canonical episode, for the viz/search scene gallery.

A card is the START of an episode and nothing else: scene geometry, the wavefront region
decomposition, the target object's 60 contact points, and which of the reachable pushes are the
right ones. No search, no model, no simulation -- the greens come straight from the canonical label
JSONs, so a card says only what the test set already asserts.

Unit = ONE EPISODE = (xml, object_id) per horizon, never the room. One room contributes up to 4
1push episodes and 2 2push episodes, each with its own target object, its own greens, and its own
difficulty tier. Geometry and regions are captured once per room (they do not depend on which
object the episode is about) and copied into every card of that room, so each card stays a
self-contained lazy fetch.

Greens, per horizon:
  1push  green = an OPENER      -- `valid` in onepush_v3.json,          out of `tried`
  2push  green = a WORKING SETUP -- `valid_first_push` in pure2push_*.json, out of `tried_1push`
Tier = the same fixed cuts the project reports everywhere (hard <5%, medium 5-30%, easy >=30%) on
that green density. For 2push the tier is READ from the canonical divisions file rather than
recomputed.

    python scripts/viz/build_scene_cards.py --out $NAMO_SCRATCH/viz/scenes            # all shards
    python scripts/viz/build_scene_cards.py --out ... --shard 3 --nshards 16          # one shard
    python scripts/viz/build_scene_cards.py --out ... --index-only                    # scenes.json
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (f"{REPO}/build_python", f"{REPO}/python", f"{REPO}/scripts",
           f"{REPO}/scripts/sandbox", f"{REPO}/scripts/pipeline"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from namo import eval_sets, paths  # noqa: E402
from viz.trace_schema import episode_filename, rle_encode  # noqa: E402

SCHEMA_VERSION = 1


# Same cuts as scripts/eval_common.py:bin_of, spelled "medium" to match the divisions file's wording.
def tier_of(density_pct):
    return "hard" if density_pct < 5 else ("medium" if density_pct < 30 else "easy")


def _key(xml, object_id):
    return episode_filename(xml, object_id)[:-len(".json")]


def episodes_1push():
    """(xml -> [episode dict]) for the canonical 1push manifest."""
    man = json.load(open(eval_sets.path("onepush_manifest")))
    out = {}
    for xml, recs in man.items():
        eps = []
        for r in recs:
            tried, valid = r["tried"], r["valid"]
            density = 100.0 * len(valid) / len(tried) if tried else 0.0
            eps.append({"horizon": "1push", "object_id": r["object_id"], "region": r.get("region"),
                        "green": valid, "tried": tried, "density_pct": round(density, 3),
                        "tier": tier_of(density), "n_green": len(valid), "n_tried": len(tried),
                        "solve_rate": r["solve_rate"]})
        out[xml] = eps
    return out


def episodes_2push():
    """(xml -> [episode dict]) for the canonical 2push manifest, tiers from the divisions file."""
    man = json.load(open(eval_sets.path("pure2push_manifest")))
    div = json.load(open(eval_sets.path("pure2push_divisions")))
    dmap = {(os.path.realpath(x), e["object_id"]): e for x, es in div.items() for e in es}
    out = {}
    for xml, recs in man.items():
        eps = []
        for r in recs:
            tried, valid = r["tried_1push"], r["valid_first_push"]
            d = dmap.get((os.path.realpath(xml), r["object_id"]))
            density = d["setup_hardness_pct"] if d else (
                100.0 * len(valid) / len(tried) if tried else 0.0)
            eps.append({"horizon": "2push", "object_id": r["object_id"], "region": r.get("region"),
                        "green": valid, "tried": tried, "density_pct": round(density, 3),
                        "tier": d["division"] if d else tier_of(density),
                        "n_green": len(valid), "n_tried": len(tried),
                        "n_setups_gt": d["n_setups_gt"] if d else None,
                        "solve_rate_1push": r["solve_rate_1push"]})
        out[xml] = eps
    return out


def capture_room(xml, make_env, extract_goal, fallback_goal, exporter_cls, cfg):
    """(scene, regions) at the room's START state -- shared by every episode of that room."""
    env = make_env(xml)
    goal = extract_goal(xml, fallback_goal)
    env.set_robot_goal(*goal)
    env.get_reachable_objects()          # warms the wavefront the snapshot reads
    info, obs = env.get_object_info(), env.get_observation()
    static = [{"name": k, "x": v["pos_x"], "y": v["pos_y"], "hw": v["size_x"], "hd": v["size_y"],
               "qw": v["quat_w"], "qz": v["quat_z"]}
              for k, v in info.items() if "pos_x" in v]
    movable = [{"name": k, "x": obs[f"{k}_pose"][0], "y": obs[f"{k}_pose"][1],
                "theta": obs[f"{k}_pose"][2], "hw": v["size_x"], "hd": v["size_y"]}
               for k, v in info.items() if k != "robot" and f"{k}_pose" in obs and "pos_x" not in v]
    scene = {"bounds": list(env.get_world_bounds()), "static": static, "movable": movable,
             "robot": list(obs["robot_pose"]), "goal": list(goal)}
    snap = exporter_cls(env).build_snapshot(xml_path=str(paths.resolve(xml)), config_path=cfg,
                                            use_current_state=True)
    rm = snap.region_map
    regions = {"nx": int(rm.shape[0]), "ny": int(rm.shape[1]), "res": float(snap.resolution),
               "origin": [float(snap.bounds[0]), float(snap.bounds[2])],
               "labels": {str(int(k)): v for k, v in snap.region_labels.items()},
               "rle": rle_encode(rm.tolist())}
    return scene, regions


def build(out_dir, shard, nshards):
    from add_contact_px import contact_offsets_world
    from namo.visualization.wavefront_snapshot import WavefrontSnapshotExporter
    from namo.core.xml_goal_parser import extract_goal_with_fallback
    from scorer_beam import CFG, FALLBACK_GOAL, make_env

    by_xml = {}
    for src in (episodes_1push(), episodes_2push()):
        for xml, eps in src.items():
            by_xml.setdefault(xml, []).extend(eps)

    xmls = sorted(by_xml)
    mine = [x for i, x in enumerate(xmls) if i % nshards == shard]
    cards_dir = os.path.join(out_dir, "cards")
    os.makedirs(cards_dir, exist_ok=True)
    t0, n = time.time(), 0
    for i, xml in enumerate(mine):
        scene, regions = capture_room(xml, make_env, extract_goal_with_fallback, FALLBACK_GOAL,
                                      WavefrontSnapshotExporter, CFG)
        byname = {m["name"]: m for m in scene["movable"]}
        for ep in by_xml[xml]:
            m = byname[ep["object_id"]]
            off = contact_offsets_world(m["hw"], m["hd"], float(m["theta"]))
            contacts = [[round(float(m["x"] + dx), 6), round(float(m["y"] + dy), 6)]
                        for dx, dy in off]
            meta = {k: v for k, v in ep.items() if k not in ("green", "tried")}
            meta.update({"xml": xml, "key": _key(xml, ep["object_id"])})
            card = {"schema_version": SCHEMA_VERSION, "meta": meta, "scene": scene,
                    "regions": regions, "contacts": contacts,
                    "green": ep["green"], "tried": ep["tried"]}
            path = os.path.join(cards_dir, f"{ep['horizon']}__{meta['key']}.json")
            json.dump(card, open(path, "w"))
            n += 1
        if (i + 1) % 25 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"shard {shard}: {i+1}/{len(mine)} rooms, {n} cards, "
                  f"{rate:.1f} rooms/s, eta {(len(mine)-i-1)/rate/60:.1f} min", flush=True)
    print(f"shard {shard}: DONE {len(mine)} rooms, {n} cards in {(time.time()-t0)/60:.1f} min")


def build_index(out_dir):
    """scenes.json = the small file the gallery page loads up front; cards are fetched lazily."""
    cards_dir = os.path.join(out_dir, "cards")
    rows = []
    for fn in sorted(os.listdir(cards_dir)):
        if not fn.endswith(".json"):
            continue
        meta = json.load(open(os.path.join(cards_dir, fn)))["meta"]
        rows.append({"file": fn, "scene": os.path.basename(meta["xml"]).replace(".xml", ""),
                     **{k: meta[k] for k in ("horizon", "object_id", "tier", "density_pct",
                                             "n_green", "n_tried", "region")}})
    # Hardest first inside a tier: that is the order you want to arrow through when hunting figures.
    rows.sort(key=lambda r: (r["horizon"], {"hard": 0, "medium": 1, "easy": 2}[r["tier"]],
                             r["density_pct"], r["scene"]))
    counts = {h: dict(Counter(r["tier"] for r in rows if r["horizon"] == h))
              for h in ("1push", "2push")}
    json.dump({"schema_version": SCHEMA_VERSION, "counts": counts, "cards": rows},
              open(os.path.join(out_dir, "scenes.json"), "w"))
    print(f"scenes.json: {len(rows)} cards  {counts}")
    for h, exp in (("1push", eval_sets.EXPECTED.get("onepush_divisions")),
                   ("2push", eval_sets.EXPECTED.get("divisions"))):
        if exp and counts[h] != exp:
            print(f"  WARNING {h} tier counts differ from eval_sets expected {exp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="gallery data root (gets cards/ and scenes.json)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--index-only", action="store_true", help="rebuild scenes.json from cards/")
    a = ap.parse_args()
    if a.index_only:
        build_index(a.out)
    else:
        build(a.out, a.shard, a.nshards)


if __name__ == "__main__":
    main()
