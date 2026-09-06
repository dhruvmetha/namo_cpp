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
import glob
import json
import os
import sys
import time
from collections import Counter, defaultdict
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


def episodes_exhaustive(sweep_dirs, remap=None, horizon_tier=False, contact_only=False):
    """(xml -> [episode dict]) straight from `exhaustive_hmax2.py` output, for the two-movable pools.

    The canonical sources above read the eval-set manifests, which only cover the single-movable
    pools. These scenes never went through that pipeline, so their labels come from the sweep files
    themselves. Same episode shape, so `build` and the gallery need no special case.

    Unit is (xml, pushed object), which for a two-movable doorway means TWO episodes per scene, one
    per block. Greens keep the meaning they have everywhere else: an opener at 1push, a working
    setup at 2push, each over that object's own reachable pushes.

    Also carries the thing these pools exist to show. `n_green_contact` counts greens whose push
    actually touched another movable, and `contact_pct` is that as a share of greens, so the gallery
    can tell a scene where the blocks interact from one where they merely sit near each other. A
    green counts as contact if the push itself collided, or, for a setup, if the finish that worked
    did. The finish half is only present in sweeps run after commit ebc7f63; older files record the
    setup push alone and will read low here rather than wrong.

    Two opt-in flags exist so the multi-movable tab shows the same set the grid counter reports,
    and both default off so the two shipped galleries keep the numbers they were built with.
    `horizon_tier` scores the 2push tier on openers PLUS setups, since a push that opens outright
    also solves within a two-push horizon, which is what `count_qualifying_grid.py` counts and what
    hmax=2 means at eval. Leaving it off scores 2push on setups alone. `contact_only` drops
    episodes whose greens never touch another movable, which is the whole reason these pools exist.
    """
    # A sweep stores the paths of the box it RAN on, so a run labelled on Amarel points at
    # /scratch/... which does not exist here. Same prefix rewrite exh_to_key.py takes, same spelling.
    rules = [tuple(x.split("=", 1)) for x in (remap or [])]
    out = {}
    missing = 0
    n_already_open = 0
    for d in sweep_dirs:
        for p in sorted(glob.glob(os.path.join(d, "*.json"))):
            r = json.load(open(p))
            cells = r.get("cells") or []
            if not cells:
                continue
            # A scene whose goal region is already reachable at the root has no region-opening
            # problem in it. The sweep still enumerates pushes and calls the ones with a finish
            # "setups", so 18 such scenes reached the gallery as 2push cards -- and every plan on
            # them is rejected at replay time for opening on push 1, which is why all 18 were the
            # cards showing "no solution replay built". v1/solo0/rb_00071 is the one Dhruv spotted.
            if r.get("goal_open_at_start"):
                n_already_open += 1
                continue
            xml = r["xml"]
            for a, b in rules:
                if xml.startswith(a):
                    xml = b + xml[len(a):]
            if not os.path.exists(xml):
                missing += 1
                continue
            per_obj = defaultdict(list)
            for c in cells:
                per_obj[c.get("object_id", r.get("object_id"))].append(c)
            eps = []
            for obj, cs in per_obj.items():
                tried = [[c["edge"], c["depth"]] for c in cs]
                n_open = sum(1 for c in cs if c["kind"] == "opener")
                for horizon, kind in (("1push", "opener"), ("2push", "setup")):
                    hits = [c for c in cs if c["kind"] == kind]
                    if not hits:
                        continue
                    touched = sum(1 for c in hits
                                  if c.get("movable_collisions")
                                  or c.get("finish_movable_collisions"))
                    if contact_only and not touched:
                        continue
                    scoring = len(hits) + (n_open if horizon_tier and kind == "setup" else 0)
                    density = 100.0 * scoring / len(tried) if tried else 0.0
                    eps.append({
                        "horizon": horizon, "object_id": obj, "region": "goal",
                        "green": [[c["edge"], c["depth"]] for c in hits], "tried": tried,
                        "density_pct": round(density, 3), "tier": tier_of(density),
                        "n_green": len(hits), "n_tried": len(tried),
                        "n_green_contact": touched,
                        "contact_pct": round(100.0 * touched / len(hits), 1),
                        "solve_rate": round(len(hits) / len(tried), 4) if tried else 0.0,
                        # The gallery prints this on any card whose horizon is "2push", so a card
                        # without it threw inside render() and killed the whole draw before the
                        # step pills were built. That is why this tab showed no "after push 2".
                        "solve_rate_1push": round(n_open / len(tried), 4) if tried else 0.0,
                    })
            if eps:
                out.setdefault(xml, []).extend(eps)
    if missing:
        print(f"  {missing} sweep file(s) skipped: xml not found on this box (check --remap)")
    if n_already_open:
        print(f"  {n_already_open} scene(s) skipped: goal region already open at the root")
    print(f"  exhaustive source: {len(out)} rooms, "
          f"{sum(len(v) for v in out.values())} episodes")
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


def build(out_dir, shard, nshards, sweep_dirs=None, remap=None,
          horizon_tier=False, contact_only=False):
    from add_contact_px import contact_offsets_world
    from namo.visualization.wavefront_snapshot import WavefrontSnapshotExporter
    from namo.core.xml_goal_parser import extract_goal_with_fallback
    from scorer_beam import CFG, FALLBACK_GOAL, make_env

    # `sweep_dirs` swaps the label source wholesale rather than adding to it. The two-movable pools
    # are not in the eval-set manifests at all, so mixing the two would just mean reading manifests
    # that cannot match a single one of these rooms.
    sources = ((episodes_exhaustive(sweep_dirs, remap, horizon_tier, contact_only),) if sweep_dirs
               else (episodes_1push(), episodes_2push()))
    by_xml = {}
    for src in sources:
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


def scene_name(meta):
    """The name shown on the card and used to tell one scene from another in the gallery.

    Three sources, in order, because the datasets disagree about where the name lives:

    A curated shortlist id (`easy_000`, `hard_006`) wins when the card has one. Those are the
    numbers the hardware build sheets use, and `meta["key"]` is where they survive. The index used
    to take the xml basename instead, which silently renamed all 600 shipped cards to "env" on any
    rebuild, since every real-table room is a directory holding a file called `env.xml`. Generated
    keys are `<base>__<object>__<hash>` and are told apart by the `__`.

    Failing that the xml basename, and failing THAT, for any room whose file is `env.xml`, the
    directory holding it plus the pool above. Room ids restart per pool, so `rb_00091` on its own
    appears in several.
    """
    key = meta.get("key") or ""
    if key and "__" not in key:
        return key
    parts = str(meta["xml"]).split("/")
    base = parts[-1].replace(".xml", "")
    if base != "env" or len(parts) < 4:
        return base
    return "/".join(parts[-4:-1])


def scene_family(xml):
    """The generator batch a room came from -- the path segment after `test/` (feb_car, aug9_car).

    Two batches of rooms were generated months apart and they are NOT interchangeable: the gallery
    filters on this so a scene's look can be traced to the batch that produced it.
    """
    parts = str(xml).split("/")
    if "test" in parts[:-1]:
        return parts[parts.index("test") + 1]
    # Two-movable pools are laid out <pool>/<variant>/rb_NNNNN/env.xml, where the variant names the
    # generator settings (hard_zig, dense_solo1, ...). That IS the batch distinction for them, so
    # the gallery filter keeps working without a second concept.
    if len(parts) >= 3 and parts[-2].startswith("rb_"):
        return parts[-3]
    return "unknown"


def patch_from_relabel(out_dir, relabel_dir, margin):
    """Rewrite every card's LABEL half from a fresh exhaustive_hmax2 run, keeping its geometry.

    Used when the labels move under a gallery that already exists: the inflation margin changed
    from 5 mm to 1 mm on 2026-09-05 and every tier, green list and solve rate in this gallery was a
    5 mm number. Geometry, regions and contact points do not depend on the margin, so re-running the
    expensive capture would be waste; only `green`, `tried` and the meta counts move.

    A room that has no problem left at the new margin keeps its card and gets `status`, so it stays
    visible and filterable instead of vanishing from the index. Two ways to have no problem:
    `goal_open` (the robot already reaches the goal region) and `no_goal_region` (the goal stopped
    being a separate region at all, so the relabel wrote no record). Their green lists are emptied
    on purpose: a sweep of an already-open room files every push with a follow-up as a setup, so
    leaving them would show a dead room as a rich 2-push one.

    ⛔ Step-through replays are NOT rebuilt here. They were recorded against 5 mm solutions and a
    card whose green list just changed may animate a push that is no longer green.
    """
    recs = {}
    for f in glob.glob(os.path.join(relabel_dir, "*.json")):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        recs["/".join(r["xml"].split("/")[-4:])] = r
    print(f"  relabel source: {len(recs)} rooms")

    n = Counter()
    for path in sorted(glob.glob(os.path.join(out_dir, "cards", "*.json"))):
        card = json.load(open(path))
        m = card["meta"]
        rec = recs.get("/".join(m["xml"].split("/")[-4:]))
        m["inflation_margin_m"] = margin
        m["replay_margin_m"] = 0.005          # what the step-through was recorded at
        if rec is None or rec.get("goal_open_at_start"):
            m["status"] = "no_goal_region" if rec is None else "goal_open"
            card["green"], m["n_green"], m["n_green_contact"] = [], 0, 0
            m["density_pct"], m["tier"], m["solve_rate"], m["contact_pct"] = 0.0, "dead", 0.0, 0.0
            n[m["status"]] += 1
        else:
            cells = [c for c in rec["cells"] if c["object_id"] == m["object_id"]]
            if not cells:
                m["status"] = "object_not_swept"
                card["green"], m["n_green"] = [], 0
                n["object_not_swept"] += 1
            else:
                kind = "opener" if m["horizon"] == "1push" else "setup"
                hits = [c for c in cells if c["kind"] == kind]
                n_open = sum(1 for c in cells if c["kind"] == "opener")
                scoring = len(hits) + (n_open if kind == "setup" else 0)
                pct = 100.0 * scoring / len(cells)
                card["green"] = [[c["edge"], c["depth"]] for c in hits]
                card["tried"] = [[c["edge"], c["depth"]] for c in cells]
                touched = sum(1 for c in hits
                              if c.get("movable_collisions") or c.get("finish_movable_collisions"))
                m.update({"status": "live" if hits else "no_solution",
                          "n_green": len(hits), "n_tried": len(cells),
                          "n_green_contact": touched,
                          "contact_pct": round(100.0 * touched / len(hits), 1) if hits else 0.0,
                          "density_pct": round(pct, 3), "tier": tier_of(pct) if hits else "dead",
                          "solve_rate": round(len(hits) / len(cells), 4),
                          "solve_rate_1push": round(n_open / len(cells), 4)})
                n[m["status"]] += 1
        json.dump(card, open(path, "w"), separators=(",", ":"))
    print(f"  patched at margin {margin} m: {dict(n)}")


def build_index(out_dir, require_replay=False):
    """scenes.json = the small file the gallery page loads up front; cards are fetched lazily.

    `require_replay` drops any card with no step-through built. Those are the cold-replay
    casualties: the sweep found the solution from a state its own earlier pushes had produced, and
    a clean start does not reproduce it, so no plan we try replays. The card is not wrong, it just
    has nothing to animate, and it renders as "no solution replay built for this episode". Dhruv
    hit those repeatedly while browsing and asked for them gone, so the gallery lists only cards
    that can show their own solution. Run the replay pass BEFORE the index when using this.
    """
    cards_dir = os.path.join(out_dir, "cards")
    replay_dir = os.path.join(out_dir, "replay")
    rows = []
    n_no_replay = 0
    for fn in sorted(os.listdir(cards_dir)):
        if not fn.endswith(".json"):
            continue
        if require_replay and not os.path.exists(os.path.join(replay_dir, fn)):
            n_no_replay += 1
            continue
        meta = json.load(open(os.path.join(cards_dir, fn)))["meta"]
        # The contact pair rides along when the card has it. It is the reason the two-movable
        # pools exist, so the gallery needs it to filter on, and the check at the bottom of this
        # function reads it. Copying only the canonical keys left that check testing a key that was
        # never in `rows`, so it silently never fired and every two-movable build printed the
        # eval-set warning it was written to suppress.
        rows.append({"file": fn, "scene": scene_name(meta),
                     "family": scene_family(meta["xml"]),
                     **{k: meta[k] for k in ("horizon", "object_id", "tier", "density_pct",
                                             "n_green", "n_tried", "region")},
                     **{k: meta[k] for k in ("n_green_contact", "contact_pct") if k in meta},
                     **{k: meta[k] for k in ("status", "inflation_margin_m") if k in meta}})
    # Hardest first inside a tier: that is the order you want to arrow through when hunting figures.
    # "dead" is a card whose room stopped being a region-opening problem at the current margin; it
    # sorts last so the browsable set stays at the front, and it keeps its row so the gallery can
    # show and filter it rather than silently dropping 204 rooms out of the index.
    rows.sort(key=lambda r: (r["horizon"],
                             {"hard": 0, "medium": 1, "easy": 2, "dead": 3}.get(r["tier"], 3),
                             r["density_pct"], r["scene"]))
    counts = {h: dict(Counter(r["tier"] for r in rows if r["horizon"] == h))
              for h in ("1push", "2push")}
    print("  families:", dict(Counter(r["family"] for r in rows)))
    json.dump({"schema_version": SCHEMA_VERSION, "counts": counts, "cards": rows},
              open(os.path.join(out_dir, "scenes.json"), "w"))
    print(f"scenes.json: {len(rows)} cards  {counts}")
    if n_no_replay:
        print(f"  {n_no_replay} card(s) left out: no step-through built (cold-replay casualties)")
    # The expected-count check only means something for the canonical pools; a two-movable gallery
    # has no eval-set entry to be compared against, and warning on every build would train the eye
    # to skip the line that matters.
    if any(r.get("n_green_contact") is not None for r in rows):
        touch = [r for r in rows if r.get("n_green_contact")]
        print(f"  greens with movable contact: {len(touch)}/{len(rows)} cards")
        return
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
    ap.add_argument("--patch-from-relabel", help="rewrite card labels from an exhaustive_hmax2 "
                                                 "output dir, keeping geometry; then --index-only")
    ap.add_argument("--margin", type=float, help="inflation margin the relabel ran at, stamped on "
                                                 "every card so a stale number is detectable")
    ap.add_argument("--require-replay", action="store_true",
                    help="leave out cards with no step-through built; they render as an empty "
                         "solution panel and are the cold-replay casualties")
    ap.add_argument("--from-exhaustive", nargs="+", metavar="DIR",
                    help="read labels from exhaustive_hmax2.py output dirs instead of the eval-set "
                         "manifests. This is how the two-movable pools get a gallery: they were "
                         "never in those manifests.")
    ap.add_argument("--remap", nargs="*", metavar="FROM=TO", default=[],
                    help="rewrite an xml path prefix recorded by the sweep, for pools labelled on "
                         "another box")
    ap.add_argument("--horizon-tier", action="store_true",
                    help="score the 2push tier on openers plus setups, matching "
                         "count_qualifying_grid.py and hmax=2, instead of setups alone")
    ap.add_argument("--contact-only", action="store_true",
                    help="keep only episodes whose greens actually touch another movable")
    a = ap.parse_args()
    if a.patch_from_relabel:
        assert a.margin is not None, "--margin is required: a card without it cannot be interpreted"
        patch_from_relabel(a.out, a.patch_from_relabel, a.margin)
        build_index(a.out, a.require_replay)
        return
    if a.index_only:
        build_index(a.out, a.require_replay)
    else:
        build(a.out, a.shard, a.nshards, a.from_exhaustive, a.remap,
              a.horizon_tier, a.contact_only)


if __name__ == "__main__":
    main()
