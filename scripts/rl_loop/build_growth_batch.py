#!/usr/bin/env python3
"""Build one +N growth batch for a growth arm (EXP-2026-07-08-rl-growth-arms).

Generic over the two arms' diets — the ONLY difference is the source manifest:
  arm N (novelty)     : v4_hq_h1/episodes_deadends_all.json  (validset, 1push)
  arm C (composition) : v4_hq_h2/labels_s30_pure2push.json   (pure2push, genuine F=None 2push)
Both sources are {xml_key: [records]} dicts that namo.rl_loop.episodes.load_pool reads directly,
so a batch is a disjoint SUBSET of the source (never re-touching rooms this arm already used, or the
testset scenes).

namo-data-pipeline invariants honored:
  - unit = (pushed object, goal region); difficulty PER-EPISODE via bin_of(solve_rate*), never a
    file label; hold out by ROOM.
  - DISJOINTNESS GATE (HARD, canonical): drop any batch xml whose FULL-SCENE signature (sorted wall
    pos+size+euler + sorted movable-obstacle geom pos+size+euler; robot/goal EXCLUDED — the
    scripts/pipeline/verify_geom_disjoint.py signature) matches a namo_testset_v1 scene. This is the
    ONLY correct gate: walls-only over-drops legit template-sharers (same floorplan, different
    obstacle layout = a different scene) and name/path is meaningless (the same physical room appears
    under different names — the exact pathology testset_v1 was rebuilt to fix). Also assert no batch
    xml sits under a /test/ dir (belt-and-suspenders). Gate history: walls-only hash -> name/path ->
    THIS full-scene signature (canonical).
  - exclude every base room already consumed by this arm (--exclude gen0_pool + prior batches +
    frozen split) so batches extend only the TRAIN side and don't re-draw own rooms (redundancy, not
    a leak — base-room name is adequate for own-reuse).
  - hold out ~--dev-frac of the batch's USED rooms as this-batch's new-batch-dev slice.
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO / "python"), str(REPO / "scripts/rl_loop"), str(REPO / "scripts/pipeline")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from eval_common import bin_of                                # noqa: E402
from build_gen0_pool import base_room                        # noqa: E402
from verify_geom_disjoint import geom_sig                    # noqa: E402  full-scene signature
from namo.paths import resolve as _resolve                   # noqa: E402


def _diff_of(rec: dict) -> str:
    sr = rec.get("solve_rate", rec.get("solve_rate_first_push"))
    return bin_of(sr) if sr is not None else "med"


def _rooms_from_json(path: str) -> set:
    """Base rooms named by an exclusion json: a pool {xml_key:[recs]} OR a split {train,dev,test}."""
    d = json.load(open(path))
    keys = []
    if isinstance(d, dict) and {"train", "dev", "test"} <= set(d.keys()):
        for part in ("train", "dev", "test"):
            keys.extend(d[part])
    elif isinstance(d, dict):
        keys.extend(d.keys())
    return {base_room(k) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="source manifest json {xml_key:[recs]}")
    ap.add_argument("--testset-sigs", required=True,
                    help="json {sigs:[...]} of testset full-scene md5s (build once via verify_geom_disjoint)")
    ap.add_argument("--exclude", nargs="*", default=[], help="pool/split jsons whose rooms to exclude")
    ap.add_argument("--n-target", type=int, default=4000, help="episodes to sample into the batch")
    ap.add_argument("--per-room-per-diff", type=int, default=2)
    ap.add_argument("--dev-frac", type=float, default=0.10, help="fraction of USED rooms held out as new-batch-dev")
    ap.add_argument("--out-batch", required=True)
    ap.add_argument("--out-devrooms", required=True)
    ap.add_argument("--out-gate-report", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    src = json.load(open(a.source))
    ts_sigs = set(json.load(open(a.testset_sigs))["sigs"])
    print(f"[batch] source keys={len(src)}  testset scene-sigs={len(ts_sigs)}", flush=True)

    excl_rooms = set()
    for e in a.exclude:
        excl_rooms |= _rooms_from_json(e)
    print(f"[batch] excluded base rooms (already used / frozen): {len(excl_rooms)}", flush=True)

    # group source episodes by base room, tag difficulty; drop excluded rooms up front
    by_room = defaultdict(list)     # base_room -> [(xml_key, rec, diff)]
    drop_excl = 0
    for xml_key, recs in src.items():
        br = base_room(xml_key)
        if br in excl_rooms:
            drop_excl += 1
            continue
        for r in recs:
            by_room[br].append((xml_key, r, _diff_of(r)))
    rooms = list(by_room)
    rng.shuffle(rooms)

    # lazy full-scene gate (cache sig per xml_key)
    sig_cache = {}

    def leaks(xml_key: str) -> bool:
        if xml_key not in sig_cache:
            full, _ = geom_sig(str(_resolve(xml_key)))
            sig_cache[xml_key] = full
        return sig_cache[xml_key] is not None and sig_cache[xml_key] in ts_sigs

    batch = defaultdict(list)      # xml_key -> [recs]
    counts = defaultdict(int)
    used_rooms = []
    n_eps = 0
    scene_dropped = 0
    for br in rooms:
        if n_eps >= a.n_target:
            break
        buckets = defaultdict(list)
        for xml_key, r, diff in by_room[br]:
            buckets[diff].append((xml_key, r))
        took = 0
        for diff, its in buckets.items():
            rng.shuffle(its)
            kept = 0
            for xml_key, r in its:
                if kept >= a.per_room_per_diff:
                    break
                assert "/test/" not in xml_key, f"LEAK: batch xml under a /test/ dir: {xml_key}"
                if leaks(xml_key):
                    scene_dropped += 1
                    continue
                batch[xml_key].append(r)
                counts[diff] += 1
                n_eps += 1
                took += 1
                kept += 1
        if took:
            used_rooms.append(br)

    # hard post-assert: NO sampled xml leaks a testset scene
    assert not any(leaks(k) for k in batch), "post-gate scene leak — bug"

    dev_n = max(1, int(round(a.dev_frac * len(used_rooms)))) if used_rooms else 0
    dev_rooms = sorted(rng.sample(used_rooms, dev_n)) if dev_n else []

    with open(a.out_batch, "w") as f:
        json.dump(batch, f)
    with open(a.out_devrooms, "w") as f:
        json.dump({"dev_rooms": dev_rooms}, f)
    report = {
        "source": a.source, "testset_sigs": a.testset_sigs, "seed": a.seed,
        "gate": "full_scene_signature (verify_geom_disjoint)",
        "drops": {"excluded_rooms": drop_excl, "testset_scene_leak_dropped": scene_dropped},
        "batch": {"n_episodes": n_eps, "n_pair_xmls": len(batch),
                  "n_used_rooms": len(used_rooms), "n_dev_rooms": len(dev_rooms),
                  "difficulty": dict(counts), "per_room_per_diff": a.per_room_per_diff,
                  "n_target": a.n_target},
    }
    with open(a.out_gate_report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[gate] excluded_rooms={drop_excl} testset_scene_leak_dropped={scene_dropped}", flush=True)
    print(f"[batch] WROTE {a.out_batch}: {n_eps} eps / {len(batch)} pair-xmls / "
          f"{len(used_rooms)} rooms ({len(dev_rooms)} dev)  diff={dict(counts)}", flush=True)


if __name__ == "__main__":
    main()
