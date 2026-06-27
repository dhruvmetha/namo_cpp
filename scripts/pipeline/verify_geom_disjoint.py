#!/usr/bin/env python3
"""GEOMETRY-based room-disjointness check (robust to path/naming schemes).

Two scenes are the SAME ROOM iff they share wall layout + movable-obstacle initial poses/sizes.
This does NOT rely on file names (train uses outputs/v3_phase1/run_NNNN..., test uses
car_envs/v3/test/.../run_NNNN — incompatible schemes, so a name-based 'overlap' is meaningless).

Signature per xml = hash( sorted(wall pos+size)  +  sorted(movable-obstacle body-pos + geom-size) ).
Robot (car) start and goal are EXCLUDED — the room is its walls + obstacle layout.

Usage:
  verify_geom_disjoint.py --train-xmls <file|h5> --test-xmls <file|json> --out report.json
"""
import sys, os, json, argparse, hashlib, re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
for _p in (f"{REPO}/build_python", f"{REPO}/python"):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)
from namo.paths import DATASETS, resolve  # noqa: E402

# Fast regex extraction (full XML parse was ~10x slower on 98k MuJoCo files).
# Two naming schemes coexist: aug9/train walls = wall_1..wall_N (no euler attr);
# feb/test "real_template" walls = wall_boundary_* / wall_inner_* (explicit euler). Both start "wall_".
# Movable obstacles: the POSE lives on the INNER <geom name="obstacle_N_movable" .../>, NOT the <body> tag.
_WALLGEOM = re.compile(r'<geom\s+name="wall_[^"]*"[^>]*?/?>')
_OBSGEOM = re.compile(r'<geom\s+name="[^"]*movable[^"]*"[^>]*?/?>')
_POS = re.compile(r'\bpos="([^"]+)"')
_SIZE = re.compile(r'\bsize="([^"]+)"')
_EUL = re.compile(r'\beuler="([^"]+)"')


def _vec(s):
    return tuple(round(float(v), 3) for v in s.split())


def _geoms(txt, pat):
    """Extract sorted (pos, size, euler) tuples for every geom matching pat. euler implicit -> (0,0,0)."""
    out = []
    for m in pat.finditer(txt):
        g = m.group(0)
        p, s, e = _POS.search(g), _SIZE.search(g), _EUL.search(g)
        if p and s:
            out.append((_vec(p.group(1)), _vec(s.group(1)), _vec(e.group(1)) if e else (0., 0., 0.)))
    out.sort()
    return tuple(out)


def geom_sig(xml_path):
    """Full ROOM signature: (walls, obstacles) each = sorted (pos,size,euler). Goal site + robot EXCLUDED.
    Returns (full_sig, walls_sig) or (None, None) if empty/unreadable.
    - full_sig  = walls + obstacle layout  -> EXACT-SCENE identity (episodes of one room differ only in goal).
    - walls_sig = walls only               -> template overlap (same floorplan, maybe different obstacles)."""
    try:
        txt = open(xml_path).read()
    except Exception:
        return None, None
    walls = _geoms(txt, _WALLGEOM)
    obs = _geoms(txt, _OBSGEOM)
    if not walls and not obs:
        return None, None
    full = hashlib.md5(repr((walls, obs)).encode()).hexdigest()
    wonly = hashlib.md5(repr(walls).encode()).hexdigest()
    return full, wonly


def _iter_xml_keys(obj):
    """Recursively yield dict keys that look like xml paths (handles flat {xml:..} and nested {bucket:{xml:..}})."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.endswith(".xml"):
                yield k
            else:
                yield from _iter_xml_keys(v)


def load_xmls(spec):
    """spec = a .txt (one path/line), a .json (xml-keyed, flat or nested by bucket), or an .h5 (xml dataset)."""
    if spec.endswith(".h5"):
        import h5py
        with h5py.File(spec, "r") as f:
            return [str(resolve(x.decode() if isinstance(x, bytes) else str(x))) for x in f["xml"][:]]
    if spec.endswith(".json"):
        return [str(resolve(k)) for k in _iter_xml_keys(json.load(open(spec)))]
    return [str(resolve(l.strip())) for l in open(spec) if l.strip() and not l.startswith("#")]


def sig_map(xmls, workers=32):
    """Dedup exact paths, hash each (threaded — I/O bound). Returns (n_parse, full_sig->[xmls], walls_sig->set)."""
    uniq = list(dict.fromkeys(xmls))        # dedup exact paths
    n_parse = 0
    full2x = {}
    walls2full = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for x, (full, wonly) in zip(uniq, ex.map(geom_sig, uniq, chunksize=64)):
            if full is None:
                continue
            n_parse += 1
            full2x.setdefault(full, []).append(x)
            walls2full.setdefault(wonly, set()).add(full)
    return n_parse, full2x, walls2full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-xmls", required=True)
    ap.add_argument("--test-xmls", required=True)
    ap.add_argument("--out", default=str(DATASETS / "policy_value_v1/stats/geom_disjoint.json"))
    a = ap.parse_args()

    print("hashing TRAIN ...", flush=True)
    tr_xmls = load_xmls(a.train_xmls)
    tr_n, tr_full2x, tr_walls2full = sig_map(tr_xmls)
    tr_full, tr_walls = set(tr_full2x), set(tr_walls2full)
    print(f"  train: {len(tr_xmls)} xmls -> {tr_n} parseable -> "
          f"{len(tr_full)} unique scenes / {len(tr_walls)} unique floorplans", flush=True)

    print("hashing TEST ...", flush=True)
    te_xmls = load_xmls(a.test_xmls)
    te_n, te_full2x, te_walls2full = sig_map(te_xmls)
    te_full, te_walls = set(te_full2x), set(te_walls2full)
    print(f"  test: {len(te_xmls)} xmls -> {te_n} parseable -> "
          f"{len(te_full)} unique scenes / {len(te_walls)} unique floorplans", flush=True)

    # EXACT-SCENE leak = same walls AND same obstacle layout in both sets -> a genuine train/test leak.
    scene_collide = te_full & tr_full
    # Floorplan overlap = same walls (obstacles may differ) -> informative, NOT necessarily a leak.
    floor_collide = te_walls & tr_walls

    scene_examples = []
    for s in list(scene_collide)[:10]:
        scene_examples.append({"test_xml": te_full2x[s][0], "train_xml": tr_full2x[s][0]})

    out = {
        "n_train_xmls": len(tr_xmls), "n_train_unique_scenes": len(tr_full), "n_train_unique_floorplans": len(tr_walls),
        "n_test_xmls": len(te_xmls), "n_test_unique_scenes": len(te_full), "n_test_unique_floorplans": len(te_walls),
        "n_train_unique_paths": len(set(tr_xmls)), "n_test_unique_paths": len(set(te_xmls)),
        "n_unparseable_train": len(set(tr_xmls)) - tr_n, "n_unparseable_test": len(set(te_xmls)) - te_n,
        "n_test_scenes_leaking_into_train": len(scene_collide),
        "n_test_floorplans_shared_with_train": len(floor_collide),
        "frac_test_floorplans_shared": round(len(floor_collide) / max(1, len(te_walls)), 4),
        "clean": len(scene_collide) == 0,
        "scene_leak_examples": scene_examples,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print("\n=== RESULT ===")
    print(json.dumps({k: v for k, v in out.items() if k != "scene_leak_examples"}, indent=2))
    print("CLEAN (no exact-scene leak)" if out["clean"]
          else f"⚠ {len(scene_collide)} TEST SCENES LEAK INTO TRAIN")
    print(f"  (floorplan overlap {len(floor_collide)}/{len(te_walls)} test floorplans — "
          f"template reuse, not a leak unless obstacles also match)")
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
