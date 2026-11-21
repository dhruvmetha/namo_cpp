#!/usr/bin/env python3
"""Copy a tree of v1 car scenes into a new tree with the v2 car (car_v2_block.py).

It never writes into the source tree: the destination must not lie inside the source or
contain it, and an existing destination file is kept unless --overwrite. Each scene is
converted only if its car <body> and <actuator> block are exactly v1's (CAR_BODY / FOOTER
in scripts/pipeline/gen_real_buildable_scenes.py, car position aside); anything else is
skipped and listed, never guessed at. Walls, blocks, floor, options and the car's position
are copied unchanged.

  python convert_v1_to_v2.py --src-root <v1 tree> --dst-root <new tree> [--limit N] [--check]
  python convert_v1_to_v2.py --selftest <one v1 scene>

--selftest rebuilds a scene with every switch at v1 and requires the compiled model to
equal the original array for array; that is the check that car_v2_block.py transcribes v1
exactly. A manifest (src, sha256 in and out, status) is written to <dst-root>/conversion_manifest.jsonl.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import car_v2_block as V  # noqa: E402


def _v1_strings():
    spec = importlib.util.spec_from_file_location(
        "gen_real_buildable_scenes", REPO / "scripts" / "pipeline" / "gen_real_buildable_scenes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CAR_BODY, mod.FOOTER


V1_CAR_BODY, V1_FOOTER = _v1_strings()


def _norm(e):
    return (e.tag, tuple(sorted(e.attrib.items())), tuple(_norm(c) for c in e))


def is_v1_car(root):
    """(ok, reason): the scene's car and actuators are v1's, position aside."""
    world = root.find("worldbody")
    car = world.find("./body[@name='car']")
    if car is None:
        return False, "no car body"
    sx, sy = car.get("pos").split()[:2]
    if _norm(car) != _norm(ET.fromstring(V1_CAR_BODY.format(sx=sx, sy=sy))):
        return False, "car body differs from v1 CAR_BODY"
    v1_act = ET.fromstring("<r>" + V1_FOOTER.split("</worldbody>")[1].split("</mujoco>")[0] + "</r>").find("actuator")
    acts = root.findall("actuator")
    if len(acts) != 1 or _norm(acts[0]) != _norm(v1_act):
        return False, "actuator block differs from v1 FOOTER"
    return True, ""


def convert_root(root, **opts):
    """A converted copy of a parsed v1 scene."""
    root = ET.fromstring(ET.tostring(root))
    world = root.find("worldbody")
    car = world.find("./body[@name='car']")
    sx, sy = car.get("pos").split()[:2]
    i = list(world).index(car)
    world.remove(car)
    world.insert(i, V.car_body_element(sx, sy, **(V.V2 | opts)))
    j = list(root).index(root.find("actuator"))
    root.remove(root.find("actuator"))
    root.insert(j, V.actuator_element(**(V.V2 | opts)))
    ET.indent(root, space="  ")
    return root


def models_equal(xml_a: str, xml_b: str):
    """Compare every numeric array of two compiled models. Returns the names that differ."""
    import mujoco
    a, b = (mujoco.MjModel.from_xml_string(x) for x in (xml_a, xml_b))
    diff = []
    for name in dir(a):
        if name.startswith("_"):
            continue
        try:
            va, vb = getattr(a, name), getattr(b, name)
        except Exception:                          # noqa: BLE001
            continue
        if isinstance(va, np.ndarray) and not np.array_equal(va, vb):
            diff.append(name)
    return diff


def selftest(scene):
    src = Path(scene).read_text()
    root = ET.fromstring(src)
    ok, why = is_v1_car(root)
    assert ok, why
    rebuilt = ET.tostring(convert_root(root, **V.V1), encoding="unicode")
    diff = models_equal(src, rebuilt)
    print(json.dumps({"scene": scene, "v1_rebuild_differs_in": diff}))
    return not diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root")
    ap.add_argument("--dst-root")
    ap.add_argument("--pattern", default="**/*.xml")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--check", action="store_true", help="compile every written scene")
    ap.add_argument("--caster", default=V.V2["caster"])
    ap.add_argument("--wheel", default=V.V2["wheel"])
    ap.add_argument("--actuator", default=V.V2["actuator"])
    ap.add_argument("--ctrl-units", default=V.V2["ctrl_units"], choices=["pwm", "omega"])
    ap.add_argument("--selftest", default=None, metavar="SCENE")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest(args.selftest) else 1)

    src, dst = Path(args.src_root).resolve(), Path(args.dst_root).resolve()
    if dst == src or src in dst.parents or dst in src.parents:
        sys.exit(f"refusing: destination {dst} overlaps source {src}")
    opts = dict(caster=args.caster, wheel=args.wheel, actuator=args.actuator, ctrl_units=args.ctrl_units)
    dst.mkdir(parents=True, exist_ok=True)
    counts = {}
    with open(dst / "conversion_manifest.jsonl", "a") as man:
        files = sorted(src.glob(args.pattern))
        for n, path in enumerate(files):
            if args.limit and n >= args.limit:
                break
            rel = path.relative_to(src)
            out = dst / rel
            raw = path.read_bytes()
            rec = {"src": str(rel), "src_sha256": hashlib.sha256(raw).hexdigest(), "opts": opts}
            try:
                root = ET.fromstring(raw)
                ok, why = is_v1_car(root)
            except ET.ParseError as e:
                ok, why = False, f"parse error: {e}"
            if not ok:
                rec["status"] = f"skipped: {why}"
            elif out.exists() and not args.overwrite:
                rec["status"] = "skipped: destination exists"
            else:
                text = "<?xml version='1.0' encoding='utf-8'?>\n" + ET.tostring(convert_root(root, **opts), encoding="unicode") + "\n"
                if args.check:
                    import mujoco
                    mujoco.MjModel.from_xml_string(text)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(text)
                rec["status"] = "converted"
                rec["dst_sha256"] = hashlib.sha256(text.encode()).hexdigest()
            counts[rec["status"]] = counts.get(rec["status"], 0) + 1
            man.write(json.dumps(rec) + "\n")
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
