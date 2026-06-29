#!/usr/bin/env python3
"""Bit-compare GATE for the render speedup: the model-input crop (render_ctx -> (5,64,64)) must be IDENTICAL
before/after any render optimization, because the model was trained on the current pipeline's crops.

  capture : render N states with the CURRENT code, save reference crops to REF.
  compare : re-render the same N states, assert np.array_equal vs REF. Reports bit-identical count + max|diff|.

Run `capture` ONCE on the original code, then `compare` after EVERY render change. Ship only if bit-identical.
Also renders each state TWICE per pass to catch within-state cache/non-determinism bugs."""
import sys, os, json, argparse
import numpy as np
from scorer_beam import BeamPlanner, make_env, FALLBACK_GOAL          # noqa: E402
from namo.core.xml_goal_parser import extract_goal_with_fallback      # noqa: E402
from namo.paths import resolve                                        # noqa: E402

CKPT = "/scratch/dm1487/sage_outputs/scorer/qfull_v3_v4hq_s1/namo-classifier/qkfk0slk/checkpoints/epoch011-val_loss0.6571.ckpt"
KEY = "/scratch/dm1487/datasets/namo_testset_v1/labels/pure2push.json"
REF = "/scratch/dm1487/eval/render_equiv/ref_crops.npz"


def sample(n):
    k = json.load(open(KEY)); xmls = list(k); step = max(1, len(xmls) // n)
    return [(xml, k[xml][0]["object_id"]) for xml in xmls[::step][:n]]


def render_all(sc, samp):
    crops = {}
    for (xml, obj) in samp:
        xmlp = str(resolve(xml)); env = make_env(xmlp); goal = extract_goal_with_fallback(xmlp, FALLBACK_GOAL)
        env.set_robot_goal(*goal); env.get_reachable_objects()
        ctx, _ = sc.render_ctx(env, obj, goal, xmlp, region_samples=None)
        ctx2, _ = sc.render_ctx(env, obj, goal, xmlp, region_samples=None)   # 2nd call = cache warm; must match
        if not np.array_equal(ctx, ctx2):
            print(f"  ⚠ WITHIN-STATE MISMATCH (cache/non-determinism bug): {os.path.basename(xml)}|{obj} "
                  f"max|diff|={np.max(np.abs(ctx-ctx2)):.2e}")
        crops[f"{os.path.basename(xml)}|{obj}"] = ctx.astype(np.float32)
    return crops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["capture", "compare"], required=True)
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(REF), exist_ok=True)
    sc = BeamPlanner(ckpt=CKPT).scorer
    crops = render_all(sc, sample(a.n))
    if a.mode == "capture":
        np.savez_compressed(REF, **crops)
        print(f"  CAPTURED {len(crops)} reference crops -> {REF}")
        return
    ref = np.load(REF); nmatch = 0; maxd = 0.0; n = len(crops)
    for k_ in crops:
        if k_ not in ref:
            print(f"  MISSING in ref: {k_}"); continue
        maxd = max(maxd, float(np.max(np.abs(crops[k_] - ref[k_]))))
        nmatch += int(np.array_equal(crops[k_], ref[k_]))
    print(f"  compared {n}: BIT-IDENTICAL={nmatch}/{n}   max|diff|={maxd:.3e}")
    print("  GATE: PASS — model input unchanged" if nmatch == n else f"  GATE: FAIL — {n-nmatch} crops differ")
    sys.exit(0 if nmatch == n else 1)


if __name__ == "__main__":
    main()
