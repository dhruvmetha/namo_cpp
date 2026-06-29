#!/usr/bin/env python3
"""WALL-CLOCK timing benchmark — Hz-v3 / NoHz-v3 / random, SAME node, interleaved, warm-only.

For each episode (stratified easy/med/hard by solve_rate_first_push), run reactive@2 for the 3 models
back-to-back from the SAME s0, timing each component with perf_counter:
  t_score = score_state (render crop + NN forward)   <- the per-state cost
  t_sim   = env.step (the physics)
  t_wall  = total per model per episode
Model load + a warmup of K forwards are UNTIMED (steady-state only). Random uses score=False (no render/NN).
Writes per-(episode,model) jsonl + prints a summary. Single process => same machine for all comparisons."""
import sys, os, json, time, argparse, random
import numpy as np
from collections import defaultdict
from scorer_beam import BeamPlanner, make_env, make_action, FALLBACK_GOAL          # noqa: E402
from eval_m3 import rank_first_pushes_h2, sample_goal_points, goal_open_pts         # noqa: E402
from namo.core.xml_goal_parser import extract_goal_with_fallback                    # noqa: E402
from namo.paths import resolve                                                      # noqa: E402

S = "/scratch/dm1487/sage_outputs/scorer"
HZ = f"{S}/qfull_v3_v4hq_s1/namo-classifier/qkfk0slk/checkpoints/epoch011-val_loss0.6571.ckpt"
NOHZ = f"{S}/qfull_nohz_v3_v4hq_s1/namo-classifier/wl8k6iyv/checkpoints/epoch012-val_loss0.6896.ckpt"
KEY = "/scratch/dm1487/datasets/namo_testset_v1/labels/pure2push.json"
PC = time.perf_counter


def tier(sr):
    return "hard" if sr < 0.05 else ("med" if sr < 0.30 else "easy")


def stratified(n_per):
    k = json.load(open(KEY)); buckets = defaultdict(list)
    for xml, recs in k.items():
        for r in recs:
            buckets[tier(r.get("solve_rate_first_push", 0.0))].append((xml, r["object_id"], r.get("region")))
    out = []
    for t in ("easy", "med", "hard"):
        out += [(x, o, rg, t) for (x, o, rg) in buckets[t][:n_per]]
    return out


def reactive_timed(pl, env, goal, xmlp, obj, s0, gp, prior, rng):
    """Reactive@2 with per-component timing. prior: 'q'=model argmax, 'uniform'=random pick (no model)."""
    ts = tsc = 0.0; ns = nsc = 0; solved = False; t0 = PC
    sc = (prior == "q")
    env.set_full_state(s0)
    t = PC; pool0 = rank_first_pushes_h2(pl, env, goal, xmlp, s0, 2, restrict_obj=obj, score=sc); tsc += PC - t; nsc += 1
    if pool0:
        o, g1, q = pool0[0] if sc else rng.choice(pool0)
        env.set_full_state(s0); t = PC; env.step(make_action(obj, g1)); ts += PC - t; ns += 1
        if goal_open_pts(env, gp):
            solved = True
        else:
            s1 = env.get_full_state()
            t = PC; pool1 = rank_first_pushes_h2(pl, env, goal, xmlp, s1, 1, restrict_obj=obj, score=sc); tsc += PC - t; nsc += 1
            if pool1:
                o2, g2, q2 = pool1[0] if sc else rng.choice(pool1)
                env.set_full_state(s1); t = PC; env.step(make_action(obj, g2)); ts += PC - t; ns += 1
                solved = bool(goal_open_pts(env, gp))
    return {"t_score": tsc, "t_sim": ts, "n_score": nsc, "n_sim": ns, "t_wall": PC - t0, "solved": solved}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-tier", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    rng = random.Random(7)
    print(f"loading scorers...", flush=True)
    pl_hz = BeamPlanner(ckpt=HZ); pl_nz = BeamPlanner(ckpt=NOHZ)
    models = [("Hz", pl_hz, "q"), ("NoHz", pl_nz, "q"), ("random", pl_nz, "uniform")]
    samp = stratified(a.n_per_tier)
    print(f"  {len(samp)} episodes; warming up {a.warmup} forwards/model (untimed)...", flush=True)
    # warmup on the first episode's env (untimed)
    x0, o0, _, _ = samp[0]; xp0 = str(resolve(x0)); e0 = make_env(xp0); g0 = extract_goal_with_fallback(xp0, FALLBACK_GOAL)
    e0.set_robot_goal(*g0); e0.get_reachable_objects(); s00 = e0.get_full_state()
    for _ in range(a.warmup):
        for pl in (pl_hz, pl_nz):
            rank_first_pushes_h2(pl, e0, g0, xp0, s00, 2, restrict_obj=o0, score=True)
    fh = open(a.out, "w"); t_start = PC
    for i, (xml, obj, reg, t) in enumerate(samp):
        try:
            xmlp = str(resolve(xml)); env = make_env(xmlp); goal = extract_goal_with_fallback(xmlp, FALLBACK_GOAL)
            env.set_robot_goal(*goal); env.get_reachable_objects(); s0 = env.get_full_state(); gp = sample_goal_points(env)
        except Exception:
            continue
        if not gp or goal_open_pts(env, gp):
            continue
        for name, pl, prior in models:                       # interleaved: all 3 on the SAME s0, adjacent in time
            r = reactive_timed(pl, env, goal, xmlp, obj, s0, gp, prior, rng)
            r.update({"model": name, "tier": t, "xml": os.path.basename(xml), "object_id": obj})
            fh.write(json.dumps(r) + "\n")
        if i % 20 == 0:
            fh.flush(); print(f"  [{i}/{len(samp)}] {PC - t_start:.0f}s", file=sys.stderr, flush=True)
    fh.close()
    print(f"DONE {a.out}  ({PC - t_start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
