#!/usr/bin/env python3
"""DEEP failure analysis of the 2x2 best-first solve results (corrected region criterion).
Answers: WHO fails (unsolved@900 tail), WHY (search-budget vs ranking), the reactive gap, ceiling vs random.
Joins leaf jsonls + pure2push_divisions.json (difficulty/n_setups) + the exhaustive (a1,a2) pairmap."""
import json, glob, os, pickle, statistics as st
from collections import defaultdict

EVAL = "/scratch/dm1487/eval"
DIV = json.load(open("/scratch/dm1487/datasets/namo_testset_v1/labels/pure2push_divisions.json"))
PM = pickle.load(open(f"{EVAL}/exhaustive_pairmap_pure2.pkl", "rb"))["pairmap"]
CELLS = {"Hz-v1": "qfull_v4hq_s1", "NoHz-v1": "qfull_nohz_v4hq_s1",
         "Hz-v2": "qfull_v2_v4hq_s1", "NoHz-v2": "qfull_nohz_v2_v4hq_s1"}


def load(run):
    rows = []
    for f in glob.glob(f"{EVAL}/bf900_{run}/shard_*.jsonl"):
        rows += [json.loads(l) for l in open(f) if l.strip()]
    return rows


# key: (realpath xml, object) -> division record
drec = {}
for xml, recs in DIV.items():
    for r in recs:
        drec[(os.path.realpath(xml), r["object_id"])] = r
    # also non-realpath
for xml, recs in DIV.items():
    for r in recs:
        drec.setdefault((xml, r["object_id"]), r)


def kf(r):
    return (os.path.realpath(r["xml"]), r["object_id"])


def needle(scene_key):
    """From the exhaustive pairmap: (#solving (a1,a2) pairs, #total tried pairs, #distinct solving a1)."""
    pm = PM.get(scene_key)
    if not pm:
        return None
    tot = sols = 0; a1s = set()
    for a1, a2map in pm.items():
        for a2, ok in a2map.items():
            tot += 1
            if ok:
                sols += 1; a1s.add(a1)
    return sols, tot, len(a1s)


data = {nm: load(run) for nm, run in CELLS.items()}
print("=== 1. SOLVE-RATE @900 + UNSOLVED COUNT, by DIVISION ===")
for nm in CELLS:
    rows = data[nm]; n = len(rows)
    bydiv = defaultdict(lambda: [0, 0])
    for r in rows:
        rec = drec.get(kf(r)); dv = rec["division"] if rec else "?"
        bydiv[dv][0] += 1; bydiv[dv][1] += int(r["solved"])
    uns = sum(1 for r in rows if not r["solved"])
    line = "  ".join(f"{dv}:{c[1]}/{c[0]}={100*c[1]/c[0]:.0f}%" for dv, c in sorted(bydiv.items()))
    print(f"  {nm:8s} n={n} unsolved={uns} ({100*uns/n:.1f}%) | {line}")

print("\n=== 2. UNSOLVED@900 OVERLAP — intrinsic-hard vs model-specific ===")
uns = {nm: {kf(r) for r in data[nm] if not r["solved"]} for nm in CELLS}
allk = set.intersection(*[{kf(r) for r in data[nm]} for nm in CELLS])
shared = set.intersection(*uns.values())
anyf = set.union(*uns.values())
print(f"  episodes common to all 4: {len(allk)}")
print(f"  unsolved by ALL 4 (intrinsic within 900): {len(shared & allk)}")
print(f"  unsolved by >=1 model: {len(anyf & allk)}  -> model-specific-only: {len(anyf & allk) - len(shared & allk)}")
for nm in CELLS:
    only = (uns[nm] & allk) - set.union(*[uns[o] for o in CELLS if o != nm])
    print(f"    {nm:8s} unsolved-only (others solved): {len(only)}")

print("\n=== 3. WHY UNSOLVED — search-budget (rare needle) vs ranking. Needle = #solving(a1,a2)/#tried (Hz-v2) ===")
solv = {kf(r): r["solved"] for r in data["Hz-v2"]}
ns_s = []; ns_u = []; dens_s = []; dens_u = []
for k, sl in solv.items():
    nd = needle(k)
    if not nd:
        continue
    sols, tot, na1 = nd
    (ns_s if sl else ns_u).append(sols)
    (dens_s if sl else dens_u).append(100 * sols / max(tot, 1))
print(f"  SOLVED  (Hz-v2): median solving-pairs={st.median(ns_s):.0f}  median needle-density={st.median(dens_s):.1f}%  n={len(ns_s)}")
print(f"  UNSOLVED(Hz-v2): median solving-pairs={st.median(ns_u):.0f}  median needle-density={st.median(dens_u):.1f}%  n={len(ns_u)}")
import numpy as np
print(f"    unsolved needle-density distribution %: p10={np.percentile(dens_u,10):.1f} p50={np.percentile(dens_u,50):.1f} p90={np.percentile(dens_u,90):.1f}")
print(f"    unsolved with <=1 solving pair (true needle-in-haystack): {sum(1 for x in ns_u if x<=1)}/{len(ns_u)}")

print("\n=== 4. SIMS-TO-SOLVE distribution (reactive vs deep search) ===")
for nm in CELLS:
    rows = data[nm]; n = len(rows)
    b = {"<=2": 0, "3-10": 0, "11-50": 0, "51-200": 0, "201-900": 0, "UNSOLVED": 0}
    for r in rows:
        if not r["solved"]: b["UNSOLVED"] += 1
        else:
            s = r["sims"]; b["<=2" if s <= 2 else "3-10" if s <= 10 else "11-50" if s <= 50 else "51-200" if s <= 200 else "201-900"] += 1
    print(f"  {nm:8s} " + " ".join(f"{k}:{100*v/n:.0f}%" for k, v in b.items()))

print("\n=== 5. REACTIVE GAP (@2 solve by division): Hz-v2 vs NoHz-v2 ===")
for nm in ("Hz-v2", "NoHz-v2"):
    bydiv = defaultdict(lambda: [0, 0])
    for r in data[nm]:
        rec = drec.get(kf(r)); dv = rec["division"] if rec else "?"
        bydiv[dv][0] += 1; bydiv[dv][1] += int(r["solved"] and r["sims"] <= 2)
    print(f"  {nm:8s} @2 by div: " + "  ".join(f"{dv}:{100*c[1]/c[0]:.0f}%" for dv, c in sorted(bydiv.items())))

print("\n=== 6. CEILING vs RANDOM — does random solve what NoHz-v1 can't? (guidance-hurts test) ===")
rnd = {}
for i in range(5):
    for r in load(f"uniform_s{i}"):
        rnd.setdefault(kf(r), 0)
        if r["solved"]: rnd[kf(r)] += 1
for nm in ("NoHz-v1", "Hz-v2"):
    u = uns[nm] & allk
    rnd_solves = sum(1 for k in u if rnd.get(k, 0) >= 1)  # any of 5 random seeds solved it
    print(f"  {nm:8s} unsolved={len(u)}; of those, random(any of 5 seeds) solved {rnd_solves} ({100*rnd_solves/max(len(u),1):.0f}%)")
