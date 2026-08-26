"""Joint sweep of the whole defensible parameter space, flat pusher.

One-at-a-time nulls are weak evidence: a knob can be inert alone and live in
combination. This runs the full product.
"""
import itertools, json, sys
from multiprocessing import Pool
sys.path.insert(0, "/common/home/dm1487/robotics_research/ktamp/namo-physics/probe")
from rig import build, run, coupling

GRID = dict(
    blk_mu   = ["0.15","0.3","0.6","1"],
    blk_tor  = ["0.0001","0.005","0.05","0.5"],
    mass     = ["0.05","0.1","0.3"],
    pmu      = ["0.05","0.3","1"],
    solref   = ["0.002 1","0.02 1","0.05 1"],
    cone     = ["elliptic","pyramidal"],
    condim   = ["3","4","6"],
    impratio = ["1","10"],
)
OFFSETS = [3.0, 3.5]

def job(a):
    off, combo = a
    kw = dict(zip(GRID, combo))
    solref = kw.pop("solref")
    try:
        t, y = run(build(offset_cm=off, blk_extra=f'solref="{solref}"', **kw), seconds=4.0)
        k, r2 = coupling(t, y)
    except Exception as e:
        return dict(off=off, err=str(e)[:80], **kw)
    return dict(off=off, solref=solref, travel=round(float(t[-1]),2),
                dyaw=round(float(y[-1]),2), k=round(float(k),4),
                r2=round(float(r2),3), **kw)

if __name__ == "__main__":
    combos = list(itertools.product(*GRID.values()))
    jobs = [(o, c) for o in OFFSETS for c in combos]
    print(f"{len(jobs)} runs", flush=True)
    with Pool(26) as p:
        res = p.map(job, jobs, chunksize=8)
    out = "/tmp/claude-89862/-common-home-dm1487-robotics-research-ktamp-namo/a699fcde-7931-4d4f-a2d6-9dc120966a3b/scratchpad/rot/joint.jsonl"
    with open(out, "w") as f:
        for r in res: f.write(json.dumps(r)+"\n")
    ok = [r for r in res if "k" in r and r["k"] == r["k"]]
    ok.sort(key=lambda r: -abs(r["k"]))
    print(f"\nran {len(ok)}; TOP 12 by |coupling| (hardware needs 2.1-2.4 deg/cm, travel near 20cm):")
    for r in ok[:12]:
        print(f"  k={r['k']:+7.3f} R2={r['r2']:5.2f} travel={r['travel']:6.2f} off={r['off']} "
              f"mu={r['blk_mu']:>4} tor={r['blk_tor']:>6} m={r['mass']:>4} pmu={r['pmu']:>4} "
              f"solref={r['solref']:>8} cone={r['cone'][:4]} cd={r['condim']} ip={r['impratio']}")
    import numpy as np
    ks = np.array([abs(r["k"]) for r in ok])
    print(f"\n|coupling| over {len(ok)} combos: max={ks.max():.3f} p99={np.percentile(ks,99):.3f} "
          f"median={np.median(ks):.3f}   (hardware 2.126-2.416)")
    print(f"combos reaching even 1.0 deg/cm: {(ks>=1.0).sum()} / {len(ks)}")
