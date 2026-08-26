import sys, math
import numpy as np
SCRATCH="/tmp/claude-89862/-common-home-dm1487-robotics-research-ktamp-namo/a699fcde-7931-4d4f-a2d6-9dc120966a3b/scratchpad/rot"

def unwrap_deg(a):
    return np.degrees(np.unwrap(np.radians(a)))

def coupling(travel_cm, dyaw_deg, lo=0.3, hi_margin=0.15):
    """Peer's fit window: drop <lo cm travel and > (max-hi_margin) cm."""
    m = (travel_cm > lo) & (travel_cm < travel_cm.max() - hi_margin)
    if m.sum() < 5:
        return float('nan'), float('nan'), 0
    x, y = travel_cm[m], dyaw_deg[m]
    A = np.vstack([x, np.ones_like(x)]).T
    sol, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ sol
    ss_res = ((y-pred)**2).sum(); ss_tot = ((y-y.mean())**2).sum()
    r2 = 1-ss_res/ss_tot if ss_tot > 0 else float('nan')
    return sol[0], r2, int(m.sum())

def load(tag):
    z = np.load(f"{SCRATCH}/trace_{tag}.npz")
    blk, byaw, cyaw = z["blk"], unwrap_deg(z["byaw"]), unwrap_deg(z["cyaw"])
    xy = blk[:, :2]
    travel = np.linalg.norm(xy - xy[0], axis=1) * 100.0
    dyaw = byaw - byaw[0]
    dcar = cyaw - cyaw[0]
    return travel, dyaw, dcar, z

if __name__ == "__main__":
    for tag in sys.argv[1:]:
        travel, dyaw, dcar, z = load(tag)
        k, r2, n = coupling(travel, dyaw)
        # per commanded step: 550 ticks each, after 100 settle
        steps = []
        for s in range(5):
            i0, i1 = 100 + s*550, 100 + (s+1)*550
            if i1 <= len(travel):
                steps.append((travel[i1-1]-travel[i0], dyaw[i1-1]-dyaw[i0]))
        print(f"{tag:22s} travel={travel[-1]:6.2f}cm dyaw={dyaw[-1]:+7.2f}deg "
              f"coupling={k:+.3f} deg/cm R2={r2:.3f} n={n} maxcar={np.abs(dcar).max():.2f}deg")
        print("   per-step (dtravel_cm, dyaw_deg): " +
              " ".join(f"({a:.2f},{b:+.2f})" for a, b in steps))
