import os, sys, json, math
import numpy as np
S="/tmp/claude-89862/-common-home-dm1487-robotics-research-ktamp-namo/a699fcde-7931-4d4f-a2d6-9dc120966a3b/scratchpad/rot"
sys.path.insert(0, os.path.dirname(__file__))
from run_push import run
from analyze import unwrap_deg, coupling

xml, cfg, edge, tag = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
try:
    r = run(xml, cfg, "obstacle_0_movable", edge, 4, tag)
except Exception as e:
    print(json.dumps(dict(tag=tag, edge=edge, error=str(e)[:200]))); sys.exit(0)
q = r["qpos"]; blk = q[:, -7:]; car = q[:, 0:7]
def yaw(row): 
    w,x,y,z=row; return math.degrees(math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)))
byaw = unwrap_deg(np.array([yaw(b[3:7]) for b in blk]))
cyaw = unwrap_deg(np.array([yaw(c[3:7]) for c in car]))
travel = np.linalg.norm(blk[:, :2]-blk[0, :2], axis=1)*100
dyaw = byaw-byaw[0]; dcar = cyaw-cyaw[0]
k, r2, n = coupling(travel, dyaw)
steps=[]
for s in range(5):
    i0,i1 = 100+s*550, 100+(s+1)*550
    if i1 <= len(travel): steps.append(round(float(dyaw[i1-1]-dyaw[i0]),2))
print(json.dumps(dict(tag=tag, edge=edge, travel=round(float(travel[-1]),2), ticks=len(q),
                      dyaw=round(float(dyaw[-1]),2), coupling=round(float(k),4),
                      r2=round(float(r2),3), maxrel=round(float(np.abs(dyaw-dcar).max()),2),
                      percm=round(float(travel[-1]/5),2), steps=steps)))
os.remove(os.path.join(S, f"qpos_{tag}.txt"))
