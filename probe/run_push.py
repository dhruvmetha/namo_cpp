"""One push through namo_rl, per-tick qpos out. Scene XML is a caller-supplied copy."""
import os, sys, json, math, tempfile
import numpy as np

SCRATCH = "/tmp/claude-89862/-common-home-dm1487-robotics-research-ktamp-namo/a699fcde-7931-4d4f-a2d6-9dc120966a3b/scratchpad/rot"

def yaw_of(q):
    w, x, y, z = q
    return math.degrees(math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))

def run(xml, cfg, obj, edge, depth, tag):
    dump = os.path.join(SCRATCH, f"qpos_{tag}.txt")
    if os.path.exists(dump):
        os.remove(dump)
    os.environ["NAMO_QPOS_DUMP"] = dump
    import namo_rl
    env = namo_rl.RLEnvironment(xml, cfg, False)
    pre = env.get_observation()[f"{obj}_pose"]
    a = namo_rl.Action()
    a.object_id = obj
    a.x, a.y, a.theta = float(pre[0]), float(pre[1]), float(pre[2])
    a.edge_idx, a.depth = edge, depth
    env.step(a)
    post = env.get_observation()[f"{obj}_pose"]
    rows = []
    with open(dump) as f:
        for line in f:
            p = line.split()
            rows.append([float(v) for v in p[2:]])
    q = np.array(rows)
    return dict(pre=list(map(float, pre)), post=list(map(float, post)), qpos=q)

if __name__ == "__main__":
    xml, cfg, obj, edge, depth, tag = sys.argv[1:7]
    r = run(xml, cfg, obj, int(edge), int(depth), tag)
    q = r["qpos"]
    # car free joint = qpos[0:7], block free joint = last 7
    car = q[:, 0:7]
    blk = q[:, -7:]
    byaw = np.array([yaw_of(row[3:7]) for row in blk])
    cyaw = np.array([yaw_of(row[3:7]) for row in car])
    np.savez(os.path.join(SCRATCH, f"trace_{tag}.npz"),
             car=car, blk=blk, byaw=byaw, cyaw=cyaw, qpos0=q[0],
             pre=r["pre"], post=r["post"])
    print(json.dumps(dict(tag=tag, ticks=len(q), pre=r["pre"], post=r["post"])))
