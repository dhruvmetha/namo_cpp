"""Replay logged wheel commands in pip MuJoCo and read the block's contact manifold."""
import sys, math
import numpy as np, mujoco
SCRATCH="/tmp/claude-89862/-common-home-dm1487-robotics-research-ktamp-namo/a699fcde-7931-4d4f-a2d6-9dc120966a3b/scratchpad/rot"

def yaw_of(q):
    w,x,y,z=q
    return math.degrees(math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)))

def load_cmds(path):
    c=[]
    for line in open(path):
        if line.startswith("[PUSH_CTRL]"):
            p=line.split(); c.append((float(p[2]),float(p[3])))
    return c

def replay(xml, cmds, qpos0, obj_geom="obstacle_0_movable", settle=100, every=25):
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    d.qpos[:] = qpos0; d.qvel[:] = 0
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, obj_geom)
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    bid = m.geom_bodyid[gid]
    d.ctrl[:] = 0
    for _ in range(settle):
        mujoco.mj_step(m, d)
    log=[]
    f6 = np.zeros(6)
    for t,(l,r) in enumerate(cmds):
        d.ctrl[0]=l; d.ctrl[1]=r
        mujoco.mj_step(m,d)
        if t % every: continue
        pusher=[]; ground=[]
        for i in range(d.ncon):
            c = d.contact[i]
            if gid not in (c.geom1, c.geom2): continue
            other = c.geom2 if c.geom1==gid else c.geom1
            mujoco.mj_contactForce(m,d,i,f6)
            rec = (c.pos.copy(), float(c.dist), float(f6[0]))
            (ground if other==floor else pusher).append(rec)
        bp = d.xpos[bid].copy(); by = yaw_of(d.xquat[bid])
        log.append(dict(t=t, pos=bp, yaw=by,
                        n_push=len(pusher), n_grnd=len(ground),
                        push=pusher, grnd=ground))
    return m, d, log

if __name__=="__main__":
    xml=sys.argv[1]; nav=sys.argv[2]; tracetag=sys.argv[3]
    z=np.load(f"{SCRATCH}/trace_{tracetag}.npz")
    qpos0=z["qpos0"] if "qpos0" in z else None
    cmds=load_cmds(nav)
    m,d,log=replay(xml,cmds,qpos0)
    print("replayed", len(cmds))
