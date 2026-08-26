"""Yaw-torque budget on the pushed block: pusher patch vs floor patch."""
import math
import numpy as np, mujoco

def yaw_of(q):
    w,x,y,z=q
    return math.degrees(math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)))

def load_cmds(path):
    c=[]
    for line in open(path):
        if line.startswith("[PUSH_CTRL]"):
            p=line.split(); c.append((float(p[2]),float(p[3])))
    return c

def probe(xml, cmds, qpos0, obj_geom="obstacle_0_movable", settle=100, every=10):
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    d.qpos[:]=qpos0; d.qvel[:]=0; d.ctrl[:]=0
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, obj_geom)
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    bid = m.geom_bodyid[gid]
    carb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "car")
    for _ in range(settle): mujoco.mj_step(m,d)
    out=[]; f6=np.zeros(6)
    for t,(l,r) in enumerate(cmds):
        d.ctrl[0]=l; d.ctrl[1]=r
        mujoco.mj_step(m,d)
        if t % every: continue
        com = d.xipos[bid].copy()
        R = d.xmat[bid].reshape(3,3)
        tz_push=tz_grnd=0.0; Fpush=0.0; cop=0.0; npush=0; ngrnd=0
        tors_push=tors_grnd=0.0
        for i in range(d.ncon):
            c=d.contact[i]
            if gid not in (c.geom1,c.geom2): continue
            other = c.geom2 if c.geom1==gid else c.geom1
            sign = 1.0 if c.geom2==gid else -1.0
            mujoco.mj_contactForce(m,d,i,f6)
            fr = c.frame.reshape(3,3)
            fw = sign * (fr.T @ f6[:3])          # world force on the block
            tw = sign * (fr.T @ f6[3:])          # world torque (torsional/rolling)
            r_ = c.pos - com
            tz = np.cross(r_, fw)[2] + tw[2]
            if other==floor:
                tz_grnd += tz; tors_grnd += tw[2]; ngrnd+=1
            else:
                tz_push += tz; tors_push += tw[2]; npush+=1
                fn = float(f6[0]); Fpush += fn
                loc = R.T @ r_                    # contact in block local frame
                cop += loc[0]*fn                  # local x = across the 7cm face
        if Fpush>1e-12: cop/=Fpush
        out.append(dict(t=t, yaw=yaw_of(d.xquat[bid]), pos=d.xpos[bid].copy(),
                        caryaw=yaw_of(d.xquat[carb]),
                        tz_push=tz_push, tz_grnd=tz_grnd, Fpush=Fpush,
                        cop_cm=cop*100, npush=npush, ngrnd=ngrnd,
                        tors_push=tors_push, tors_grnd=tors_grnd))
    return out
