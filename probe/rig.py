"""Kinematic-pusher rig: the limiting case of our car.

The pusher is a box on a single slide joint, velocity-servoed. It cannot yaw,
cannot slip, cannot lose traction. Any failure to rotate here is the contact
model, not the controller. Block geometry/mass/friction match the pool scene.
"""
import math
import numpy as np, mujoco

TPL = """
<mujoco model="rig">
  <compiler angle="degree"/>
  <option timestep="0.002" integrator="implicitfast" iterations="100" cone="{cone}" impratio="{impratio}"/>
  <default><geom density="1"/></default>
  <worldbody>
    <geom name="floor" type="plane" condim="4" friction="{floor_mu} 0.005 0.001" size="0 0 0.05" {floor_extra}/>
    <body name="pusher" pos="0 {py} 0.0475">
      <joint name="slide" type="slide" axis="0 -1 0"/>
      <inertial pos="0 0 0" mass="10.0" diaginertia="1 1 1"/>
      {pusher_geom}
    </body>
    <body name="block" pos="0 0 0.020">
      <joint type="free"/>
      <geom name="block" type="box" size="0.035 0.075 0.020" condim="{condim}"
            friction="{blk_mu} {blk_tor} 0.0001" mass="{mass}" priority="1" {blk_extra}/>
    </body>
  </worldbody>
  <actuator>
    <velocity name="drive" joint="slide" kv="500" ctrlrange="-1 1" forcerange="-200 200"/>
  </actuator>
</mujoco>
"""

FLAT = ('<geom name="face" type="box" pos="{ox} -0.0175 0" size="0.035 0.0175 0.0325" '
        'condim="3" priority="2" friction="{pmu} 0.005 0.0001" {extra}/>')

def build(offset_cm=3.0, blk_mu="1", blk_tor="0.005", mass="0.1", floor_mu="0.5",
          cone="elliptic", impratio="1", condim="4", pusher="flat", pmu="1",
          blk_extra="", floor_extra="", pusher_yaw_deg=0.0, gap=0.005):
    """offset_cm: pusher centre offset along the block's 7cm face. +x."""
    ox = offset_cm/100.0
    if pusher == "flat":
        pg = FLAT.format(ox=ox, pmu=pmu, extra=f'euler="0 0 {pusher_yaw_deg}"' if pusher_yaw_deg else "")
    elif pusher == "point":     # thin vertical bar: single-point contact, no flat mate
        pg = (f'<geom name="face" type="cylinder" pos="{ox} -0.0175 0" size="0.0175 0.0325" '
              f'condim="3" priority="2" friction="{pmu} 0.005 0.0001"/>')
    elif pusher == "narrow":    # 1cm wide flat face
        pg = (f'<geom name="face" type="box" pos="{ox} -0.0175 0" size="0.005 0.0175 0.0325" '
              f'condim="3" friction="{pmu} 0.005 0.0001"/>')
    xml = TPL.format(py=0.075+gap+0.0175, pusher_geom=pg, blk_mu=blk_mu, blk_tor=blk_tor,
                     mass=mass, floor_mu=floor_mu, cone=cone, impratio=impratio,
                     condim=condim, blk_extra=blk_extra, floor_extra=floor_extra)
    return xml

def run(xml, speed=0.05357, seconds=5.5, every=5):
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "block")
    for _ in range(100):
        mujoco.mj_step(m, d)          # settle onto the floor
    d.ctrl[0] = speed
    n = int(seconds/m.opt.timestep)
    pos, yaw = [], []
    for t in range(n):
        mujoco.mj_step(m, d)
        if t % every: continue
        q = d.xquat[bid]
        pos.append(d.xpos[bid][:2].copy())
        yaw.append(math.degrees(math.atan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]**2+q[3]**2))))
    pos = np.array(pos); yaw = np.degrees(np.unwrap(np.radians(np.array(yaw))))
    travel = np.linalg.norm(pos-pos[0], axis=1)*100
    dyaw = yaw-yaw[0]
    return travel, dyaw

def coupling(travel, dyaw, lo=0.3, hi=0.15):
    msk = (travel > lo) & (travel < travel.max()-hi)
    if msk.sum() < 5: return float('nan'), float('nan')
    x, y = travel[msk], dyaw[msk]
    A = np.vstack([x, np.ones_like(x)]).T
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A@sol; ss=((y-pred)**2).sum(); st=((y-y.mean())**2).sum()
    return sol[0], (1-ss/st if st>0 else float('nan'))
