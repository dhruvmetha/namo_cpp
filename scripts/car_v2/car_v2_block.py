"""The diff-drive car, v2: one file that the v1->v2 converter and future scene generators share.

Every robot constant below carries its source:
  v1           the v1 block, CAR_BODY / FOOTER in scripts/pipeline/gen_real_buildable_scenes.py
  published    microMVP V3, https://people.cs.rutgers.edu/~jy512/mvp/ and /pna.html, and the
               Pololu pages it links (gearmotor https://www.pololu.com/product/1512,
               wheels https://www.pololu.com/product/1088), read 2026-11-21
  robot repo   ktamp/robot_control at c3f4820a (config/real.yaml, config/controller.yaml,
               src/robot_control/controller/navigation.py)
  fitted       fitted by fit_drivetrain.py to the robot's free driving in robot_control
               push_calibration (33 real trials, approach and retreat, no block contact),
               2026-11-21; numbers in $NAMO_SCRATCH/car_motion_probe_20261121/drivetrain_fit/
  placeholder  not measured on our robot and not published

Switches, so each change can be measured alone and together (V1 and V2 below):
  body       "v1": v1's 0.5 kg car with 1.5 cm x 1 mm wheels on a 7.5 cm track.
             "mvp": the published build, just under 100 g, 32 x 7 mm wheels, and a 7 x 7 cm
             collision box (user 2026-11-22: true car size 7x7). It is the body e2e_5dd3a674's
             v2_7x7 arm ran, element for element.
             "mvp_6x6": the same build with a 6 x 6 cm collision box, from the robot repo
             real.yaml:28 comment "7x7 robot footprint (0.5 cm safety buffer over physical 6x6 cm)".
             No measurement backs that comment, and the user gives 7 x 7. namo_main asked for it
             2026-11-21 after the v2 car's 7 x 7 corners hit a wall turning in place; kept as an option.
             Either way the planner keeps 7 x 7: the C++ wavefront, inflation, edge points and
             controllers take planning.robot_size from the config, not from this box.
  caster     "v1": two 2.5 mm spheres on ball joints, friction "0 0 0". MuJoCo uses the larger
             friction of two geoms, so they meet the floor at 0.5 and skid, dragging 1.34 N
             (27% of v1's weight). "one_frictionless" (M1): one caster, as the published build
             has, with priority 1 and condim 1 so its floor contact has no friction, no joint,
             and the chassis centre of mass moved toward it so the car rests on three points.
  wheel      "v1": hinge damping 0.01, 78% of v1's steady wheel torque. "v2" (M2): damping 0.
  actuator   "v1": velocity servo kv 0.75, +-0.5 N m. "pwm" (M3): a PWM-driven DC motor,
             torque = STALL_TORQUE*u - (STALL_TORQUE/NO_LOAD_SPEED)*omega within +-STALL_TORQUE,
             plus wheel-joint frictionloss GEARBOX_FRICTION_PWM*STALL_TORQUE (Coulomb gearbox
             friction). Of three forms for the low-PWM loss (this; a driver dead zone; an affine
             command remap) the logs reject the dead zone (31% vs 2.4% error on steady speeds)
             and cannot separate the other two, so the XML-only form is used.
  ctrl_units "pwm": ctrl is the PWM in [-1, 1]. "omega": the v1 C++ controller writes wheel
             speeds in rad/s (cruise 0.05357/r) into ctrl, so the gain is scaled to make that
             number mean the PWM the real firmware sends at cruise (cruise -> CRUISE_PWM 0.4).
             Same torque for the same physical PWM; only the meaning of ctrl differs.

The names the C++ DiffDriveAdapter looks up are kept (car_freejoint, left/right_wheel_drive,
left_wheel_collision, left/right_wheel); a single caster is named rear_support_body so the
adapter's skip list still keeps it out of the object list.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

# ---- chassis box ----
SPAWN_Z = 0.01                          # v1: body origin starts 1 cm up and drops onto the wheels
CHASSIS_CENTER_Z = 0.0375               # v1: 5 mm ground clearance + 3.25 cm half height (all bodies)


def _box_inertia(m, hx, hy, hz):
    return (m * (hy * hy + hz * hz) / 3, m * (hx * hx + hz * hz) / 3, m * (hx * hx + hy * hy) / 3)


BODY = {
    "v1": dict(
        chassis_half=(0.035, 0.035, 0.0325),            # v1: 7 x 7 x 6.5 cm
        marker_pos="0.034 0 0.0505",                    # v1 (visual only)
        chassis_mass=0.35,                              # v1
        chassis_inertia=(0.000266, 0.000266, 0.000286), # v1
        wheel_radius=0.015,                             # v1
        wheel_half_width=0.0005,                        # v1: 1 mm wheels
        wheel_y=0.0375,                                 # v1: 7.5 cm track
        wheel_mass=0.05,                                # v1
        wheel_inertia=(0.000003, 0.000006, 0.000003),   # v1
        caster_mass=0.025,                              # v1
        caster_inertia=(0.000001, 0.000001, 0.000001),  # v1
    ),
    "mvp": dict(
        chassis_half=(0.035, 0.035, 0.0325),            # user 2026-11-22: true car size 7x7;
                                                        # height is v1's 6.5 cm (placeholder, not published)
        marker_pos="0.034 0 0.0505",                    # v1 (visual only)
        # total 0.095 kg: published "just under 100 grams"; split below is a placeholder
        chassis_mass=0.087,                             # 0.095 - 2 wheels - caster
        chassis_inertia=(0.000266 * 0.087 / 0.35, 0.000266 * 0.087 / 0.35, 0.000286 * 0.087 / 0.35),
                                                        # placeholder: v1's 7 x 7 x 6.5 cm box inertia (a uniform
                                                        # box, rounded) scaled to the mass, as e2e_5dd3a674 ran it
        wheel_radius=0.016,                             # published: Pololu 32 mm wheels
        wheel_half_width=0.0035,                        # published: 7 mm wide
        wheel_y=0.02,                                   # fitted: 4.0 cm track from the in-place rotation
                                                        # rates (20.8 deg/s at PWM 0.15, 53.4 at 0.25); the
                                                        # log analysis's own table fit gave 3.96 cm
        wheel_mass=0.003,                               # placeholder: not published
        wheel_inertia=(2.04e-7, 3.84e-7, 2.04e-7),      # solid 3 g disc, r 1.6 cm, 7 mm wide
        caster_mass=0.002,                              # placeholder: not published
        caster_inertia=(1e-9, 1e-9, 1e-9),              # small ball; irrelevant without a joint
    ),
}
BODY["mvp_6x6"] = BODY["mvp"] | dict(
    chassis_half=(0.03, 0.03, 0.0325),                  # robot repo real.yaml:28 comment "physical 6x6 cm";
                                                        # height is v1's 6.5 cm (placeholder, not published)
    marker_pos="0.029 0 0.0505",                        # v1's marker moved onto the 6 cm front face (visual only)
    chassis_inertia=_box_inertia(0.087, 0.03, 0.03, 0.0325),
                                                        # placeholder: uniform solid box of the 6 x 6 x 6.5 cm chassis
)
ARMATURE = 0.0001       # v1 value, kept as a placeholder reflected rotor inertia (~1e-8 kg m^2 rotor x 120^2)

# ---- casters ----
CASTER_X = 0.03         # v1: 3 cm from the axle; v1 has one each side, the one-caster layout keeps the rear
CASTER_RADIUS = 0.0025  # v1; the published metal ball's size is not stated
ONE_CASTER_COM_X = -0.01    # placeholder: centre of mass 1 cm behind the axle, toward the single rear caster,
                            # so the car rests on three points; published build does not say where it sits
V1_CASTER_DAMPING = 0.0001  # v1 ball-joint damping

# ---- wheel joints and motor ----
V1_WHEEL_DAMPING = 0.01     # v1
STALL_TORQUE = 0.1765 * 3.7 / 4.5   # published Pololu 1512 (120:1): 1.8 kg cm stall at 4.5 V, scaled to the
                                    # 1S LiPo's 3.7 V nominal = 0.145 N m. The logs cannot see it (no stalls,
                                    # 15 Hz camera), so it stays at the datasheet value.
NO_LOAD_SPEED = 7.5         # fitted, rad/s at the wheel at PWM 1. The datasheet's 150 rpm at 4.5 V scales
                            # to 12.9 rad/s at 3.7 V; the real car delivers 58% of that.
GEARBOX_FRICTION_PWM = 0.06 # fitted. NOT the robot repo's 0.3 (navigation.py PHYSICAL_MOVE_COMMAND,
                            # "Nothing moves below it"): the logs show the car moving steadily at
                            # commanded 0.14-0.15 (22 reverses, 29 in-place rotations).
FRICTION_SOLREF = (0.004, 1.0)          # stiff gearbox friction: MuJoCo's default lets a wheel creep at
FRICTION_SOLIMP = (0.999, 0.9999, 0.001, 0.5, 2.0)   # 0.3-0.4 rad/s under the friction; this cuts it to 0.001
CRUISE_PWM = 0.4            # robot repo controller.yaml push.max_speed. Free driving at 0.4 in the logs runs
                            # 3.5-3.7 cm/s; the 5.125 cm/s in the namo config comes from a May 23 straight
                            # pure-pursuit session whose data is not on this filesystem.
CRUISE_MS = 0.05357         # v1 controller cruise (push_tracker_max_speed x kCarWheelMaxSpeedMs)
# Not modelled: the firmware's RIGHT_GAIN 0.87 right-wheel trim (robot repo controller.yaml), which
# offsets a motor mismatch on the real car; v2's two motors are identical, so it has nothing to cancel.

V1_KV, V1_FORCE, V1_CTRL = 0.75, 0.5, 25.0      # v1 velocity actuator

V2 = dict(body="mvp", caster="one_frictionless", wheel="v2", actuator="pwm", ctrl_units="pwm")
V1 = dict(body="v1", caster="v1", wheel="v1", actuator="v1", ctrl_units="pwm")


def _f(*xs):
    return " ".join(repr(float(x)) for x in xs)


def car_body_element(sx, sy, body="mvp", caster="one_frictionless", wheel="v2", actuator="pwm", **_) -> ET.Element:
    """The car <body>, built element for element in v1's order so all-v1 switches compile to v1."""
    B = BODY[body]
    hx, hy, hz = B["chassis_half"]
    cz = CHASSIS_CENTER_Z
    com_x = ONE_CASTER_COM_X if caster == "one_frictionless" else 0.0
    car = ET.Element("body", {"name": "car", "pos": f"{sx} {sy} {SPAWN_Z!r}"})
    ET.SubElement(car, "freejoint", {"name": "car_freejoint"})
    ET.SubElement(car, "inertial", {"pos": _f(com_x, 0, cz), "mass": repr(B["chassis_mass"]),
                                    "diaginertia": _f(*B["chassis_inertia"])})
    for side, x, rgba in (("front", hx / 2, "0.3 0.3 0.7 1"), ("rear", -hx / 2, "0.25 0.25 0.6 1")):
        ET.SubElement(car, "geom", {"name": f"{side}_chassis_collision", "type": "box", "pos": _f(x, 0, cz),
                                    "size": _f(hx / 2, hy, hz), "rgba": rgba})
    ET.SubElement(car, "geom", {"name": "front_marker", "type": "box", "pos": B["marker_pos"],
                                "size": "0.002 0.015 0.01", "rgba": "1 0.2 0.2 1", "contype": "0", "conaffinity": "0"})
    casters = (("rear", -CASTER_X, "0.7 0.1 0.1 1"),) if caster == "one_frictionless" else \
        (("rear", -CASTER_X, "0.7 0.1 0.1 1"), ("front", CASTER_X, "0.1 0.7 0.1 1"))
    for side, x, rgba in casters:
        b = ET.SubElement(car, "body", {"name": f"{side}_support_body", "pos": _f(x, 0, CASTER_RADIUS)})
        if caster == "v1":
            ET.SubElement(b, "joint", {"name": f"{side}_caster_joint", "type": "ball", "damping": repr(V1_CASTER_DAMPING)})
        ET.SubElement(b, "inertial", {"pos": "0 0 0", "mass": repr(B["caster_mass"]), "diaginertia": _f(*B["caster_inertia"])})
        g = {"name": f"{side}_support", "type": "sphere", "pos": "0 0 0", "size": repr(CASTER_RADIUS),
             "friction": "0 0 0", "rgba": rgba}
        if caster != "v1":
            g |= {"priority": "1", "condim": "1"}
        ET.SubElement(b, "geom", g)
    r = B["wheel_radius"]
    for side, y in (("left", B["wheel_y"]), ("right", -B["wheel_y"])):
        b = ET.SubElement(car, "body", {"name": f"{side}_wheel", "pos": _f(0, y, r)})
        ET.SubElement(b, "inertial", {"pos": "0 0 0", "mass": repr(B["wheel_mass"]), "diaginertia": _f(*B["wheel_inertia"])})
        j = {"name": f"{side}_wheel_joint", "type": "hinge", "axis": "0 1 0",
             "damping": repr(V1_WHEEL_DAMPING if wheel == "v1" else 0.0), "armature": repr(ARMATURE)}
        if actuator == "pwm":
            j |= {"frictionloss": repr(GEARBOX_FRICTION_PWM * STALL_TORQUE),
                  "solreffriction": _f(*FRICTION_SOLREF), "solimpfriction": _f(*FRICTION_SOLIMP)}
        ET.SubElement(b, "joint", j)
        ET.SubElement(b, "geom", {"name": f"{side}_wheel_collision", "type": "cylinder",
                                  "size": _f(r, B["wheel_half_width"]), "euler": "90 0 0", "rgba": "0.1 0.1 0.1 1"})
    return car


def actuator_element(body="mvp", actuator="pwm", ctrl_units="pwm", **_) -> ET.Element:
    act = ET.Element("actuator")
    cruise_omega = CRUISE_MS / BODY[body]["wheel_radius"]
    for side in ("left", "right"):
        if actuator == "v1":
            ET.SubElement(act, "velocity", {"name": f"{side}_wheel_drive", "joint": f"{side}_wheel_joint",
                                            "ctrlrange": _f(-V1_CTRL, V1_CTRL), "kv": repr(V1_KV),
                                            "forcerange": _f(-V1_FORCE, V1_FORCE)})
            continue
        pwm_per_ctrl = 1.0 if ctrl_units == "pwm" else CRUISE_PWM / cruise_omega
        ET.SubElement(act, "general", {
            "name": f"{side}_wheel_drive", "joint": f"{side}_wheel_joint",
            "ctrllimited": "true", "ctrlrange": _f(-1 / pwm_per_ctrl, 1 / pwm_per_ctrl),
            "forcelimited": "true", "forcerange": _f(-STALL_TORQUE, STALL_TORQUE),
            "gaintype": "fixed", "gainprm": repr(STALL_TORQUE * pwm_per_ctrl),
            "biastype": "affine", "biasprm": _f(0, 0, -STALL_TORQUE / NO_LOAD_SPEED),
        })
    return act


def car_body_xml(sx, sy, **opts) -> str:
    """Text form for generators that assemble scenes as strings (as CAR_BODY is used today)."""
    car = car_body_element(sx, sy, **(V2 | opts))
    ET.indent(car, space="  ", level=2)
    return "    " + ET.tostring(car, encoding="unicode")


def actuator_xml(**opts) -> str:
    act = actuator_element(**(V2 | opts))
    ET.indent(act, space="  ", level=1)
    return "  " + ET.tostring(act, encoding="unicode")
