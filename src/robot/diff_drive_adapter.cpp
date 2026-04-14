#include "robot/diff_drive_adapter.hpp"
#include <stdexcept>
#include <string>

namespace namo {

DiffDriveAdapter::DiffDriveAdapter(const mjModel* m, const std::array<double, 3>& init_pos)
    : init_pos_(init_pos) {
    // Look up freejoint
    int fj_id = mj_name2id(m, mjOBJ_JOINT, "car_freejoint");
    if (fj_id < 0) {
        throw std::runtime_error("DiffDriveAdapter: joint 'car_freejoint' not found in model");
    }
    freejoint_qpos_adr_ = m->jnt_qposadr[fj_id];
    freejoint_qvel_adr_ = m->jnt_dofadr[fj_id];

    // Look up actuators by name
    left_actuator_idx_ = mj_name2id(m, mjOBJ_ACTUATOR, "left_wheel_drive");
    right_actuator_idx_ = mj_name2id(m, mjOBJ_ACTUATOR, "right_wheel_drive");
    if (left_actuator_idx_ < 0 || right_actuator_idx_ < 0) {
        throw std::runtime_error("DiffDriveAdapter: wheel actuators not found in model");
    }

    // Look up wheel hinge joints — used to read actual ω for the PI loop.
    int left_wj = mj_name2id(m, mjOBJ_JOINT, "left_wheel_joint");
    int right_wj = mj_name2id(m, mjOBJ_JOINT, "right_wheel_joint");
    if (left_wj < 0 || right_wj < 0) {
        throw std::runtime_error("DiffDriveAdapter: wheel hinge joints not found");
    }
    left_wheel_qvel_adr_ = m->jnt_dofadr[left_wj];
    right_wheel_qvel_adr_ = m->jnt_dofadr[right_wj];

    // Get wheel radius from the wheel geom
    int wheel_geom = mj_name2id(m, mjOBJ_GEOM, "left_wheel_collision");
    if (wheel_geom >= 0) {
        wheel_radius_ = m->geom_size[wheel_geom * 3];  // cylinder radius
    } else {
        wheel_radius_ = 0.015;  // fallback: 1.5cm
    }

    // Get wheelbase from wheel body positions (distance between left and right)
    int left_body = mj_name2id(m, mjOBJ_BODY, "left_wheel");
    int right_body = mj_name2id(m, mjOBJ_BODY, "right_wheel");
    if (left_body >= 0 && right_body >= 0) {
        // body_pos is relative to parent — for wheels, y offsets differ by wheelbase
        double ly = m->body_pos[left_body * 3 + 1];
        double ry = m->body_pos[right_body * 3 + 1];
        wheelbase_ = std::abs(ly - ry);
    } else {
        wheelbase_ = 0.075;  // fallback: 7.5cm for 7cm car
    }

    // init_pos_[2] provides the initial z for teleportation
}

std::vector<std::string> DiffDriveAdapter::get_skip_body_names() const {
    // Skip the car body and all its children, plus world
    return {"world", "car", "rear_support_body", "front_support_body",
            "left_wheel", "right_wheel"};
}

std::array<double, 2> DiffDriveAdapter::get_xy(const mjModel* m, const mjData* d) const {
    // Freejoint qpos is ABSOLUTE world position (not relative to body origin)
    return {d->qpos[freejoint_qpos_adr_ + 0],
            d->qpos[freejoint_qpos_adr_ + 1]};
}

double DiffDriveAdapter::get_theta(const mjModel* m, const mjData* d) const {
    // Quaternion starts at qpos[adr+3] in MuJoCo freejoint: (x,y,z,w,qx,qy,qz)
    return quat_to_yaw(&d->qpos[freejoint_qpos_adr_ + 3]);
}

void DiffDriveAdapter::set_xy(const mjModel* m, mjData* d, double x, double y) const {
    // Freejoint qpos is absolute world position
    d->qpos[freejoint_qpos_adr_ + 0] = x;
    d->qpos[freejoint_qpos_adr_ + 1] = y;
    // Keep current z and quaternion

    // Zero freejoint velocities
    for (int i = 0; i < 6; i++) {
        d->qvel[freejoint_qvel_adr_ + i] = 0.0;
    }
    mj_forward(const_cast<mjModel*>(m), d);
}

void DiffDriveAdapter::set_se2(const mjModel* m, mjData* d,
                                double x, double y, double theta) const {
    d->qpos[freejoint_qpos_adr_ + 0] = x;
    d->qpos[freejoint_qpos_adr_ + 1] = y;
    d->qpos[freejoint_qpos_adr_ + 2] = init_pos_[2];  // maintain ground height

    auto q = yaw_to_quat(theta);
    d->qpos[freejoint_qpos_adr_ + 3] = q[0];  // w
    d->qpos[freejoint_qpos_adr_ + 4] = q[1];  // x
    d->qpos[freejoint_qpos_adr_ + 5] = q[2];  // y
    d->qpos[freejoint_qpos_adr_ + 6] = q[3];  // z

    // Zero all robot velocities (freejoint + wheels)
    for (int i = 0; i < 6; i++) {
        d->qvel[freejoint_qvel_adr_ + i] = 0.0;
    }

    // Teleport invalidates integrator state.
    cmd_omega_left_ = 0.0;
    cmd_omega_right_ = 0.0;
    integral_left_ = 0.0;
    integral_right_ = 0.0;

    mj_forward(const_cast<mjModel*>(m), d);
}

void DiffDriveAdapter::apply_control(const mjModel* m, mjData* d,
                                      double vx, double vy) const {
    // Car heading was set at teleport. Both wheels get equal speed.
    // Use magnitude of (vx, vy) as the desired forward speed.
    double speed = std::sqrt(vx * vx + vy * vy);
    double omega = speed / wheel_radius_;
    apply_wheel_control(m, d, omega, omega);
}

void DiffDriveAdapter::zero_control(const mjModel* m, mjData* d) const {
    // Motor actuators: zero ctrl = zero torque (passive coast). Also clear
    // the commanded ω and integrator state so the PI inner loop stops
    // actively driving and the wheels free-coast.
    (void)m;
    d->ctrl[left_actuator_idx_] = 0.0;
    d->ctrl[right_actuator_idx_] = 0.0;
    cmd_omega_left_ = 0.0;
    cmd_omega_right_ = 0.0;
    integral_left_ = 0.0;
    integral_right_ = 0.0;
}

void DiffDriveAdapter::apply_wheel_control(const mjModel* m, mjData* d,
                                            double omega_left, double omega_right) const {
    // Store commanded wheel ω. The PI loop runs in inner_control_update
    // every physics step (standard fast inner-velocity / slow outer-planner
    // cascade). Immediately evaluate once so this call never leaves stale
    // torque in ctrl.
    cmd_omega_left_ = omega_left;
    cmd_omega_right_ = omega_right;
    inner_control_update(m, d);
}

void DiffDriveAdapter::inner_control_update(const mjModel* m, mjData* d) const {
    // Per-wheel PI velocity controller with anti-windup (clamp on integral),
    // running at the physics rate matches ros2_control velocity_controllers,
    // Drake's JointVelocityController, and typical motor-driver firmware.
    //   err = ω_cmd - ω_measured
    //   I   = clamp(I + err·dt, ±τ_max/Ki)
    //   τ   = clamp(Kp·err + Ki·I, ±τ_max)
    const double dt = m->opt.timestep;
    const double I_max = kTauMax / kPiKi;

    // --- Left wheel ---
    double actual_l = d->qvel[left_wheel_qvel_adr_];
    double err_l = cmd_omega_left_ - actual_l;
    integral_left_ += err_l * dt;
    if (integral_left_ > I_max) integral_left_ = I_max;
    else if (integral_left_ < -I_max) integral_left_ = -I_max;
    double tau_l = kPiKp * err_l + kPiKi * integral_left_;
    if (tau_l > kTauMax) tau_l = kTauMax;
    else if (tau_l < -kTauMax) tau_l = -kTauMax;
    d->ctrl[left_actuator_idx_] = tau_l;

    // --- Right wheel ---
    double actual_r = d->qvel[right_wheel_qvel_adr_];
    double err_r = cmd_omega_right_ - actual_r;
    integral_right_ += err_r * dt;
    if (integral_right_ > I_max) integral_right_ = I_max;
    else if (integral_right_ < -I_max) integral_right_ = -I_max;
    double tau_r = kPiKp * err_r + kPiKi * integral_right_;
    if (tau_r > kTauMax) tau_r = kTauMax;
    else if (tau_r < -kTauMax) tau_r = -kTauMax;
    d->ctrl[right_actuator_idx_] = tau_r;
}

std::array<double, 4> DiffDriveAdapter::yaw_to_quat(double theta) {
    // Rotation around z-axis: q = (cos(θ/2), 0, 0, sin(θ/2))
    double half = theta * 0.5;
    return {std::cos(half), 0.0, 0.0, std::sin(half)};
}

double DiffDriveAdapter::quat_to_yaw(const double* quat) {
    // MuJoCo quaternion: (w, x, y, z)
    // yaw = atan2(2(wz + xy), 1 - 2(y² + z²))
    double w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    return std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
}

} // namespace namo
