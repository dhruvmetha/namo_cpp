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

    // Get wheel radius from the wheel geom
    int wheel_geom = mj_name2id(m, mjOBJ_GEOM, "left_wheel_collision");
    if (wheel_geom >= 0) {
        wheel_radius_ = m->geom_size[wheel_geom * 3];  // cylinder radius
    } else {
        wheel_radius_ = 0.015;  // fallback: 1.5cm
    }

    // init_pos_[2] provides the initial z for teleportation
}

std::vector<std::string> DiffDriveAdapter::get_skip_body_names() const {
    // Skip the car body and all its children, plus world
    return {"world", "car", "rear_support_body", "front_support_body",
            "left_wheel", "right_wheel"};
}

std::array<double, 2> DiffDriveAdapter::get_xy(const mjModel* m, const mjData* d) const {
    // Freejoint qpos is displacement from body origin. Add init_pos for world frame.
    return {d->qpos[freejoint_qpos_adr_ + 0] + init_pos_[0],
            d->qpos[freejoint_qpos_adr_ + 1] + init_pos_[1]};
}

double DiffDriveAdapter::get_theta(const mjModel* m, const mjData* d) const {
    // Quaternion starts at qpos[adr+3] in MuJoCo freejoint: (x,y,z,w,qx,qy,qz)
    return quat_to_yaw(&d->qpos[freejoint_qpos_adr_ + 3]);
}

void DiffDriveAdapter::set_xy(const mjModel* m, mjData* d, double x, double y) const {
    // Convert world coordinates to qpos displacement from body origin
    d->qpos[freejoint_qpos_adr_ + 0] = x - init_pos_[0];
    d->qpos[freejoint_qpos_adr_ + 1] = y - init_pos_[1];
    // Keep current z and quaternion

    // Zero freejoint velocities
    for (int i = 0; i < 6; i++) {
        d->qvel[freejoint_qvel_adr_ + i] = 0.0;
    }
    mj_forward(const_cast<mjModel*>(m), d);
}

void DiffDriveAdapter::set_se2(const mjModel* m, mjData* d,
                                double x, double y, double theta) const {
    d->qpos[freejoint_qpos_adr_ + 0] = x - init_pos_[0];
    d->qpos[freejoint_qpos_adr_ + 1] = y - init_pos_[1];
    d->qpos[freejoint_qpos_adr_ + 2] = 0.0;  // keep at body-origin z (ground height)

    auto q = yaw_to_quat(theta);
    d->qpos[freejoint_qpos_adr_ + 3] = q[0];  // w
    d->qpos[freejoint_qpos_adr_ + 4] = q[1];  // x
    d->qpos[freejoint_qpos_adr_ + 5] = q[2];  // y
    d->qpos[freejoint_qpos_adr_ + 6] = q[3];  // z

    // Zero all robot velocities (freejoint + wheels)
    for (int i = 0; i < 6; i++) {
        d->qvel[freejoint_qvel_adr_ + i] = 0.0;
    }

    mj_forward(const_cast<mjModel*>(m), d);
}

void DiffDriveAdapter::apply_control(const mjModel* m, mjData* d,
                                      double vx, double vy) const {
    // Car heading was set at teleport. Both wheels get equal speed.
    // Use magnitude of (vx, vy) as the desired forward speed.
    double speed = std::sqrt(vx * vx + vy * vy);
    // Convert linear speed to angular wheel velocity: omega = speed / radius
    double omega = speed / wheel_radius_;
    d->ctrl[left_actuator_idx_] = omega;
    d->ctrl[right_actuator_idx_] = omega;
}

void DiffDriveAdapter::zero_control(const mjModel* m, mjData* d) const {
    d->ctrl[left_actuator_idx_] = 0.0;
    d->ctrl[right_actuator_idx_] = 0.0;
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
