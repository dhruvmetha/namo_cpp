#pragma once

#include "robot/robot_adapter.hpp"
#include <array>
#include <cmath>

namespace namo {

/// Adapter for the diff-drive car robot.
///
/// The car uses a freejoint (7 qpos: x,y,z,qw,qx,qy,qz) for the chassis
/// plus two hinge joints for left/right wheels. Actuators are velocity
/// controllers on the wheel joints.
///
/// For pushing: the car is teleported to the push start with correct heading,
/// then both wheels drive at equal velocity (straight-line push).
class DiffDriveAdapter : public RobotAdapter {
public:
    /// Construct from a loaded MuJoCo model.
    /// @param m       MuJoCo model (for joint/actuator lookups)
    /// @param init_pos  Initial car body world position (from d->xpos after mj_forward).
    ///                  Freejoint qpos is relative to this origin.
    DiffDriveAdapter(const mjModel* m, const std::array<double, 3>& init_pos);

    // Identity
    std::string get_body_name() const override { return "car"; }
    bool use_body_pose() const override { return true; }
    std::string get_pose_source_name() const override { return "car"; }
    std::vector<std::string> get_skip_body_names() const override;
    std::vector<std::string> get_collision_body_names() const override;

    // Pose
    std::array<double, 2> get_xy(const mjModel* m, const mjData* d) const override;
    double get_theta(const mjModel* m, const mjData* d) const override;
    double get_yaw_rate(const mjModel* m, const mjData* d) const override;
    double get_speed(const mjModel* m, const mjData* d) const override;

    // Teleport
    void set_xy(const mjModel* m, mjData* d, double x, double y) const override;
    void set_se2(const mjModel* m, mjData* d,
                 double x, double y, double theta) const override;

    // Control
    void apply_control(const mjModel* m, mjData* d,
                       double vx, double vy) const override;
    void zero_control(const mjModel* m, mjData* d) const override;

    // Diff-drive navigation
    bool is_diff_drive() const override { return true; }
    void apply_wheel_control(const mjModel* m, mjData* d,
                             double omega_left, double omega_right) const override;
    double get_wheelbase() const override { return wheelbase_; }
    double get_wheel_radius() const override { return wheel_radius_; }

private:
    int freejoint_qpos_adr_;   // qpos index for freejoint (x,y,z,qw,qx,qy,qz)
    int freejoint_qvel_adr_;   // qvel index for freejoint (6 DOF)
    int left_actuator_idx_;    // ctrl index for left wheel
    int right_actuator_idx_;   // ctrl index for right wheel
    double wheel_radius_;      // wheel radius for velocity conversion
    double wheelbase_;         // distance between left and right wheels
    std::array<double, 3> init_pos_;  // body origin in world frame (qpos is relative to this)

    /// Convert yaw angle to MuJoCo quaternion (w,x,y,z) for rotation around z-axis
    static std::array<double, 4> yaw_to_quat(double theta);
    /// Extract yaw from MuJoCo quaternion (w,x,y,z)
    static double quat_to_yaw(const double* quat);
};

} // namespace namo
