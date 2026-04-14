#pragma once

#include "robot/robot_adapter.hpp"
#include <array>
#include <cmath>

namespace namo {

/// Adapter for the diff-drive car robot.
///
/// The car uses a freejoint (7 qpos: x,y,z,qw,qx,qy,qz) for the chassis
/// plus two hinge joints for left/right wheels. Actuators are torque
/// motors on the wheel joints; this adapter runs a per-wheel PI velocity
/// loop (anti-windup clamp) matching the standard diff-drive firmware
/// pattern (ROS2 velocity_controllers, Drake, Isaac).
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

    // Pose
    std::array<double, 2> get_xy(const mjModel* m, const mjData* d) const override;
    double get_theta(const mjModel* m, const mjData* d) const override;

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
    void inner_control_update(const mjModel* m, mjData* d) const override;
    double get_wheelbase() const override { return wheelbase_; }
    double get_wheel_radius() const override { return wheel_radius_; }

private:
    int freejoint_qpos_adr_;   // qpos index for freejoint (x,y,z,qw,qx,qy,qz)
    int freejoint_qvel_adr_;   // qvel index for freejoint (6 DOF)
    int left_actuator_idx_;    // ctrl index for left wheel
    int right_actuator_idx_;   // ctrl index for right wheel
    int left_wheel_qvel_adr_;  // qvel index for left wheel hinge (measures ω)
    int right_wheel_qvel_adr_; // qvel index for right wheel hinge
    double wheel_radius_;      // wheel radius for velocity conversion
    double wheelbase_;         // distance between left and right wheels
    std::array<double, 3> init_pos_;  // body origin in world frame (qpos is relative to this)

    // PI velocity controller state (per wheel).
    // Mutable: control methods are logically const but must update integrator.
    mutable double integral_left_ = 0.0;
    mutable double integral_right_ = 0.0;
    mutable double cmd_omega_left_ = 0.0;
    mutable double cmd_omega_right_ = 0.0;

    // PI gains — tuned for the 7cm car (wheel inertia ~6e-6 kg·m², radius 1.5cm,
    // max static friction torque ≈ 0.033 Nm). Gains are deliberately small
    // compared to the old MuJoCo velocity actuator's kv=0.75 because:
    //   - MuJoCo's velocity actuator uses implicit discretization internally
    //     (stable at high gain). Our PI is explicit, so high gain saturates
    //     torque, slips the wheel, and the next tick reverses into braking.
    //   - We want τ_initial ≈ 0.05 Nm (just above slip threshold) to let the
    //     car accelerate without breaking static friction → Kp ≈ 0.05 / ω_target.
    // kControlDt is the outer control period (NAMO applies ctrl once per 10ms
    // and then runs 5 physics steps); the PI integrator advances at that rate.
    // Inner loop runs at physics rate. Kp=0.75 matches the old MuJoCo
    // velocity actuator's kv gain. Ki is set to a small value — the large
    // gains tried initially caused asymmetric wheel drift (each wheel's
    // integral grows independently → heading drift during drive).
    static constexpr double kPiKp = 0.75;
    static constexpr double kPiKi = 0.5;
    static constexpr double kTauMax = 0.3;   // matches forcerange in XML

    /// Convert yaw angle to MuJoCo quaternion (w,x,y,z) for rotation around z-axis
    static std::array<double, 4> yaw_to_quat(double theta);
    /// Extract yaw from MuJoCo quaternion (w,x,y,z)
    static double quat_to_yaw(const double* quat);
};

} // namespace namo
