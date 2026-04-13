#pragma once

#include "robot/robot_adapter.hpp"
#include <array>

namespace namo {

/// Adapter for the existing 2-DOF holonomic point robot.
///
/// The point robot uses two slide joints (x, y). Its qpos layout is:
///   qpos[0] = x displacement from body origin
///   qpos[1] = y displacement from body origin
/// Robot has no heading — theta is always 0.
///
/// Control: ctrl[0] = vx, ctrl[1] = vy (motor actuators on slide joints)
class HolonomicAdapter : public RobotAdapter {
public:
    /// @param init_pos  Initial robot position from XML (body origin in world frame).
    ///                  The point robot's qpos are displacements from this origin.
    explicit HolonomicAdapter(const std::array<double, 3>& init_pos);

    // Identity
    std::string get_body_name() const override { return "robot"; }
    std::string get_pose_source_name() const override { return "robot"; }
    std::vector<std::string> get_skip_body_names() const override {
        return {"robot", "world"};
    }

    // Pose
    std::array<double, 2> get_xy(const mjModel* m, const mjData* d) const override;
    double get_theta(const mjModel* m, const mjData* d) const override { return 0.0; }

    // Teleport
    void set_xy(const mjModel* m, mjData* d, double x, double y) const override;
    void set_se2(const mjModel* m, mjData* d,
                 double x, double y, double theta) const override;

    // Control
    void apply_control(const mjModel* m, mjData* d,
                       double vx, double vy) const override;
    void zero_control(const mjModel* m, mjData* d) const override;

private:
    std::array<double, 3> init_pos_;  // XML body origin (world frame)
};

} // namespace namo
