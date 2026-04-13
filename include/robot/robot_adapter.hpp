#pragma once

#include <array>
#include <string>
#include <vector>

extern "C" {
#include "mujoco/mujoco.h"
}

namespace namo {

/// Abstract interface for robot-specific behavior.
///
/// Encapsulates how to read/write robot pose, apply control, and identify
/// robot bodies in the MuJoCo model. Implementations exist for:
///   - HolonomicAdapter: 2-DOF point robot (slide joints)
///   - DiffDriveAdapter: diff-drive car (freejoint + wheel hinges)
class RobotAdapter {
public:
    virtual ~RobotAdapter() = default;

    // ── Identity ──────────────────────────────────────────────────────
    /// Primary MuJoCo body name (e.g. "robot" or "car")
    virtual std::string get_body_name() const = 0;
    /// Geom name used for pose queries via get_geom_pose()
    virtual std::string get_pose_geom_name() const = 0;
    /// Body names to skip when enumerating environment objects
    virtual std::vector<std::string> get_skip_body_names() const = 0;

    // ── Pose (SE2) ────────────────────────────────────────────────────
    /// Read robot (x, y) from MuJoCo state
    virtual std::array<double, 2> get_xy(const mjModel* m, const mjData* d) const = 0;
    /// Read robot heading (radians). Returns 0 for holonomic.
    virtual double get_theta(const mjModel* m, const mjData* d) const = 0;

    // ── Teleport ──────────────────────────────────────────────────────
    /// Place robot at world-frame (x, y), preserving current theta.
    virtual void set_xy(const mjModel* m, mjData* d, double x, double y) const = 0;
    /// Place robot at world-frame (x, y, theta).
    virtual void set_se2(const mjModel* m, mjData* d,
                         double x, double y, double theta) const = 0;

    // ── Control ───────────────────────────────────────────────────────
    /// Apply a push control signal. The caller provides the desired push
    /// direction as (vx, vy). The adapter maps this to actuator commands:
    ///   - Holonomic: ctrl = (vx, vy)
    ///   - Diff-drive: both wheels at ||vx,vy|| / wheel_radius (heading
    ///     was set at teleport time, so direction is implicit)
    virtual void apply_control(const mjModel* m, mjData* d,
                               double vx, double vy) const = 0;
    /// Zero all robot actuators.
    virtual void zero_control(const mjModel* m, mjData* d) const = 0;
};

} // namespace namo
