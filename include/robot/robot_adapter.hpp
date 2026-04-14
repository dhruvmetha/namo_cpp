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
    /// Whether to use body pose (true) or geom pose (false) for position queries.
    /// Car uses body pose (no single geom at center); point robot uses geom pose.
    virtual bool use_body_pose() const { return false; }
    /// Name used for pose queries — body name if use_body_pose(), geom name otherwise.
    virtual std::string get_pose_source_name() const = 0;
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

    // ── Diff-drive navigation support ─────────────────────────────────
    /// Whether this robot supports diff-drive style navigation
    /// (rotate-drive-rotate). Holonomic robots return false (use teleport).
    virtual bool is_diff_drive() const { return false; }

    /// Apply left/right wheel angular velocities [rad/s].
    /// Only meaningful for diff-drive robots; no-op for holonomic.
    virtual void apply_wheel_control(const mjModel* m, mjData* d,
                                     double omega_left, double omega_right) const {
        (void)m; (void)d; (void)omega_left; (void)omega_right;
    }

    /// Wheelbase (distance between wheels) [m]. Used by pure pursuit.
    virtual double get_wheelbase() const { return 0.0; }

    /// Wheel radius [m]. Used to convert linear speed to wheel ω.
    virtual double get_wheel_radius() const { return 0.0; }

    /// Inner control update — called once per physics step from step_control_tick.
    /// Diff-drive adapters override this to run their PI velocity loop at the
    /// physics rate (matches real firmware). Holonomic is a no-op.
    virtual void inner_control_update(const mjModel* m, mjData* d) const {
        (void)m; (void)d;
    }
};

} // namespace namo
