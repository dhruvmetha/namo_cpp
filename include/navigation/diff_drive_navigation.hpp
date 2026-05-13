#pragma once

#include "navigation/navigation_strategy.hpp"
#include <memory>

namespace namo {

class ConfigManager;

/// Three-phase diff-drive navigation, closed-loop:
///   Phase 1: PD on chassis yaw to face path start
///   Phase 2: P on along-path distance (per straight segment)
///   Phase 3: PD on chassis yaw to push target heading
///
/// Each phase commands the wheel-velocity inner loop with a continuously
/// updated reference, so control never jumps to zero. This avoids the
/// brake-induced friction reversal that produced rebound under bang-bang
/// control with a passive wait phase.
///
/// Outer-loop PD on yaw (rotate phases):
///   ω_cmd = clamp(Kp_yaw · err - Kd_yaw · yaw_rate, ±angular_speed)
///   wheel_left  = -ω_cmd · b/2 / r
///   wheel_right = +ω_cmd · b/2 / r
/// Outer-loop P on along-distance (drive phase):
///   v_cmd = clamp(Kp_drive · along, 0, linear_speed)
///   wheel_left = wheel_right = v_cmd / r
///
/// Reference: standard joint-impedance / virtual stiffness-damping form
/// (Hogan 1985). Inner wheel-velocity loop is the existing <velocity kv=…>
/// MuJoCo actuator; outer loop is what this file implements.
class DiffDriveNavigation : public NavigationStrategy {
public:
    /// Controller for rotate / drive phases.
    /// PD: outer-loop PD on chassis pose. Reactive feedback. Works on
    ///     high-friction plants; prone to limit-cycle on low friction.
    /// TRAPEZOIDAL: open-loop velocity profile (ramp-up / cruise / ramp-down)
    ///     respecting plant brake limit α_max. No reactive feedback. Robust
    ///     to actuator + contact saturation. Default.
    enum class Mode { PD, TRAPEZOIDAL };

    struct Params {
        // Choose controller for rotate_in_place / drive_straight_to
        Mode mode = Mode::TRAPEZOIDAL;

        // ── Saturation / max speeds ──────────────────────────────────────
        double linear_speed = 0.10;      // m/s — drive saturation
        double angular_speed = 1.0;      // rad/s — rotate saturation

        // ── Trapezoidal profile ──────────────────────────────────────────
        // Maximum chassis angular acceleration / deceleration to use when
        // shaping commanded ω(t). Must be ≤ what the plant can deliver
        // smoothly. Empirical from brake test: ~5 rad/s² is well within
        // the smooth regime on the matched-friction plant (peak step-brake
        // exhibits ~25 Hz oscillation; staying well below that bandwidth
        // avoids exciting it).
        double alpha_max = 5.0;          // rad/s²
        // Same idea for linear motion.
        double accel_max = 0.5;          // m/s²

        // ── Outer-loop gains ─────────────────────────────────────────────
        // PD on chassis yaw → commanded chassis ω. Kd dominates because the
        // inner wheel-velocity loop (kv=0.75) introduces phase lag in the
        // loaded env; higher damping suppresses the resulting oscillation.
        double Kp_yaw = 3.0;             // [1/s] — proportional yaw gain
        double Kd_yaw = 3.0;             // [-]  — derivative gain (damping)
        // P on along-path distance → commanded forward velocity
        double Kp_drive = 5.0;           // [1/s] — proportional drive gain

        // ── Convergence ──────────────────────────────────────────────────
        // A rotation phase exits when |yaw_err| AND |yaw_rate| are both small.
        double theta_converged = 0.01;   // rad — ~0.6°
        double rate_converged  = 0.05;   // rad/s
        // A drive phase exits when along-distance AND chassis speed are both small.
        double xy_converged    = 0.005;  // m — 5mm
        double speed_converged = 0.01;   // m/s

        // ── Skip-if-already-at-goal (top of execute()) ───────────────────
        double xy_tolerance = 0.05;      // m
        double theta_tolerance = 0.22;   // rad

        // ── Pure pursuit (legacy, not on the live path) ──────────────────
        double lookahead = 0.10;
        double sharp_turn_exit = 0.15;

        // ── Safety ───────────────────────────────────────────────────────
        int max_nav_steps = 6000;
        double max_path_deviation = 0.15;

        // Path segmentation: split waypoints when heading changes by more
        // than this, so each segment is a straight corridor.
        double sharp_turn_threshold = 0.35;
    };

    explicit DiffDriveNavigation(const Params& params);
    explicit DiffDriveNavigation(std::shared_ptr<ConfigManager> config);

    NavigationResult execute(
        NAMOEnvironment& env,
        const std::vector<std::array<double, 2>>& path,
        double target_theta,
        const std::string& target_object = ""
    ) override;

    const Params& params() const { return params_; }

private:
    Params params_;
};

} // namespace namo
