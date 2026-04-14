#pragma once

#include "navigation/navigation_strategy.hpp"
#include <memory>

namespace namo {

class ConfigManager;

/// Three-phase diff-drive navigation:
///   Phase 1: rotate in place to face path start
///   Phase 2: pure pursuit along path (constant linear speed)
///   Phase 3: rotate in place to target heading
///
/// Each phase has:
///   - Constant control signal (no ramps) for clean dynamics
///   - Active termination check + zero-control settle phase
///   - Collision checks (robot vs walls, robot vs any movable)
///   - Timeout guard
///
/// Pure pursuit math follows the standard formulation:
///   α = angle to lookahead point in robot frame
///   κ = 2·sin(α) / L   (curvature)
///   (v, ω) = (linear_speed, κ·linear_speed)
///   (v_left, v_right) wheel speeds = (v - ω·b/2, v + ω·b/2) / r
///
/// Reference: Coulter 1992, "Implementation of the Pure Pursuit Path
/// Tracking Algorithm". Also mirrors ROS Nav2 regulated_pure_pursuit_controller
/// (without regulation; we use constant speed).
class DiffDriveNavigation : public NavigationStrategy {
public:
    struct Params {
        // Constant speeds during each phase
        double linear_speed = 0.10;      // m/s during straight driving
        double angular_speed = 1.0;      // rad/s — fast; overshoot is bounded by physics, not config

        // Pure pursuit
        double lookahead = 0.10;         // m — larger = smoother steering

        // Exit thresholds (trigger zero-control + wait).
        double xy_threshold = 0.015;     // m — exit drive when within 1.5cm of segment endpoint
        double theta_threshold = 0.05;   // rad — exit rotation at ~2.9°

        // Post-wait final tolerance.
        // theta_tolerance accommodates ~10° physics overshoot during wait
        // (wheels grip while chassis still rotating → small reverse bounce).
        double xy_tolerance = 0.05;      // m — 5cm
        double theta_tolerance = 0.22;   // rad — ~12.5°

        // Wait period after each phase (rotation or linear drive).
        // Zero control + step simulation, letting wheel brakes and caster
        // momentum dissipate before transitioning to the next phase.
        // Not active braking — just passive coast to rest.
        int wait_steps = 30;              // 0.30s of zero-control coast

        // Linear deceleration ramp before wait (smooth velocity transition).
        int decel_steps = 50;             // 0.5s ramp
        int settle_steps = 20;            // unused
        double velocity_tolerance = 0.01; // unused

        // Safety
        int max_nav_steps = 6000;         // control steps total
        double max_path_deviation = 0.15; // m; abort if drifting off path

        // Sharp-turn recovery: if the steering angle to lookahead exceeds
        // this, switch to in-place rotation until aligned again.
        double sharp_turn_threshold = 0.35;  // ~20 deg
        double sharp_turn_exit = 0.15;       // resume when under ~8.6 deg
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
