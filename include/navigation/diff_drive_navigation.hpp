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
        double linear_speed = 0.10;      // m/s during pure pursuit — slow for stability
        double angular_speed = 0.25;     // rad/s during rotation — low → small settle overshoot

        // Pure pursuit
        double lookahead = 0.10;         // m — larger = smoother steering

        // Exit thresholds (trigger zero-control + settle).
        // Set higher than the typical pure-pursuit oscillation amplitude so
        // we stop driving before entering the unstable regime near the goal.
        double xy_threshold = 0.03;      // m — exit when within 3cm of goal
        double theta_threshold = 0.10;   // rad — exit rotation at ~5.7°

        // Post-settle final tolerance (allows for coast distance + drift).
        double xy_tolerance = 0.05;      // m — 5cm final tolerance
        double theta_tolerance = 0.20;   // rad — ~11.5°

        // Smooth deceleration ramp before settle.
        // Commanded velocity decays linearly from full speed to 0 over this
        // many control steps. Eliminates jerk at phase exit.
        int decel_steps = 25;             // 25 ticks @ 0.01s = 0.25s

        // Settling
        int settle_steps = 20;
        double velocity_tolerance = 0.01;

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
