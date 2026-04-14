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
        double angular_speed = 0.5;      // rad/s during in-place rotation

        // Pure pursuit
        double lookahead = 0.10;         // m — larger = smoother steering

        // Exit thresholds (trigger zero-control + settle)
        double xy_threshold = 0.01;      // m
        double theta_threshold = 0.05;   // rad

        // Post-settle final tolerance
        double xy_tolerance = 0.03;      // m — 3cm tolerance
        double theta_tolerance = 0.15;   // rad — ~8.6°

        // Settling
        int settle_steps = 20;
        double velocity_tolerance = 0.01;

        // Safety
        int max_nav_steps = 6000;         // control steps total
        double max_path_deviation = 0.15; // m; abort if drifting off path
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
