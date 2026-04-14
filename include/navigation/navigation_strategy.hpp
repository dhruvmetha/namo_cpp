#pragma once

#include <array>
#include <string>
#include <vector>

namespace namo {

class NAMOEnvironment;

/// Result of a navigation attempt.
struct NavigationResult {
    bool success = false;
    std::string failure_reason;
    std::string collision_object;   // If collision caused failure
    int steps_used = 0;
    /// Sampled robot trajectory during navigation: (x, y, theta, phase).
    /// Phase: 0 = rotate_start, 1 = pure_pursuit, 2 = rotate_end.
    std::vector<std::array<double, 4>> trajectory;
};

/// Abstract interface for getting the robot to a target (x, y, theta) pose.
///
/// Implementations:
///   - HolonomicNavigation: instant teleport (point robot).
///   - DiffDriveNavigation: rotate → pure pursuit → rotate (car).
class NavigationStrategy {
public:
    virtual ~NavigationStrategy() = default;

    /// Navigate the robot to the final waypoint in `path` with heading `target_theta`.
    /// The path is a sequence of (x, y) waypoints in world frame from current
    /// position to goal. Must not be empty; an empty path indicates goal is
    /// unreachable and the caller should fail the push attempt.
    /// `target_object` is the movable object we're about to push; collisions
    /// with it are ignored during the final rotation phase (near-contact expected).
    virtual NavigationResult execute(
        NAMOEnvironment& env,
        const std::vector<std::array<double, 2>>& path,
        double target_theta,
        const std::string& target_object = ""
    ) = 0;
};

} // namespace namo
