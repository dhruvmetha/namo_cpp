#include "navigation/holonomic_navigation.hpp"
#include "environment/namo_environment.hpp"

namespace namo {

NavigationResult HolonomicNavigation::execute(
    NAMOEnvironment& env,
    const std::vector<std::array<double, 2>>& path,
    double target_theta,
    const std::string& /*target_object*/) {

    NavigationResult result;
    if (path.empty()) {
        result.failure_reason = "empty path (goal unreachable)";
        return result;
    }

    // Teleport to final waypoint with target heading.
    // The push controller does its own placement-collision check after this.
    const auto& goal = path.back();
    env.set_robot_se2(goal[0], goal[1], target_theta);

    result.success = true;
    return result;
}

} // namespace namo
