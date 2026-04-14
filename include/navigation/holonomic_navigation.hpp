#pragma once

#include "navigation/navigation_strategy.hpp"

namespace namo {

/// Holonomic (point robot) navigation: teleport to the final waypoint.
/// Preserves the original point-robot behavior exactly.
class HolonomicNavigation : public NavigationStrategy {
public:
    NavigationResult execute(
        NAMOEnvironment& env,
        const std::vector<std::array<double, 2>>& path,
        double target_theta,
        const std::string& target_object = ""
    ) override;
};

} // namespace namo
