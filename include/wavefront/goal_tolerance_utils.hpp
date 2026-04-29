#pragma once

#include <algorithm>
#include <vector>

namespace namo {

constexpr double kDefaultWavefrontRobotRadiusM = 0.15;
constexpr double kDefaultWavefrontTier1MarginM = 0.005;

inline double compute_goal_tolerance_m(
    const std::vector<double>& robot_size,
    double tier1_inflation_margin_m) {

    double robot_radius_m = kDefaultWavefrontRobotRadiusM;
    if (!robot_size.empty()) {
        if (robot_size.size() >= 2) {
            robot_radius_m = std::max(robot_size[0], robot_size[1]);
        } else {
            robot_radius_m = robot_size[0];
        }
    }

    if (robot_radius_m <= 0.0) {
        robot_radius_m = kDefaultWavefrontRobotRadiusM;
    }

    const double margin_m =
        (tier1_inflation_margin_m >= 0.0) ? tier1_inflation_margin_m : kDefaultWavefrontTier1MarginM;

    return robot_radius_m + margin_m;
}

}  // namespace namo
