#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

namespace namo {

constexpr double kDefaultWavefrontRobotRadiusM = 0.15;
constexpr double kDefaultWavefrontTier1MarginM = 0.005;

inline double compute_rotation_safe_robot_radius_m(const std::vector<double>& robot_size) {
    double robot_radius_m = kDefaultWavefrontRobotRadiusM;
    if (!robot_size.empty()) {
        if (robot_size.size() >= 2) {
            const double hx = std::abs(robot_size[0]);
            const double hy = std::abs(robot_size[1]);
            robot_radius_m = std::sqrt(hx * hx + hy * hy);
        } else {
            robot_radius_m = std::abs(robot_size[0]);
        }
    }

    if (robot_radius_m <= 0.0) {
        robot_radius_m = kDefaultWavefrontRobotRadiusM;
    }
    return robot_radius_m;
}

inline double compute_wavefront_inflation_radius_m(
    const std::vector<double>& robot_size,
    double tier1_inflation_margin_m) {
    const double robot_radius_m = compute_rotation_safe_robot_radius_m(robot_size);
    const double margin_m =
        (tier1_inflation_margin_m >= 0.0) ? tier1_inflation_margin_m : kDefaultWavefrontTier1MarginM;
    return robot_radius_m + margin_m;
}

inline double compute_goal_tolerance_m(
    const std::vector<double>& robot_size,
    double tier1_inflation_margin_m) {
    return compute_wavefront_inflation_radius_m(robot_size, tier1_inflation_margin_m);
}

}  // namespace namo
