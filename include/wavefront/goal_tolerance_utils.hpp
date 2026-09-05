#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

namespace namo {

constexpr double kDefaultWavefrontRobotRadiusM = 0.15;
// Fallback tier-1 inflation margin when wavefront_inflation.yaml is not
// found. Must equal tier1.base_inflation_margin_m in config/wavefront_inflation.yaml
// (1 mm, the real_2mov study value) so a missing sidecar cannot make the
// C++ grid disagree with the robot_control grid (BUG-001).
constexpr double kDefaultWavefrontTier1MarginM = 0.001;

inline double compute_rotation_safe_robot_radius_m(const std::vector<double>& robot_size) {
    // Axis-aligned max(hx, hy) — the pre-merge local convention. The function
    // name is kept for API compatibility but no longer returns a diagonal;
    // the diagonal proved too conservative for our scenes (effective inflation
    // of 4.95 cm for a 7×7 robot crowded out push approaches).
    double robot_radius_m = kDefaultWavefrontRobotRadiusM;
    if (!robot_size.empty()) {
        if (robot_size.size() >= 2) {
            const double hx = std::abs(robot_size[0]);
            const double hy = std::abs(robot_size[1]);
            robot_radius_m = std::max(hx, hy);
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
