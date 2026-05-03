#include "environment/namo_environment.hpp"
#include "wavefront/wavefront_planner.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namo::NAMOEnvironment;
using namo::WavefrontPlanner;

constexpr double kResolution = 0.02;
constexpr double kEpsilon = 1e-9;

std::filesystem::path find_repo_root() {
    std::filesystem::path cursor = std::filesystem::absolute(__FILE__).parent_path();
    for (int i = 0; i < 8; ++i) {
        if (std::filesystem::exists(cursor / "include" / "wavefront" / "wavefront_planner.hpp") &&
            std::filesystem::exists(cursor / "python" / "tests" / "data" / "geometric_transport_priority_fixture.xml")) {
            return cursor;
        }
        cursor = cursor.parent_path();
    }
    throw std::runtime_error("Failed to locate repository root from test source path");
}

bool nearly_equal(double a, double b, double eps = kEpsilon) {
    return std::abs(a - b) <= eps;
}

bool validate_case(
    WavefrontPlanner& planner,
    NAMOEnvironment& env,
    const std::array<double, 2>& start,
    const std::array<double, 2>& goal) {

    planner.update_wavefront(env, std::vector<double>{start[0], start[1]});
    const auto path = planner.extract_path(start, goal);
    if (path.empty()) {
        std::cerr << "Path is empty for start=(" << start[0] << ", " << start[1]
                  << ") goal=(" << goal[0] << ", " << goal[1] << ")\n";
        return false;
    }

    const auto& bounds = planner.get_bounds();
    const double resolution = planner.get_resolution();

    const int start_gx = planner.world_to_grid_x(start[0]);
    const int start_gy = planner.world_to_grid_y(start[1]);
    const int goal_gx = planner.world_to_grid_x(goal[0]);
    const int goal_gy = planner.world_to_grid_y(goal[1]);

    int prev_gx = -1;
    int prev_gy = -1;
    for (size_t i = 0; i < path.size(); ++i) {
        const auto& w = path[i];
        const int gx = planner.world_to_grid_x(w[0]);
        const int gy = planner.world_to_grid_y(w[1]);

        const double center_x = bounds[0] + (static_cast<double>(gx) + 0.5) * resolution;
        const double center_y = bounds[2] + (static_cast<double>(gy) + 0.5) * resolution;

        if (!nearly_equal(w[0], center_x) || !nearly_equal(w[1], center_y)) {
            std::cerr << "Waypoint not at cell center at index " << i
                      << ": waypoint=(" << w[0] << ", " << w[1]
                      << "), center=(" << center_x << ", " << center_y << ")\n";
            return false;
        }

        if (i == 0 && (gx != start_gx || gy != start_gy)) {
            std::cerr << "First waypoint cell does not match start cell: got=("
                      << gx << ", " << gy << "), expected=("
                      << start_gx << ", " << start_gy << ")\n";
            return false;
        }

        if (i > 0) {
            const int dx = std::abs(gx - prev_gx);
            const int dy = std::abs(gy - prev_gy);
            if (dx > 1 || dy > 1 || (dx == 0 && dy == 0)) {
                std::cerr << "Non 8-connected step between waypoint " << (i - 1)
                          << " and " << i << ": prev=(" << prev_gx << ", " << prev_gy
                          << "), cur=(" << gx << ", " << gy << ")\n";
                return false;
            }
        }

        prev_gx = gx;
        prev_gy = gy;
    }

    const auto& final_w = path.back();
    const int final_gx = planner.world_to_grid_x(final_w[0]);
    const int final_gy = planner.world_to_grid_y(final_w[1]);
    if (final_gx != goal_gx || final_gy != goal_gy) {
        std::cerr << "Final waypoint cell mismatch: final=(" << final_gx << ", " << final_gy
                  << "), goal_cell=(" << goal_gx << ", " << goal_gy << ")\n";
        return false;
    }

    return true;
}

}  // namespace

int main() {
    try {
        setenv("MUJOCO_GL", "egl", 0);

        const auto repo_root = find_repo_root();
        const auto xml_path = repo_root / "python" / "tests" / "data" / "geometric_transport_priority_fixture.xml";

        if (!std::filesystem::exists(xml_path)) {
            std::cerr << "Missing fixture XML: " << xml_path << "\n";
            return 1;
        }

        NAMOEnvironment env(xml_path.string(), false, false);
        const auto robot_half_extents = env.get_robot_planning_half_extents();
        WavefrontPlanner planner(
            kResolution,
            env,
            std::vector<double>{robot_half_extents[0], robot_half_extents[1]},
            0.005);

        // Straight, diagonal, and corridor-like goals in one scaled-scene test binary.
        const std::vector<std::pair<std::array<double, 2>, std::array<double, 2>>> cases = {
            {{0.0, 0.0}, {1.0, 0.0}},
            {{0.0, 0.0}, {3.5, 0.6}},
            {{0.0, 0.0}, {4.0, 0.0}},
            {{0.0, 0.0}, {3.5, -0.6}},
        };

        for (const auto& [start, goal] : cases) {
            if (!validate_case(planner, env, start, goal)) {
                return 1;
            }
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
}
