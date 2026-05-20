/**
 * @file test_nav_state_machine.cpp
 * @brief Pure free-space A → B navigation test using the C++ pipeline.
 *
 * No Python, no push, no skill. Just:
 *   - Load NAMOEnvironment with a scene XML.
 *   - Place the car at start (x, y, theta).
 *   - Compute wavefront from start to goal via WavefrontPlanner.
 *   - Extract the 2D path.
 *   - Run DiffDriveNavigation::execute on it (state machine: rotate-drive
 *     per segment + final rotate).
 *   - Print outcome and final pose error.
 *
 * Use NAMO_QPOS_DUMP=path NAMO_NAV_LOG=1 to capture trajectory + path
 * for video render with render_nav_video.py.
 *
 * Usage:
 *   test_nav_state_machine <xml_path> <start_x> <start_y> <start_theta_deg>
 *                          <goal_x> <goal_y> <target_theta_deg>
 */

#include "environment/namo_environment.hpp"
#include "wavefront/wavefront_planner.hpp"
#include "navigation/diff_drive_navigation.hpp"
#include "navigation/holonomic_navigation.hpp"
#include "navigation/qpos_dump.hpp"
#include "robot/robot_adapter.hpp"
#include "config/config_manager.hpp"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <string>

extern "C" {
#include "mujoco/mujoco.h"
}

namespace {

double wrap_angle(double a) {
    while (a >  M_PI) a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 8) {
        std::cerr << "Usage: " << argv[0]
                  << " <xml_path> <start_x> <start_y> <start_theta_deg>"
                  << " <goal_x> <goal_y> <target_theta_deg>\n";
        return 1;
    }
    const std::string xml_path = argv[1];
    const double sx = std::atof(argv[2]);
    const double sy = std::atof(argv[3]);
    const double s_theta = std::atof(argv[4]) * M_PI / 180.0;
    const double gx = std::atof(argv[5]);
    const double gy = std::atof(argv[6]);
    const double t_theta = std::atof(argv[7]) * M_PI / 180.0;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Loading: " << xml_path << "\n";
    std::cout << "  start: (" << sx << ", " << sy << "), heading "
              << argv[4] << " deg\n";
    std::cout << "  goal : (" << gx << ", " << gy << "), heading "
              << argv[7] << " deg\n\n";

    // 1. Load environment (no visualization)
    namo::NAMOEnvironment env(xml_path, /*visualize=*/false);

    // 2. Place car at start
    env.set_robot_se2(sx, sy, s_theta);
    // Settle a few ticks at zero control so contacts stabilize
    auto* sim = env.get_mujoco_wrapper();
    auto* model = sim->model();
    auto* data = sim->data();
    auto* adapter = env.get_robot_adapter();
    if (adapter && adapter->is_diff_drive()) {
        adapter->zero_control(model, data);
    }
    for (int i = 0; i < 200; i++) sim->step();
    env.update_object_states();

    auto rs0 = env.get_robot_state();
    std::cout << "Settled at: (" << rs0->position[0] << ", "
              << rs0->position[1] << "), yaw = "
              << adapter->get_theta(model, data) * 180.0 / M_PI << " deg\n";

    // 3. Wavefront plan
    // Matches HighLevelPlanner construction: 0.05 m grid, robot half-extents
    // from the namo_config_complete_skill15.yaml defaults. Robot size only
    // governs obstacle-inflation footprint for path extraction here.
    const double WAVEFRONT_RESOLUTION_M = 0.05;
    const std::vector<double> ROBOT_HALF_EXTENTS_M = {0.21, 0.21};
    namo::WavefrontPlanner planner(WAVEFRONT_RESOLUTION_M, env, ROBOT_HALF_EXTENTS_M);
    std::vector<double> start_pos = {rs0->position[0], rs0->position[1]};
    if (!planner.update_wavefront(env, start_pos)) {
        std::cerr << "Wavefront update failed\n";
        return 2;
    }
    auto path = planner.extract_path({start_pos[0], start_pos[1]}, {gx, gy});
    if (path.empty()) {
        std::cerr << "No collision-free path from start to goal.\n";
        return 3;
    }
    std::cout << "Wavefront path: " << path.size() << " waypoints\n";

    // Emit path for visualizer if NAMO_NAV_LOG is set (matches push_controller)
    if (std::getenv("NAMO_NAV_LOG")) {
        std::cerr << "[NAV_PATH]";
        for (const auto& p : path) std::cerr << " " << p[0] << "," << p[1];
        std::cerr << "\n";
    }

    // 4. Build nav strategy (trapezoidal default)
    std::unique_ptr<namo::NavigationStrategy> nav;
    if (adapter && adapter->is_diff_drive()) {
        nav = std::make_unique<namo::DiffDriveNavigation>(
            namo::DiffDriveNavigation::Params{});
    } else {
        nav = std::make_unique<namo::HolonomicNavigation>();
    }

    // 5. Run state machine
    auto result = nav->execute(env, path, t_theta, /*target_object=*/"");

    // 6. Report
    auto rs1 = env.get_robot_state();
    const double end_yaw = adapter->get_theta(model, data);
    const double pos_err = std::hypot(gx - rs1->position[0], gy - rs1->position[1]);
    const double yaw_err = std::abs(wrap_angle(t_theta - end_yaw)) * 180.0 / M_PI;

    std::cout << "\n=== Result ===\n";
    std::cout << "  success         : " << (result.success ? "yes" : "no") << "\n";
    if (!result.success) {
        std::cout << "  failure_reason  : " << result.failure_reason << "\n";
        std::cout << "  collision_obj   : " << result.collision_object << "\n";
    }
    std::cout << "  steps_used      : " << result.steps_used
              << "  (" << result.steps_used * 0.01 << " s)\n";
    std::cout << "  end pos         : (" << rs1->position[0] << ", "
              << rs1->position[1] << ")\n";
    std::cout << "  end yaw (deg)   : " << end_yaw * 180.0 / M_PI << "\n";
    std::cout << "  pos error (mm)  : " << pos_err * 1000.0 << "\n";
    std::cout << "  yaw error (deg) : " << yaw_err << "\n";

    return result.success ? 0 : 4;
}
