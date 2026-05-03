#include "config/config_manager.hpp"
#include "environment/namo_environment.hpp"
#include "planning/mpc_executor.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

using namo::ConfigManager;
using namo::MPCExecutor;
using namo::NAMOEnvironment;
using namo::PlanStep;
using namo::SE2State;

std::filesystem::path find_repo_root() {
    std::filesystem::path cursor = std::filesystem::absolute(__FILE__).parent_path();
    for (int i = 0; i < 8; ++i) {
        if (std::filesystem::exists(cursor / "include" / "planning" / "mpc_executor.hpp") &&
            std::filesystem::exists(cursor / "test_xml" / "little-car-modeling-package" / "artifacts" / "nav_env.xml")) {
            return cursor;
        }
        cursor = cursor.parent_path();
    }
    throw std::runtime_error("Failed to locate repository root");
}

void require(bool cond, const std::string& msg) {
    if (!cond) {
        throw std::runtime_error(msg);
    }
}

std::filesystem::path write_temp_config(const std::string& stem, const std::string& yaml_text) {
    const auto temp_dir = std::filesystem::temp_directory_path();
    const auto file_path = temp_dir / (stem + ".yaml");
    std::ofstream out(file_path);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open temp config for writing: " + file_path.string());
    }
    out << yaml_text;
    out.close();
    return file_path;
}

std::shared_ptr<ConfigManager> load_config(const std::filesystem::path& cfg_path) {
    return std::shared_ptr<ConfigManager>(ConfigManager::create_from_file(cfg_path.string()).release());
}

double yaw_from_quat_xyzw(const std::array<double, 4>& q) {
    const double x = q[0];
    const double y = q[1];
    const double z = q[2];
    const double w = q[3];
    return std::atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z));
}

void run_forced_nav_timeout_case(const std::filesystem::path& xml_path) {
    const std::string yaml_text =
        "planning:\n"
        "  robot_type: \"diff_drive\"\n"
        "  robot_size: [0.035, 0.038]\n"
        "  skill_level_resolution: 0.005\n"
        "skill:\n"
        "  max_push_steps: 10\n"
        "  control_steps_per_push: 250\n"
        "  force_scaling: 1.0\n"
        "  points_per_face: 15\n"
        "  check_object_collision: false\n"
        "navigation:\n"
        "  diff_drive:\n"
        "    max_nav_steps: 1\n";

    const auto cfg_path = write_temp_config("namo_failure_diag_nav_timeout", yaml_text);
    const auto config = load_config(cfg_path);

    NAMOEnvironment env(xml_path.string(), config, false, false);
    const auto robot_half_extents = env.get_robot_planning_half_extents();
    MPCExecutor executor(
        env,
        config->planning().skill_level_resolution,
        {robot_half_extents[0], robot_half_extents[1]},
        config->planning().wavefront_tier1_inflation_margin,
        config->skill().max_push_steps,
        config->skill().control_steps_per_push,
        config->skill().force_scaling,
        config->skill().points_per_face,
        config->skill().check_object_collision,
        config
    );

    const std::string object_name = "obstacle_1_movable";
    auto reachable_edges = executor.get_reachable_edges_with_wavefront(object_name);
    require(!reachable_edges.empty(), "Expected reachable edges for obstacle_1_movable");
    const int edge_idx = reachable_edges.front();

    const auto* state = env.get_object_state(object_name);
    require(state != nullptr, "Expected object state for obstacle_1_movable");
    const double theta = yaw_from_quat_xyzw(state->quaternion);
    const SE2State target_pose(state->position[0] + 0.6, state->position[1], theta);

    const std::vector<PlanStep> single_step = {PlanStep(edge_idx, 1, target_pose)};
    const auto exec_result = executor.execute_plan(object_name, single_step);

    require(!exec_result.success, "Expected forced navigation timeout failure");
    require(exec_result.failure_reason.find("Primitive step") == std::string::npos,
            "Failure reason should not collapse to generic primitive-step message");
    require(exec_result.failure_diagnostics.code == "navigation_failed",
            "Expected navigation_failed diagnostic code");
    require(exec_result.failure_diagnostics.step_index_1based == 1,
            "Expected failure step index to be 1");

    std::error_code ec;
    std::filesystem::remove(cfg_path, ec);
}

}  // namespace

int main() {
    try {
        const auto repo_root = find_repo_root();
        const auto xml_path = repo_root / "test_xml" / "little-car-modeling-package" / "artifacts" / "nav_env.xml";
        require(std::filesystem::exists(xml_path), "Missing fixture XML: " + xml_path.string());
        run_forced_nav_timeout_case(xml_path);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_mpc_failure_diagnostics failed: " << e.what() << "\n";
        return 1;
    }
}
