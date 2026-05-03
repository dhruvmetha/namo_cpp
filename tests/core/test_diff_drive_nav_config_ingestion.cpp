#include "config/config_manager.hpp"
#include "environment/namo_environment.hpp"
#include "navigation/diff_drive_navigation.hpp"
#include "planning/mpc_executor.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

using namo::ConfigManager;
using namo::DiffDriveNavigation;
using namo::MPCExecutor;
using namo::NAMOEnvironment;

constexpr double kEps = 1e-12;

std::filesystem::path find_repo_root() {
    std::filesystem::path cursor = std::filesystem::absolute(__FILE__).parent_path();
    for (int i = 0; i < 8; ++i) {
        if (std::filesystem::exists(cursor / "include" / "config" / "config_manager.hpp") &&
            std::filesystem::exists(cursor / "test_xml" / "little-car-modeling-package" / "artifacts" / "nav_env.xml")) {
            return cursor;
        }
        cursor = cursor.parent_path();
    }
    throw std::runtime_error("Failed to locate repository root");
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool nearly_equal(double a, double b) {
    return std::abs(a - b) <= kEps;
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

DiffDriveNavigation::Params build_executor_and_get_params(
    const std::filesystem::path& xml_path,
    const std::shared_ptr<ConfigManager>& config) {
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

    const auto params_opt = executor.get_controller().get_diff_drive_params_for_debug();
    require(params_opt.has_value(), "Expected diff-drive params from controller debug accessor");
    return *params_opt;
}

void assert_params_equal(const DiffDriveNavigation::Params& got, const DiffDriveNavigation::Params& expected, const std::string& label) {
    require(nearly_equal(got.linear_speed, expected.linear_speed), label + ": linear_speed mismatch");
    require(nearly_equal(got.angular_speed, expected.angular_speed), label + ": angular_speed mismatch");
    require(nearly_equal(got.lookahead, expected.lookahead), label + ": lookahead mismatch");
    require(nearly_equal(got.xy_threshold, expected.xy_threshold), label + ": xy_threshold mismatch");
    require(nearly_equal(got.theta_threshold, expected.theta_threshold), label + ": theta_threshold mismatch");
    require(nearly_equal(got.xy_tolerance, expected.xy_tolerance), label + ": xy_tolerance mismatch");
    require(nearly_equal(got.theta_tolerance, expected.theta_tolerance), label + ": theta_tolerance mismatch");
    require(got.wait_steps == expected.wait_steps, label + ": wait_steps mismatch");
    require(got.decel_steps == expected.decel_steps, label + ": decel_steps mismatch");
    require(got.settle_steps == expected.settle_steps, label + ": settle_steps mismatch");
    require(nearly_equal(got.velocity_tolerance, expected.velocity_tolerance), label + ": velocity_tolerance mismatch");
    require(got.max_nav_steps == expected.max_nav_steps, label + ": max_nav_steps mismatch");
    require(nearly_equal(got.max_path_deviation, expected.max_path_deviation), label + ": max_path_deviation mismatch");
    require(nearly_equal(got.sharp_turn_threshold, expected.sharp_turn_threshold), label + ": sharp_turn_threshold mismatch");
    require(nearly_equal(got.sharp_turn_exit, expected.sharp_turn_exit), label + ": sharp_turn_exit mismatch");
}

void run_override_case(const std::filesystem::path& xml_path) {
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
        "    max_nav_steps: 12345\n"
        "    wait_steps: 77\n";

    const auto cfg_path = write_temp_config("namo_diff_nav_override", yaml_text);
    const auto config = load_config(cfg_path);
    const auto got = build_executor_and_get_params(xml_path, config);

    DiffDriveNavigation::Params expected;
    expected.max_nav_steps = 12345;
    expected.wait_steps = 77;
    assert_params_equal(got, expected, "override_case");

    std::error_code ec;
    std::filesystem::remove(cfg_path, ec);
}

void run_default_case(const std::filesystem::path& xml_path) {
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
        "  check_object_collision: false\n";

    const auto cfg_path = write_temp_config("namo_diff_nav_defaults", yaml_text);
    const auto config = load_config(cfg_path);
    const auto got = build_executor_and_get_params(xml_path, config);

    const DiffDriveNavigation::Params expected;
    assert_params_equal(got, expected, "default_case");

    std::error_code ec;
    std::filesystem::remove(cfg_path, ec);
}

}  // namespace

int main() {
    try {
        const auto repo_root = find_repo_root();
        const auto xml_path = repo_root / "test_xml" / "little-car-modeling-package" / "artifacts" / "nav_env.xml";
        require(std::filesystem::exists(xml_path), "Missing fixture XML: " + xml_path.string());

        run_override_case(xml_path);
        run_default_case(xml_path);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_diff_drive_nav_config_ingestion failed: " << e.what() << "\n";
        return 1;
    }
}
