#include "config/config_manager.hpp"
#include "environment/namo_environment.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namo::ConfigManager;
using namo::NAMOEnvironment;

constexpr double kEps = 1e-6;
constexpr double kExpectedHalfExtentX = 0.035;
constexpr double kExpectedHalfExtentY = 0.038;
constexpr double kExpectedHalfExtentZ = 0.070;

std::filesystem::path find_repo_root() {
    std::filesystem::path cursor = std::filesystem::absolute(__FILE__).parent_path();
    for (int i = 0; i < 8; ++i) {
        if (std::filesystem::exists(cursor / "include" / "environment" / "namo_environment.hpp") &&
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

bool nearly_equal(double a, double b, double eps = kEps) {
    return std::abs(a - b) <= eps;
}

std::filesystem::path write_temp_file(const std::string& stem, const std::string& suffix, const std::string& text) {
    const auto path = std::filesystem::temp_directory_path() / (stem + suffix);
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open temp file for writing: " + path.string());
    }
    out << text;
    out.close();
    return path;
}

std::shared_ptr<ConfigManager> load_config(const std::filesystem::path& cfg_path) {
    return std::shared_ptr<ConfigManager>(ConfigManager::create_from_file(cfg_path.string()).release());
}

struct CerrCapture {
    std::streambuf* original = nullptr;
    std::ostringstream buffer;

    CerrCapture() : original(std::cerr.rdbuf(buffer.rdbuf())) {}
    ~CerrCapture() { std::cerr.rdbuf(original); }

    std::string str() const { return buffer.str(); }
};

std::string read_file(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + path.string());
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

void replace_all(std::string& text, const std::string& needle, const std::string& replacement) {
    std::string::size_type pos = 0;
    while ((pos = text.find(needle, pos)) != std::string::npos) {
        text.replace(pos, needle.size(), replacement);
        pos += replacement.size();
    }
}

std::filesystem::path write_noncollidable_robot_xml(const std::filesystem::path& src_xml) {
    std::string xml_text = read_file(src_xml);
    const std::vector<std::string> robot_geom_names = {
        "front_chassis_collision",
        "rear_chassis_collision",
        "rear_support",
        "front_support",
        "left_wheel_collision",
        "right_wheel_collision",
    };
    for (const auto& geom_name : robot_geom_names) {
        const std::string needle = "name=\"" + geom_name + "\"";
        const std::string replacement = needle + " contype=\"0\" conaffinity=\"0\"";
        replace_all(xml_text, needle, replacement);
    }
    return write_temp_file("namo_robot_footprint_no_collide", ".xml", xml_text);
}

void run_geometry_derived_case(const std::filesystem::path& xml_path) {
    const std::string yaml_text =
        "planning:\n"
        "  robot_type: \"diff_drive\"\n"
        "  robot_size: [0.500000, 0.500000]\n";
    const auto cfg_path = write_temp_file("namo_robot_footprint_mismatch", ".yaml", yaml_text);
    const auto config = load_config(cfg_path);

    std::string warnings;
    {
        CerrCapture capture;
        NAMOEnvironment env(xml_path.string(), config, false, false);

        const auto half_extents = env.get_robot_planning_half_extents();
        require(nearly_equal(half_extents[0], kExpectedHalfExtentX), "Expected geometry-derived half_extent_x");
        require(nearly_equal(half_extents[1], kExpectedHalfExtentY), "Expected geometry-derived half_extent_y including wheel geoms");

        const auto& robot_info = env.get_robot_info();
        require(nearly_equal(robot_info.size[0], kExpectedHalfExtentX), "Robot info size_x mismatch");
        require(nearly_equal(robot_info.size[1], kExpectedHalfExtentY), "Robot info size_y mismatch");
        require(nearly_equal(robot_info.size[2], kExpectedHalfExtentZ), "Robot info size_z mismatch");

        const auto all_info = env.get_all_object_info();
        const auto robot_it = all_info.find("robot");
        require(robot_it != all_info.end(), "Expected robot entry in get_all_object_info()");
        require(nearly_equal(robot_it->second.at("size_x"), kExpectedHalfExtentX), "Exported size_x mismatch");
        require(nearly_equal(robot_it->second.at("size_y"), kExpectedHalfExtentY), "Exported size_y mismatch");
        require(nearly_equal(robot_it->second.at("size_z"), kExpectedHalfExtentZ), "Exported size_z mismatch");

        const auto csv_path = std::filesystem::temp_directory_path() / "namo_robot_footprint_objects.csv";
        env.save_objects_to_file(csv_path.string());
        const std::string csv_text = read_file(csv_path);
        require(csv_text.find("2,robot,0.035000,0.038000") != std::string::npos,
                "Robot CSV export should contain distinct x/y half-extents");
        std::error_code csv_ec;
        std::filesystem::remove(csv_path, csv_ec);

        const auto bounds = env.get_environment_bounds();
        const auto* robot_state = env.get_robot_state();
        require(robot_state != nullptr, "Expected robot state");
        const double robot_radius = std::sqrt(
            half_extents[0] * half_extents[0] + half_extents[1] * half_extents[1]);
        require(bounds[0] <= robot_state->position[0] - robot_radius,
                "Bounds should include robot x-min with rotation-safe radius");
        require(bounds[1] >= robot_state->position[0] + robot_radius,
                "Bounds should include robot x-max with rotation-safe radius");
        require(bounds[2] <= robot_state->position[1] - robot_radius,
                "Bounds should include robot y-min with rotation-safe radius");
        require(bounds[3] >= robot_state->position[1] + robot_radius,
                "Bounds should include robot y-max with rotation-safe radius");

        warnings = capture.str();
    }

    require(warnings.find("planning.robot_size is legacy fallback only") != std::string::npos,
            "Expected legacy-fallback warning for explicit planning.robot_size");
    require(warnings.find("geometry-derived") != std::string::npos,
            "Expected geometry-derived wording in warning");

    std::error_code ec;
    std::filesystem::remove(cfg_path, ec);
}

void run_fallback_case(const std::filesystem::path& xml_path) {
    const std::string yaml_text =
        "planning:\n"
        "  robot_type: \"diff_drive\"\n"
        "  robot_size: [0.123000, 0.124000]\n";
    const auto cfg_path = write_temp_file("namo_robot_footprint_fallback", ".yaml", yaml_text);
    const auto xml_no_collide = write_noncollidable_robot_xml(xml_path);
    const auto config = load_config(cfg_path);

    std::string warnings;
    {
        CerrCapture capture;
        NAMOEnvironment env(xml_no_collide.string(), config, false, false);
        const auto half_extents = env.get_robot_planning_half_extents();
        require(nearly_equal(half_extents[0], 0.123), "Fallback size_x should come from planning.robot_size");
        require(nearly_equal(half_extents[1], 0.124), "Fallback size_y should come from planning.robot_size");
        warnings = capture.str();
    }

    require(warnings.find("falling back to planning.robot_size") != std::string::npos,
            "Expected fallback warning when no collidable robot geoms are available");

    std::error_code ec;
    std::filesystem::remove(cfg_path, ec);
    std::filesystem::remove(xml_no_collide, ec);
}

}  // namespace

int main() {
    try {
        const auto repo_root = find_repo_root();
        const auto xml_path = repo_root / "test_xml" / "little-car-modeling-package" / "artifacts" / "nav_env.xml";
        require(std::filesystem::exists(xml_path), "Missing fixture XML: " + xml_path.string());

        run_geometry_derived_case(xml_path);
        run_fallback_case(xml_path);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_robot_planning_footprint failed: " << e.what() << "\n";
        return 1;
    }
}
