#include "config/config_manager.hpp"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

void write_text_file(const std::filesystem::path& path, const std::string& contents) {
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open temporary config file: " + path.string());
    }
    out << contents;
}

void test_deprecated_key_rejected(const std::filesystem::path& temp_dir) {
    const auto config_path = temp_dir / "deprecated_edge_offset.yaml";
    write_text_file(
        config_path,
        "planning:\n"
        "  robot_size: [0.15, 0.15]\n"
        "skill:\n"
        "  object_clearance: 0.1\n");

    bool raised = false;
    try {
        (void)namo::ConfigManager::create_from_file(config_path.string());
    } catch (const namo::ConfigSchemaError& err) {
        raised = true;
        const std::string message = err.what();
        assert(message.find("skill.object_clearance") != std::string::npos);
        assert(message.find("planning.wavefront_edge_offset_margin") != std::string::npos);
    }
    assert(raised && "Expected deprecated config key to raise ConfigSchemaError");
}

void test_planning_edge_offset_loaded(const std::filesystem::path& temp_dir) {
    const auto config_path = temp_dir / "canonical_edge_offset.yaml";
    write_text_file(
        config_path,
        "planning:\n"
        "  robot_size: [0.15, 0.15]\n"
        "  wavefront_edge_offset_margin: 0.03\n");

    auto config = namo::ConfigManager::create_from_file(config_path.string());
    assert(config);
    assert(std::fabs(config->planning().wavefront_edge_offset_margin - 0.03) < 1e-12);
}

}  // namespace

int main() {
    try {
        const auto temp_dir =
            std::filesystem::temp_directory_path() / "namo_config_manager_schema_test";
        std::filesystem::create_directories(temp_dir);

        test_deprecated_key_rejected(temp_dir);
        test_planning_edge_offset_loaded(temp_dir);

        std::filesystem::remove_all(temp_dir);
        return 0;
    } catch (const std::exception& err) {
        std::cerr << "ConfigManager schema test failed: " << err.what() << std::endl;
        return 1;
    }
}
