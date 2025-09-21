#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <string>
#include <filesystem>
#include <cstdint>
#include <memory>
#include <algorithm>

#include "../include/core/parameter_loader.hpp"
#include "../include/environment/namo_environment.hpp"
#include "../include/planning/namo_push_controller.hpp"
#include "../include/wavefront/wavefront_planner.hpp"

using namespace namo;

// Scene configuration for primitive generation
struct SceneConfig {
    std::string name;
    std::string xml_path;
    std::string description;
};

// Primitive data structure for binary storage
struct __attribute__((packed)) NominalPrimitive {
    float delta_x;        // Position change in x
    float delta_y;        // Position change in y  
    float delta_theta;    // Rotation change (yaw)
    uint8_t edge_idx;     // Push direction (0-63)
    uint8_t push_steps;   // Push step number (1-10)
};

// Verify struct size
static_assert(sizeof(NominalPrimitive) == 14, "NominalPrimitive must be 14 bytes");

// Helper function to add suffix to filename
std::string add_suffix_to_filename(const std::string& base_path, const std::string& suffix) {
    auto dot_pos = base_path.find_last_of('.');
    if (dot_pos == std::string::npos) {
        return base_path + "_" + suffix;
    }
    return base_path.substr(0, dot_pos) + "_" + suffix + base_path.substr(dot_pos);
}

// Generate primitives for a single scene
std::vector<NominalPrimitive> generate_primitives_for_scene(
    const SceneConfig& scene_config,
    bool visualize,
    double resolution,
    int points_per_face,
    int control_steps,
    int max_push_steps,
    double force_scaling
) {
    std::cout << "\n=== Generating primitives for " << scene_config.name << " ===" << std::endl;
    std::cout << "XML: " << scene_config.xml_path << std::endl;
    std::cout << "Description: " << scene_config.description << std::endl;
    
    // Create NAMO environment for this scene
    NAMOEnvironment env(scene_config.xml_path, visualize, false);
    
    // Get robot info
    const auto& robot_info = env.get_robot_info();
    std::vector<double> robot_size = {robot_info.size[0], robot_info.size[1]};
    
    // Create wavefront planner (heap allocation to avoid 32MB stack array)
    auto wavefront_planner = std::make_unique<WavefrontPlanner>(resolution, env, robot_size);
    
    // Set robot goal (fixed for nominal primitive generation)
    std::array<double, 2> robot_goal = {0.0, 0.0};
    env.set_robot_goal(robot_goal);
    
    // Create push controller
    NAMOPushController push_controller(env, *wavefront_planner, max_push_steps, control_steps, force_scaling, points_per_face);
    
    // Get movable objects (should be our nominal object)
    std::array<std::string, 20> reachable_objects;
    size_t reachable_count;
    size_t num_reachable = push_controller.get_reachable_objects(reachable_objects, reachable_count);
    
    if (num_reachable == 0) {
        throw std::runtime_error("No reachable objects found in scene: " + scene_config.xml_path);
    }
    
    std::string target_object = reachable_objects[0];
    std::cout << "Using object: " << target_object << std::endl;
    
    // Get edge points for this object
    std::array<std::array<double, 2>, 64> edge_points;
    std::array<std::array<double, 2>, 64> mid_points;
    size_t edge_count, mid_count;
    size_t num_edges = push_controller.generate_edge_points(target_object, edge_points, mid_points, edge_count, mid_count);
    
    std::cout << "Generated " << num_edges << " edge points" << std::endl;
    
    // Position camera for good view if visualizing
    if (visualize) {
        auto obj_state = env.get_object_state(target_object);
        if (obj_state) {
            std::array<double, 3> focus_point = {obj_state->position[0], obj_state->position[1], 0.0};
            env.set_camera_lookat(focus_point);
            env.set_camera_position(6.0, 0.0, -45.0);
        }
    }
    
    // Get initial object state
    auto initial_obj_state = env.get_object_state(target_object);
    if (!initial_obj_state) {
        throw std::runtime_error("Failed to get initial object state!");
    }
    
    std::array<double, 3> initial_pos = initial_obj_state->position;
    std::array<double, 4> initial_quat = initial_obj_state->quaternion;
    
    std::cout << "Initial object position: [" << initial_pos[0] << ", " << initial_pos[1] << ", " << initial_pos[2] << "]" << std::endl;
    
    // Generate primitives for each edge
    std::vector<NominalPrimitive> all_primitives;
    all_primitives.reserve(num_edges * max_push_steps);
    
    for (size_t edge_idx = 0; edge_idx < num_edges; edge_idx++) {
        std::cout << "Generating primitives for edge " << edge_idx << " / " << num_edges << std::endl;
        
        // Reset environment to initial state
        env.reset();
        env.step_simulation();
        
        // Generate primitives for all step counts (1 to max_push_steps) - pyramid approach
        for (int push_steps = 1; push_steps <= max_push_steps; push_steps++) {
            // Reset environment to initial state for each primitive
            env.reset();
            env.step_simulation();
            
            // Execute push primitive for this number of steps
            bool success = push_controller.execute_push_primitive(target_object, edge_idx, push_steps);
            
            // Get object state after push sequence
            auto final_obj_state = env.get_object_state(target_object);
            if (!final_obj_state) {
                continue;
            }
            
            // Calculate displacement from initial position
            NominalPrimitive primitive;
            primitive.delta_x = final_obj_state->position[0] - initial_pos[0];
            primitive.delta_y = final_obj_state->position[1] - initial_pos[1];
            
            // Calculate rotation change (simple yaw extraction)
            auto quat_to_yaw = [](const std::array<double, 4>& q) -> double {
                return std::atan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 
                                1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]));
            };
            
            primitive.delta_theta = quat_to_yaw(final_obj_state->quaternion) - quat_to_yaw(initial_quat);
            primitive.edge_idx = edge_idx;
            primitive.push_steps = push_steps;
            
            all_primitives.push_back(primitive);
            
            // Render final state if visualizing (only for last step to avoid too much output)
            if (visualize && push_steps == max_push_steps) {
                env.render();
            }
        }
        
        // Pause between edges for observation
        if (visualize) {
            std::cout << "Press Enter to continue to next edge..." << std::endl;
            std::cin.get();
        }
    }
    
    std::cout << "Generated " << all_primitives.size() << " primitives for " << scene_config.name << std::endl;
    return all_primitives;
}

// Save primitives to binary file
void save_primitives_to_file(const std::string& output_file, const std::vector<NominalPrimitive>& primitives) {
    std::ofstream file(output_file, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to create output file: " + output_file);
    }
    
    std::cout << "Saving primitives to: " << output_file << std::endl;
    
    // Write header
    uint32_t count = primitives.size();
    file.write(reinterpret_cast<const char*>(&count), sizeof(count));
    
    // Write primitives
    file.write(reinterpret_cast<const char*>(primitives.data()), 
              count * sizeof(NominalPrimitive));
    
    file.close();
    
    std::cout << "Saved " << count << " primitives to: " << output_file << std::endl;
    std::cout << "File size: " << std::filesystem::file_size(output_file) << " bytes" << std::endl;
}

int main() {
    std::cout << "=== Multi-Scene Nominal Motion Primitive Generator ===" << std::endl;
    std::cout << "Generating primitives for multiple object shapes" << std::endl;
    
    try {
        // Prefer unified config if present, fallback to minimal local config
        std::string config_path = "config/namo_config_complete_skill15.yaml";
        bool using_unified_config = std::filesystem::exists(config_path);
        
        if (!using_unified_config) {
            // Fallback: create minimal config for standalone use
            std::string config_content = R"(visualize=false

[data_collection]
enabled=false

[wavefront_planner]
resolution=0.05
)";
            
            // Write temporary config file
            std::ofstream config_file("tools/primitive_gen_config.yaml");
            config_file << config_content;
            config_file.close();
            config_path = "tools/primitive_gen_config.yaml";
        }
        
        // Load configuration using our parameter loader
        FastParameterLoader params(config_path);
        std::cout << "Configuration loaded from: " << config_path << std::endl;
        
        // Define the three scenes to generate primitives for
        std::vector<SceneConfig> scenes = {
            {"square", "data/nominal_primitive_scene_square.xml", "Square object (0.35x0.35m)"},
            {"wide", "data/nominal_primitive_scene_wide.xml", "Wide object (0.45x0.25m)"},
            {"tall", "data/nominal_primitive_scene_tall.xml", "Tall object (0.25x0.45m)"}
        };
        
        // Filter to only existing files, with fallback to legacy
        std::vector<SceneConfig> existing_scenes;
        for (const auto& scene : scenes) {
            if (std::filesystem::exists(scene.xml_path)) {
                existing_scenes.push_back(scene);
                std::cout << "Found scene: " << scene.name << " -> " << scene.xml_path << std::endl;
            } else {
                std::cout << "Scene XML not found, skipping: " << scene.xml_path << std::endl;
            }
        }
        
        // Fallback to legacy file if no variants found
        if (existing_scenes.empty()) {
            std::string legacy_xml = "data/nominal_primitive_scene.xml";
            if (std::filesystem::exists(legacy_xml)) {
                std::cout << "Using legacy single scene: " << legacy_xml << std::endl;
                existing_scenes.push_back({"square", legacy_xml, "Legacy single scene"});
            } else {
                throw std::runtime_error("No valid scene XML files found!");
            }
        }
        
        std::cout << "Found " << existing_scenes.size() << " scene(s) to process" << std::endl;
        
        // Get generation parameters with fallbacks
        bool visualize = false;
        if (params.has_key("visualize")) {
            visualize = params.get_bool("visualize");
        } else if (params.has_key("system.enable_visualization")) {
            visualize = params.get_bool("system.enable_visualization");
        }
        
        double resolution = 0.05;
        if (params.has_key("wavefront_planner.resolution")) {
            resolution = params.get_double("wavefront_planner.resolution");
        }
        
        // Edge sampling density (points per object face)
        int points_per_face = 3; // Default fallback
        if (params.has_key("skill.points_per_face")) {
            points_per_face = params.get_int("skill.points_per_face");
        } else if (params.has_key("skill.num_edge_points")) {
            int total = params.get_int("skill.num_edge_points");
            points_per_face = std::max(1, total / 4);
        }
        // Clamp to respect MAX_EDGE_POINTS capacity (4 faces * 16 points = 64 max)
        points_per_face = std::clamp(points_per_face, 1, 16);
        
        // Push controller parameters
        int control_steps = 250;
        if (params.has_key("skill.control_steps_per_push")) {
            control_steps = params.get_int("skill.control_steps_per_push");
        }
        
        int max_push_steps = 10;
        if (params.has_key("motion_primitives.max_push_steps")) {
            max_push_steps = params.get_int("motion_primitives.max_push_steps");
        } else if (params.has_key("skill.max_push_steps")) {
            max_push_steps = params.get_int("skill.max_push_steps");
        }
        
        double force_scaling = 1.0;
        if (params.has_key("skill.force_scaling")) {
            force_scaling = params.get_double("skill.force_scaling");
        }
        
        // Determine base output file
        std::string base_output = "data/motion_primitives.dat";
        if (params.has_key("system.motion_primitives_file")) {
            base_output = params.get_string("system.motion_primitives_file");
        }
        
        std::cout << "Generation parameters:" << std::endl;
        std::cout << "  Visualize: " << (visualize ? "true" : "false") << std::endl;
        std::cout << "  Resolution: " << resolution << std::endl;
        std::cout << "  Points per face: " << points_per_face << std::endl;
        std::cout << "  Control steps: " << control_steps << std::endl;
        std::cout << "  Max push steps: " << max_push_steps << std::endl;
        std::cout << "  Force scaling: " << force_scaling << std::endl;
        std::cout << "  Base output: " << base_output << std::endl;
        
        // Generate primitives for each scene
        for (const auto& scene : existing_scenes) {
            try {
                auto primitives = generate_primitives_for_scene(
                    scene, visualize, resolution, points_per_face, 
                    control_steps, max_push_steps, force_scaling
                );
                
                // Save to suffixed output file
                std::string output_file = add_suffix_to_filename(base_output, scene.name);
                save_primitives_to_file(output_file, primitives);
                
                // For backward compatibility: if this is the square scene, also write to base file
                if (scene.name == "square" && output_file != base_output) {
                    try {
                        save_primitives_to_file(base_output, primitives);
                        std::cout << "Also saved square primitives to base file: " << base_output << std::endl;
                    } catch (const std::exception& e) {
                        std::cout << "Warning: Failed to write base file: " << e.what() << std::endl;
                        // Continue - the suffixed file is the primary output
                    }
                }
                
            } catch (const std::exception& e) {
                std::cerr << "Failed to generate primitives for scene " << scene.name << ": " << e.what() << std::endl;
                // Continue with other scenes
            }
        }
        
        // Clean up temporary config if we created one
        if (!using_unified_config) {
            std::filesystem::remove("tools/primitive_gen_config.yaml");
        }
        
        std::cout << "\n=== Generation Complete ===" << std::endl;
        std::cout << "Processed " << existing_scenes.size() << " scene(s)" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}