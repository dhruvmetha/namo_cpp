#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <string>
#include <filesystem>
#include <cstdint>

#include "../include/core/parameter_loader.hpp"
#include "../include/environment/namo_environment.hpp"
#include "../include/planning/namo_push_controller.hpp"
#include "../include/planning/incremental_wavefront_planner.hpp"

using namespace namo;

// Primitive data structure for binary storage
struct __attribute__((packed)) NominalPrimitive {
    float delta_x;        // Position change in x
    float delta_y;        // Position change in y  
    float delta_theta;    // Rotation change (yaw)
    uint8_t edge_idx;     // Push direction (0-11)
    uint8_t push_steps;   // Push step number (1-10)
};

// Verify struct size
static_assert(sizeof(NominalPrimitive) == 14, "NominalPrimitive must be 14 bytes");

int main() {
    std::cout << "=== Nominal Motion Primitive Generator ===" << std::endl;
    std::cout << "Using existing NAMO infrastructure" << std::endl;
    std::cout << "Generating 12 directions × 10 step variants = 120 primitives (pyramid approach)" << std::endl;
    
    try {
        // Create simple config for nominal scene
        std::string config_content = R"(xml_path=data/nominal_primitive_scene.xml
visualize=false
robot_goal=[0.0, 0.0]

[data_collection]
enabled=false

[wavefront_planner]
resolution=0.05
)";
        
        // Write temporary config file
        std::ofstream config_file("tools/primitive_gen_config.yaml");
        config_file << config_content;
        config_file.close();
        
        // Load configuration using our parameter loader
        FastParameterLoader params("tools/primitive_gen_config.yaml");
        std::cout << "Configuration loaded!" << std::endl;
        
        // Get settings
        std::string xml_path = params.get_string("xml_path");
        bool visualize = params.get_bool("visualize");
        double resolution = params.get_double("wavefront_planner.resolution");
        std::array<double, 2> robot_goal = params.get_array<2>("robot_goal");
        
        std::cout << "Settings:" << std::endl;
        std::cout << "  XML: " << xml_path << std::endl;
        std::cout << "  Visualize: " << (visualize ? "true" : "false") << std::endl;
        std::cout << "  Resolution: " << resolution << std::endl;
        
        // Create NAMO environment (this handles MuJoCo setup and visualization)
        NAMOEnvironment env(xml_path, visualize, false);
        std::cout << "Environment created successfully!" << std::endl;
        
        // Get robot info
        const auto& robot_info = env.get_robot_info();
        std::vector<double> robot_size = {robot_info.size[0], robot_info.size[1]};
        
        // Create wavefront planner
        IncrementalWavefrontPlanner wavefront_planner(resolution, env, robot_size);
        
        // Set robot goal
        env.set_robot_goal(robot_goal);
        
        // Create push controller (this has the primitive execution logic)
        NAMOPushController push_controller(env, wavefront_planner, 10, 250, 1.0);
        std::cout << "Push controller created with parameters: 10 steps, 250 control_steps, 1.0 scaling" << std::endl;
        
        // Get movable objects (should be our nominal 0.35x0.35 object)
        std::array<std::string, 20> reachable_objects;
        size_t reachable_count;
        size_t num_reachable = push_controller.get_reachable_objects(reachable_objects, reachable_count);
        
        if (num_reachable == 0) {
            std::cerr << "No reachable objects found!" << std::endl;
            return 1;
        }
        
        std::string target_object = reachable_objects[0];
        std::cout << "Using object: " << target_object << std::endl;
        
        // Get edge points for this object
        std::array<std::array<double, 2>, 16> edge_points;
        std::array<std::array<double, 2>, 16> mid_points;
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
        
        // Generate primitives using existing push controller
        std::vector<NominalPrimitive> all_primitives;
        
        // Get initial object state
        auto initial_obj_state = env.get_object_state(target_object);
        if (!initial_obj_state) {
            std::cerr << "Failed to get initial object state!" << std::endl;
            return 1;
        }
        
        std::array<double, 3> initial_pos = initial_obj_state->position;
        std::array<double, 4> initial_quat = initial_obj_state->quaternion;
        
        std::cout << "Initial object position: [" << initial_pos[0] << ", " << initial_pos[1] << ", " << initial_pos[2] << "]" << std::endl;
        
        // Generate primitives for each edge
        for (size_t edge_idx = 0; edge_idx < num_edges && edge_idx < 12; edge_idx++) {
            std::cout << "\n--- Generating primitives for edge " << edge_idx << " ---" << std::endl;
            std::cout << "Edge point: [" << edge_points[edge_idx][0] << ", " << edge_points[edge_idx][1] << "]" << std::endl;
            std::cout << "Mid point: [" << mid_points[edge_idx][0] << ", " << mid_points[edge_idx][1] << "]" << std::endl;
            
            // Reset environment to initial state
            env.reset();
            env.step_simulation();
            
            // Generate primitives for all step counts (1 to 10) - pyramid approach
            for (int push_steps = 1; push_steps <= 10; push_steps++) {
                // Reset environment to initial state for each primitive
                env.reset();
                env.step_simulation();
                
                // Execute push primitive for this number of steps
                bool success = push_controller.execute_push_primitive(target_object, edge_idx, push_steps);
                
                // Get object state after push sequence
                auto final_obj_state = env.get_object_state(target_object);
                if (!final_obj_state) {
                    std::cout << "Edge " << edge_idx << ", Steps " << push_steps << ": Failed to get final object state [FAILED]" << std::endl;
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
                
                std::cout << "Edge " << edge_idx << ", Steps " << push_steps 
                          << ": dx=" << primitive.delta_x 
                          << ", dy=" << primitive.delta_y
                          << ", dθ=" << primitive.delta_theta
                          << (success ? " [SUCCESS]" : " [FAILED]") << std::endl;
                
                // Render final state if visualizing (only for last step to avoid too much output)
                if (visualize && push_steps == 10) {
                    env.render();
                }
            }
            
            // Pause between edges for observation
            if (visualize) {
                std::cout << "Press Enter to continue to next edge..." << std::endl;
                std::cin.get();
            }
        }
        
        // Save primitives to binary file
        std::string output_file = "data/motion_primitives.dat";
        std::ofstream file(output_file, std::ios::binary);
        if (!file) {
            std::cerr << "Failed to create output file: " << output_file << std::endl;
            return 1;
        }
        
        // Write header
        uint32_t count = all_primitives.size();
        file.write(reinterpret_cast<const char*>(&count), sizeof(count));
        
        // Write primitives
        file.write(reinterpret_cast<const char*>(all_primitives.data()), 
                  count * sizeof(NominalPrimitive));
        
        file.close();
        
        std::cout << "\n=== Generation Complete ===" << std::endl;
        std::cout << "Generated " << all_primitives.size() << " primitives" << std::endl;
        std::cout << "Saved to: " << output_file << std::endl;
        std::cout << "File size: " << std::filesystem::file_size(output_file) << " bytes" << std::endl;
        
        // Clean up temporary config
        std::filesystem::remove("tools/primitive_gen_config.yaml");
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}