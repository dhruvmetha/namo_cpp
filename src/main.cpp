#include <iostream>
#include <string>

#include "core/parameter_loader.hpp"
#include "core/mujoco_wrapper.hpp"
#include "core/memory_manager.hpp"
#include "environment/namo_environment.hpp"
#include "wavefront/wavefront_planner.hpp"
#include "planning/namo_push_controller.hpp"

using namespace namo;

int main(int argc, char* argv[]) {
    // std::cout << "NAMO Standalone - High-Performance Navigation Among Movable Obstacles" << std::endl;
    // std::cout << "=================================================================" << std::endl;
    
    // Check command line arguments
    if (argc != 2) {
        // std::cout << "Usage: " << argv[0] << " <config.yaml>" << std::endl;
        // std::cout << "Example: " << argv[0] << " config/namo_config.yaml" << std::endl;
        return 1;
    }
    
    try {
        // Load configuration
        // std::cout << "Loading configuration from: " << argv[1] << std::endl;
        FastParameterLoader params(argv[1]);
        // std::cout << "Configuration loaded successfully!" << std::endl;
        
        // Initialize memory manager
        NAMOMemoryManager memory_manager;
        // std::cout << "Memory manager initialized" << std::endl;
        
        // Create environment
        // std::cout << "Loading xml_path..." << std::endl;
        std::string xml_path;
        try {
            xml_path = params.get_string("xml_path");
            // std::cout << "Loaded xml_path: " << xml_path << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Failed to load xml_path: " << e.what() << std::endl;
            return 1;
        }
        
        // std::cout << "Checking visualize setting..." << std::endl;
        bool visualize = false;
        try {
            // std::cout << "has_key('visualize'): " << params.has_key("visualize") << std::endl;
            if (params.has_key("visualize")) {
                // std::cout << "visualize key found, getting value..." << std::endl;
                visualize = params.get_bool("visualize");
                // std::cout << "visualize value: " << visualize << std::endl;
            } else {
                // std::cout << "visualize key not found, using default false" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error checking visualize: " << e.what() << std::endl;
            visualize = false;
        }
        
        // std::cout << "Checking data collection setting..." << std::endl;
        bool enable_logging = false;
        if (params.has_key("data_collection.enabled")) {
            // std::cout << "data_collection.enabled key found, getting value..." << std::endl;
            enable_logging = params.get_bool("data_collection.enabled");
        } else {
            // std::cout << "data_collection.enabled key not found, using default false" << std::endl;
        }
        
        // std::cout << "Creating NAMO environment..." << std::endl;
        // std::cout << "  XML path: " << xml_path << std::endl;
        // std::cout << "  Visualization: " << (visualize ? "enabled" : "disabled") << std::endl;
        // std::cout << "  Logging: " << (enable_logging ? "enabled" : "disabled") << std::endl;
        
        NAMOEnvironment env(xml_path, visualize, enable_logging);
        // std::cout << "Environment created successfully!" << std::endl;
        
        // Get robot size for wavefront planner
        const auto& robot_info = env.get_robot_info();
        std::vector<double> robot_size = {robot_info.size[0], robot_info.size[1]};
        
        // Create incremental wavefront planner
        // std::cout << "Attempting to load wavefront planner resolution..." << std::endl;
        double resolution;
        try {
            resolution = params.get_double("wavefront_planner.resolution");
            // std::cout << "Loaded resolution: " << resolution << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Failed to load resolution: " << e.what() << std::endl;
            std::cerr << "Using default resolution: 0.1" << std::endl;
            resolution = 0.1;
        }
        // std::cout << "Creating wavefront planner with resolution: " << resolution << std::endl;
        
        WavefrontPlanner wavefront_planner(resolution, env, robot_size);
        // std::cout << "Wavefront planner created successfully!" << std::endl;
        
        // Get goal position
        std::array<double, 2> robot_goal = params.get_array<2>("robot_goal");
        env.set_robot_goal(robot_goal);
        // std::cout << "Robot goal set to: [" << robot_goal[0] << ", " << robot_goal[1] << "]" << std::endl;
        
        // Create NAMO push controller
        // std::cout << "Creating NAMO push controller..." << std::endl;
        NAMOPushController push_controller(env, wavefront_planner, 10, 250, 1.0);
        // std::cout << "Push controller created successfully!" << std::endl;
        
        // Run basic test
        // std::cout << "\n--- Running Basic Test ---" << std::endl;
        
        // Test environment bounds
        auto bounds = env.get_environment_bounds();
        // std::cout << "Environment bounds: [" 
                  // << bounds[0] << ", " << bounds[1] << "] x [" 
                  // << bounds[2] << ", " << bounds[3] << "]" << std::endl;
        
        // Test object information
        // std::cout << "Environment contains:" << std::endl;
        // std::cout << "  Static objects: " << env.get_num_static() << std::endl;
        // std::cout << "  Movable objects: " << env.get_num_movable() << std::endl;
        
        // List objects
        const auto& static_objects = env.get_static_objects();
        for (size_t i = 0; i < env.get_num_static(); i++) {
            const auto& obj = static_objects[i];
            // std::cout << "    Static: " << obj.name 
                      // << " at [" << obj.position[0] << ", " << obj.position[1] << "]"
                      // << " size [" << obj.size[0] << ", " << obj.size[1] << "]" << std::endl;
        }
        
        const auto& movable_objects = env.get_movable_objects();
        for (size_t i = 0; i < env.get_num_movable(); i++) {
            const auto& obj = movable_objects[i];
            // std::cout << "    Movable: " << obj.name 
                      // << " at [" << obj.position[0] << ", " << obj.position[1] << "]"
                      // << " size [" << obj.size[0] << ", " << obj.size[1] << "]" << std::endl;
        }
        
        // Test robot position
        const auto* robot_state = env.get_robot_state();
        if (robot_state) {
            // std::cout << "Robot at: [" << robot_state->position[0] << ", " 
                      // << robot_state->position[1] << ", " << robot_state->position[2] << "]" << std::endl;
        }
        
        // Test wavefront computation
        // std::cout << "\n--- Testing Wavefront Computation ---" << std::endl;
        std::vector<double> start_pos = {robot_state->position[0], robot_state->position[1]};
        
        auto start_time = std::chrono::high_resolution_clock::now();
        bool updated = wavefront_planner.update_wavefront(env, start_pos);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        // std::cout << "Wavefront update took: " << duration.count() << " microseconds" << std::endl;
        // std::cout << "Wavefront updated: " << (updated ? "yes" : "no") << std::endl;
        
        // Test goal reachability
        bool goal_reachable = wavefront_planner.is_goal_reachable(robot_goal);
        // std::cout << "Goal reachable: " << (goal_reachable ? "yes" : "no") << std::endl;
        
        // Save initial wavefront for visualization
        wavefront_planner.save_wavefront_iteration("debug_wavefront", 0);
        // std::cout << "Initial wavefront saved for debugging" << std::endl;
        
        // Test push controller
        std::array<std::string, 20> reachable_objects;
        size_t reachable_count;
        size_t num_reachable = push_controller.get_reachable_objects(reachable_objects, reachable_count);

        
        if (num_reachable > 0) {
            
            // Generate edge points for the first reachable object
            std::array<std::array<double, 2>, 16> edge_points;
            std::array<std::array<double, 2>, 16> mid_points;
            size_t edge_count, mid_count;
            size_t num_edges = push_controller.generate_edge_points(reachable_objects[0], edge_points, mid_points, edge_count, mid_count);
            
            if (num_edges > 0) {
                // Get object state for comprehensive debugging
                auto obj_state = env.get_object_state(reachable_objects[0]);
                auto robot_state = env.get_robot_state();
                
                // Position camera to follow the action if visualizing
                if (visualize) {
                    // Focus camera on the object being pushed
                    auto obj_state = env.get_object_state(reachable_objects[0]);
                    if (obj_state) {
                        std::array<double, 3> focus_point = {
                            obj_state->position[0], obj_state->position[1], 0.0
                        };
                        env.set_camera_lookat(focus_point);
                        env.set_camera_position(6.0, 0.0, -45.0);  // Angled top-down view
                    }
                }

                env.set_zero_velocity();
                env.step_simulation();

                // std::cout << "\n--- MPC-style Push Sequence with Wavefront Debugging ---" << std::endl;
                
                // Array of edge indices to test in sequence (simulating MPC decisions)
                int edge_indices[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
                int durations[] = {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};  // Different durations for variety
                
                for (int i = 0; i < 12; i++) {
                    // std::cout << "\n--- MPC Iteration " << (i+1) << " ---" << std::endl;
                    
                    // Execute push primitive (simulating MPC control step)
                    bool push_success = push_controller.execute_push_primitive(
                        reachable_objects[0], edge_indices[i], durations[i]);
                    
                    // std::cout << "Push primitive " << edge_indices[i] << " (duration=" << durations[i] 
                              // << "): " << (push_success ? "SUCCESS" : "FAILED") << std::endl;
                    
                    // Update robot state and position
                    robot_state = env.get_robot_state();
                    start_pos = {robot_state->position[0], robot_state->position[1]};
                    // std::cout << "Robot position: [" << start_pos[0] << ", " << start_pos[1] << "]" << std::endl;
                    
                    // Update wavefront (recompute reachability)
                    updated = wavefront_planner.update_wavefront(env, start_pos);
                    // std::cout << "Wavefront updated: " << (updated ? "yes" : "no") << std::endl;
                    
                    // Save wavefront for this MPC iteration
                    wavefront_planner.save_wavefront_iteration("mpc_wavefront", i+1);
                    
                    // Optional: test goal reachability after each step
                    bool goal_still_reachable = wavefront_planner.is_goal_reachable(robot_goal);
                    // std::cout << "Goal still reachable: " << (goal_still_reachable ? "yes" : "no") << std::endl;
                }

            }
        }
        
        // // Push controller memory statistics
        size_t primitives_used, states_used;
        push_controller.get_memory_stats(primitives_used, states_used);
        // std::cout << "Push controller memory: " << primitives_used << " primitives, " << states_used << " states" << std::endl;
        
        // Test memory pool statistics
        // std::cout << "\n--- Memory Pool Statistics ---" << std::endl;
        memory_manager.print_statistics();
        
        // Performance statistics
        const auto& stats = wavefront_planner.get_statistics();
        // std::cout << "\n--- Performance Statistics ---" << std::endl;
        // std::cout << "Total planning time: " << stats.total_planning_time << " ms" << std::endl;
        // std::cout << "Wavefront time: " << stats.wavefront_time << " ms" << std::endl;
        // std::cout << "Change detection time: " << stats.change_detection_time << " ms" << std::endl;
        // std::cout << "Wavefront updates: " << stats.wavefront_updates << std::endl;
        
        // Interactive mode if visualization is enabled
        if (visualize) {
            // std::cout << "\n--- Interactive Mode ---" << std::endl;
            // std::cout << "Visualization enabled. Use mouse to interact with the scene." << std::endl;
            // std::cout << "Press CTRL+C to exit." << std::endl;

                        
            // Set up top-down camera view
            const auto* robot_state = env.get_robot_state();
            if (robot_state) {
                // Position camera to look down at the robot from above
                std::array<double, 3> robot_center = {
                    robot_state->position[0], 
                    robot_state->position[1], 
                    0.0  // Ground level
                };
                
                // Set camera to look at robot position
                env.set_camera_lookat(robot_center);
                
                // Top-down view: negative elevation to look down from above
                env.set_camera_position(8.0, 0.0, -60.0);
                
                // std::cout << "Camera positioned for top-down view at robot: [" 
                          // << robot_state->position[0] << ", " << robot_state->position[1] << "]" << std::endl;
            }
            
            int frame_count = 0;
            while (!env.should_close()) {
                env.render();
                
                // Periodic wavefront updates for testing
                if (frame_count % 60 == 0) {  // Every ~1 second at 60 FPS
                    wavefront_planner.update_wavefront(env, start_pos);
                }
                
                frame_count++;
                
                // Simple exit condition
                if (frame_count > 10000) break;
            }
        }
        
        // std::cout << "\n--- Test Completed Successfully ---" << std::endl;
        // std::cout << "All basic functionality verified!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}