/**
 * @file test_visual_markers.cpp
 * @brief Simple test for visual edge markers functionality
 */

#include "environment/namo_environment.hpp"
#include "planning/namo_push_controller.hpp"
#include "wavefront/wavefront_planner.hpp"
#include <iostream>

using namespace namo;

int main() {
    try {
        // std::cout << "=== Visual Edge Markers Test ===" << std::endl;
        
        // Initialize system without visualization (to avoid GLFW issues)
        NAMOEnvironment env("data/nominal_primitive_scene.xml", false);
        
        auto movable_objects = env.get_movable_objects();
        if (movable_objects.empty()) {
            std::cerr << "No movable objects found" << std::endl;
            return 1;
        }
        
        std::string object_name = movable_objects[0].name;
        // std::cout << "Testing with object: " << object_name << std::endl;
        
        // Test with simple simulated reachable edges instead of using NAMOPushController
        std::vector<int> reachable_edges = {1, 3, 5, 7, 9, 11};  // Simulate pattern from previous tests
        // std::cout << "Using simulated reachable edges: " << reachable_edges.size() << "/12 [";
        for (size_t i = 0; i < reachable_edges.size(); ++i) {
            // std::cout << reachable_edges[i];
            if (i < reachable_edges.size() - 1) std::cout << ", ";
        }
        // std::cout << "]" << std::endl;
        
        // Test visualization call (should not crash even without GLFW)
        // std::cout << "Testing visualization call..." << std::endl;
        env.visualize_edge_reachability(object_name, reachable_edges);
        // std::cout << "✓ Visualization call completed successfully" << std::endl;
        
        // std::cout << "🎉 Visual markers test completed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}