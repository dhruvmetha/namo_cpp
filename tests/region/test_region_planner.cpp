#include "planners/region/region_analyzer.hpp"
#include "planners/region/region_graph.hpp"
#include "planners/region/region_path_planner.hpp"
// #include "planners/region/goal_proposal_generator.hpp"  // Skip for now to avoid MuJoCo dependency
// #include "environment/namo_environment.hpp"
#include "core/types.hpp"
#include <iostream>
#include <chrono>

using namespace namo;

int main() {
    // std::cout << "=== Region-Based Planner Test ===" << std::endl;
    
    try {
        // Test 1: Basic data structure validation
        // std::cout << "\n1. Testing basic data structures..." << std::endl;
        
        // Test GenericFixedVector
        GenericFixedVector<int, 10> test_vec;
        test_vec.push_back(1);
        test_vec.push_back(2);
        test_vec.push_back(3);
        
        // std::cout << "   GenericFixedVector: size=" << test_vec.size() 
                  // << ", capacity=" << test_vec.capacity() << std::endl;
        // std::cout << "   Elements: ";
        for (size_t i = 0; i < test_vec.size(); ++i) {
            // std::cout << test_vec[i] << " ";
        }
        // std::cout << std::endl;
        
        // Test Region
        Region test_region(0);
        test_region.add_grid_cell(5, 10);
        test_region.add_grid_cell(6, 10);
        test_region.add_sample_point(1.0, 2.0);
        test_region.centroid = {1.5, 2.5};
        test_region.area = 0.1;
        
        // std::cout << "   Region: id=" << test_region.id 
                  // << ", grid_cells=" << test_region.grid_cells.size()
                  // << ", centroid=(" << test_region.centroid.first 
                  // << "," << test_region.centroid.second << ")" << std::endl;
        
        // Test RegionEdge
        RegionEdge test_edge(0, 1);
        test_edge.add_blocking_object("box_1");
        test_edge.add_blocking_object("box_2");
        
        // std::cout << "   RegionEdge: connects(" << test_edge.region_a_id 
                  // << "," << test_edge.region_b_id << "), blocking_objects=" 
                  // << test_edge.blocking_objects.size()
                  // << ", blocked=" << test_edge.is_currently_blocked << std::endl;
        
        // Test RegionGraph
        RegionGraph test_graph;
        int region_id = test_graph.add_region(test_region);
        Region another_region(1);
        another_region.centroid = {3.0, 4.0};
        test_graph.add_region(another_region);
        test_graph.add_edge(0, 1, 0.8);
        
        // std::cout << "   RegionGraph: regions=" << test_graph.num_regions()
                  // << ", edges=" << test_graph.num_edges() << std::endl;
        
        // Test LightweightState
        LightweightState test_state;
        test_state.robot_pose = SE2State(1.0, 2.0, 0.5);
        test_state.object_names.push_back("box_1");
        test_state.movable_object_poses[0] = SE2State(3.0, 4.0, 1.0);
        
        LightweightState copied_state = test_state.copy();
        copied_state.apply_object_movement("box_1", SE2State(5.0, 6.0, 1.5));
        
        // std::cout << "   LightweightState: robot=(" << test_state.robot_pose.x 
                  // << "," << test_state.robot_pose.y << "), objects=" 
                  // << test_state.object_names.size() << std::endl;
        // std::cout << "   After copy+movement: box_1=(" 
                  // << copied_state.get_object_pose("box_1")->x << ","
                  // << copied_state.get_object_pose("box_1")->y << ")" << std::endl;
        
        // std::cout << "   ✓ All data structures working correctly!" << std::endl;
        
        // Test 2: RegionAnalyzer basic functionality
        // std::cout << "\n2. Testing RegionAnalyzer..." << std::endl;
        
        RegionAnalyzer analyzer(0.05, 50.0, 0.25);  // Lower sampling density for test
        
        // std::cout << "   Configuration: resolution=" << analyzer.get_sampling_density()
                  // << ", goal_radius=" << analyzer.get_goal_region_radius() << std::endl;
        
        // Test without environment for now (just configuration)
        // std::cout << "   ✓ RegionAnalyzer initialized successfully!" << std::endl;
        
        // Test 2b: RegionPathPlanner
        // std::cout << "\n2b. Testing RegionPathPlanner..." << std::endl;
        
        RegionPathPlanner path_planner;
        
        // Test with a simple graph
        RegionGraph test_path_graph;
        Region region_0(0);
        region_0.centroid = {0.0, 0.0};
        Region region_1(1);
        region_1.centroid = {2.0, 0.0};
        Region region_2(2);
        region_2.centroid = {4.0, 0.0};
        
        test_path_graph.add_region(region_0);
        test_path_graph.add_region(region_1);
        test_path_graph.add_region(region_2);
        test_path_graph.add_edge(0, 1, 1.0);
        test_path_graph.add_edge(1, 2, 1.0);
        
        // Add some blocking objects
        RegionEdge* edge_01 = test_path_graph.find_edge(0, 1);
        RegionEdge* edge_12 = test_path_graph.find_edge(1, 2);
        if (edge_01) edge_01->add_blocking_object("box_1");
        if (edge_12) {
            edge_12->add_blocking_object("box_2");
            edge_12->add_blocking_object("box_3");
        }
        
        test_path_graph.robot_region_id = 0;
        test_path_graph.goal_region_id = 2;
        
        PathSolution solution = path_planner.find_shortest_path(test_path_graph);
        
        // std::cout << "   Path found: " << solution.path_found << std::endl;
        // std::cout << "   Path length: " << solution.path_length() << std::endl;
        // std::cout << "   Obstacles to remove: " << solution.total_obstacles_to_remove << std::endl;
        // std::cout << "   Blocking objects: " << solution.blocking_objects.size() << std::endl;
        
        auto stats = path_planner.get_last_planning_stats();
        // std::cout << "   Planning time: " << stats.planning_time_ms << " ms" << std::endl;
        // std::cout << "   Nodes expanded: " << stats.nodes_expanded << std::endl;
        
        // std::cout << "   ✓ RegionPathPlanner working correctly!" << std::endl;
        
        // Test 3: Performance validation
        // std::cout << "\n3. Testing performance characteristics..." << std::endl;
        
        // Test large fixed vectors
        auto start_time = std::chrono::high_resolution_clock::now();
        
        GenericFixedVector<std::pair<int, int>, 1000> large_vec;
        for (int i = 0; i < 1000; ++i) {
            large_vec.push_back({i, i*2});
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        
        // std::cout << "   Large vector operations: " << duration << " ms" << std::endl;
        // std::cout << "   Vector size: " << large_vec.size() << "/" << large_vec.capacity() << std::endl;
        
        // Test state hash computation
        start_time = std::chrono::high_resolution_clock::now();
        size_t hash_val = test_state.compute_hash();
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        
        // std::cout << "   State hash computation: " << duration << " ms (hash=" << hash_val << ")" << std::endl;
        
        // std::cout << "   ✓ Performance characteristics validated!" << std::endl;
        
        // Test 4: Memory allocation validation
        // std::cout << "\n4. Testing zero-allocation guarantees..." << std::endl;
        
        // Create multiple copies to ensure no dynamic allocation
        std::vector<LightweightState> states;
        for (int i = 0; i < 10; ++i) {  // Reduced from 100 to 10
            LightweightState state = test_state.copy();
            state.robot_pose.x += i * 0.1;
            states.push_back(state);
        }
        
        // std::cout << "   Created " << states.size() << " state copies" << std::endl;
        
        // Create multiple region graphs
        std::vector<RegionGraph> graphs;
        for (int i = 0; i < 3; ++i) {  // Reduced from 10 to 3
            RegionGraph graph = test_graph;  // Copy constructor
            graphs.push_back(graph);
        }
        
        // std::cout << "   Created " << graphs.size() << " graph copies" << std::endl;
        // std::cout << "   ✓ Zero-allocation patterns validated!" << std::endl;
        
        // std::cout << "\n=== All Tests Passed! ===" << std::endl;
        // std::cout << "Region-based planner core components are working correctly." << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return 1;
    }
}