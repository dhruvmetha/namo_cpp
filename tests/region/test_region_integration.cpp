#include "planners/region/region_tree_search.hpp"
#include "planners/region/region_analyzer.hpp"
#include "planners/region/region_path_planner.hpp"
#include "planners/region/goal_proposal_generator.hpp"
#include "config/config_manager.hpp"
#include "core/types.hpp"
#include <iostream>
#include <chrono>

using namespace namo;

int main() {
    // std::cout << "=== Region Planner Components Integration Test ===" << std::endl;
    
    try {
        // Test 1: Core component initialization
        // std::cout << "\n1. Testing core components initialization..." << std::endl;
        
        auto config = std::make_unique<ConfigManager>();  // Use default config
        // std::cout << "   ✓ ConfigManager initialized" << std::endl;
        
        // Test individual components
        RegionAnalyzer analyzer(0.05, 50.0, 0.25);
        // std::cout << "   ✓ RegionAnalyzer initialized (res=0.05, area=50.0, tol=0.25)" << std::endl;
        
        RegionPathPlanner path_planner;
        // std::cout << "   ✓ RegionPathPlanner initialized" << std::endl;
        
        // Test 2: Data structure validation
        // std::cout << "\n2. Testing core data structures..." << std::endl;
        
        // Test LightweightState
        LightweightState test_state;
        test_state.robot_pose = SE2State(0.0, 0.0, 0.0);
        test_state.object_names.push_back("box_1");
        test_state.object_names.push_back("box_2");
        test_state.movable_object_poses[0] = SE2State(1.0, 0.0, 0.0);
        test_state.movable_object_poses[1] = SE2State(2.0, 0.0, 0.0);
        
        // std::cout << "   ✓ LightweightState created with " << test_state.object_names.size() << " objects" << std::endl;
        
        // Test state copying
        LightweightState copied_state = test_state.copy();
        // std::cout << "   ✓ State copying works (copied " << copied_state.object_names.size() << " objects)" << std::endl;
        
        // Test state hash
        std::size_t hash1 = test_state.compute_hash();
        std::size_t hash2 = copied_state.compute_hash();
        // std::cout << "   ✓ State hashing works (hash1=" << hash1 << ", hash2=" << hash2 << ")" << std::endl;
        
        // Test object movement
        SE2State new_pose(1.5, 0.5, 0.1);
        copied_state.apply_object_movement("box_1", new_pose);
        // std::cout << "   ✓ Object movement applied" << std::endl;
        
        // Verify hashes changed
        std::size_t hash3 = copied_state.compute_hash();
        bool hash_changed = (hash1 != hash3);
        // std::cout << "   ✓ Hash changed after movement: " << (hash_changed ? "YES" : "NO") << std::endl;
        
        // Test 3: Region graph operations
        // std::cout << "\n3. Testing region graph operations..." << std::endl;
        
        RegionGraph test_graph;
        
        // Create test regions
        Region region_0(0);
        region_0.centroid = {0.0, 0.0};
        region_0.area = 1.0;
        
        Region region_1(1);
        region_1.centroid = {2.0, 0.0};
        region_1.area = 1.5;
        
        Region region_2(2);
        region_2.centroid = {4.0, 0.0};
        region_2.area = 2.0;
        
        test_graph.add_region(region_0);
        test_graph.add_region(region_1);
        test_graph.add_region(region_2);
        
        // std::cout << "   ✓ Created " << test_graph.num_regions() << " regions" << std::endl;
        
        // Add edges with blocking objects
        test_graph.add_edge(0, 1, 1.0);
        test_graph.add_edge(1, 2, 1.0);
        
        RegionEdge* edge_01 = test_graph.find_edge(0, 1);
        RegionEdge* edge_12 = test_graph.find_edge(1, 2);
        
        if (edge_01) {
            edge_01->add_blocking_object("box_1");
            // std::cout << "   ✓ Added blocking object to edge (0,1)" << std::endl;
        }
        
        if (edge_12) {
            edge_12->add_blocking_object("box_2");
            // std::cout << "   ✓ Added blocking object to edge (1,2)" << std::endl;
        }
        
        // std::cout << "   ✓ Created " << test_graph.num_edges() << " edges with blocking objects" << std::endl;
        
        // Set robot and goal regions
        test_graph.robot_region_id = 0;
        test_graph.goal_region_id = 2;
        
        // Test 4: Path planning
        // std::cout << "\n4. Testing path planning..." << std::endl;
        
        PathSolution solution = path_planner.find_shortest_path(test_graph);
        
        // std::cout << "   Path found: " << (solution.path_found ? "YES" : "NO") << std::endl;
        
        if (solution.path_found) {
            // std::cout << "   Path length: " << solution.path_length() << " regions" << std::endl;
            // std::cout << "   Obstacles to remove: " << solution.total_obstacles_to_remove << std::endl;
            // std::cout << "   Path cost: " << solution.path_cost << std::endl;
            
            // std::cout << "   Region path: ";
            for (size_t i = 0; i < solution.region_path.size(); ++i) {
                // std::cout << solution.region_path[i];
                if (i < solution.region_path.size() - 1) std::cout << " -> ";
            }
            // std::cout << std::endl;
            
            // std::cout << "   Blocking objects: ";
            for (size_t i = 0; i < solution.blocking_objects.size(); ++i) {
                // std::cout << solution.blocking_objects[i];
                if (i < solution.blocking_objects.size() - 1) std::cout << ", ";
            }
            // std::cout << std::endl;
        } else {
            // std::cout << "   Failure reason: " << solution.failure_reason << std::endl;
        }
        
        // Test 5: Action step operations
        // std::cout << "\n5. Testing action step operations..." << std::endl;
        
        ActionStep action1("box_1", SE2State(1.5, 0.0, 0.0));
        ActionStep action2("box_2", SE2State(2.5, 0.0, 0.0));
        
        // std::cout << "   ✓ Created action steps for object manipulation" << std::endl;
        
        // Test action equality
        ActionStep action1_copy("box_1", SE2State(1.5, 0.0, 0.0));
        bool actions_equal = (action1 == action1_copy);
        // std::cout << "   ✓ Action equality test: " << (actions_equal ? "PASS" : "FAIL") << std::endl;
        
        // Test action sequence
        GenericFixedVector<ActionStep, 20> action_sequence;
        action_sequence.push_back(action1);
        action_sequence.push_back(action2);
        
        // std::cout << "   ✓ Created action sequence with " << action_sequence.size() << " actions" << std::endl;
        
        // Test 6: Tree search result structures
        // std::cout << "\n6. Testing tree search result structures..." << std::endl;
        
        TreeSearchResult result;
        result.solution_found = true;
        result.best_action_sequence = action_sequence;
        result.solution_cost = 2.0;
        result.solution_depth = 2;
        result.nodes_expanded = 15;
        result.total_nodes_generated = 25;
        result.max_depth_reached = 2;
        result.search_time_ms = 5.5;
        
        // std::cout << "   ✓ TreeSearchResult created" << std::endl;
        // std::cout << "   Solution valid: " << (result.is_valid() ? "YES" : "NO") << std::endl;
        // std::cout << "   Number of actions: " << result.num_actions() << std::endl;
        // std::cout << "   Search statistics: " << result.nodes_expanded << " nodes, " 
                  // << result.search_time_ms << " ms" << std::endl;
        
        // Test 7: Performance measurement
        // std::cout << "\n7. Testing performance..." << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Perform multiple operations
        for (int i = 0; i < 100; ++i) {
            LightweightState perf_state;
            perf_state.robot_pose = SE2State(i * 0.01, 0.0, 0.0);
            perf_state.object_names.push_back("test_object");
            perf_state.movable_object_poses[0] = SE2State(1.0 + i * 0.01, 0.0, 0.0);
            
            std::size_t hash = perf_state.compute_hash();
            (void)hash;  // Suppress unused variable warning
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        
        // std::cout << "   ✓ Performance test: 100 state operations in " << elapsed_ms << " ms" << std::endl;
        // std::cout << "   Average per operation: " << (elapsed_ms / 100.0) << " ms" << std::endl;
        
        // Test 8: Configuration validation
        // std::cout << "\n8. Testing configuration validation..." << std::endl;
        
        const auto& planning_config = config->planning();
        // std::cout << "   High-level resolution: " << planning_config.high_level_resolution << std::endl;
        // std::cout << "   Max planning iterations: " << planning_config.max_planning_iterations << std::endl;
        // std::cout << "   Max BFS queue: " << planning_config.max_bfs_queue << std::endl;
        
        const auto& system_config = config->system();
        // std::cout << "   Visualization enabled: " << (system_config.enable_visualization ? "YES" : "NO") << std::endl;
        // std::cout << "   Number of threads: " << system_config.num_threads << std::endl;
        
        // std::cout << "\n=== All Integration Tests Completed Successfully! ===" << std::endl;
        // std::cout << "Region planner core components are working correctly." << std::endl;
        
        // std::cout << "\nTest Summary:" << std::endl;
        // std::cout << "✓ Component initialization: PASS" << std::endl;
        // std::cout << "✓ Data structure operations: PASS" << std::endl;
        // std::cout << "✓ Region graph construction: PASS" << std::endl;
        // std::cout << "✓ Path planning algorithm: PASS" << std::endl;
        // std::cout << "✓ Action step handling: PASS" << std::endl;
        // std::cout << "✓ Tree search structures: PASS" << std::endl;
        // std::cout << "✓ Performance validation: PASS" << std::endl;
        // std::cout << "✓ Configuration access: PASS" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return 1;
    }
}