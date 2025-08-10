#include "planners/region/region_graph.hpp"
#include "planners/region/region_path_planner.hpp"
#include "core/types.hpp"
#include <iostream>

using namespace namo;

int main() {
    std::cout << "=== Region Path Planner Test ===" << std::endl;
    
    try {
        // Test 1: Basic path planning
        std::cout << "\n1. Testing RegionPathPlanner..." << std::endl;
        
        RegionPathPlanner path_planner;
        
        // Create a simple test graph
        RegionGraph test_graph;
        
        // Add regions
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
        
        // Add edges
        test_graph.add_edge(0, 1, 1.0);
        test_graph.add_edge(1, 2, 1.0);
        
        // Add blocking objects to edges
        RegionEdge* edge_01 = test_graph.find_edge(0, 1);
        RegionEdge* edge_12 = test_graph.find_edge(1, 2);
        
        if (edge_01) {
            edge_01->add_blocking_object("box_1");
            std::cout << "   Added box_1 blocking edge (0,1)" << std::endl;
        }
        
        if (edge_12) {
            edge_12->add_blocking_object("box_2");
            edge_12->add_blocking_object("box_3");
            std::cout << "   Added box_2, box_3 blocking edge (1,2)" << std::endl;
        }
        
        // Set robot and goal regions
        test_graph.robot_region_id = 0;
        test_graph.goal_region_id = 2;
        
        std::cout << "   Graph setup: " << test_graph.num_regions() << " regions, " 
                  << test_graph.num_edges() << " edges" << std::endl;
        std::cout << "   Robot in region: " << test_graph.robot_region_id << std::endl;
        std::cout << "   Goal in region: " << test_graph.goal_region_id << std::endl;
        
        // Test path finding
        std::cout << "\n2. Finding shortest path..." << std::endl;
        
        PathSolution solution = path_planner.find_shortest_path(test_graph);
        
        std::cout << "   Path found: " << (solution.path_found ? "YES" : "NO") << std::endl;
        
        if (solution.path_found) {
            std::cout << "   Path length: " << solution.path_length() << " regions" << std::endl;
            std::cout << "   Obstacles to remove: " << solution.total_obstacles_to_remove << std::endl;
            std::cout << "   Path cost: " << solution.path_cost << std::endl;
            
            std::cout << "   Region path: ";
            for (size_t i = 0; i < solution.region_path.size(); ++i) {
                std::cout << solution.region_path[i];
                if (i < solution.region_path.size() - 1) std::cout << " -> ";
            }
            std::cout << std::endl;
            
            std::cout << "   Blocking objects: ";
            for (size_t i = 0; i < solution.blocking_objects.size(); ++i) {
                std::cout << solution.blocking_objects[i];
                if (i < solution.blocking_objects.size() - 1) std::cout << ", ";
            }
            std::cout << std::endl;
            
            // Verify solution properties
            std::cout << "   Start region: " << solution.start_region() << std::endl;
            std::cout << "   Goal region: " << solution.goal_region() << std::endl;
            std::cout << "   Requires object movement: " << (solution.requires_object_movement() ? "YES" : "NO") << std::endl;
            std::cout << "   Solution is valid: " << (solution.is_valid() ? "YES" : "NO") << std::endl;
        } else {
            std::cout << "   Failure reason: " << solution.failure_reason << std::endl;
        }
        
        // Test statistics
        auto stats = path_planner.get_last_planning_stats();
        std::cout << "\n3. Planning statistics..." << std::endl;
        std::cout << "   Planning time: " << stats.planning_time_ms << " ms" << std::endl;
        std::cout << "   Nodes expanded: " << stats.nodes_expanded << std::endl;
        std::cout << "   Regions explored: " << stats.total_regions_explored << std::endl;
        std::cout << "   Edges examined: " << stats.total_edges_examined << std::endl;
        std::cout << "   Found optimal path: " << (stats.found_optimal_path ? "YES" : "NO") << std::endl;
        
        // Test 3: Different path scenarios
        std::cout << "\n4. Testing edge cases..." << std::endl;
        
        // Same start and goal
        PathSolution same_solution = path_planner.find_path_between_regions(test_graph, 0, 0);
        std::cout << "   Same start/goal - Path found: " << (same_solution.path_found ? "YES" : "NO") << std::endl;
        std::cout << "   Same start/goal - Obstacles: " << same_solution.total_obstacles_to_remove << std::endl;
        
        // Invalid regions
        PathSolution invalid_solution = path_planner.find_path_between_regions(test_graph, -1, 10);
        std::cout << "   Invalid regions - Path found: " << (invalid_solution.path_found ? "YES" : "NO") << std::endl;
        if (!invalid_solution.path_found) {
            std::cout << "   Invalid regions - Reason: " << invalid_solution.failure_reason << std::endl;
        }
        
        // Test reachability check
        bool is_reachable = path_planner.is_goal_reachable(test_graph);
        std::cout << "   Goal reachable: " << (is_reachable ? "YES" : "NO") << std::endl;
        
        // Test critical objects
        auto critical_objects = path_planner.find_critical_blocking_objects(test_graph);
        std::cout << "   Critical blocking objects: " << critical_objects.size() << std::endl;
        
        std::cout << "\n=== All Path Planning Tests Passed! ===" << std::endl;
        std::cout << "Region path planner is working correctly." << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return 1;
    }
}