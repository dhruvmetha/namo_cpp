#include "planners/region/region_path_planner.hpp"
#include <chrono>
#include <climits>
#include <algorithm>

namespace namo {

PathSolution RegionPathPlanner::find_shortest_path(const RegionGraph& graph) {
    if (!graph.is_valid()) {
        PathSolution solution;
        solution.failure_reason = "Invalid region graph (robot or goal region not set)";
        return solution;
    }
    
    return find_path_between_regions(graph, graph.robot_region_id, graph.goal_region_id);
}

PathSolution RegionPathPlanner::find_path_between_regions(const RegionGraph& graph, 
                                                        int start_region_id, 
                                                        int goal_region_id) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Reset statistics
    last_stats_ = PlanningStats{};
    
    // Validate inputs
    if (!is_valid_region_id(start_region_id, graph) || 
        !is_valid_region_id(goal_region_id, graph)) {
        PathSolution solution;
        solution.failure_reason = "Invalid start or goal region ID";
        update_statistics(start_time, std::chrono::high_resolution_clock::now(), false);
        return solution;
    }
    
    // Check if already at goal
    if (start_region_id == goal_region_id) {
        PathSolution solution;
        solution.path_found = true;
        solution.region_path.push_back(start_region_id);
        solution.total_obstacles_to_remove = 0;
        solution.path_cost = 0.0;
        update_statistics(start_time, std::chrono::high_resolution_clock::now(), true);
        return solution;
    }
    
    // Run BFS to find optimal path
    PathSolution solution = run_bfs(graph, start_region_id, goal_region_id);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    update_statistics(start_time, end_time, solution.path_found);
    
    return solution;
}

bool RegionPathPlanner::is_goal_reachable(const RegionGraph& graph) {
    PathSolution solution = find_shortest_path(graph);
    return solution.path_found;
}

std::vector<std::string> RegionPathPlanner::find_critical_blocking_objects(const RegionGraph& graph) {
    PathSolution solution = find_shortest_path(graph);
    
    std::vector<std::string> blocking_objects;
    for (size_t i = 0; i < solution.blocking_objects.size(); ++i) {
        blocking_objects.push_back(solution.blocking_objects[i]);
    }
    
    return blocking_objects;
}

PathSolution RegionPathPlanner::run_bfs(const RegionGraph& graph, int start_region, int goal_region) {
    // Initialize search state
    reset_search_state();
    reset_bfs_queue();
    
    // Start BFS
    visited_[start_region] = true;
    distance_[start_region] = 0;
    obstacles_count_[start_region] = 0;
    parent_[start_region] = -1;
    enqueue_region(start_region);
    
    last_stats_.nodes_expanded = 1;
    
    while (!is_queue_empty()) {
        int current_region = dequeue_region();
        last_stats_.total_regions_explored++;
        
        // Check if we reached the goal
        if (current_region == goal_region) {
            last_stats_.found_optimal_path = true;
            return reconstruct_path(graph, start_region, goal_region);
        }
        
        // Explore neighbors
        const auto& neighbors = graph.get_neighbors(current_region);
        for (size_t i = 0; i < neighbors.size(); ++i) {
            int neighbor_region = neighbors[i];
            last_stats_.total_edges_examined++;
            
            if (!visited_[neighbor_region]) {
                // Find the edge between current and neighbor
                // TODO: Update to use new RegionGraph structure
                // For now, use simple cost of 1 per edge
                const auto& blocking_objects = graph.get_blocking_objects(current_region, neighbor_region);
                int edge_cost = static_cast<int>(blocking_objects.size());
                int new_obstacles_count = obstacles_count_[current_region] + edge_cost;
                int new_distance = distance_[current_region] + 1;
                
                // Mark as visited and update path info
                visited_[neighbor_region] = true;
                distance_[neighbor_region] = new_distance;
                obstacles_count_[neighbor_region] = new_obstacles_count;
                parent_[neighbor_region] = current_region;
                
                enqueue_region(neighbor_region);
                last_stats_.nodes_expanded++;
            }
        }
    }
    
    // No path found
    PathSolution solution;
    solution.failure_reason = "No path exists from robot region to goal region";
    return solution;
}

PathSolution RegionPathPlanner::reconstruct_path(const RegionGraph& graph, 
                                                int start_region, int goal_region) {
    PathSolution solution;
    solution.path_found = true;
    
    // Reconstruct path by following parent pointers
    GenericFixedVector<int, MAX_REGIONS> reverse_path;
    int current = goal_region;
    
    while (current != -1) {
        reverse_path.push_back(current);
        current = parent_[current];
    }
    
    // Reverse the path to get start -> goal order
    for (int i = static_cast<int>(reverse_path.size()) - 1; i >= 0; --i) {
        solution.region_path.push_back(reverse_path[i]);
    }
    
    // Set path metrics
    solution.total_obstacles_to_remove = obstacles_count_[goal_region];
    solution.path_cost = static_cast<double>(obstacles_count_[goal_region]);
    
    // Extract blocking objects from the path
    extract_blocking_objects_from_path(graph, solution);
    
    return solution;
}

void RegionPathPlanner::extract_blocking_objects_from_path(const RegionGraph& graph, 
                                                         PathSolution& solution) {
    // Clear existing blocking objects
    solution.blocking_objects.clear();
    
    // Walk through path edges and collect all blocking objects
    for (size_t i = 0; i < solution.region_path.size() - 1; ++i) {
        int region_a = solution.region_path[i];
        int region_b = solution.region_path[i + 1];
        
        // TODO: Update to use new RegionGraph structure
        const auto& blocking_objects = graph.get_blocking_objects(region_a, region_b);
        if (!blocking_objects.empty()) {
            // Add all blocking objects from this edge
            for (size_t j = 0; j < blocking_objects.size(); ++j) {
                const std::string& obj_name = blocking_objects[j];
                
                // Avoid duplicates (object might block multiple edges)
                bool already_added = false;
                for (size_t k = 0; k < solution.blocking_objects.size(); ++k) {
                    if (solution.blocking_objects[k] == obj_name) {
                        already_added = true;
                        break;
                    }
                }
                
                if (!already_added && solution.blocking_objects.size() < solution.blocking_objects.capacity()) {
                    solution.blocking_objects.push_back(obj_name);
                }
            }
        }
    }
    
    // Verify obstacle count matches
    if (static_cast<int>(solution.blocking_objects.size()) != solution.total_obstacles_to_remove) {
        // This might happen due to duplicate objects blocking multiple edges
        // Update the count to match actual unique objects
        solution.total_obstacles_to_remove = static_cast<int>(solution.blocking_objects.size());
        solution.path_cost = static_cast<double>(solution.total_obstacles_to_remove);
    }
}

void RegionPathPlanner::update_statistics(const std::chrono::high_resolution_clock::time_point& start,
                                         const std::chrono::high_resolution_clock::time_point& end,
                                         bool found_path) const {
    last_stats_.planning_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    last_stats_.found_optimal_path = found_path;
}

} // namespace namo