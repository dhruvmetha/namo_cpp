#include "planners/region/region_tree_search.hpp"
#include "environment/namo_environment.hpp"
#include <chrono>
#include <cmath>
#include <algorithm>

namespace namo {

RegionTreeSearch::RegionTreeSearch(NAMOEnvironment& env, int max_depth, int max_goal_proposals)
    : env_(env)
    , max_depth_(max_depth)
    , max_goal_proposals_(max_goal_proposals)
    , goal_tolerance_(0.25)
    , max_nodes_(MAX_SEARCH_NODES)
    , active_nodes_(0) {
    
    // Initialize components
    region_analyzer_ = std::make_unique<RegionAnalyzer>(0.05, 50.0, goal_tolerance_);
    path_planner_ = std::make_unique<RegionPathPlanner>();
    goal_generator_ = std::make_unique<GoalProposalGenerator>(env, 2.0, 50);
}

TreeSearchResult RegionTreeSearch::search(const LightweightState& initial_state, 
                                         const SE2State& robot_goal,
                                         int depth_limit) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Use provided depth limit or default
    int effective_depth = (depth_limit > 0) ? depth_limit : max_depth_;
    
    // Reset search state
    reset_search_state();
    
    // Run the tree search
    TreeSearchResult result = run_tree_search(initial_state, robot_goal, effective_depth);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    update_search_statistics(start_time, end_time);
    
    last_result_ = result;
    return result;
}

GenericFixedVector<std::string, 10> RegionTreeSearch::get_blocking_objects(
    const LightweightState& state, const SE2State& robot_goal) {
    
    // Build region graph for current state
    RegionGraph graph = region_analyzer_->discover_regions(env_, robot_goal);
    
    // Update with current object positions
    // TODO: Implement lightweight state → environment update
    
    // Find shortest path
    PathSolution path = path_planner_->find_shortest_path(graph);
    
    GenericFixedVector<std::string, 10> blocking_objects;
    if (path.path_found) {
        for (size_t i = 0; i < path.blocking_objects.size() && i < blocking_objects.capacity(); ++i) {
            blocking_objects.push_back(path.blocking_objects[i]);
        }
    }
    
    return blocking_objects;
}

bool RegionTreeSearch::is_robot_at_goal(const LightweightState& state, 
                                       const SE2State& robot_goal,
                                       double tolerance) {
    double distance = std::sqrt(
        (state.robot_pose.x - robot_goal.x) * (state.robot_pose.x - robot_goal.x) +
        (state.robot_pose.y - robot_goal.y) * (state.robot_pose.y - robot_goal.y)
    );
    
    return distance <= tolerance;
}

TreeSearchResult RegionTreeSearch::run_tree_search(const LightweightState& initial_state,
                                                  const SE2State& robot_goal,
                                                  int depth_limit) {
    TreeSearchResult result;
    
    // Check if already at goal
    if (is_robot_at_goal(initial_state, robot_goal)) {
        result.solution_found = true;
        result.solution_cost = 0.0;
        result.solution_depth = 0;
        return result;
    }
    
    // Create root node
    int root_idx = allocate_node();
    if (root_idx < 0) {
        result.failure_reason = "Failed to allocate root node";
        return result;
    }
    
    RegionSearchNode& root = get_node(root_idx);
    root.state = initial_state;
    root.depth = 0;
    root.parent_node_idx = -1;
    
    // Evaluate root node
    if (!evaluate_node(root, robot_goal)) {
        result.failure_reason = "Root node evaluation failed - no path to goal exists";
        return result;
    }
    
    // Initialize search queue
    search_queue_.clear();
    search_queue_.push_back(root_idx);
    
    result.nodes_expanded = 1;
    result.total_nodes_generated = 1;
    
    // BFS through tree
    while (!search_queue_.empty() && result.nodes_expanded < max_nodes_) {
        // Get next node from queue
        int current_idx = search_queue_[0];
        
        // Remove from front of queue (shift elements)
        for (size_t i = 0; i < search_queue_.size() - 1; ++i) {
            search_queue_[i] = search_queue_[i + 1];
        }
        search_queue_.resize(search_queue_.size() - 1);
        
        RegionSearchNode& current = get_node(current_idx);
        
        // Check if this is a solution
        if (current.is_goal_reached) {
            extract_best_solution(current_idx);
            result.solution_found = true;
            result.solution_depth = current.depth;
            result.solution_cost = current.cost_estimate;
            break;
        }
        
        // Expand node if within depth limit
        if (current.depth < depth_limit) {
            auto children = expand_node(current_idx, robot_goal);
            
            for (size_t i = 0; i < children.size(); ++i) {
                int child_idx = children[i];
                RegionSearchNode& child = get_node(child_idx);
                
                result.total_nodes_generated++;
                result.max_depth_reached = std::max(result.max_depth_reached, child.depth);
                
                // Add valid children to search queue
                if (child.is_valid && !is_duplicate_state(child.state, child_idx)) {
                    if (search_queue_.size() < search_queue_.capacity()) {
                        search_queue_.push_back(child_idx);
                    }
                }
            }
        }
        
        result.nodes_expanded++;
    }
    
    if (!result.solution_found) {
        if (result.nodes_expanded >= max_nodes_) {
            result.failure_reason = "Search exhausted node limit without finding solution";
        } else {
            result.failure_reason = "No solution found within depth limit";
        }
    }
    
    return result;
}

GenericFixedVector<int, 100> RegionTreeSearch::expand_node(int node_idx, const SE2State& robot_goal) {
    GenericFixedVector<int, 100> children;
    RegionSearchNode& parent = get_node(node_idx);
    
    // Get blocking objects for this state
    blocking_objects_ = get_blocking_objects(parent.state, robot_goal);
    
    if (blocking_objects_.empty()) {
        // No blocking objects means robot can reach goal directly
        parent.is_goal_reached = true;
        return children;
    }
    
    // For each blocking object, generate goal proposals
    for (size_t obj_idx = 0; obj_idx < blocking_objects_.size(); ++obj_idx) {
        const std::string& object_name = blocking_objects_[obj_idx];
        
        // Generate goal proposals for this object
        goal_proposals_ = goal_generator_->generate_proposals(
            object_name, parent.state, max_goal_proposals_);
        
        // Create child node for each proposal
        for (size_t prop_idx = 0; prop_idx < goal_proposals_.size(); ++prop_idx) {
            int child_idx = allocate_node();
            if (child_idx < 0) break;  // Out of nodes
            
            RegionSearchNode& child = get_node(child_idx);
            
            // Initialize child
            child.state = parent.state.copy();
            child.depth = parent.depth + 1;
            child.parent_node_idx = node_idx;
            child.action_sequence = parent.action_sequence;
            
            // Apply the action
            ActionStep action(object_name, goal_proposals_[prop_idx]);
            child.add_action(action);
            child.state.apply_object_movement(object_name, goal_proposals_[prop_idx]);
            
            // Evaluate child node
            if (evaluate_node(child, robot_goal)) {
                children.push_back(child_idx);
            } else {
                // Mark as invalid but keep allocated for potential debugging
                child.is_valid = false;
            }
            
            if (children.size() >= children.capacity()) break;
        }
        
        if (children.size() >= children.capacity()) break;
    }
    
    return children;
}

bool RegionTreeSearch::evaluate_node(RegionSearchNode& node, const SE2State& robot_goal) {
    // Check if robot reached goal in this state
    node.is_goal_reached = is_robot_at_goal(node.state, robot_goal);
    
    if (node.is_goal_reached) {
        node.cost_estimate = static_cast<double>(node.depth);
        node.num_obstacles_remaining = 0;
        node.path_exists = true;
        return true;
    }
    
    // Update region graph for this state
    update_node_region_graph(node, robot_goal);
    
    if (!node.region_graph.is_valid()) {
        node.is_valid = false;
        return false;
    }
    
    // Check if path to goal exists
    PathSolution path = path_planner_->find_shortest_path(node.region_graph);
    node.path_exists = path.path_found;
    
    if (!node.path_exists) {
        node.is_valid = false;
        return false;
    }
    
    // Update node metrics
    node.num_obstacles_remaining = path.total_obstacles_to_remove;
    node.cost_estimate = compute_node_cost(node);
    
    return true;
}

void RegionTreeSearch::update_node_region_graph(RegionSearchNode& node, const SE2State& robot_goal) {
    // For now, recompute region graph from scratch
    // TODO: Implement incremental update for better performance
    node.region_graph = region_analyzer_->discover_regions(env_, robot_goal);
    
    // Update blocking objects based on current state
    // TODO: Apply lightweight state to environment before region analysis
}

double RegionTreeSearch::compute_node_cost(const RegionSearchNode& node) {
    // Simple cost: depth + remaining obstacles
    return static_cast<double>(node.depth) + static_cast<double>(node.num_obstacles_remaining) * 0.5;
}

int RegionTreeSearch::allocate_node() {
    if (active_nodes_ >= MAX_SEARCH_NODES) {
        return -1;  // No more nodes available
    }
    
    return static_cast<int>(active_nodes_++);
}

void RegionTreeSearch::reset_search_state() {
    active_nodes_ = 0;
    search_queue_.clear();
    blocking_objects_.clear();
    goal_proposals_.clear();
    
    // Clear result
    last_result_ = TreeSearchResult{};
}

void RegionTreeSearch::extract_best_solution(int solution_node_idx) {
    last_result_.best_action_sequence = reconstruct_action_sequence(solution_node_idx);
    
    const RegionSearchNode& solution_node = get_node(solution_node_idx);
    last_result_.solution_cost = solution_node.cost_estimate;
    last_result_.solution_depth = solution_node.depth;
}

GenericFixedVector<ActionStep, 20> RegionTreeSearch::reconstruct_action_sequence(int node_idx) {
    GenericFixedVector<ActionStep, 20> sequence;
    
    // The action sequence is already stored in the node
    const RegionSearchNode& node = get_node(node_idx);
    return node.action_sequence;
}

bool RegionTreeSearch::is_duplicate_state(const LightweightState& state, int current_node_idx) {
    // Simple duplicate detection: compare hash with all existing nodes
    std::size_t state_hash = state.compute_hash();
    
    for (size_t i = 0; i < active_nodes_; ++i) {
        if (static_cast<int>(i) != current_node_idx) {
            const RegionSearchNode& other = get_node(static_cast<int>(i));
            if (other.is_valid && other.state.compute_hash() == state_hash) {
                return true;  // Duplicate found
            }
        }
    }
    
    return false;
}

void RegionTreeSearch::update_search_statistics(
    const std::chrono::high_resolution_clock::time_point& start,
    const std::chrono::high_resolution_clock::time_point& end) {
    
    last_result_.search_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
}

} // namespace namo