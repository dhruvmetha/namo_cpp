#include "planning/greedy_planner.hpp"
#include <cmath>
#include <iostream>
#include <algorithm>

namespace namo {

GreedyPlanner::GreedyPlanner() : node_count_(0) {
    // Set default distance and goal check functions
    distance_func_ = default_distance;
    goal_check_func_ = default_goal_check;
}

bool GreedyPlanner::initialize(const std::string& primitive_filepath) {
    return primitive_loader_.load_primitives(primitive_filepath);
}

std::vector<PlanStep> GreedyPlanner::plan_push_sequence(
    const SE2State& start_state,
    const SE2State& goal_state,
    const std::vector<int>& allowed_edges,
    int expansion_limit) {
    
    if (!primitive_loader_.is_loaded()) {
        std::cerr << "Primitives not loaded" << std::endl;
        return {};
    }
    
    // Clear previous search data
    clear_search_data();
    
    // Transform goal to local coordinate frame (following old implementation)
    SE2State local_goal = transform_to_local_frame(start_state, goal_state);
    
    // Initialize search from origin (transformed start state)
    SE2State origin(0.0, 0.0, 0.0);
    double initial_cost = distance_func_(origin, local_goal);
    
    search_nodes_[0] = SearchNode(origin, initial_cost, -1, 0, nullptr);
    open_set_.push(&search_nodes_[0]);
    node_count_ = 1;
    
    // Track best node (closest to goal) in case exact goal is not reached
    SearchNode* best_node = &search_nodes_[0];
    double best_cost = initial_cost;
    
    int iterations = 0;
    
    while (!open_set_.empty() && iterations < expansion_limit) {
        SearchNode* current = open_set_.top();
        open_set_.pop();
        
        // Check if we reached the goal
        if (goal_check_func_(current->state, local_goal)) {
            std::cout << "Goal reached in " << iterations << " iterations" << std::endl;
            return reconstruct_path(current, start_state);
        }
        
        // Update best node if this one is closer
        if (current->cost < best_cost) {
            best_node = current;
            best_cost = current->cost;
        }
        
        // Expand current state using primitives
        const auto& all_primitives = primitive_loader_.get_all_primitives();
        
        for (size_t i = 0; i < primitive_loader_.size(); i++) {
            const LoadedPrimitive& primitive = all_primitives[i];
            
            // Skip if this edge is not allowed
            if (!allowed_edges.empty()) {
                bool edge_allowed = std::find(allowed_edges.begin(), allowed_edges.end(), 
                                            primitive.edge_idx) != allowed_edges.end();
                if (!edge_allowed) continue;
            }
            
            // Skip primitives with 0 push steps
            if (primitive.push_steps <= 0) continue;
            
            // Apply primitive to get new state
            SE2State new_state = apply_primitive(current->state, primitive);
            double new_cost = distance_func_(new_state, local_goal);
            
            // Create new search node if we have space
            if (node_count_ < MAX_SEARCH_NODES) {
                search_nodes_[node_count_] = SearchNode(
                    new_state, new_cost, primitive.edge_idx, primitive.push_steps, current);
                open_set_.push(&search_nodes_[node_count_]);
                node_count_++;
            }
        }
        
        iterations++;
    }
    
    // Always return path to best node found (matching old implementation behavior)
    // The old implementation never returns empty - it always returns the best path found
    if (best_node != &search_nodes_[0]) {
        std::cout << "Exact goal not reached. Returning path to closest state (distance: " 
                  << best_cost << ")" << std::endl;
        return reconstruct_path(best_node, start_state);
    } else {
        // If no progress made, return a single step with the best available primitive
        // This matches old implementation behavior of always returning something
        std::cout << "No improvement found. Returning single best primitive step" << std::endl;
        return get_fallback_primitive_step(origin, local_goal, start_state, allowed_edges);
    }
}

SE2State GreedyPlanner::transform_to_local_frame(const SE2State& reference, const SE2State& target) {
    // Following old implementation approach (best_first_search_planner.hpp:238-255)
    double dx = target.x - reference.x;
    double dy = target.y - reference.y;
    double dtheta = normalize_angle(target.theta - reference.theta);
    
    // Rotate point by -reference.theta
    double cos_ref = std::cos(-reference.theta);
    double sin_ref = std::sin(-reference.theta);
    
    return SE2State(
        dx * cos_ref - dy * sin_ref,
        dx * sin_ref + dy * cos_ref,
        dtheta
    );
}

SE2State GreedyPlanner::transform_to_global_frame(const SE2State& reference, const SE2State& local) {
    // Following old implementation approach (best_first_search_planner.hpp:277-293)
    double cos_ref = std::cos(reference.theta);
    double sin_ref = std::sin(reference.theta);
    
    // Rotate point by reference.theta
    double x = local.x * cos_ref - local.y * sin_ref;
    double y = local.x * sin_ref + local.y * cos_ref;
    
    SE2State result(
        reference.x + x,
        reference.y + y,
        normalize_angle(reference.theta + local.theta)
    );
    
    return result;
}

SE2State GreedyPlanner::apply_primitive(const SE2State& current_state, const LoadedPrimitive& primitive) {
    // Following old implementation approach (best_first_search_planner.hpp:257-275)
    // Transform primitive effect to current state frame
    double cos_theta = std::cos(current_state.theta);
    double sin_theta = std::sin(current_state.theta);
    
    double dx = primitive.delta_x;
    double dy = primitive.delta_y;
    
    SE2State result(
        current_state.x + dx * cos_theta - dy * sin_theta,
        current_state.y + dx * sin_theta + dy * cos_theta,
        normalize_angle(current_state.theta + primitive.delta_theta)
    );
    
    return result;
}

double GreedyPlanner::default_distance(const SE2State& state, const SE2State& goal) {
    // Position distance
    double dx = state.x - goal.x;
    double dy = state.y - goal.y;
    double pos_dist = std::sqrt(dx*dx + dy*dy);
    
    // Angular distance
    double angle_diff = normalize_angle(state.theta - goal.theta);
    double ang_dist = std::abs(angle_diff);
    
    // Weight rotation in the distance calculation (following old implementation)
    return pos_dist + 1.0 * ang_dist;
}

bool GreedyPlanner::default_goal_check(const SE2State& state, const SE2State& goal) {
    // Much tighter thresholds to match old implementation behavior
    const double position_threshold = 0.005;  // 5mm (was 5cm)
    const double angle_threshold = 0.05;      // ~3 degrees (was ~6 degrees)
    
    double dx = state.x - goal.x;
    double dy = state.y - goal.y;
    double pos_dist = std::sqrt(dx*dx + dy*dy);
    
    double angle_diff = std::abs(normalize_angle(state.theta - goal.theta));
    
    return pos_dist < position_threshold && angle_diff < angle_threshold;
}

double GreedyPlanner::normalize_angle(double angle) {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

void GreedyPlanner::clear_search_data() {
    // Clear priority queue
    std::priority_queue<SearchNode*, std::vector<SearchNode*>, SearchNodeCompare> empty_queue;
    open_set_.swap(empty_queue);
    
    // Reset node count
    node_count_ = 0;
}

std::vector<PlanStep> GreedyPlanner::reconstruct_path(SearchNode* goal_node, const SE2State& start_state) {
    std::vector<PlanStep> plan_sequence;
    std::vector<SearchNode*> path;
    
    // Collect nodes in reverse order
    SearchNode* current = goal_node;
    while (current != nullptr) {
        path.push_back(current);
        current = current->parent;
    }
    
    // Reverse to get correct order (start to goal)
    std::reverse(path.begin(), path.end());
    
    // Convert to plan steps with global coordinates
    for (SearchNode* node : path) {
        if (node->parent == nullptr) {
            // Skip start node (has no primitive action)
            continue;
        }
        
        // Transform local coordinates back to global
        SE2State global_pose = transform_to_global_frame(start_state, node->state);
        
        plan_sequence.emplace_back(
            node->primitive_edge,
            node->primitive_steps,
            global_pose
        );
    }
    
    return plan_sequence;
}

std::vector<PlanStep> GreedyPlanner::get_fallback_primitive_step(
    const SE2State& origin, 
    const SE2State& local_goal, 
    const SE2State& start_state,
    const std::vector<int>& allowed_edges) {
    
    // Find the primitive that gets closest to the goal, but only if it makes meaningful progress
    // Added threshold to avoid executing primitives that move object in wrong direction
    
    if (!primitive_loader_.is_loaded()) {
        return {};
    }
    
    // Current distance from origin to goal
    double current_distance = distance_func_(origin, local_goal);
    
    double best_distance = std::numeric_limits<double>::max();
    LoadedPrimitive best_primitive;
    bool found_primitive = false;
    
    // Check all available primitives to find the one that gets closest
    for (int edge = 0; edge < 12; edge++) {
        // Skip if this edge is not allowed (respect reachable edges constraint)
        if (!allowed_edges.empty()) {
            bool edge_allowed = std::find(allowed_edges.begin(), allowed_edges.end(), edge) != allowed_edges.end();
            if (!edge_allowed) continue;
        }
        
        for (int steps = 1; steps <= 10; steps++) {
            const LoadedPrimitive& primitive = primitive_loader_.get_primitive(edge, steps);
            
            // Skip invalid primitives
            if (primitive.push_steps <= 0) continue;
            
            // Apply primitive to origin and see how close we get to goal
            SE2State result_state = apply_primitive(origin, primitive);
            double distance = distance_func_(result_state, local_goal);
            
            // Only consider primitives that actually improve distance to goal
            // This prevents moving objects in wrong direction when no good path exists
            if (distance < current_distance && distance < best_distance) {
                best_distance = distance;
                best_primitive = primitive;
                found_primitive = true;
            }
        }
    }
    
    if (!found_primitive) {
        std::cout << "No primitive improves distance to goal. Returning empty plan." << std::endl;
        return {};
    }
    
    // Create a single-step plan with the best primitive
    SE2State result_state = apply_primitive(origin, best_primitive);
    SE2State global_pose = transform_to_global_frame(start_state, result_state);
    
    std::vector<PlanStep> fallback_plan;
    fallback_plan.emplace_back(best_primitive.edge_idx, best_primitive.push_steps, global_pose);
    
    return fallback_plan;
}

} // namespace namo