#pragma once

#include "core/types.hpp"
#include "planning/primitive_loader.hpp"
#include <vector>
#include <array>
#include <queue>
#include <functional>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace namo {

/**
 * @brief Search node for greedy best-first search
 */
struct SearchNode {
    SE2State state;
    double cost;                // Heuristic cost to goal
    int primitive_edge;         // Edge index of primitive that led to this state (-1 for start)
    int primitive_steps;        // Push steps of primitive that led to this state (0 for start)
    SearchNode* parent;         // Parent node for path reconstruction
    
    // Default constructor for array initialization
    SearchNode() : state(), cost(0.0), primitive_edge(-1), primitive_steps(0), parent(nullptr) {}
    
    SearchNode(const SE2State& state_, double cost_, int edge = -1, int steps = 0, SearchNode* parent_ = nullptr)
        : state(state_), cost(cost_), primitive_edge(edge), primitive_steps(steps), parent(parent_) {}
};

/**
 * @brief Priority queue comparator for min-heap (lowest cost first)
 */
struct SearchNodeCompare {
    bool operator()(const SearchNode* a, const SearchNode* b) const {
        return a->cost > b->cost;  // Min heap - lowest cost has highest priority
    }
};

/**
 * @brief Plan step result containing primitive action and resulting pose
 */
struct PlanStep {
    int edge_idx;
    int push_steps;
    SE2State pose;  // Resulting object pose after this primitive
    
    PlanStep(int edge, int steps, const SE2State& pose_)
        : edge_idx(edge), push_steps(steps), pose(pose_) {}
};

/**
 * @brief Greedy best-first search planner following old implementation approach
 * 
 * Plans in empty environment using geometric primitive application.
 * No physics simulation - pure displacement transforms.
 */
class GreedyPlanner {
private:
    static constexpr size_t MAX_SEARCH_NODES = 10000;
    static constexpr int DEFAULT_EXPANSION_LIMIT = 5000;
    
    PrimitiveLoader primitive_loader_;
    
    // Pre-allocated search structures
    std::array<SearchNode, MAX_SEARCH_NODES> search_nodes_;
    size_t node_count_;
    std::priority_queue<SearchNode*, std::vector<SearchNode*>, SearchNodeCompare> open_set_;
    
    // Distance and goal check functions
    std::function<double(const SE2State&, const SE2State&)> distance_func_;
    std::function<bool(const SE2State&, const SE2State&)> goal_check_func_;
    
public:
    GreedyPlanner();
    
    /**
     * @brief Initialize planner with primitive database
     * 
     * @param primitive_filepath Path to binary primitive file
     * @return bool True if initialization succeeded
     */
    bool initialize(const std::string& primitive_filepath);
    
    /**
     * @brief Set distance function for heuristic
     * 
     * @param func Function that computes distance between two SE2 states
     */
    void set_distance_function(std::function<double(const SE2State&, const SE2State&)> func) {
        distance_func_ = func;
    }
    
    /**
     * @brief Set goal check function
     * 
     * @param func Function that checks if a state is close enough to goal
     */
    void set_goal_check_function(std::function<bool(const SE2State&, const SE2State&)> func) {
        goal_check_func_ = func;
    }
    
    /**
     * @brief Plan push sequence from start to goal state
     * 
     * Follows old implementation approach:
     * 1. Transform goal to local coordinate frame
     * 2. Search in empty environment using geometric primitive application
     * 3. Return sequence of primitives that reach goal
     * 
     * @param start_state Initial object pose
     * @param goal_state Target object pose  
     * @param allowed_edges Vector of allowed edge indices (empty = use all)
     * @param expansion_limit Maximum search iterations
     * @return std::vector<PlanStep> Sequence of primitive actions
     */
    std::vector<PlanStep> plan_push_sequence(
        const SE2State& start_state,
        const SE2State& goal_state, 
        const std::vector<int>& allowed_edges = {},
        int expansion_limit = DEFAULT_EXPANSION_LIMIT
    );
    
    /**
     * @brief Transform coordinates to local frame (public for testing)
     */
    SE2State transform_to_local_frame(const SE2State& reference, const SE2State& target);
    
    /**
     * @brief Transform coordinates to global frame (public for testing)
     */
    SE2State transform_to_global_frame(const SE2State& reference, const SE2State& local);
    
private:
    
    /**
     * @brief Apply primitive displacement to current state
     * Pure geometric transform - no physics
     */
    SE2State apply_primitive(const SE2State& current_state, const LoadedPrimitive& primitive);
    
    /**
     * @brief Default distance function - Euclidean + angular distance
     */
    static double default_distance(const SE2State& state, const SE2State& goal);
    
    /**
     * @brief Default goal check function - within distance and angle thresholds
     */
    static bool default_goal_check(const SE2State& state, const SE2State& goal);
    
    /**
     * @brief Normalize angle to [-π, π]
     */
    static double normalize_angle(double angle);
    
    /**
     * @brief Clear search data structures for new search
     */
    void clear_search_data();
    
    /**
     * @brief Reconstruct path from goal node back to start
     */
    std::vector<PlanStep> reconstruct_path(SearchNode* goal_node, const SE2State& start_state);
    
    /**
     * @brief Get fallback primitive step when no progress can be made
     * Returns the primitive that gets closest to the goal, matching old implementation
     */
    std::vector<PlanStep> get_fallback_primitive_step(
        const SE2State& origin, 
        const SE2State& local_goal, 
        const SE2State& start_state,
        const std::vector<int>& allowed_edges);
};

} // namespace namo