#pragma once

#include "planners/region/region_graph.hpp"
#include "planners/region/region_path_planner.hpp"
#include "planners/region/goal_proposal_generator.hpp"
#include "planners/region/region_analyzer.hpp"
#include "core/types.hpp"
#include <vector>
#include <string>
#include <array>

namespace namo {

// Forward declaration
class NAMOEnvironment;

/**
 * @brief Represents a single action in the tree search
 * 
 * Each action corresponds to moving one object to a specific goal position.
 */
struct ActionStep {
    std::string object_name;         // Object to move
    SE2State target_pose;           // Where to move the object
    bool execution_success = false; // Was execution successful (filled during real execution)
    double execution_time = 0.0;   // Time taken to execute (filled during real execution)
    
    ActionStep() = default;
    ActionStep(const std::string& obj_name, const SE2State& pose) 
        : object_name(obj_name), target_pose(pose) {}
    
    bool operator==(const ActionStep& other) const {
        return object_name == other.object_name && 
               std::abs(target_pose.x - other.target_pose.x) < 1e-6 &&
               std::abs(target_pose.y - other.target_pose.y) < 1e-6 &&
               std::abs(target_pose.theta - other.target_pose.theta) < 1e-6;
    }
};

/**
 * @brief Represents a node in the region tree search
 * 
 * Each node contains a state and the sequence of actions that led to that state.
 */
struct RegionSearchNode {
    // Current state representation
    LightweightState state;
    
    // Region graph for this state
    RegionGraph region_graph;
    
    // Sequence of actions from root to this node
    GenericFixedVector<ActionStep, 20> action_sequence;  // MAX_DEPTH × MAX_PROPOSALS
    
    // Tree structure
    int depth = 0;
    int parent_node_idx = -1;  // Index in node pool
    
    // Evaluation metrics
    double cost_estimate = 0.0;
    bool is_goal_reached = false;
    bool is_valid = true;
    
    // Node statistics
    int num_obstacles_remaining = 0;  // From path planner
    bool path_exists = false;
    
    RegionSearchNode() = default;
    
    // Check if this node represents a solution
    bool is_solution() const {
        return is_goal_reached && is_valid;
    }
    
    // Get the last action in the sequence
    const ActionStep* get_last_action() const {
        return action_sequence.empty() ? nullptr : &action_sequence.back();
    }
    
    // Add an action to the sequence
    void add_action(const ActionStep& action) {
        if (action_sequence.size() < action_sequence.capacity()) {
            action_sequence.push_back(action);
        }
    }
    
    // Compute hash for duplicate detection
    std::size_t compute_hash() const {
        return state.compute_hash();
    }
};

/**
 * @brief Complete tree search result
 */
struct TreeSearchResult {
    // Success information
    bool solution_found = false;
    GenericFixedVector<ActionStep, 20> best_action_sequence;
    
    // Solution quality
    double solution_cost = 0.0;
    int solution_depth = 0;
    
    // Search statistics
    int nodes_expanded = 0;
    int total_nodes_generated = 0;
    int max_depth_reached = 0;
    double search_time_ms = 0.0;
    
    // Failure information
    std::string failure_reason;
    
    TreeSearchResult() = default;
    
    bool is_valid() const {
        return solution_found && !best_action_sequence.empty();
    }
    
    size_t num_actions() const {
        return best_action_sequence.size();
    }
};

/**
 * @brief N-depth limited tree search for region-based planning
 * 
 * Performs alternating branching between object selection and goal proposal
 * to find action sequences that allow the robot to reach its goal.
 */
class RegionTreeSearch {
public:
    /**
     * @brief Constructor
     * @param env Environment reference for region analysis and goal proposals
     * @param max_depth Maximum search depth (default 2)
     * @param max_goal_proposals Number of goal proposals per object (default 5)
     */
    RegionTreeSearch(NAMOEnvironment& env, int max_depth = 2, int max_goal_proposals = 5);
    
    /**
     * @brief Perform tree search to find action sequence
     * @param initial_state Initial lightweight state
     * @param robot_goal Robot's target position
     * @param depth_limit Search depth limit (overrides default if specified)
     * @return Complete search result with best action sequence
     */
    TreeSearchResult search(const LightweightState& initial_state, 
                           const SE2State& robot_goal,
                           int depth_limit = -1);  // -1 uses default
    
    /**
     * @brief Find all reachable objects from current state
     * @param state Current state
     * @param robot_goal Robot goal for path analysis
     * @return List of objects that block the robot's path to goal
     */
    GenericFixedVector<std::string, 10> get_blocking_objects(const LightweightState& state,
                                                            const SE2State& robot_goal);
    
    /**
     * @brief Check if robot has reached its goal in the given state
     * @param state State to check
     * @param robot_goal Goal position
     * @param tolerance Goal tolerance (default 0.25m)
     * @return True if robot is at goal
     */
    bool is_robot_at_goal(const LightweightState& state, 
                         const SE2State& robot_goal,
                         double tolerance = 0.25);
    
    /**
     * @brief Configuration methods
     */
    void set_max_depth(int depth) { max_depth_ = depth; }
    void set_max_goal_proposals(int proposals) { max_goal_proposals_ = proposals; }
    void set_goal_tolerance(double tolerance) { goal_tolerance_ = tolerance; }
    void set_max_nodes(int max_nodes) { max_nodes_ = max_nodes; }
    
    int get_max_depth() const { return max_depth_; }
    int get_max_goal_proposals() const { return max_goal_proposals_; }
    double get_goal_tolerance() const { return goal_tolerance_; }
    
    /**
     * @brief Get search statistics from last run
     */
    const TreeSearchResult& get_last_search_result() const { return last_result_; }

private:
    // Environment and component references
    NAMOEnvironment& env_;
    std::unique_ptr<RegionAnalyzer> region_analyzer_;
    std::unique_ptr<RegionPathPlanner> path_planner_;
    std::unique_ptr<GoalProposalGenerator> goal_generator_;
    
    // Configuration
    int max_depth_;
    int max_goal_proposals_;
    double goal_tolerance_;
    int max_nodes_;
    
    // Pre-allocated search workspace
    static constexpr size_t MAX_SEARCH_NODES = 10000;
    std::array<RegionSearchNode, MAX_SEARCH_NODES> node_pool_;
    size_t active_nodes_ = 0;
    
    // Search queue (BFS-style)
    GenericFixedVector<int, MAX_SEARCH_NODES> search_queue_;
    
    // Workspace for object selection and goal proposals
    GenericFixedVector<std::string, 10> blocking_objects_;
    GenericFixedVector<SE2State, 20> goal_proposals_;
    
    // Results tracking
    TreeSearchResult last_result_;
    
    // Core search algorithm
    TreeSearchResult run_tree_search(const LightweightState& initial_state,
                                    const SE2State& robot_goal,
                                    int depth_limit);
    
    // Node expansion
    GenericFixedVector<int, 100> expand_node(int node_idx, const SE2State& robot_goal);
    
    // State evaluation
    bool evaluate_node(RegionSearchNode& node, const SE2State& robot_goal);
    void update_node_region_graph(RegionSearchNode& node, const SE2State& robot_goal);
    double compute_node_cost(const RegionSearchNode& node);
    
    // Node management
    int allocate_node();
    void reset_search_state();
    RegionSearchNode& get_node(int node_idx) { return node_pool_[node_idx]; }
    const RegionSearchNode& get_node(int node_idx) const { return node_pool_[node_idx]; }
    
    // Solution extraction
    void extract_best_solution(int solution_node_idx);
    GenericFixedVector<ActionStep, 20> reconstruct_action_sequence(int node_idx);
    
    // Utilities
    bool is_duplicate_state(const LightweightState& state, int current_node_idx);
    void update_search_statistics(const std::chrono::high_resolution_clock::time_point& start,
                                 const std::chrono::high_resolution_clock::time_point& end);
};

} // namespace namo