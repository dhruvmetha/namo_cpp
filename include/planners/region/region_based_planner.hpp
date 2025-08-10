#pragma once

#include "planners/region/region_tree_search.hpp"
#include "planners/region/region_analyzer.hpp"
#include "planners/region/region_path_planner.hpp"
#include "planners/region/goal_proposal_generator.hpp"
#include "skills/namo_push_skill.hpp"
#include "environment/namo_environment.hpp"
#include "config/config_manager.hpp"
#include "core/types.hpp"
#include <memory>

namespace namo {

/**
 * @brief Result of region-based planning
 */
struct RegionPlanningResult {
    // Success information
    bool success = false;
    
    // Planning information
    GenericFixedVector<ActionStep, 20> planned_actions;
    int planning_iterations = 0;
    double planning_time_ms = 0.0;
    
    // Execution information  
    GenericFixedVector<ActionStep, 20> executed_actions;
    int execution_iterations = 0;
    double execution_time_ms = 0.0;
    
    // Combined metrics
    double total_time_ms = 0.0;
    
    // Failure information
    std::string failure_reason;
    
    RegionPlanningResult() = default;
    
    bool has_planned_actions() const {
        return !planned_actions.empty();
    }
    
    bool has_executed_actions() const {
        return !executed_actions.empty();
    }
    
    size_t num_planned_actions() const {
        return planned_actions.size();
    }
    
    size_t num_executed_actions() const {
        return executed_actions.size();
    }
    
    bool is_complete_success() const {
        return success && num_planned_actions() == num_executed_actions();
    }
};

/**
 * @brief Main region-based high-level planner
 * 
 * Coordinates all components to provide a complete region-based planning solution:
 * 1. Analyzes environment to discover regions
 * 2. Uses tree search to find optimal action sequences
 * 3. Executes actions using the skill system
 * 4. Provides comprehensive results and statistics
 */
class RegionBasedPlanner {
public:
    /**
     * @brief Constructor
     * @param env Environment reference
     * @param config Configuration manager (optional)
     */
    RegionBasedPlanner(NAMOEnvironment& env, 
                      std::unique_ptr<ConfigManager> config = nullptr);
    
    /**
     * @brief Destructor
     */
    ~RegionBasedPlanner() = default;
    
    /**
     * @brief Plan and execute to reach robot goal
     * @param robot_goal Target position for robot
     * @param max_depth Search depth limit (default -1 uses config)
     * @param execute_actions Whether to execute planned actions (default true)
     * @return Complete planning and execution result
     */
    RegionPlanningResult plan_to_goal(const SE2State& robot_goal, 
                                     int max_depth = -1,
                                     bool execute_actions = true);
    
    /**
     * @brief Plan only (no execution)
     * @param robot_goal Target position for robot
     * @param max_depth Search depth limit
     * @return Planning result with action sequence
     */
    RegionPlanningResult plan_only(const SE2State& robot_goal, int max_depth = -1);
    
    /**
     * @brief Execute a pre-planned action sequence
     * @param actions Action sequence to execute
     * @return Execution result
     */
    RegionPlanningResult execute_actions(const GenericFixedVector<ActionStep, 20>& actions);
    
    /**
     * @brief Check if robot can reach goal (reachability analysis)
     * @param robot_goal Target position
     * @return True if goal is reachable (possibly requiring object movement)
     */
    bool is_goal_reachable(const SE2State& robot_goal);
    
    /**
     * @brief Get objects that block robot's path to goal
     * @param robot_goal Target position
     * @return List of blocking objects in priority order
     */
    std::vector<std::string> get_blocking_objects(const SE2State& robot_goal);
    
    /**
     * @brief Configuration methods
     */
    void set_max_depth(int depth);
    void set_goal_proposals_per_object(int proposals);
    void set_goal_tolerance(double tolerance);
    void set_sampling_density(double density);
    
    int get_max_depth() const;
    int get_goal_proposals_per_object() const;
    double get_goal_tolerance() const;
    double get_sampling_density() const;
    
    /**
     * @brief Component access for advanced usage
     */
    RegionTreeSearch& get_tree_search() { return *tree_search_; }
    RegionAnalyzer& get_region_analyzer() { return *region_analyzer_; }
    RegionPathPlanner& get_path_planner() { return *path_planner_; }
    GoalProposalGenerator& get_goal_generator() { return *goal_generator_; }
    NAMOPushSkill& get_push_skill() { return *push_skill_; }
    
    const RegionTreeSearch& get_tree_search() const { return *tree_search_; }
    const RegionAnalyzer& get_region_analyzer() const { return *region_analyzer_; }
    const RegionPathPlanner& get_path_planner() const { return *path_planner_; }
    const GoalProposalGenerator& get_goal_generator() const { return *goal_generator_; }
    const NAMOPushSkill& get_push_skill() const { return *push_skill_; }
    
    /**
     * @brief Statistics and debugging
     */
    struct PlannerStats {
        // Region analysis
        int regions_discovered = 0;
        int region_edges = 0;
        double region_analysis_time_ms = 0.0;
        
        // Tree search
        int search_nodes_expanded = 0;
        int search_max_depth_reached = 0;
        double tree_search_time_ms = 0.0;
        
        // Execution
        int actions_executed = 0;
        int execution_failures = 0;
        double action_execution_time_ms = 0.0;
        
        // Overall
        double total_planning_time_ms = 0.0;
        bool last_planning_success = false;
    };
    
    const PlannerStats& get_last_stats() const { return last_stats_; }
    void reset_statistics() { last_stats_ = PlannerStats{}; }
    
    /**
     * @brief Reset planner state
     */
    void reset();

private:
    // Environment and configuration
    NAMOEnvironment& env_;
    std::unique_ptr<ConfigManager> config_;
    
    // Core components
    std::unique_ptr<RegionAnalyzer> region_analyzer_;
    std::unique_ptr<RegionPathPlanner> path_planner_;
    std::unique_ptr<GoalProposalGenerator> goal_generator_;
    std::unique_ptr<RegionTreeSearch> tree_search_;
    std::unique_ptr<NAMOPushSkill> push_skill_;
    
    // Configuration cache
    int max_depth_;
    int goal_proposals_per_object_;
    double goal_tolerance_;
    double sampling_density_;
    
    // Statistics tracking
    mutable PlannerStats last_stats_;
    
    // Helper methods
    void initialize_components();
    void load_configuration();
    void update_component_configuration();
    
    LightweightState create_lightweight_state();
    RegionPlanningResult execute_single_action(const ActionStep& action);
    
    void update_planning_statistics(const TreeSearchResult& search_result);
    void update_execution_statistics(const std::vector<SkillResult>& execution_results);
    
    // Validation
    bool validate_robot_goal(const SE2State& robot_goal) const;
    bool validate_action_sequence(const GenericFixedVector<ActionStep, 20>& actions) const;
};

} // namespace namo