#include "planners/region/region_based_planner.hpp"
#include <chrono>
#include <iostream>

namespace namo {

RegionBasedPlanner::RegionBasedPlanner(NAMOEnvironment& env, 
                                     std::unique_ptr<ConfigManager> config)
    : env_(env)
    , config_(std::move(config))
    , max_depth_(2)
    , goal_proposals_per_object_(5)
    , goal_tolerance_(0.25)
    , sampling_density_(100.0) {
    
    // Load configuration if provided
    if (config_) {
        load_configuration();
    }
    
    // Initialize all components
    initialize_components();
}

RegionPlanningResult RegionBasedPlanner::plan_to_goal(const SE2State& robot_goal, 
                                                     int max_depth,
                                                     bool execute_actions) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Reset statistics
    reset_statistics();
    
    // Validate inputs
    if (!validate_robot_goal(robot_goal)) {
        RegionPlanningResult result;
        result.failure_reason = "Invalid robot goal position";
        return result;
    }
    
    // Use provided depth or default
    int effective_depth = (max_depth > 0) ? max_depth : max_depth_;
    
    // Plan first
    RegionPlanningResult planning_result = plan_only(robot_goal, effective_depth);
    
    if (!planning_result.success || !execute_actions) {
        auto end_time = std::chrono::high_resolution_clock::now();
        planning_result.total_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        return planning_result;
    }
    
    // Execute planned actions
    RegionPlanningResult execution_result = this->execute_actions(planning_result.planned_actions);
    
    // Combine results
    RegionPlanningResult combined_result = planning_result;
    combined_result.executed_actions = execution_result.executed_actions;
    combined_result.execution_iterations = execution_result.execution_iterations;
    combined_result.execution_time_ms = execution_result.execution_time_ms;
    combined_result.success = execution_result.success;
    
    if (!execution_result.success) {
        combined_result.failure_reason = execution_result.failure_reason;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    combined_result.total_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    return combined_result;
}

RegionPlanningResult RegionBasedPlanner::plan_only(const SE2State& robot_goal, int max_depth) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    RegionPlanningResult result;
    
    // Use provided depth or default
    int effective_depth = (max_depth > 0) ? max_depth : max_depth_;
    
    try {
        // Create lightweight state from current environment
        LightweightState initial_state = create_lightweight_state();
        
        // Perform tree search
        TreeSearchResult search_result = tree_search_->search(initial_state, robot_goal, effective_depth);
        
        // Update planning statistics
        update_planning_statistics(search_result);
        
        if (search_result.solution_found) {
            result.success = true;
            result.planned_actions = search_result.best_action_sequence;
            result.planning_iterations = search_result.nodes_expanded;
        } else {
            result.success = false;
            result.failure_reason = search_result.failure_reason;
        }
        
    } catch (const std::exception& e) {
        result.success = false;
        result.failure_reason = std::string("Planning exception: ") + e.what();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.planning_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    return result;
}

RegionPlanningResult RegionBasedPlanner::execute_actions(const GenericFixedVector<ActionStep, 20>& actions) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    RegionPlanningResult result;
    
    if (!validate_action_sequence(actions)) {
        result.failure_reason = "Invalid action sequence";
        auto end_time = std::chrono::high_resolution_clock::now();
        result.execution_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        return result;
    }
    
    std::vector<SkillResult> execution_results;
    
    try {
        // Execute each action in sequence
        for (size_t i = 0; i < actions.size(); ++i) {
            const ActionStep& action = actions[i];
            
            // Execute single action
            RegionPlanningResult single_result = execute_single_action(action);
            
            if (single_result.success && !single_result.executed_actions.empty()) {
                // Copy executed action to our result
                result.executed_actions.push_back(single_result.executed_actions[0]);
                result.execution_iterations++;
            } else {
                // Execution failed
                result.success = false;
                result.failure_reason = "Action execution failed: " + single_result.failure_reason;
                break;
            }
        }
        
        if (result.execution_iterations == static_cast<int>(actions.size())) {
            result.success = true;
        }
        
    } catch (const std::exception& e) {
        result.success = false;
        result.failure_reason = std::string("Execution exception: ") + e.what();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.execution_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    // Update execution statistics
    update_execution_statistics(execution_results);
    
    return result;
}

bool RegionBasedPlanner::is_goal_reachable(const SE2State& robot_goal) {
    try {
        // Create lightweight state
        LightweightState current_state = create_lightweight_state();
        
        // Get blocking objects
        auto blocking_objects = tree_search_->get_blocking_objects(current_state, robot_goal);
        
        // If no blocking objects, goal is directly reachable
        if (blocking_objects.empty()) {
            return true;
        }
        
        // Otherwise, run a shallow search to check if goal is reachable with object movements
        TreeSearchResult result = tree_search_->search(current_state, robot_goal, 1);
        return result.solution_found;
        
    } catch (const std::exception& e) {
        return false;
    }
}

std::vector<std::string> RegionBasedPlanner::get_blocking_objects(const SE2State& robot_goal) {
    try {
        // Discover regions and build connectivity graph
        RegionGraph graph = region_analyzer_->discover_regions(env_, robot_goal);
        
        // Use the new heuristic: BFS to find next obstacles
        return graph.get_next_obstacles_to_move();
        
    } catch (const std::exception& e) {
        std::cerr << "Error in get_blocking_objects: " << e.what() << std::endl;
        return {};  // Return empty vector on error
    }
}

void RegionBasedPlanner::set_max_depth(int depth) {
    max_depth_ = depth;
    update_component_configuration();
}

void RegionBasedPlanner::set_goal_proposals_per_object(int proposals) {
    goal_proposals_per_object_ = proposals;
    update_component_configuration();
}

void RegionBasedPlanner::set_goal_tolerance(double tolerance) {
    goal_tolerance_ = tolerance;
    update_component_configuration();
}

void RegionBasedPlanner::set_sampling_density(double density) {
    sampling_density_ = density;
    update_component_configuration();
}

int RegionBasedPlanner::get_max_depth() const { return max_depth_; }
int RegionBasedPlanner::get_goal_proposals_per_object() const { return goal_proposals_per_object_; }
double RegionBasedPlanner::get_goal_tolerance() const { return goal_tolerance_; }
double RegionBasedPlanner::get_sampling_density() const { return sampling_density_; }

void RegionBasedPlanner::reset() {
    // Reset statistics
    reset_statistics();
    
    // Reset components if needed
    // (Most components are stateless or reset themselves on each use)
}

void RegionBasedPlanner::initialize_components() {
    std::cout << "RegionBasedPlanner: Initializing components..." << std::endl;
    
    // Create region analyzer
    region_analyzer_ = std::make_unique<RegionAnalyzer>(0.05, sampling_density_, goal_tolerance_);
    std::cout << "RegionBasedPlanner: Created region analyzer" << std::endl;
    
    // Create path planner
    path_planner_ = std::make_unique<RegionPathPlanner>();
    std::cout << "RegionBasedPlanner: Created path planner" << std::endl;
    
    // Create goal proposal generator
    goal_generator_ = std::make_unique<GoalProposalGenerator>(env_, goal_proposals_per_object_);
    std::cout << "RegionBasedPlanner: Created goal generator" << std::endl;
    
    // Create tree search with configuration
    tree_search_ = std::make_unique<RegionTreeSearch>(env_, max_depth_, goal_proposals_per_object_);
    std::cout << "RegionBasedPlanner: Created tree search" << std::endl;
    
    // Configure tree search
    tree_search_->set_goal_tolerance(goal_tolerance_);
    
    // Create push skill for action execution  
    push_skill_ = std::make_unique<NAMOPushSkill>(env_);  // TODO: Pass config when available
    std::cout << "RegionBasedPlanner: Created push skill" << std::endl;
    
    std::cout << "RegionBasedPlanner: All components initialized successfully" << std::endl;
}

void RegionBasedPlanner::load_configuration() {
    if (!config_) return;
    
    try {
        // For now, use default values since region planner config is not yet integrated
        // TODO: Add region planner configuration section to ConfigManager
        
        // Use planning section for some values
        const auto& planning_config = config_->planning();
        
        // Adapt existing configuration values where possible
        goal_tolerance_ = std::min(goal_tolerance_, planning_config.position_threshold * 1000.0); // Convert to reasonable scale
        
        std::cout << "RegionBasedPlanner: Using default configuration with some adapted values" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "RegionBasedPlanner: Warning - Failed to load configuration: " << e.what() << std::endl;
        std::cerr << "RegionBasedPlanner: Using all default values" << std::endl;
    }
}

void RegionBasedPlanner::update_component_configuration() {
    if (tree_search_) {
        tree_search_->set_max_depth(max_depth_);
        tree_search_->set_max_goal_proposals(goal_proposals_per_object_);
        tree_search_->set_goal_tolerance(goal_tolerance_);
    }
}

LightweightState RegionBasedPlanner::create_lightweight_state() {
    LightweightState state;
    state.initialize_from_environment(env_);
    return state;
}

RegionPlanningResult RegionBasedPlanner::execute_single_action(const ActionStep& action) {
    RegionPlanningResult result;
    
    try {
        // Prepare skill parameters
        std::map<std::string, SkillParameterValue> params;
        params["object_name"] = action.object_name;
        params["target_pose"] = action.target_pose;
        
        // Check if skill is applicable
        if (!push_skill_->is_applicable(params)) {
            result.failure_reason = "Push skill not applicable for object: " + action.object_name;
            return result;
        }
        
        // Check preconditions
        std::vector<std::string> precondition_failures = push_skill_->check_preconditions(params);
        if (!precondition_failures.empty()) {
            result.failure_reason = "Precondition check failed: " + precondition_failures[0];
            return result;
        }
        
        // Execute the skill
        SkillResult skill_result = push_skill_->execute(params);
        
        if (skill_result.success) {
            result.success = true;
            
            // Create executed action with results
            ActionStep executed_action = action;
            executed_action.execution_success = true;
            executed_action.execution_time = skill_result.execution_time.count();
            
            result.executed_actions.push_back(executed_action);
            result.execution_iterations = 1;
            result.execution_time_ms = skill_result.execution_time.count();
            
        } else {
            result.failure_reason = "Skill execution failed: " + skill_result.failure_reason;
        }
        
    } catch (const std::exception& e) {
        result.failure_reason = std::string("Single action execution exception: ") + e.what();
    }
    
    return result;
}

void RegionBasedPlanner::update_planning_statistics(const TreeSearchResult& search_result) {
    last_stats_.search_nodes_expanded = search_result.nodes_expanded;
    last_stats_.search_max_depth_reached = search_result.max_depth_reached;
    last_stats_.tree_search_time_ms = search_result.search_time_ms;
    last_stats_.total_planning_time_ms += search_result.search_time_ms;
    last_stats_.last_planning_success = search_result.solution_found;
}

void RegionBasedPlanner::update_execution_statistics(const std::vector<SkillResult>& execution_results) {
    last_stats_.actions_executed = static_cast<int>(execution_results.size());
    
    double total_execution_time = 0.0;
    int failures = 0;
    
    for (const auto& result : execution_results) {
        total_execution_time += result.execution_time.count();
        if (!result.success) {
            failures++;
        }
    }
    
    last_stats_.action_execution_time_ms = total_execution_time;
    last_stats_.execution_failures = failures;
}

bool RegionBasedPlanner::validate_robot_goal(const SE2State& robot_goal) const {
    // Basic bounds checking
    if (std::isnan(robot_goal.x) || std::isnan(robot_goal.y) || std::isnan(robot_goal.theta)) {
        return false;
    }
    
    // Check reasonable bounds (adjust based on environment)
    if (std::abs(robot_goal.x) > 100.0 || std::abs(robot_goal.y) > 100.0) {
        return false;
    }
    
    return true;
}

bool RegionBasedPlanner::validate_action_sequence(const GenericFixedVector<ActionStep, 20>& actions) const {
    if (actions.empty()) {
        return false;
    }
    
    for (size_t i = 0; i < actions.size(); ++i) {
        const ActionStep& action = actions[i];
        
        // Validate object name
        if (action.object_name.empty()) {
            return false;
        }
        
        // Validate target pose
        if (!validate_robot_goal(action.target_pose)) {
            return false;
        }
    }
    
    return true;
}

} // namespace namo