#pragma once

#include "planners/strategies/selection_strategy.hpp"
#include "wavefront/wavefront_planner.hpp"
#include "skills/namo_push_skill.hpp"
#include "environment/namo_environment.hpp"
#include "config/config_manager.hpp"
#include <memory>

namespace namo {

struct PlanningResult {
    bool success;
    int iterations_used;
    double total_time;
    std::vector<std::string> objects_pushed;
    std::string failure_reason;
    
    PlanningResult() : success(false), iterations_used(0), total_time(0.0) {}
};

// High-level NAMO planner implementing single-stream planning algorithm
// Based on legacy interface_namo.cpp planning loop (lines 303-707)
class HighLevelPlanner {
private:
    // Core components
    NAMOEnvironment& environment_;
    std::unique_ptr<SelectionStrategy> strategy_;
    std::unique_ptr<NAMOPushSkill> push_skill_;
    
    // Separate wavefront planner for high-level reachability analysis
    // This is different from the one used inside the skill for local planning
    std::unique_ptr<WavefrontPlanner> high_level_wavefront_;
    
    // Configuration management
    std::shared_ptr<ConfigManager> config_;
    
    // Internal state
    SE2State robot_goal_;
    std::vector<std::string> execution_log_;
    
    // Helper methods
    std::vector<std::string> computeReachableObjects();
    bool isRobotGoalReachable();
    void updateHighLevelWavefront();
    void logIteration(int iteration, const SelectionResult& selection, 
                     const SkillResult& result);

public:
    /**
     * @brief Constructor with configuration
     * @param env Environment reference
     * @param strategy Selection strategy
     * @param config Configuration manager (optional, uses default if not provided)
     */
    HighLevelPlanner(NAMOEnvironment& env, 
                    std::unique_ptr<SelectionStrategy> strategy,
                    std::shared_ptr<ConfigManager> config = nullptr);
    
    ~HighLevelPlanner();
    
    // Main planning interface
    PlanningResult planToGoal(const SE2State& robot_goal, 
                             int max_iterations = -1);  // -1 uses config default
    
    // Strategy management
    void setStrategy(std::unique_ptr<SelectionStrategy> strategy);
    SelectionStrategy* getStrategy() const { return strategy_.get(); }
    
    // Debug access
    const WavefrontPlanner* getHighLevelWavefront() const { return high_level_wavefront_.get(); }
    
    // Configuration access
    const ConfigManager& getConfig() const { return *config_; }
    void updateConfig(std::shared_ptr<ConfigManager> new_config);
    
    // State queries
    const std::vector<std::string>& getExecutionLog() const { return execution_log_; }
    bool isGoalReachable(const SE2State& goal);
    
    // Reset state
    void reset();
};

} // namespace namo