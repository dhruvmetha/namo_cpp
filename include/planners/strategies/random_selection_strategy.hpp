#pragma once

#include "planners/strategies/selection_strategy.hpp"
#include "config/config_manager.hpp"
#include <random>
#include <map>
#include <memory>

namespace namo {

// Implementation of the legacy random object + goal selection strategy
class RandomSelectionStrategy : public SelectionStrategy {
private:
    mutable std::random_device rd_;
    mutable std::mt19937 gen_;
    
    // Configuration management
    std::shared_ptr<ConfigManager> config_;
    
    // Helper methods
    std::string selectRandomObject(const std::vector<std::string>& objects) const;
    SE2State generateRandomGoal(const NAMOEnvironment& env, 
                               const std::string& object_name) const;
    bool isGoalValid(const NAMOEnvironment& env, const SE2State& goal) const;
    SE2State tryPolarGoalGeneration(const NAMOEnvironment& env,
                                   const std::string& object_name) const;
    SE2State tryCardinalDirections(const NAMOEnvironment& env,
                                  const std::string& object_name) const;

public:
    /**
     * @brief Constructor with optional configuration
     * @param config Configuration manager (uses default if not provided)
     */
    explicit RandomSelectionStrategy(std::shared_ptr<ConfigManager> config = nullptr);
    
    SelectionResult selectObjectAndGoal(
        const NAMOEnvironment& env,
        const std::vector<std::string>& reachable_objects,
        const SE2State& robot_goal
    ) override;
    
    std::string getStrategyName() const override {
        return "RandomSelection";
    }
    
    void configure(const std::map<std::string, double>& params) override;
    
    // Configuration access
    void updateConfig(std::shared_ptr<ConfigManager> config) { config_ = config; }
};

} // namespace namo