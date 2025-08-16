#pragma once

#include <vector>
#include <string>
#include <map>
#include <memory>
#include <chrono>
#include "core/types.hpp"

namespace namo {

// Forward declarations
class NAMOEnvironment;
class WavefrontPlanner;
class NAMOPushSkill;
class ConfigManager;

/**
 * @brief Represents a single action step in an optimization sequence
 */
struct ActionStep {
    std::string object_name;
    SE2State target_state;      // Where we want to move the object
    SE2State final_state;       // Where the object actually ended up
    bool execution_success;     // Whether the action succeeded
    
    ActionStep(const std::string& name, const SE2State& target, const SE2State& final, bool success)
        : object_name(name), target_state(target), final_state(final), execution_success(success) {}
};

/**
 * @brief Result of sequence optimization containing all found minimal sequences
 */
struct OptimizationResult {
    std::vector<std::vector<std::string>> minimal_sequences;  // All found minimal sequences (object names)
    std::vector<std::vector<int>> minimal_indices;           // Same sequences as indices into original
    double optimization_time_seconds;
    int sequences_tested;
    int original_sequence_length;
    int minimal_sequence_length;
    bool optimization_successful;
    std::string failure_reason;
    
    OptimizationResult() 
        : optimization_time_seconds(0.0), sequences_tested(0), 
          original_sequence_length(0), minimal_sequence_length(0),
          optimization_successful(false) {}
};

/**
 * @brief Optimization algorithm types - extensible for future algorithms
 */
enum class OptimizationMethod {
    EXHAUSTIVE,           // optimizeActionSequence - full exhaustive search
    REVERSE_ORDER,        // optimizeActionSequence2 - backward combinatorial search  
    GREEDY_REMOVAL,       // optimizeActionSequenceGreedy - greedy iterative removal
    NONE                  // No optimization
};

/**
 * @brief Abstract base class for optimization strategies (Strategy Pattern)
 */
class OptimizationStrategy {
public:
    virtual ~OptimizationStrategy() = default;
    
    virtual OptimizationResult optimize(
        const std::vector<ActionStep>& action_steps,
        const SE2State& robot_goal,
        NAMOEnvironment& env,
        WavefrontPlanner& wavefront_planner,
        NAMOPushSkill& skill
    ) = 0;
    
    virtual std::string get_method_name() const = 0;
    virtual OptimizationMethod get_method_type() const = 0;
};

/**
 * @brief Reverse-order combinatorial optimization strategy (optimizeActionSequence2)
 */
class ReverseOrderStrategy : public OptimizationStrategy {
private:
    double timeout_seconds_;
    int max_sequence_length_;
    
public:
    ReverseOrderStrategy(double timeout = 30.0, int max_length = 20)
        : timeout_seconds_(timeout), max_sequence_length_(max_length) {}
    
    OptimizationResult optimize(
        const std::vector<ActionStep>& action_steps,
        const SE2State& robot_goal,
        NAMOEnvironment& env,
        WavefrontPlanner& wavefront_planner,
        NAMOPushSkill& skill
    ) override;
    
    std::string get_method_name() const override { return "ReverseOrder"; }
    OptimizationMethod get_method_type() const override { return OptimizationMethod::REVERSE_ORDER; }
};

/**
 * @brief Exhaustive search optimization strategy (optimizeActionSequence)
 */
class ExhaustiveStrategy : public OptimizationStrategy {
private:
    double timeout_seconds_;
    int max_sequence_length_;
    
public:
    ExhaustiveStrategy(double timeout = 60.0, int max_length = 15)
        : timeout_seconds_(timeout), max_sequence_length_(max_length) {}
    
    OptimizationResult optimize(
        const std::vector<ActionStep>& action_steps,
        const SE2State& robot_goal,
        NAMOEnvironment& env,
        WavefrontPlanner& wavefront_planner,
        NAMOPushSkill& skill
    ) override;
    
    std::string get_method_name() const override { return "Exhaustive"; }
    OptimizationMethod get_method_type() const override { return OptimizationMethod::EXHAUSTIVE; }
};

/**
 * @brief Greedy removal optimization strategy (optimizeActionSequenceGreedy)  
 */
class GreedyRemovalStrategy : public OptimizationStrategy {
private:
    double timeout_seconds_;
    
public:
    GreedyRemovalStrategy(double timeout = 10.0)
        : timeout_seconds_(timeout) {}
    
    OptimizationResult optimize(
        const std::vector<ActionStep>& action_steps,
        const SE2State& robot_goal,
        NAMOEnvironment& env,
        WavefrontPlanner& wavefront_planner,
        NAMOPushSkill& skill
    ) override;
    
    std::string get_method_name() const override { return "GreedyRemoval"; }
    OptimizationMethod get_method_type() const override { return OptimizationMethod::GREEDY_REMOVAL; }
};

/**
 * @brief Main action sequence optimizer - uses Strategy Pattern for different optimization methods
 */
class ActionSequenceOptimizer {
private:
    std::unique_ptr<OptimizationStrategy> strategy_;
    std::shared_ptr<ConfigManager> config_;
    
    // Environment state management
    bool has_saved_state_;
    
    // Helper methods
    bool test_action_sequence(
        const std::vector<int>& sequence_indices,
        const std::vector<ActionStep>& action_steps,
        const SE2State& robot_goal,
        NAMOEnvironment& env,
        WavefrontPlanner& wavefront_planner,
        NAMOPushSkill& skill
    );
    
    bool execute_single_action(
        const ActionStep& action,
        NAMOEnvironment& env,
        NAMOPushSkill& skill
    );
    
public:
    explicit ActionSequenceOptimizer(std::shared_ptr<ConfigManager> config);
    ~ActionSequenceOptimizer() = default;
    
    // Strategy management
    void set_optimization_method(OptimizationMethod method);
    void set_custom_strategy(std::unique_ptr<OptimizationStrategy> strategy);
    OptimizationMethod get_current_method() const;
    std::string get_current_method_name() const;
    
    // Main optimization interface
    OptimizationResult optimize_sequence(
        const std::vector<std::string>& object_sequence,
        const std::map<std::string, SE2State>& target_states,
        const std::map<std::string, SE2State>& final_states,
        const SE2State& robot_goal,
        NAMOEnvironment& env,
        WavefrontPlanner& wavefront_planner,
        NAMOPushSkill& skill
    );
    
    // Convenience method for PlanningResult integration
    OptimizationResult optimize_planning_result(
        const struct PlanningResult& planning_result,
        const SE2State& robot_goal,
        NAMOEnvironment& env,
        WavefrontPlanner& wavefront_planner,
        NAMOPushSkill& skill
    );
    
    // Utility methods
    bool is_optimization_enabled() const;
    void print_optimization_summary(const OptimizationResult& result) const;
    
    // Static factory methods for easy strategy creation
    static std::unique_ptr<OptimizationStrategy> create_strategy(
        OptimizationMethod method,
        const std::shared_ptr<ConfigManager>& config = nullptr
    );
};

} // namespace namo