/**
 * @file skill_demo.cpp
 * @brief Demonstration of NAMO Skill System for High-Level Planners
 * 
 * This example shows how any high-level planner can use the NAMO skill
 * system to perform object manipulation without knowing internal details.
 */

#include "skills/namo_push_skill.hpp"
#include "environment/namo_environment.hpp"
#include <iostream>
#include <iomanip>

namespace namo {

class SkillDemo {
private:
    std::unique_ptr<NAMOEnvironment> env_;
    std::unique_ptr<NAMOPushSkill> skill_;
    
public:
    void run() {
        std::cout << "=== NAMO Skill System Demonstration ===\n" << std::endl;
        
        setup_environment();
        demonstrate_skill_interface();
        demonstrate_planning_integration();
        demonstrate_error_handling();
        
        std::cout << "\n=== Demonstration Complete ===\n" << std::endl;
    }
    
private:
    void setup_environment() {
        std::cout << "1. Setting up environment and skill..." << std::endl;
        
        // Initialize environment (without visualization for demo)
        env_ = std::make_unique<NAMOEnvironment>("data/test_scene.xml", false);
        
        // Configure skill with custom parameters
        NAMOPushSkill::Config config;
        config.tolerance = 0.02;           // 2cm tolerance
        config.max_planning_attempts = 3;  // Try 3 times
        config.planning_timeout = std::chrono::seconds(5);
        
        skill_ = std::make_unique<NAMOPushSkill>(*env_, config);
        
        std::cout << "   ✓ Environment loaded with " 
                  << env_->get_num_movable() << " movable objects" << std::endl;
        std::cout << "   ✓ Skill configured with 2cm tolerance" << std::endl;
    }
    
    void demonstrate_skill_interface() {
        std::cout << "\n2. Demonstrating skill interface..." << std::endl;
        
        // Get skill metadata
        std::cout << "   Skill Name: " << skill_->get_name() << std::endl;
        std::cout << "   Description: " << skill_->get_description() << std::endl;
        
        // Show parameter schema
        auto schema = skill_->get_parameter_schema();
        std::cout << "   Parameters (" << schema.size() << "):" << std::endl;
        for (const auto& [name, spec] : schema) {
            std::cout << "     - " << std::setw(15) << std::left << name 
                      << ": " << spec.description 
                      << (spec.required ? " (required)" : " (optional)") << std::endl;
        }
        
        // Show current world state
        auto world_state = skill_->get_world_state();
        std::cout << "   World State (" << world_state.size() << " items):" << std::endl;
        for (const auto& [key, value] : world_state) {
            std::cout << "     - " << key << std::endl;
        }
    }
    
    void demonstrate_planning_integration() {
        std::cout << "\n3. Demonstrating high-level planner integration..." << std::endl;
        
        // Example: Task planner wants to move object to (1.0, 1.0, 0.0)
        std::map<std::string, SkillParameterValue> task_params = {
            {"object_name", std::string("obstacle_1_movable")},
            {"target_pose", SE2State(1.0, 1.0, 0.0)}
        };
        
        // 1. Check applicability (action selection)
        bool applicable = skill_->is_applicable(task_params);
        std::cout << "   Applicability check: " << (applicable ? "✓ Applicable" : "✗ Not applicable") << std::endl;
        
        // 2. Check preconditions (detailed analysis)
        auto preconditions = skill_->check_preconditions(task_params);
        if (preconditions.empty()) {
            std::cout << "   Preconditions: ✓ All met" << std::endl;
        } else {
            std::cout << "   Preconditions: ✗ Unmet (" << preconditions.size() << "):" << std::endl;
            for (const auto& condition : preconditions) {
                std::cout << "     - " << condition << std::endl;
            }
        }
        
        // 3. Duration estimation (temporal planning)
        if (applicable) {
            auto duration = skill_->estimate_duration(task_params);
            std::cout << "   Estimated duration: " << duration.count() << "ms" << std::endl;
        }
        
        // 4. Execution (if applicable)
        if (applicable && preconditions.empty()) {
            std::cout << "   Executing skill..." << std::endl;
            auto result = skill_->execute(task_params);
            
            if (result.success) {
                std::cout << "   ✓ Execution successful!" << std::endl;
                std::cout << "     - Actual duration: " << result.execution_time.count() << "ms" << std::endl;
                std::cout << "     - Steps executed: " << std::get<int>(result.outputs.at("steps_executed")) << std::endl;
                std::cout << "     - Robot goal reached: " << std::get<bool>(result.outputs.at("robot_goal_reached")) << std::endl;
            } else {
                std::cout << "   ✗ Execution failed: " << result.failure_reason << std::endl;
            }
        } else {
            std::cout << "   Skipping execution due to unmet preconditions" << std::endl;
        }
    }
    
    void demonstrate_error_handling() {
        std::cout << "\n4. Demonstrating error handling..." << std::endl;
        
        // Example: Invalid parameters
        std::map<std::string, SkillParameterValue> invalid_params = {
            {"object_name", std::string("nonexistent_object")},
            {"target_pose", SE2State(100.0, 100.0, 0.0)}  // Way outside bounds
        };
        
        std::cout << "   Testing with invalid parameters..." << std::endl;
        
        bool applicable = skill_->is_applicable(invalid_params);
        std::cout << "   Applicability: " << (applicable ? "✓" : "✗") << " (should be false)" << std::endl;
        
        auto preconditions = skill_->check_preconditions(invalid_params);
        std::cout << "   Precondition failures (" << preconditions.size() << "):" << std::endl;
        for (const auto& condition : preconditions) {
            std::cout << "     - " << condition << std::endl;
        }
        
        // Example: Graceful degradation
        std::cout << "\n   Demonstrating graceful degradation..." << std::endl;
        std::map<std::string, SkillParameterValue> challenging_params = {
            {"object_name", std::string("obstacle_1_movable")},
            {"target_pose", SE2State(3.0, 3.0, M_PI)},  // Far but valid
            {"tolerance", 0.001},  // Very strict tolerance
            {"max_attempts", 1}    // Only one attempt
        };
        
        auto result = skill_->execute(challenging_params);
        if (!result.success) {
            std::cout << "   First attempt failed: " << result.failure_reason << std::endl;
            std::cout << "   Trying with relaxed parameters..." << std::endl;
            
            // Relax parameters
            challenging_params["tolerance"] = 0.05;  // More lenient
            challenging_params["max_attempts"] = 5;  // More attempts
            
            result = skill_->execute(challenging_params);
            std::cout << "   Second attempt: " << (result.success ? "✓ Success" : "✗ Failed") << std::endl;
        } else {
            std::cout << "   ✓ Challenging parameters succeeded on first try!" << std::endl;
        }
    }
};

} // namespace namo

// Example integration patterns for different planner types
namespace examples {

// Example 1: PDDL Planner Integration
class PDDLExecutor {
public:
    struct PDDLAction {
        std::string name;
        std::string object;
        double x, y, theta;
    };
    
    bool execute_action(namo::NAMOPushSkill& skill, const PDDLAction& action) {
        if (action.name == "push") {
            std::map<std::string, namo::SkillParameterValue> params = {
                {"object_name", action.object},
                {"target_pose", namo::SE2State(action.x, action.y, action.theta)}
            };
            
            if (skill.is_applicable(params)) {
                auto result = skill.execute(params);
                return result.success;
            }
        }
        return false;
    }
};

// Example 2: Behavior Tree Node
class PushObjectNode {
private:
    namo::NAMOPushSkill& skill_;
    std::string object_name_;
    namo::SE2State target_pose_;
    
public:
    PushObjectNode(namo::NAMOPushSkill& skill, const std::string& object, const namo::SE2State& target)
        : skill_(skill), object_name_(object), target_pose_(target) {}
    
    enum class NodeStatus { SUCCESS, FAILURE, RUNNING };
    
    NodeStatus tick() {
        std::map<std::string, namo::SkillParameterValue> params = {
            {"object_name", object_name_},
            {"target_pose", target_pose_}
        };
        
        if (!skill_.is_applicable(params)) {
            return NodeStatus::FAILURE;
        }
        
        auto result = skill_.execute(params);
        return result.success ? NodeStatus::SUCCESS : NodeStatus::FAILURE;
    }
};

// Example 3: RL Environment Interface
class RLEnvironment {
private:
    namo::NAMOPushSkill& skill_;
    
public:
    struct Action {
        std::string object_id;
        double x, y, theta;
    };
    
    struct StepResult {
        bool done;
        double reward;
        std::string info;
    };
    
    explicit RLEnvironment(namo::NAMOPushSkill& skill) : skill_(skill) {}
    
    StepResult step(const Action& action) {
        std::map<std::string, namo::SkillParameterValue> params = {
            {"object_name", action.object_id},
            {"target_pose", namo::SE2State(action.x, action.y, action.theta)}
        };
        
        // Check applicability for reward shaping
        if (!skill_.is_applicable(params)) {
            return {false, -1.0, "Invalid action"};
        }
        
        // Execute and compute reward
        auto result = skill_.execute(params);
        double reward = result.success ? 1.0 : -0.1;
        
        return {result.success, reward, result.failure_reason};
    }
};

} // namespace examples

int main() {
    try {
        namo::SkillDemo demo;
        demo.run();
        
        std::cout << "\nIntegration examples available in source code:" << std::endl;
        std::cout << "  - PDDLExecutor: PDDL action execution" << std::endl;
        std::cout << "  - PushObjectNode: Behavior tree node" << std::endl;
        std::cout << "  - RLEnvironment: Reinforcement learning interface" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Demo failed: " << e.what() << std::endl;
        return 1;
    }
}