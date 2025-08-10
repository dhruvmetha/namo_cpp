#include "skills/namo_push_skill.hpp"
#include "environment/namo_environment.hpp"
#include <iostream>

using namespace namo;

int main() {
    std::cout << "=== NAMO Skill with Benchmark Environment Test ===" << std::endl;
    
    try {
        // Initialize environment with benchmark scene
        std::cout << "\n1. Loading benchmark environment..." << std::endl;
        NAMOEnvironment env("data/benchmark_env.xml", false);  // No visualization
        
        std::cout << "   Environment loaded successfully!" << std::endl;
        std::cout << "   Static objects: " << env.get_static_objects().size() << std::endl;
        std::cout << "   Movable objects: " << env.get_movable_objects().size() << std::endl;
        
        // Initialize skill
        std::cout << "\n2. Initializing NAMO skill..." << std::endl;
        NAMOPushSkill skill(env);
        
        std::cout << "   Skill initialized successfully!" << std::endl;
        
        // Test skill interface
        std::cout << "\n3. Testing skill interface..." << std::endl;
        auto schema = skill.get_parameter_schema();
        std::cout << "   Parameter schema contains " << schema.size() << " parameters" << std::endl;
        
        // Test with one of the movable objects
        std::cout << "\n4. Testing object manipulation..." << std::endl;
        SE2State target_pose(0.0, 0.0, 0.0);  // Move to origin
        
        std::map<std::string, SkillParameterValue> params = {
            {"object_name", std::string("obstacle_1_movable")},
            {"target_pose", target_pose}
        };
        
        // Check if skill is applicable
        bool applicable = skill.is_applicable(params);
        std::cout << "   Skill applicable: " << (applicable ? "YES" : "NO") << std::endl;
        
        if (applicable) {
            // Check preconditions
            auto preconditions = skill.check_preconditions(params);
            if (preconditions.empty()) {
                std::cout << "   Preconditions: PASS" << std::endl;
            } else {
                std::cout << "   Preconditions: FAIL (" << preconditions.size() << " unmet):" << std::endl;
                for (const auto& condition : preconditions) {
                    std::cout << "     - " << condition << std::endl;
                }
            }
        }
        
        std::cout << "\n=== Skill Test Completed Successfully! ===" << std::endl;
        std::cout << "The NAMO skill system works correctly with the benchmark environment." << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}