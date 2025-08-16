#include "skills/namo_push_skill.hpp"
#include <iostream>
#include <cassert>

namespace namo {

class NAMOSkillTest {
private:
    std::unique_ptr<NAMOEnvironment> env_;
    std::unique_ptr<NAMOPushSkill> skill_;
    
public:
    void run_test() {
        // std::cout << "Testing NAMO Skill System" << std::endl;
        // std::cout << "=========================" << std::endl;
        
        try {
            setup_environment();
            test_interface();
            test_applicability();
            // std::cout << "\n✅ NAMO Skill tests passed!" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "\n❌ Test failed: " << e.what() << std::endl;
            throw;
        }
    }
    
private:
    void setup_environment() {
        // std::cout << "Setting up environment..." << std::endl;
        
        std::string xml_path = "data/test_scene.xml";
        env_ = std::make_unique<NAMOEnvironment>(xml_path, true);
        
        // Use default ConfigManager
        auto config = ConfigManager::create_default();
        skill_ = std::make_unique<NAMOPushSkill>(*env_, std::shared_ptr<ConfigManager>(config.release()));
        
        // std::cout << "✓ Environment and skill created" << std::endl;
    }
    
    void test_interface() {
        // std::cout << "\n=== Testing Interface ===" << std::endl;
        
        // Test metadata
        assert(skill_->get_name() == "namo_push");
        assert(!skill_->get_description().empty());
        
        // std::cout << "Name: " << skill_->get_name() << std::endl;
        // std::cout << "Description: " << skill_->get_description() << std::endl;
        
        // Test schema
        auto schema = skill_->get_parameter_schema();
        // std::cout << "Parameters (" << schema.size() << "):" << std::endl;
        
        for (const auto& [key, spec] : schema) {
            // std::cout << "  - " << key << ": " << spec.description 
                      // << (spec.required ? " (required)" : " (optional)") << std::endl;
        }
        
        assert(schema.find("object_name") != schema.end());
        assert(schema.find("target_pose") != schema.end());
        assert(schema.at("object_name").required);
        assert(schema.at("target_pose").required);
        assert(!schema.at("robot_goal").required);
        
        // std::cout << "✓ Interface tests passed" << std::endl;
    }
    
    void test_applicability() {
        // std::cout << "\n=== Testing Applicability ===" << std::endl;
        
        // Test missing parameters
        std::map<std::string, SkillParameterValue> empty_params;
        assert(!skill_->is_applicable(empty_params));
        // std::cout << "✓ Empty parameters correctly rejected" << std::endl;
        
        // Test with valid parameters
        std::map<std::string, SkillParameterValue> valid_params = {
            {"object_name", std::string("obstacle_1_movable")},
            {"target_pose", SE2State(1.0, 1.0, 0.0)}
        };
        
        // Check if this object exists
        bool applicable = skill_->is_applicable(valid_params);
        // std::cout << "Valid parameters applicable: " << (applicable ? "yes" : "no") << std::endl;
        
        // Test preconditions
        auto preconditions = skill_->check_preconditions(valid_params);
        // std::cout << "Preconditions (" << preconditions.size() << "):" << std::endl;
        for (const auto& condition : preconditions) {
            // std::cout << "  - " << condition << std::endl;
        }
        
        // Test state observation
        auto state = skill_->get_world_state();
        // std::cout << "World state (" << state.size() << " items):" << std::endl;
        for (const auto& [key, value] : state) {
            // std::cout << "  - " << key << std::endl;
        }
        
        // Test duration estimation
        if (applicable) {
            auto duration = skill_->estimate_duration(valid_params);
            // std::cout << "Estimated duration: " << duration.count() << "ms" << std::endl;
        }
        
        // std::cout << "✓ Applicability tests passed" << std::endl;
    }
};

} // namespace namo

int main() {
    try {
        namo::NAMOSkillTest test;
        test.run_test();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}