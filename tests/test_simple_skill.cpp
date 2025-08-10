#include "skills/namo_push_skill.hpp"
#include "core/parameter_loader.hpp"
#include "core/mujoco_wrapper.hpp"
#include <iostream>
#include <cassert>

namespace namo {

/**
 * @brief Simple test to validate skill interface basics
 */
class SimpleSkillTest {
private:
    std::unique_ptr<NAMOEnvironment> env_;
    std::unique_ptr<NAMOPushSkill> skill_;
    
public:
    void run_test() {
        std::cout << "Simple Skill Interface Test" << std::endl;
        std::cout << "===========================" << std::endl;
        
        try {
            setup_environment();
            test_basic_interface();
            std::cout << "\n🎉 Simple skill test passed!" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "\n❌ Test failed: " << e.what() << std::endl;
            throw;
        }
    }
    
private:
    void setup_environment() {
        std::cout << "Setting up test environment..." << std::endl;
        
        std::string xml_path = "data/test_scene.xml";
        bool visualize = false;
        
        // Initialize environment
        env_ = std::make_unique<NAMOEnvironment>(xml_path, visualize);
        
        // Initialize skill
        skill_ = std::make_unique<NAMOPushSkill>(*env_);
        
        std::cout << "✓ Environment setup complete" << std::endl;
    }
    
    void test_basic_interface() {
        std::cout << "\n=== Testing Basic Interface ===" << std::endl;
        
        // Test skill metadata
        std::string name = skill_->get_name();
        std::string desc = skill_->get_description();
        
        std::cout << "Skill name: " << name << std::endl;
        std::cout << "Description: " << desc << std::endl;
        
        assert(name == "namo_push");
        assert(!desc.empty());
        
        // Test parameter schema
        auto schema = skill_->get_parameter_schema();
        std::cout << "Parameter schema (" << schema.size() << " parameters):" << std::endl;
        
        for (const auto& param : schema) {
            std::cout << "  - " << param.first << ": " << param.second.description << std::endl;
        }
        
        // Verify required parameters exist
        assert(schema.find("object_name") != schema.end());
        assert(schema.find("target_pose") != schema.end());
        
        std::cout << "✓ Basic interface tests passed" << std::endl;
    }
};

} // namespace namo

int main() {
    try {
        namo::SimpleSkillTest test;
        test.run_test();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}