/**
 * @file test_complete_planning.cpp
 * @brief Integration test for complete NAMO planning pipeline
 * 
 * Tests the two-stage approach:
 * 1. Abstract planning in empty environment (GreedyPlanner)
 * 2. MPC execution with real physics (MPCExecutor)
 */

#include "core/parameter_loader.hpp"
#include "environment/namo_environment.hpp"
#include "planning/namo_push_controller.hpp"
#include "planning/primitive_loader.hpp"
#include "planning/greedy_planner.hpp"
#include "planning/mpc_executor.hpp"
#include <iostream>
#include <chrono>
#include <vector>

using namespace namo;

/**
 * @brief Test case structure
 */
struct TestCase {
    std::string name;
    SE2State start_state;
    SE2State goal_state;
    std::vector<int> allowed_edges;  // Empty = use all edges
    bool expect_success;
    double max_planning_time_ms;
    
    TestCase(const std::string& n, const SE2State& start, const SE2State& goal, 
             bool success = true, double max_time = 1000.0)
        : name(n), start_state(start), goal_state(goal), expect_success(success), max_planning_time_ms(max_time) {}
};

/**
 * @brief Planning test results
 */
struct TestResult {
    bool success;
    double planning_time_ms;
    double execution_time_ms;
    int plan_length;
    int executed_steps;
    bool robot_goal_reached;
    std::string failure_reason;
};

/**
 * @brief Complete planning system tester
 */
class PlanningTester {
private:
    NAMOEnvironment env_;
    NAMOPushController controller_;
    PrimitiveLoader loader_;
    GreedyPlanner planner_;
    MPCExecutor executor_;
    
    std::vector<TestCase> test_cases_;
    
public:
    PlanningTester(const std::string& scene_path, bool visualize = false) 
        : env_(scene_path, visualize), controller_(env_), executor_(env_, controller_) {
        
        // Initialize test cases
        setup_test_cases();
    }
    
    /**
     * @brief Initialize the planning system
     */
    bool initialize(const std::string& primitive_filepath) {
        // std::cout << "Initializing planning system..." << std::endl;
        
        // Load primitives
        if (!loader_.load_primitives(primitive_filepath)) {
            std::cerr << "Failed to load primitives from " << primitive_filepath << std::endl;
            return false;
        }
        
        // Initialize planner
        if (!planner_.initialize(primitive_filepath)) {
            std::cerr << "Failed to initialize planner" << std::endl;
            return false;
        }
        
        // Set MPC parameters
        executor_.set_parameters(10, 0.02, 0.15, 3);  // max_steps, pos_thresh, ang_thresh, stuck_limit
        
        // std::cout << "Planning system initialized successfully" << std::endl;
        return true;
    }
    
    /**
     * @brief Run all test cases
     */
    void run_all_tests() {
        // std::cout << "\n=== Running Complete Planning Pipeline Tests ===" << std::endl;
        // std::cout << "Test approach: Abstract Planning → MPC Execution" << std::endl;
        // std::cout << "Total test cases: " << test_cases_.size() << "\n" << std::endl;
        
        int passed = 0;
        int failed = 0;
        
        for (size_t i = 0; i < test_cases_.size(); i++) {
            const TestCase& test = test_cases_[i];
            
            // std::cout << "--- Test " << (i+1) << "/" << test_cases_.size() 
                      // << ": " << test.name << " ---" << std::endl;
            
            TestResult result = run_single_test(test);
            
            if (result.success == test.expect_success) {
                // std::cout << "✓ PASSED" << std::endl;
                passed++;
            } else {
                // std::cout << "✗ FAILED: " << result.failure_reason << std::endl;
                failed++;
            }
            
            print_test_result(result);
            // std::cout << std::endl;
        }
        
        // Summary
        // std::cout << "=== Test Summary ===" << std::endl;
        // std::cout << "Passed: " << passed << "/" << test_cases_.size() << std::endl;
        // std::cout << "Failed: " << failed << "/" << test_cases_.size() << std::endl;
        
        if (failed == 0) {
            // std::cout << "🎉 All tests passed!" << std::endl;
        }
    }
    
    /**
     * @brief Test primitive loading performance
     */
    void test_primitive_loading() {
        // std::cout << "\n=== Primitive Loading Test ===" << std::endl;
        
        const auto& primitives = loader_.get_all_primitives();
        // std::cout << "Total primitives: " << loader_.size() << std::endl;
        
        // Test lookup performance
        auto start = std::chrono::high_resolution_clock::now();
        
        int lookup_count = 0;
        for (int edge = 0; edge < 12; edge++) {
            auto valid_steps = loader_.get_valid_steps_for_edge(edge);
            // std::cout << "Edge " << edge << ": " << valid_steps.size() << " valid steps";
            
            for (int step : valid_steps) {
                const LoadedPrimitive& prim = loader_.get_primitive(edge, step);
                lookup_count++;
                
                if (valid_steps.size() <= 3) {  // Print details for edges with few primitives
                    // std::cout << " [" << step << "steps: dx=" << prim.delta_x 
                              // << " dy=" << prim.delta_y << " dθ=" << prim.delta_theta << "]";
                }
            }
            // std::cout << std::endl;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // std::cout << "Performed " << lookup_count << " primitive lookups in " 
                  // << duration.count() << " μs" << std::endl;
        // std::cout << "Average lookup time: " << (duration.count() / lookup_count) << " μs" << std::endl;
    }
    
private:
    /**
     * @brief Setup test cases covering different scenarios
     */
    void setup_test_cases() {
        // Test 1: Small displacement (should be easy)
        test_cases_.emplace_back(
            "Small displacement",
            SE2State(0.0, 0.0, 0.0),
            SE2State(0.1, 0.1, 0.2),
            true, 500.0
        );
        
        // Test 2: Medium displacement
        test_cases_.emplace_back(
            "Medium displacement", 
            SE2State(0.0, 0.0, 0.0),
            SE2State(0.3, 0.2, 0.5),
            true, 1000.0
        );
        
        // Test 3: Large displacement (may require multiple primitives)
        test_cases_.emplace_back(
            "Large displacement",
            SE2State(0.0, 0.0, 0.0), 
            SE2State(0.5, 0.4, 1.0),
            true, 2000.0
        );
        
        // Test 4: Pure rotation
        test_cases_.emplace_back(
            "Pure rotation",
            SE2State(0.0, 0.0, 0.0),
            SE2State(0.0, 0.0, 1.57),  // 90 degrees
            true, 800.0
        );
        
        // Test 5: Constrained edges (only allow subset of push directions)
        TestCase constrained_test(
            "Constrained edges",
            SE2State(0.0, 0.0, 0.0),
            SE2State(0.2, 0.15, 0.3),
            true, 1500.0
        );
        constrained_test.allowed_edges = {0, 3, 6, 9};  // Only 4 directions
        test_cases_.push_back(constrained_test);
        
        // Test 6: Impossible goal (very far away)
        test_cases_.emplace_back(
            "Impossible goal",
            SE2State(0.0, 0.0, 0.0),
            SE2State(2.0, 2.0, 0.0),  // Way outside primitive reach
            false, 3000.0
        );
    }
    
    /**
     * @brief Run a single test case
     */
    TestResult run_single_test(const TestCase& test) {
        TestResult result;
        result.success = false;
        result.plan_length = 0;
        result.executed_steps = 0;
        result.robot_goal_reached = false;
        
        // Reset environment to clean state
        env_.reset();
        
        // Get the movable object (assume first one for testing)
        auto movable_objects = env_.get_movable_objects();
        if (movable_objects.empty()) {
            result.failure_reason = "No movable objects in environment";
            return result;
        }
        
        std::string object_name = movable_objects[0].name;
        // std::cout << "Testing with object: " << object_name << std::endl;
        
        // Set object to start state
        // Note: This would require environment API to set object pose
        // For now, we'll work with whatever the current state is
        
        // Stage 1: Abstract Planning
        // std::cout << "Stage 1: Abstract planning in empty environment..." << std::endl;
        auto planning_start = std::chrono::high_resolution_clock::now();
        
        std::vector<PlanStep> plan = planner_.plan_push_sequence(
            test.start_state,
            test.goal_state,
            test.allowed_edges,
            5000  // expansion limit
        );
        
        auto planning_end = std::chrono::high_resolution_clock::now();
        result.planning_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(
            planning_end - planning_start).count() / 1000.0;
        
        result.plan_length = plan.size();
        
        if (plan.empty()) {
            result.failure_reason = "No plan found in abstract planning stage";
            return result;
        }
        
        // std::cout << "Generated plan with " << plan.size() << " primitive steps" << std::endl;
        
        // Print plan details
        for (size_t i = 0; i < std::min(plan.size(), size_t(5)); i++) {
            const PlanStep& step = plan[i];
            // std::cout << "  Step " << (i+1) << ": Edge " << step.edge_idx 
                      // << ", " << step.push_steps << " steps → ["
                      // << step.pose.x << ", " << step.pose.y << ", " << step.pose.theta << "]" << std::endl;
        }
        if (plan.size() > 5) {
            // std::cout << "  ... and " << (plan.size() - 5) << " more steps" << std::endl;
        }
        
        // Stage 2: MPC Execution  
        // std::cout << "Stage 2: MPC execution with real physics..." << std::endl;
        auto execution_start = std::chrono::high_resolution_clock::now();
        
        ExecutionResult exec_result = executor_.execute_plan(object_name, plan);
        
        auto execution_end = std::chrono::high_resolution_clock::now();
        result.execution_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            execution_end - execution_start).count();
        
        result.executed_steps = exec_result.steps_executed;
        result.robot_goal_reached = exec_result.robot_goal_reached;
        result.success = exec_result.success;
        
        if (!exec_result.success) {
            result.failure_reason = "MPC execution failed: " + exec_result.failure_reason;
        }
        
        return result;
    }
    
    /**
     * @brief Print detailed test result
     */
    void print_test_result(const TestResult& result) {
        // std::cout << "  Planning time: " << result.planning_time_ms << " ms" << std::endl;
        // std::cout << "  Execution time: " << result.execution_time_ms << " ms" << std::endl;
        // std::cout << "  Plan length: " << result.plan_length << " primitives" << std::endl;
        // std::cout << "  Executed steps: " << result.executed_steps << "/" << result.plan_length << std::endl;
        // std::cout << "  Robot goal reached: " << (result.robot_goal_reached ? "Yes" : "No") << std::endl;
    }
};

/**
 * @brief Main function
 */
int main(int argc, char* argv[]) {
    if (argc != 2) {
        // std::cout << "Usage: " << argv[0] << " <config_file>" << std::endl;
        return 1;
    }
    
    try {
        // Load configuration
        FastParameterLoader params(argv[1]);
        
        std::string scene_path = params.get_parameter("scene_path", "data/test_scene.xml");
        bool visualize = params.get_bool("visualize", false);
        std::string primitive_path = params.get_parameter("primitive_path", "data/motion_primitives.dat");
        
        // std::cout << "=== NAMO Complete Planning Pipeline Test ===" << std::endl;
        // std::cout << "Scene: " << scene_path << std::endl;
        // std::cout << "Primitives: " << primitive_path << std::endl;
        // std::cout << "Visualization: " << (visualize ? "enabled" : "disabled") << std::endl;
        // std::cout << std::endl;
        
        // Create tester
        PlanningTester tester(scene_path, visualize);
        
        // Initialize system
        if (!tester.initialize(primitive_path)) {
            std::cerr << "Failed to initialize planning system" << std::endl;
            return 1;
        }
        
        // Test primitive loading
        tester.test_primitive_loading();
        
        // Run all integration tests
        tester.run_all_tests();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}