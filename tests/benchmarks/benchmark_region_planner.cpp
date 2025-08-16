#include "planners/region/region_based_planner.hpp"
#include "planners/region/region_tree_search.hpp"
#include "environment/namo_environment.hpp"
#include "config/config_manager.hpp"
#include "core/types.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iomanip>

using namespace namo;

struct BenchmarkResult {
    std::string test_name;
    bool success;
    double planning_time_ms;
    int actions_found;
    int nodes_expanded;
    int max_depth_reached;
    std::string failure_reason;
};

struct BenchmarkStats {
    double mean_time_ms = 0.0;
    double median_time_ms = 0.0;
    double min_time_ms = 0.0;
    double max_time_ms = 0.0;
    double std_dev_ms = 0.0;
    double success_rate = 0.0;
    double mean_actions = 0.0;
    double mean_nodes_expanded = 0.0;
};

class RegionPlannerBenchmark {
public:
    RegionPlannerBenchmark() {
        // Initialize with minimal mock environment
        // std::cout << "Initializing benchmark environment..." << std::endl;
    }
    
    BenchmarkStats run_benchmark(const std::string& test_suite_name,
                                const std::vector<SE2State>& goals,
                                int max_depth,
                                int iterations_per_goal = 5) {
        
        // std::cout << "\n=== Running " << test_suite_name << " Benchmark ===" << std::endl;
        // std::cout << "Goals: " << goals.size() << ", Iterations per goal: " << iterations_per_goal << std::endl;
        // std::cout << "Max depth: " << max_depth << std::endl;
        
        std::vector<BenchmarkResult> all_results;
        
        // Create minimal config for testing
        auto config = create_test_config();
        
        // Mock environment (since we can't easily create real MuJoCo environment for unit tests)
        try {
            // For this benchmark, we'll focus on the core algorithmic performance
            // without requiring full MuJoCo setup
            
            for (size_t goal_idx = 0; goal_idx < goals.size(); ++goal_idx) {
                const SE2State& goal = goals[goal_idx];
                
                // std::cout << "Testing goal " << (goal_idx + 1) << "/" << goals.size() 
                          // << " at (" << goal.x << ", " << goal.y << ", " << goal.theta << ")" << std::endl;
                
                for (int iter = 0; iter < iterations_per_goal; ++iter) {
                    BenchmarkResult result = run_single_test(goal, max_depth, config.get());
                    result.test_name = test_suite_name + "_goal" + std::to_string(goal_idx) + "_iter" + std::to_string(iter);
                    all_results.push_back(result);
                    
                    if (iter == 0) {  // Print details for first iteration only
                        // std::cout << "  Iteration 1: " << (result.success ? "SUCCESS" : "FAILED")
                                  // << " (" << std::fixed << std::setprecision(3) << result.planning_time_ms << " ms";
                        if (result.success) {
                            // std::cout << ", " << result.actions_found << " actions, " 
                                     // << result.nodes_expanded << " nodes)";
                        } else {
                            // std::cout << ", " << result.failure_reason << ")";
                        }
                        // std::cout << std::endl;
                    }
                }
            }
            
        } catch (const std::exception& e) {
            std::cerr << "Benchmark error: " << e.what() << std::endl;
            // Return empty stats on error
            return BenchmarkStats{};
        }
        
        return compute_statistics(all_results);
    }
    
private:
    std::unique_ptr<ConfigManager> create_test_config() {
        // Create minimal config for testing
        auto config = std::make_unique<ConfigManager>();
        
        // Set reasonable defaults for benchmarking
        // Note: region planner parameters are handled internally by the planner
        // We can use the existing configuration methods or implement specific setters if needed
        
        return config;
    }
    
    BenchmarkResult run_single_test(const SE2State& goal, int max_depth, ConfigManager* config) {
        BenchmarkResult result;
        
        try {
            // For unit testing, we'll create a minimal simulation without full MuJoCo
            // This focuses on the algorithmic performance of the region-based planner
            
            // Create lightweight state for testing
            LightweightState test_state;
            test_state.robot_pose = SE2State(0.0, 0.0, 0.0);  // Start at origin
            
            // Add some test objects
            test_state.object_names.push_back("box_1");
            test_state.object_names.push_back("box_2");
            test_state.movable_object_poses[0] = SE2State(1.0, 0.0, 0.0);
            test_state.movable_object_poses[1] = SE2State(2.0, 0.0, 0.0);
            
            // Time the core search algorithm
            auto start_time = std::chrono::high_resolution_clock::now();
            
            // Create mock environment for testing
            // Note: This is a simplified test that focuses on the search algorithm
            // without requiring full MuJoCo integration
            
            MockNAMOEnvironment mock_env;
            RegionTreeSearch tree_search(mock_env, max_depth, 5);
            
            TreeSearchResult search_result = tree_search.search(test_state, goal, max_depth);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            
            result.success = search_result.solution_found;
            result.planning_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            result.actions_found = search_result.num_actions();
            result.nodes_expanded = search_result.nodes_expanded;
            result.max_depth_reached = search_result.max_depth_reached;
            
            if (!result.success) {
                result.failure_reason = search_result.failure_reason;
            }
            
        } catch (const std::exception& e) {
            result.success = false;
            result.failure_reason = e.what();
            result.planning_time_ms = 0.0;
        }
        
        return result;
    }
    
    BenchmarkStats compute_statistics(const std::vector<BenchmarkResult>& results) {
        BenchmarkStats stats;
        
        if (results.empty()) {
            return stats;
        }
        
        // Extract timing data from successful runs
        std::vector<double> times;
        std::vector<int> actions;
        std::vector<int> nodes;
        int successful_runs = 0;
        
        for (const auto& result : results) {
            if (result.success) {
                times.push_back(result.planning_time_ms);
                actions.push_back(result.actions_found);
                nodes.push_back(result.nodes_expanded);
                successful_runs++;
            }
        }
        
        if (!times.empty()) {
            // Sort for median calculation
            std::vector<double> sorted_times = times;
            std::sort(sorted_times.begin(), sorted_times.end());
            
            // Basic statistics
            stats.mean_time_ms = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
            stats.median_time_ms = sorted_times[sorted_times.size() / 2];
            stats.min_time_ms = *std::min_element(times.begin(), times.end());
            stats.max_time_ms = *std::max_element(times.begin(), times.end());
            
            // Standard deviation
            double variance = 0.0;
            for (double time : times) {
                variance += (time - stats.mean_time_ms) * (time - stats.mean_time_ms);
            }
            stats.std_dev_ms = std::sqrt(variance / times.size());
            
            // Action and node statistics
            stats.mean_actions = std::accumulate(actions.begin(), actions.end(), 0.0) / actions.size();
            stats.mean_nodes_expanded = std::accumulate(nodes.begin(), nodes.end(), 0.0) / nodes.size();
        }
        
        stats.success_rate = static_cast<double>(successful_runs) / results.size();
        
        return stats;
    }
    
    // Mock environment for testing without full MuJoCo setup
    class MockNAMOEnvironment : public NAMOEnvironment {
    public:
        MockNAMOEnvironment() : NAMOEnvironment("", false) {
            // Minimal mock setup for algorithmic testing
        }
    };
};

void print_benchmark_stats(const std::string& test_name, const BenchmarkStats& stats) {
    // std::cout << "\n--- " << test_name << " Results ---" << std::endl;
    // std::cout << std::fixed << std::setprecision(3);
    // std::cout << "Success Rate:      " << (stats.success_rate * 100.0) << "%" << std::endl;
    // std::cout << "Mean Time:         " << stats.mean_time_ms << " ms" << std::endl;
    // std::cout << "Median Time:       " << stats.median_time_ms << " ms" << std::endl;
    // std::cout << "Min Time:          " << stats.min_time_ms << " ms" << std::endl;
    // std::cout << "Max Time:          " << stats.max_time_ms << " ms" << std::endl;
    // std::cout << "Std Dev:           " << stats.std_dev_ms << " ms" << std::endl;
    // std::cout << "Mean Actions:      " << stats.mean_actions << std::endl;
    // std::cout << "Mean Nodes:        " << stats.mean_nodes_expanded << std::endl;
}

int main() {
    // std::cout << "=== Region-Based Planner Performance Benchmark ===" << std::endl;
    
    RegionPlannerBenchmark benchmark;
    
    // Define test scenarios
    std::vector<SE2State> simple_goals = {
        SE2State(1.5, 0.0, 0.0),    // Simple goal requiring 1 object move
        SE2State(2.5, 0.0, 0.0),    // Goal requiring 2 object moves
        SE2State(0.5, 1.0, 0.0),    // Goal in different direction
        SE2State(3.0, 0.0, 0.0),    // Challenging goal
    };
    
    std::vector<SE2State> complex_goals = {
        SE2State(4.0, 0.0, 0.0),    // Far goal requiring multiple moves
        SE2State(2.0, 2.0, 0.0),    // Goal requiring object rearrangement
        SE2State(3.0, -1.0, 0.0),   // Goal with negative y component
        SE2State(1.0, 1.5, 1.57),   // Goal with rotation
    };
    
    try {
        // Benchmark 1: Simple scenarios with depth 1
        BenchmarkStats simple_depth1 = benchmark.run_benchmark(
            "Simple_Goals_Depth1", simple_goals, 1, 3);
        print_benchmark_stats("Simple Goals (Depth 1)", simple_depth1);
        
        // Benchmark 2: Simple scenarios with depth 2
        BenchmarkStats simple_depth2 = benchmark.run_benchmark(
            "Simple_Goals_Depth2", simple_goals, 2, 3);
        print_benchmark_stats("Simple Goals (Depth 2)", simple_depth2);
        
        // Benchmark 3: Complex scenarios with depth 2
        BenchmarkStats complex_depth2 = benchmark.run_benchmark(
            "Complex_Goals_Depth2", complex_goals, 2, 3);
        print_benchmark_stats("Complex Goals (Depth 2)", complex_depth2);
        
        // Benchmark 4: Scalability test with increasing depth
        // std::cout << "\n=== Scalability Analysis ===" << std::endl;
        
        SE2State scalability_goal(2.5, 0.0, 0.0);
        std::vector<SE2State> single_goal = {scalability_goal};
        
        for (int depth = 1; depth <= 3; ++depth) {
            BenchmarkStats scalability_stats = benchmark.run_benchmark(
                "Scalability_Depth" + std::to_string(depth), single_goal, depth, 5);
            
            // std::cout << "Depth " << depth << ": " 
                     // << std::fixed << std::setprecision(3)
                     // << scalability_stats.mean_time_ms << " ms avg, "
                     // << (scalability_stats.success_rate * 100.0) << "% success, "
                     // << scalability_stats.mean_nodes_expanded << " nodes avg" << std::endl;
        }
        
        // Summary
        // std::cout << "\n=== Benchmark Summary ===" << std::endl;
        // std::cout << "✓ Algorithm performance measured across multiple scenarios" << std::endl;
        // std::cout << "✓ Scalability analyzed for different search depths" << std::endl;
        // std::cout << "✓ Success rates and timing statistics computed" << std::endl;
        // std::cout << "✓ Region-based planner core algorithms validated" << std::endl;
        
        // std::cout << "\n=== All Benchmarks Completed Successfully! ===" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Benchmark error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown benchmark error occurred" << std::endl;
        return 1;
    }
}