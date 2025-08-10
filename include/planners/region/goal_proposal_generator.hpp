#pragma once

#include "planners/region/region_graph.hpp"
#include "environment/namo_environment.hpp"
#include "core/types.hpp"
#include <random>
#include <vector>
#include <chrono>

namespace namo {

/**
 * @brief Generates goal proposals for objects during tree search
 * 
 * For each object that needs to be moved, this class generates multiple
 * candidate goal positions in free space. These proposals create the
 * branching factor in the tree search.
 */
class GoalProposalGenerator {
public:
    /**
     * @brief Constructor
     * @param env Environment reference for collision checking
     * @param sampling_radius Default radius around object for sampling
     * @param max_attempts Maximum attempts to find valid positions
     */
    GoalProposalGenerator(NAMOEnvironment& env, 
                         double sampling_radius = 2.0,
                         int max_attempts = 50);
    
    /**
     * @brief Generate goal proposals for a specific object
     * @param object_name Name of object to move
     * @param current_state Current lightweight state
     * @param num_proposals Number of proposals to generate (default 5)
     * @return Vector of valid goal positions for the object
     */
    GenericFixedVector<SE2State, 20> generate_proposals(const std::string& object_name,
                                                       const LightweightState& current_state,
                                                       int num_proposals = 5);
    
    /**
     * @brief Generate proposals for object at specific index (faster)
     * @param object_index Index of object in movable objects array
     * @param current_state Current lightweight state
     * @param num_proposals Number of proposals to generate
     * @return Vector of valid goal positions
     */
    GenericFixedVector<SE2State, 20> generate_proposals_by_index(int object_index,
                                                               const LightweightState& current_state,
                                                               int num_proposals = 5);
    
    /**
     * @brief Check if a position is valid for placing an object
     * @param object_name Name of object to place
     * @param pose Proposed pose for the object
     * @param current_state Current state (for collision checking)
     * @return True if position is valid
     */
    bool is_valid_goal_position(const std::string& object_name,
                               const SE2State& pose,
                               const LightweightState& current_state);
    
    /**
     * @brief Generate strategic proposals that help clear path to goal
     * @param object_name Object to move
     * @param current_state Current state
     * @param robot_goal Robot's target position
     * @param num_proposals Number of proposals
     * @return Strategic goal proposals
     */
    GenericFixedVector<SE2State, 20> generate_strategic_proposals(const std::string& object_name,
                                                                const LightweightState& current_state,
                                                                const SE2State& robot_goal,
                                                                int num_proposals = 5);
    
    /**
     * @brief Configuration methods
     */
    void set_sampling_radius(double radius) { sampling_radius_ = radius; }
    void set_max_attempts(int attempts) { max_attempts_ = attempts; }
    void set_clearance_threshold(double clearance) { clearance_threshold_ = clearance; }
    
    double get_sampling_radius() const { return sampling_radius_; }
    int get_max_attempts() const { return max_attempts_; }
    double get_clearance_threshold() const { return clearance_threshold_; }
    
    /**
     * @brief Statistics for monitoring performance
     */
    struct GenerationStats {
        int total_attempts = 0;
        int valid_proposals_found = 0;
        int collision_rejections = 0;
        int out_of_bounds_rejections = 0;
        double generation_time_ms = 0.0;
    };
    
    const GenerationStats& get_last_generation_stats() const { return last_stats_; }
    void reset_statistics() { last_stats_ = GenerationStats{}; }

private:
    // Environment reference
    NAMOEnvironment& env_;
    
    // Configuration
    double sampling_radius_;      // Radius around object to sample goals
    int max_attempts_;           // Maximum attempts to find valid positions
    double clearance_threshold_; // Minimum clearance from obstacles
    
    // Random number generation
    mutable std::mt19937 rng_;
    
    // Statistics tracking
    mutable GenerationStats last_stats_;
    
    // Pre-allocated workspace
    static constexpr size_t MAX_CANDIDATE_POSITIONS = 100;
    GenericFixedVector<SE2State, MAX_CANDIDATE_POSITIONS> candidate_positions_;
    
    // Core generation methods
    GenericFixedVector<SE2State, 20> generate_random_proposals(const std::string& object_name,
                                                              const LightweightState& current_state,
                                                              int num_proposals);
    
    GenericFixedVector<SE2State, 20> generate_edge_based_proposals(const std::string& object_name,
                                                                  const LightweightState& current_state,
                                                                  int num_proposals);
    
    // Sampling methods
    SE2State sample_around_object(const SE2State& object_pose, double radius) const;
    SE2State sample_in_free_space(const LightweightState& current_state) const;
    SE2State sample_away_from_robot_path(const LightweightState& current_state, 
                                        const SE2State& robot_goal) const;
    
    // Validation methods
    bool is_position_in_bounds(const SE2State& pose) const;
    bool is_position_collision_free(const std::string& object_name,
                                   const SE2State& pose,
                                   const LightweightState& current_state) const;
    bool has_sufficient_clearance(const SE2State& pose, const LightweightState& current_state) const;
    
    // Utility methods
    const ObjectInfo* get_object_info(const std::string& object_name) const;
    std::vector<double> get_environment_bounds() const;
    double calculate_distance(const SE2State& pose1, const SE2State& pose2) const;
    
    // Statistics helpers
    void update_statistics(const std::chrono::high_resolution_clock::time_point& start,
                          const std::chrono::high_resolution_clock::time_point& end,
                          int proposals_found) const;
};

} // namespace namo