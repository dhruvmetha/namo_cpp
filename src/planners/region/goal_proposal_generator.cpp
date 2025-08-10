#include "planners/region/goal_proposal_generator.hpp"
#include <chrono>
#include <cmath>
#include <algorithm>

namespace namo {

GoalProposalGenerator::GoalProposalGenerator(NAMOEnvironment& env, 
                                           double sampling_radius,
                                           int max_attempts)
    : env_(env)
    , sampling_radius_(sampling_radius)
    , max_attempts_(max_attempts)
    , clearance_threshold_(0.1)
    , rng_(std::random_device{}()) {
}

GenericFixedVector<SE2State, 20> GoalProposalGenerator::generate_proposals(
    const std::string& object_name,
    const LightweightState& current_state,
    int num_proposals) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Reset statistics
    last_stats_ = GenerationStats{};
    
    // Try different generation strategies
    GenericFixedVector<SE2State, 20> proposals;
    
    // Strategy 1: Random sampling around object (50% of proposals)
    int random_count = std::max(1, num_proposals / 2);
    auto random_proposals = generate_random_proposals(object_name, current_state, random_count);
    for (size_t i = 0; i < random_proposals.size() && proposals.size() < proposals.capacity(); ++i) {
        proposals.push_back(random_proposals[i]);
    }
    
    // Strategy 2: Edge-based proposals (remaining proposals)
    int edge_count = num_proposals - static_cast<int>(proposals.size());
    if (edge_count > 0) {
        auto edge_proposals = generate_edge_based_proposals(object_name, current_state, edge_count);
        for (size_t i = 0; i < edge_proposals.size() && proposals.size() < proposals.capacity(); ++i) {
            proposals.push_back(edge_proposals[i]);
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    update_statistics(start_time, end_time, static_cast<int>(proposals.size()));
    
    return proposals;
}

GenericFixedVector<SE2State, 20> GoalProposalGenerator::generate_proposals_by_index(
    int object_index,
    const LightweightState& current_state,
    int num_proposals) {
    
    if (object_index < 0 || object_index >= static_cast<int>(current_state.object_names.size())) {
        return GenericFixedVector<SE2State, 20>{};
    }
    
    return generate_proposals(current_state.object_names[object_index], current_state, num_proposals);
}

bool GoalProposalGenerator::is_valid_goal_position(const std::string& object_name,
                                                 const SE2State& pose,
                                                 const LightweightState& current_state) {
    return is_position_in_bounds(pose) &&
           is_position_collision_free(object_name, pose, current_state) &&
           has_sufficient_clearance(pose, current_state);
}

GenericFixedVector<SE2State, 20> GoalProposalGenerator::generate_strategic_proposals(
    const std::string& object_name,
    const LightweightState& current_state,
    const SE2State& robot_goal,
    int num_proposals) {
    
    // For now, implement as enhanced random sampling that avoids robot's path to goal
    GenericFixedVector<SE2State, 20> proposals;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    last_stats_ = GenerationStats{};
    
    int attempts = 0;
    while (static_cast<int>(proposals.size()) < num_proposals && attempts < max_attempts_) {
        SE2State candidate = sample_away_from_robot_path(current_state, robot_goal);
        
        if (is_valid_goal_position(object_name, candidate, current_state)) {
            proposals.push_back(candidate);
            last_stats_.valid_proposals_found++;
        } else {
            last_stats_.collision_rejections++;
        }
        
        attempts++;
        last_stats_.total_attempts++;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    update_statistics(start_time, end_time, static_cast<int>(proposals.size()));
    
    return proposals;
}

GenericFixedVector<SE2State, 20> GoalProposalGenerator::generate_random_proposals(
    const std::string& object_name,
    const LightweightState& current_state,
    int num_proposals) {
    
    GenericFixedVector<SE2State, 20> proposals;
    
    // Get current object position
    const SE2State* current_pose = current_state.get_object_pose(object_name);
    if (!current_pose) {
        return proposals;
    }
    
    int attempts = 0;
    while (static_cast<int>(proposals.size()) < num_proposals && attempts < max_attempts_) {
        SE2State candidate = sample_around_object(*current_pose, sampling_radius_);
        
        if (is_valid_goal_position(object_name, candidate, current_state)) {
            proposals.push_back(candidate);
            last_stats_.valid_proposals_found++;
        } else {
            last_stats_.collision_rejections++;
        }
        
        attempts++;
        last_stats_.total_attempts++;
    }
    
    return proposals;
}

GenericFixedVector<SE2State, 20> GoalProposalGenerator::generate_edge_based_proposals(
    const std::string& object_name,
    const LightweightState& current_state,
    int num_proposals) {
    
    GenericFixedVector<SE2State, 20> proposals;
    
    // Get object info for size
    const ObjectInfo* obj_info = get_object_info(object_name);
    if (!obj_info) {
        return proposals;
    }
    
    // Get current object position
    const SE2State* current_pose = current_state.get_object_pose(object_name);
    if (!current_pose) {
        return proposals;
    }
    
    // Generate proposals at different angles around the object
    int attempts = 0;
    double angle_step = 2.0 * M_PI / std::max(4, num_proposals * 2);  // More angles than needed
    
    for (double angle = 0.0; angle < 2.0 * M_PI && static_cast<int>(proposals.size()) < num_proposals; angle += angle_step) {
        if (attempts >= max_attempts_) break;
        
        // Sample at edge of object's influence radius
        double radius = std::max(obj_info->size[0], obj_info->size[1]) * 1.5 + sampling_radius_;
        
        SE2State candidate;
        candidate.x = current_pose->x + radius * std::cos(angle);
        candidate.y = current_pose->y + radius * std::sin(angle);
        candidate.theta = angle;  // Orient object toward its old position
        
        if (is_valid_goal_position(object_name, candidate, current_state)) {
            proposals.push_back(candidate);
            last_stats_.valid_proposals_found++;
        } else {
            last_stats_.collision_rejections++;
        }
        
        attempts++;
        last_stats_.total_attempts++;
    }
    
    return proposals;
}

SE2State GoalProposalGenerator::sample_around_object(const SE2State& object_pose, double radius) const {
    std::uniform_real_distribution<double> angle_dist(0.0, 2.0 * M_PI);
    std::uniform_real_distribution<double> radius_dist(0.0, radius);
    std::uniform_real_distribution<double> theta_dist(-M_PI, M_PI);
    
    double angle = angle_dist(rng_);
    double r = radius_dist(rng_);
    
    SE2State sample;
    sample.x = object_pose.x + r * std::cos(angle);
    sample.y = object_pose.y + r * std::sin(angle);
    sample.theta = theta_dist(rng_);
    
    return sample;
}

SE2State GoalProposalGenerator::sample_in_free_space(const LightweightState& current_state) const {
    auto bounds = get_environment_bounds();
    
    std::uniform_real_distribution<double> x_dist(bounds[0], bounds[1]);
    std::uniform_real_distribution<double> y_dist(bounds[2], bounds[3]);
    std::uniform_real_distribution<double> theta_dist(-M_PI, M_PI);
    
    SE2State sample;
    sample.x = x_dist(rng_);
    sample.y = y_dist(rng_);
    sample.theta = theta_dist(rng_);
    
    return sample;
}

SE2State GoalProposalGenerator::sample_away_from_robot_path(const LightweightState& current_state,
                                                          const SE2State& robot_goal) const {
    // Sample in free space, but prefer positions away from robot's direct path to goal
    SE2State sample = sample_in_free_space(current_state);
    
    // Calculate distance from robot's direct path to goal
    double robot_x = current_state.robot_pose.x;
    double robot_y = current_state.robot_pose.y;
    double goal_x = robot_goal.x;
    double goal_y = robot_goal.y;
    
    // If sample is too close to robot's path, try to move it away
    double line_length_sq = (goal_x - robot_x) * (goal_x - robot_x) + (goal_y - robot_y) * (goal_y - robot_y);
    if (line_length_sq > 1e-6) {
        // Project sample onto robot's path line
        double t = ((sample.x - robot_x) * (goal_x - robot_x) + (sample.y - robot_y) * (goal_y - robot_y)) / line_length_sq;
        t = std::max(0.0, std::min(1.0, t));
        
        double closest_x = robot_x + t * (goal_x - robot_x);
        double closest_y = robot_y + t * (goal_y - robot_y);
        
        double dist_to_path = std::sqrt((sample.x - closest_x) * (sample.x - closest_x) + 
                                       (sample.y - closest_y) * (sample.y - closest_y));
        
        // If too close to path, move sample away
        double min_clearance = 1.0;  // 1 meter clearance from robot path
        if (dist_to_path < min_clearance) {
            double push_distance = min_clearance - dist_to_path + 0.5;
            
            // Direction perpendicular to robot's path
            double path_dx = goal_x - robot_x;
            double path_dy = goal_y - robot_y;
            double path_length = std::sqrt(path_dx * path_dx + path_dy * path_dy);
            
            if (path_length > 1e-6) {
                double perp_x = -path_dy / path_length;
                double perp_y = path_dx / path_length;
                
                sample.x += perp_x * push_distance;
                sample.y += perp_y * push_distance;
            }
        }
    }
    
    return sample;
}

bool GoalProposalGenerator::is_position_in_bounds(const SE2State& pose) const {
    auto bounds = get_environment_bounds();
    return pose.x >= bounds[0] && pose.x <= bounds[1] &&
           pose.y >= bounds[2] && pose.y <= bounds[3];
}

bool GoalProposalGenerator::is_position_collision_free(const std::string& object_name,
                                                     const SE2State& pose,
                                                     const LightweightState& current_state) const {
    // Get object info for size
    const ObjectInfo* obj_info = get_object_info(object_name);
    if (!obj_info) {
        return false;
    }
    
    // Check collision with other movable objects
    for (size_t i = 0; i < current_state.object_names.size(); ++i) {
        if (current_state.object_names[i] == object_name) {
            continue;  // Don't check collision with self
        }
        
        const SE2State& other_pose = current_state.movable_object_poses[i];
        const ObjectInfo* other_info = get_object_info(current_state.object_names[i]);
        
        if (other_info) {
            // Simple bounding box collision check
            double min_distance = (obj_info->size[0] + other_info->size[0]) * 0.6 +
                                 (obj_info->size[1] + other_info->size[1]) * 0.6;
            
            double distance = calculate_distance(pose, other_pose);
            if (distance < min_distance) {
                return false;
            }
        }
    }
    
    // Check collision with static objects (simplified)
    const auto& static_objects = env_.get_static_objects();
    size_t num_static = env_.get_num_static();
    
    for (size_t i = 0; i < num_static; ++i) {
        const ObjectInfo& static_obj = static_objects[i];
        
        // Simple distance check
        double min_distance = (obj_info->size[0] + static_obj.size[0]) * 0.6 +
                             (obj_info->size[1] + static_obj.size[1]) * 0.6;
        
        SE2State static_pose;
        static_pose.x = static_obj.position[0];
        static_pose.y = static_obj.position[1];
        static_pose.theta = 0.0;  // Assume static objects don't rotate
        
        double distance = calculate_distance(pose, static_pose);
        if (distance < min_distance) {
            return false;
        }
    }
    
    return true;
}

bool GoalProposalGenerator::has_sufficient_clearance(const SE2State& pose, 
                                                   const LightweightState& current_state) const {
    // Check clearance from robot
    double robot_clearance = calculate_distance(pose, current_state.robot_pose);
    if (robot_clearance < clearance_threshold_) {
        return false;
    }
    
    return true;
}

const ObjectInfo* GoalProposalGenerator::get_object_info(const std::string& object_name) const {
    return env_.get_object_info(object_name);
}

std::vector<double> GoalProposalGenerator::get_environment_bounds() const {
    return env_.get_environment_bounds();
}

double GoalProposalGenerator::calculate_distance(const SE2State& pose1, const SE2State& pose2) const {
    return std::sqrt((pose2.x - pose1.x) * (pose2.x - pose1.x) + 
                     (pose2.y - pose1.y) * (pose2.y - pose1.y));
}

void GoalProposalGenerator::update_statistics(const std::chrono::high_resolution_clock::time_point& start,
                                             const std::chrono::high_resolution_clock::time_point& end,
                                             int proposals_found) const {
    last_stats_.generation_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    last_stats_.valid_proposals_found = proposals_found;
}

} // namespace namo