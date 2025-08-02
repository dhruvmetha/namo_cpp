#pragma once

#include <array>
#include <string>
#include <vector>

namespace namo {

/**
 * @brief Structure representing a loaded motion primitive
 * 
 * Contains SE(2) displacement information from universal primitive database
 */
struct LoadedPrimitive {
    double delta_x;      // Position displacement in x
    double delta_y;      // Position displacement in y  
    double delta_theta;  // Orientation displacement (radians)
    int edge_idx;        // Edge index (0-11)
    int push_steps;      // Number of push steps (1-10)
    
    LoadedPrimitive() : delta_x(0.0), delta_y(0.0), delta_theta(0.0), edge_idx(-1), push_steps(0) {}
    
    LoadedPrimitive(double dx, double dy, double dtheta, int edge, int steps)
        : delta_x(dx), delta_y(dy), delta_theta(dtheta), edge_idx(edge), push_steps(steps) {}
};

/**
 * @brief Fast primitive loader with pre-allocated storage
 * 
 * Loads universal motion primitives from binary database for zero-allocation runtime.
 * Follows old implementation approach - no scaling, primitives used as pure displacement vectors.
 */
class PrimitiveLoader {
private:
    static constexpr size_t MAX_PRIMITIVES = 120;  // 12 edges × 10 step variants
    static constexpr size_t MAX_EDGES = 12;
    static constexpr size_t MAX_STEPS = 10;
    
    std::array<LoadedPrimitive, MAX_PRIMITIVES> primitives_;
    std::array<std::array<int, MAX_STEPS>, MAX_EDGES> lookup_table_;  // [edge][step-1] -> primitive_index
    size_t loaded_count_;
    bool is_loaded_;
    
public:
    PrimitiveLoader();
    
    /**
     * @brief Load primitives from binary file
     * 
     * @param filepath Path to binary primitive database
     * @return bool True if loading succeeded
     */
    bool load_primitives(const std::string& filepath);
    
    /**
     * @brief Get primitive by edge index and push steps
     * 
     * @param edge_idx Edge index (0-11)
     * @param push_steps Number of push steps (1-10)
     * @return const LoadedPrimitive& Reference to primitive
     */
    const LoadedPrimitive& get_primitive(int edge_idx, int push_steps) const;
    
    /**
     * @brief Check if primitives are loaded
     * 
     * @return bool True if primitives are loaded
     */
    bool is_loaded() const { return is_loaded_; }
    
    /**
     * @brief Get number of loaded primitives
     * 
     * @return size_t Number of primitives
     */
    size_t size() const { return loaded_count_; }
    
    /**
     * @brief Get all primitives (for iteration)
     * 
     * @return const std::array<LoadedPrimitive, MAX_PRIMITIVES>& Reference to primitive array
     */
    const std::array<LoadedPrimitive, MAX_PRIMITIVES>& get_all_primitives() const { 
        return primitives_; 
    }
    
    /**
     * @brief Get valid primitive indices for a given edge
     * 
     * @param edge_idx Edge index (0-11)
     * @return std::vector<int> Valid push step values for this edge
     */
    std::vector<int> get_valid_steps_for_edge(int edge_idx) const;
};

} // namespace namo