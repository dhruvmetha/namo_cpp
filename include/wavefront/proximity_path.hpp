#pragma once

#include <array>
#include <vector>

namespace namo {

// Port of the real robot's path extraction: robot_control
// planner/wavefront_path_planner.py:77-179 over utils/wavefront.py WavefrontPlanner. Free cells
// within proximity_distance_m of an inflated obstacle cost up to 1 + proximity_weight times more
// to enter (wavefront.py:157-227), Dijkstra runs from the goal over 8-connected moves
// (wavefront.py:583-623), and the path follows the distance gradient down from the start
// (wavefront.py:625-674). A blocked start gets the same trapped-start recovery first
// (wavefront.py:240-323).
//
// The grid is NAMO's inflated occupancy grid, indexed [x][y], -1 = obstacle. The Python builds its
// own from marker boxes; this takes the grid the BFS reachability already uses, so both agree on
// which cells are free. Waypoints are cell centres, as in the Python (NAMO's BFS path uses cell
// corners). The Python's NumPy 2 arithmetic runs in float32 (Python floats are weak scalars next
// to the float32 cost grid), so this does too, in a file built without -ffast-math so the sums
// round the same way.
//
// Returns the start cell through the goal cell, or an empty path if either is blocked or they are
// not connected.
std::vector<std::array<double, 2>> plan_proximity_path(
    std::vector<std::vector<int>> grid,
    double origin_x,
    double origin_y,
    double resolution,
    const std::array<double, 2>& start,
    const std::array<double, 2>& goal,
    double proximity_distance_m,
    double proximity_weight);

}  // namespace namo
