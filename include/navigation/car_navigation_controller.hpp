#pragma once

#include "navigation/push_path_follower.hpp"

#include <array>
#include <optional>
#include <string>
#include <vector>

namespace namo {

// C++ port of robot_control's NavigationController
// (robot_control/src/robot_control/controller/navigation.py), the state machine
// IDLE -> PRE_ROTATING -> FOLLOWING -> POST_ROTATING -> FINISHED that drives the real car
// between pushes. Path following is delegated to PushPathFollower with align_mode on, which is
// the port of FollowPathController.
//
// Units follow the Python: headings in degrees, times in seconds. Lengths are in whatever unit
// car_size and wheel_base are given in, and Params::cm says how long one centimetre is in that
// unit, for the watchdog's absolute 0.5 cm threshold. The parity test runs it in cm (cm = 1.0)
// against the Python; the simulator runs it in metres (cm = 0.01).
//
// Not ported: RETREATING (navigation.py:477-533). When the no-progress watchdog fires, the Python
// backs out to a free cell and then reports FAILED; this goes to FAILED straight away. Also not
// ported: the standalone ROTATING state (rotate_to), which navigate_to never enters.
class CarNavigationController {
public:
    using Point = std::array<double, 2>;

    enum class State { IDLE, PRE_ROTATING, FOLLOWING, POST_ROTATING, FINISHED, FAILED };

    struct Params {
        double car_width = 7.0;
        double car_height = 7.0;
        double wheel_base = 7.0;
        double cm = 1.0;
        // controller.yaml navigation: block.
        double max_speed = 0.4;
        double pre_rotation_skip_angle = 45.0;
        double goal_tolerance_ratio = 0.15;
        double rotation_tolerance_deg = 4.0;
        double rotation_stable_time = 0.5;
        double rotation_speed_max = 0.25;
        double rotation_speed_min = 0.15;
        double wheel_deadband = 0.05;
    };

    struct Pose {
        double x = 0.0;
        double y = 0.0;
        double theta_deg = 0.0;
    };

    struct Command {
        double left_speed = 0.0;
        double right_speed = 0.0;
    };

    explicit CarNavigationController(const Params& params);

    // navigation.py:368-475 with the caller supplying the path (the `path=` argument there). The
    // Python's goal_x/goal_y only feed a log line, so they are left out; the path ends at the goal.
    bool navigate_to(std::optional<double> goal_theta_deg, const Point& current_pos,
                     const std::vector<Point>& path);
    Command step(const Pose& pose, double timestamp_s);

    State state() const { return state_; }
    bool is_done() const { return state_ == State::FINISHED || state_ == State::FAILED; }
    const std::string& failure_cause() const { return failure_cause_; }
    const PushPathFollower& follower() const { return follower_; }
    static const char* state_name(State state);

private:
    // navigation.py:139-189, _NoProgressWatchdog.
    struct Watchdog {
        bool has_anchor = false;
        Pose anchor;
        double anchor_time = 0.0;
        bool commanded_to_move = false;
        void reset() { has_anchor = false; anchor_time = 0.0; commanded_to_move = false; }
    };

    std::optional<Command> handle_rotation(double heading_error_deg, double now);
    Command rotation_command(double heading_error_deg) const;
    // Returns "blocked" or "under_commanded" once the window passes without progress, else "".
    const char* update_watchdog(const Pose& pose, const Command& command, double now);

    static double wrap_to_180(double deg);

    Params params_;
    double car_size_ = 0.0;
    PushPathFollower follower_;
    State state_ = State::IDLE;
    Watchdog watchdog_;
    std::string failure_cause_;
    std::optional<double> rotation_stable_start_;
    std::optional<double> target_theta_;
    std::optional<double> pre_rotation_target_;
};

}  // namespace namo
