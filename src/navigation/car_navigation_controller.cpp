#include "navigation/car_navigation_controller.hpp"

#include <algorithm>
#include <cmath>
#include <tuple>

namespace namo {

namespace {

// navigation.py:109-125. Position in cm, scaled by Params::cm.
constexpr double kNoProgressPositionCm = 0.5;
constexpr double kNoProgressHeadingDeg = 3.0;
constexpr double kNoProgressWindowS = 3.0;
constexpr double kPhysicalMoveCommand = 0.3;

PushPathFollower::Params follower_params(const CarNavigationController::Params& p) {
    // navigation.py:283-294: lookahead 1.0 x car_size, goal tolerance ratio x car_size,
    // FollowPathController defaults for the rest (follow_path.py:110-118, WHEEL_DEADBAND 0.05).
    const double car_size = std::max(std::abs(p.car_width), std::abs(p.car_height));
    PushPathFollower::Params fp;
    fp.robot_width_m = p.car_width;
    fp.robot_height_m = p.car_height;
    fp.wheel_base_m = p.wheel_base;
    fp.lookahead_distance_m = 1.0 * car_size;
    fp.goal_tolerance_m = p.goal_tolerance_ratio * car_size;
    fp.max_speed = p.max_speed;
    fp.max_point_gap_ratio = 0.01;
    fp.no_skip_ratio = 0.5;
    fp.wheel_deadband = 0.05;
    fp.align_mode = true;
    return fp;
}

}  // namespace

CarNavigationController::CarNavigationController(const Params& params)
    : params_(params),
      car_size_(std::max(std::abs(params.car_width), std::abs(params.car_height))),
      follower_(follower_params(params)) {}

bool CarNavigationController::navigate_to(std::optional<double> goal_theta_deg,
                                          const Point& current_pos,
                                          const std::vector<Point>& raw_path) {
    // navigation.py:90-102, _filter_duplicate_points with min_dist 0.1 x car_size.
    std::vector<Point> path;
    if (!raw_path.empty()) {
        path.push_back(raw_path.front());
        for (size_t i = 1; i < raw_path.size(); ++i) {
            const double dx = raw_path[i][0] - path.back()[0];
            const double dy = raw_path[i][1] - path.back()[1];
            if (std::hypot(dx, dy) > 0.1 * car_size_) {
                path.push_back(raw_path[i]);
            }
        }
    }
    if (path.size() < 2) {
        state_ = State::FAILED;
        return false;
    }

    target_theta_ = goal_theta_deg;

    const Point* first_target = nullptr;
    for (const auto& pt : path) {
        if (std::hypot(pt[0] - current_pos[0], pt[1] - current_pos[1]) > 0.5 * car_size_) {
            first_target = &pt;
            break;
        }
    }
    if (first_target == nullptr) {
        state_ = target_theta_ ? State::POST_ROTATING : State::FINISHED;
        return true;
    }

    const double dx = (*first_target)[0] - current_pos[0];
    const double dy = (*first_target)[1] - current_pos[1];
    follower_.set_path(path);
    watchdog_.reset();
    failure_cause_.clear();
    pre_rotation_target_ = std::atan2(dy, dx) * (180.0 / M_PI);
    state_ = State::PRE_ROTATING;
    return true;
}

CarNavigationController::Command CarNavigationController::step(const Pose& pose, double now) {
    if (state_ == State::IDLE || state_ == State::FINISHED || state_ == State::FAILED) {
        return {};
    }

    if (state_ == State::PRE_ROTATING) {
        const double heading_error = wrap_to_180(*pre_rotation_target_ - pose.theta_deg);
        if (std::abs(heading_error) < params_.pre_rotation_skip_angle) {
            rotation_stable_start_.reset();
            state_ = State::FOLLOWING;
        } else {
            const auto command = handle_rotation(heading_error, now);
            if (!command) {
                rotation_stable_start_.reset();
                state_ = State::FOLLOWING;
            } else {
                return *command;
            }
        }
    }

    if (state_ == State::FOLLOWING) {
        // navigation.py:661-683. A follower that finished last tick moves us on, but this tick
        // still returns the follower's own command, as in the Python.
        if (follower_.is_done()) {
            if (target_theta_) {
                rotation_stable_start_.reset();
                state_ = State::POST_ROTATING;
            } else {
                state_ = State::FINISHED;
                return {};
            }
        }
        PushPathFollower::Pose follower_pose;
        follower_pose.x_m = pose.x;
        follower_pose.y_m = pose.y;
        follower_pose.theta_rad = pose.theta_deg * (M_PI / 180.0);
        const auto out = follower_.step(follower_pose, now);
        const Command command{out.left_speed, out.right_speed};
        const char* cause = update_watchdog(pose, command, now);
        if (*cause != '\0') {
            failure_cause_ = cause;
            state_ = State::FAILED;
            return {};
        }
        return command;
    }

    if (state_ == State::POST_ROTATING) {
        if (!target_theta_) {
            state_ = State::FINISHED;
            return {};
        }
        const double heading_error = wrap_to_180(*target_theta_ - pose.theta_deg);
        const auto command = handle_rotation(heading_error, now);
        if (!command) {
            rotation_stable_start_.reset();
            state_ = State::FINISHED;
            return {};
        }
        return *command;
    }

    return {};
}

std::optional<CarNavigationController::Command> CarNavigationController::handle_rotation(
    double heading_error_deg, double now) {
    // navigation.py:706-735. The Python times the stable hold with time.time(); this uses the
    // timestamp it is handed, which in the simulator is sim time.
    if (std::abs(heading_error_deg) <= params_.rotation_tolerance_deg) {
        if (!rotation_stable_start_) {
            rotation_stable_start_ = now;
            return Command{};
        }
        if (now - *rotation_stable_start_ >= params_.rotation_stable_time) {
            return std::nullopt;
        }
        return Command{};
    }
    rotation_stable_start_.reset();
    return rotation_command(heading_error_deg);
}

CarNavigationController::Command CarNavigationController::rotation_command(
    double heading_error_deg) const {
    // navigation.py:737-783.
    const double abs_error = std::abs(heading_error_deg);
    const double speed_max = params_.rotation_speed_max;
    const double speed_min = params_.rotation_speed_min;
    double speed = speed_min;
    if (abs_error > 45.0) {
        speed = speed_max;
    } else if (abs_error > 15.0) {
        const double t = (abs_error - 15.0) / 30.0;
        speed = speed_min + t * (speed_max - speed_min);
    }
    double left = (heading_error_deg > 0.0) ? -speed : speed;
    double right = (heading_error_deg > 0.0) ? speed : -speed;
    std::tie(left, right) = PushPathFollower::deadband_scale(left, right, params_.wheel_deadband);
    return {PushPathFollower::clamp(left, -1.0, 1.0), PushPathFollower::clamp(right, -1.0, 1.0)};
}

const char* CarNavigationController::update_watchdog(const Pose& pose, const Command& command,
                                                     double now) {
    if (!watchdog_.has_anchor) {
        watchdog_.has_anchor = true;
        watchdog_.anchor = pose;
        watchdog_.anchor_time = now;
        watchdog_.commanded_to_move = false;
        return "";
    }
    if (std::max(std::abs(command.left_speed), std::abs(command.right_speed)) >=
        kPhysicalMoveCommand) {
        watchdog_.commanded_to_move = true;
    }
    const double moved = std::hypot(pose.x - watchdog_.anchor.x, pose.y - watchdog_.anchor.y);
    const double turned = std::abs(wrap_to_180(pose.theta_deg - watchdog_.anchor.theta_deg));
    if (moved >= kNoProgressPositionCm * params_.cm || turned >= kNoProgressHeadingDeg) {
        watchdog_.anchor = pose;
        watchdog_.anchor_time = now;
        watchdog_.commanded_to_move = false;
        return "";
    }
    if (now - watchdog_.anchor_time < kNoProgressWindowS) {
        return "";
    }
    return watchdog_.commanded_to_move ? "blocked" : "under_commanded";
}

double CarNavigationController::wrap_to_180(double deg) {
    while (deg >= 180.0) {
        deg -= 360.0;
    }
    while (deg < -180.0) {
        deg += 360.0;
    }
    return deg;
}

const char* CarNavigationController::state_name(State state) {
    switch (state) {
    case State::IDLE:
        return "IDLE";
    case State::PRE_ROTATING:
        return "PRE_ROTATING";
    case State::FOLLOWING:
        return "FOLLOWING";
    case State::POST_ROTATING:
        return "POST_ROTATING";
    case State::FINISHED:
        return "FINISHED";
    case State::FAILED:
        return "FAILED";
    }
    return "UNKNOWN";
}

}  // namespace namo
