#include "navigation/push_path_follower.hpp"

#include <algorithm>
#include <cmath>
#include <tuple>

namespace namo {

namespace {

constexpr double kAlignEnterDeg = 60.0;
constexpr double kAlignExitDeg = 45.0;

}  // namespace

PushPathFollower::PushPathFollower(const Params& params)
    : params_(params) {
    car_size_m_ = std::max(std::abs(params_.robot_width_m), std::abs(params_.robot_height_m));
    if (car_size_m_ <= 0.0) {
        car_size_m_ = 0.07;
    }

    lookahead_distance_m_ =
        (params_.lookahead_distance_m > 0.0) ? params_.lookahead_distance_m : (0.5 * car_size_m_);
    goal_tolerance_m_ =
        (params_.goal_tolerance_m > 0.0) ? params_.goal_tolerance_m : (0.2 * car_size_m_);
    max_speed_ = clamp(params_.max_speed, 0.0, 1.0);
}

void PushPathFollower::set_path(const std::vector<Point>& path) {
    path_raw_ = path;
    const double max_step =
        std::max(1e-6, std::max(1e-4, params_.max_point_gap_ratio) * car_size_m_);
    path_ = resample_path(path_raw_, max_step);
    build_prefix_s();

    path_index_ = 0;
    align_active_ = false;
    has_prev_cte_ = false;
    prev_cte_ = 0.0;
    prev_cte_time_ = 0.0;
    cte_dot_filt_ = 0.0;

    mode_ = path_.empty() ? Mode::IDLE : Mode::TRACK;
}

void PushPathFollower::clear_path() {
    path_raw_.clear();
    path_.clear();
    path_s_.clear();
    path_index_ = 0;
    align_active_ = false;
    has_prev_cte_ = false;
    prev_cte_ = 0.0;
    prev_cte_time_ = 0.0;
    cte_dot_filt_ = 0.0;
    mode_ = Mode::IDLE;
}

void PushPathFollower::reset() {
    clear_path();
    set_speed(params_.max_speed);
}

void PushPathFollower::set_speed(double speed) {
    max_speed_ = clamp(speed, 0.0, 1.0);
}

PushPathFollower::StepOutput PushPathFollower::step(const Pose& pose, double timestamp_s) {
    StepOutput output;
    output.mode = Mode::IDLE;

    if (path_.empty()) {
        mode_ = Mode::IDLE;
        return output;
    }

    const Point final_point = path_.back();
    const double dist_to_goal = std::hypot(pose.x_m - final_point[0], pose.y_m - final_point[1]);
    if (dist_to_goal < goal_tolerance_m_) {
        mode_ = Mode::FINISHED;
        output.mode = mode_;
        output.done = true;
        output.target_point = final_point;
        output.path_index = path_index_;
        return output;
    }

    const Point target_point = find_target_point(pose.x_m, pose.y_m);
    const ReferencePoint ref = find_closest_reference({pose.x_m, pose.y_m});
    path_index_ = std::max(path_index_, ref.segment_index);

    const CurvatureResult curvature_pp = pure_pursuit_curvature(
        pose.x_m, pose.y_m, pose.theta_rad, target_point);

    const double heading_deg = std::abs(curvature_pp.heading_error_rad) * 180.0 / M_PI;
    if (align_active_) {
        if (heading_deg <= kAlignExitDeg) {
            align_active_ = false;
        }
    } else if (heading_deg >= kAlignEnterDeg) {
        align_active_ = true;
    }

    output.target_point = target_point;
    output.path_index = path_index_;
    output.heading_error_rad = curvature_pp.heading_error_rad;
    output.cte = ref.cte;

    if (align_active_) {
        const double w = 0.6 * max_speed_;
        double left_speed = (curvature_pp.heading_error_rad > 0.0) ? -w : w;
        double right_speed = (curvature_pp.heading_error_rad > 0.0) ? w : -w;
        std::tie(left_speed, right_speed) = enforce_deadband_scale(left_speed, right_speed);

        mode_ = Mode::ALIGN;
        output.left_speed = clamp(left_speed, -1.0, 1.0);
        output.right_speed = clamp(right_speed, -1.0, 1.0);
        output.mode = mode_;
        return output;
    }

    const double dist_to_path = std::abs(ref.cte);
    const double cte_pd_enable_dist = 2.5 * car_size_m_;
    const double cte_clip = 1.0 * car_size_m_;
    const double cte_deadband = 0.10 * car_size_m_;
    const double curv_pd_max = 2.0 / std::max(car_size_m_, 1e-9);
    const double curvature_max = 3.5 / std::max(car_size_m_, 1e-9);
    const double rotate_in_place_rad = 110.0 * M_PI / 180.0;

    double turn_factor = std::cos(std::abs(curvature_pp.heading_error_rad));
    turn_factor = std::max(min_turn_factor_, turn_factor);
    double base_speed = max_speed_ * turn_factor;

    const bool use_pd = (dist_to_path <= cte_pd_enable_dist);

    double cte_used = ref.cte;
    if (std::abs(cte_used) < cte_deadband) {
        cte_used = 0.0;
    }
    cte_used = clamp(cte_used, -cte_clip, cte_clip);

    double cte_pd_curv = 0.0;
    if (!use_pd || !std::isfinite(timestamp_s)) {
        has_prev_cte_ = false;
        prev_cte_ = 0.0;
        prev_cte_time_ = 0.0;
        cte_dot_filt_ = 0.0;
    } else {
        double cte_dot = 0.0;
        if (has_prev_cte_) {
            const double dt = timestamp_s - prev_cte_time_;
            if (dt > 1e-6) {
                cte_dot = (cte_used - prev_cte_) / dt;
            }
        }

        cte_dot_filt_ =
            ((1.0 - cte_dot_alpha_) * cte_dot_filt_) + (cte_dot_alpha_ * cte_dot);
        prev_cte_ = cte_used;
        prev_cte_time_ = timestamp_s;
        has_prev_cte_ = true;

        const double kp = 1.2;
        const double kd = 0.35;
        const double cte_n = cte_used / std::max(car_size_m_, 1e-9);
        const double cte_dot_n = cte_dot_filt_ / std::max(car_size_m_, 1e-9);
        cte_pd_curv =
            (kp * cte_n + kd * cte_dot_n) / std::max(car_size_m_, 1e-9);
        cte_pd_curv = clamp(cte_pd_curv, -curv_pd_max, curv_pd_max);
    }

    double curvature_cmd = curvature_pp.curvature + cte_pd_curv;
    curvature_cmd = clamp(curvature_cmd, -curvature_max, curvature_max);
    output.curvature_command = curvature_cmd;

    double curv_slow = 1.0 - 0.65 * (std::abs(curvature_cmd) / std::max(curvature_max, 1e-9));
    curv_slow = clamp(curv_slow, 0.35, 1.0);
    base_speed *= curv_slow;

    if (!use_pd && (std::abs(curvature_pp.heading_error_rad) > rotate_in_place_rad)) {
        const double w = 0.6 * max_speed_;
        double left_speed = (curvature_pp.heading_error_rad > 0.0) ? -w : w;
        double right_speed = (curvature_pp.heading_error_rad > 0.0) ? w : -w;
        std::tie(left_speed, right_speed) = enforce_deadband_scale(left_speed, right_speed);

        mode_ = Mode::ROTATE_IN_PLACE;
        output.left_speed = clamp(left_speed, -1.0, 1.0);
        output.right_speed = clamp(right_speed, -1.0, 1.0);
        output.mode = mode_;
        return output;
    }

    const double diff = curvature_cmd * params_.wheel_base_m / 2.0;
    double left_speed = base_speed * (1.0 - diff);
    double right_speed = base_speed * (1.0 + diff);

    const double max_wheel = std::max(std::abs(left_speed), std::abs(right_speed));
    if (max_wheel > max_speed_ && max_speed_ > 1e-9) {
        const double scale = max_speed_ / max_wheel;
        left_speed *= scale;
        right_speed *= scale;
    }

    std::tie(left_speed, right_speed) = enforce_deadband_scale(left_speed, right_speed);

    output.left_speed = clamp(left_speed, -1.0, 1.0);
    output.right_speed = clamp(right_speed, -1.0, 1.0);
    mode_ = use_pd ? Mode::TRACK : Mode::ACQUIRE;
    output.mode = mode_;
    return output;
}

const char* PushPathFollower::mode_name(Mode mode) {
    switch (mode) {
    case Mode::IDLE:
        return "IDLE";
    case Mode::ALIGN:
        return "ALIGN";
    case Mode::ROTATE_IN_PLACE:
        return "ROTATE_IN_PLACE";
    case Mode::ACQUIRE:
        return "ACQUIRE";
    case Mode::TRACK:
        return "TRACK";
    case Mode::FINISHED:
        return "FINISHED";
    }
    return "UNKNOWN";
}

std::vector<PushPathFollower::Point> PushPathFollower::resample_path(
    const std::vector<Point>& path,
    double max_step) const {
    if (path.size() <= 1) {
        return path;
    }

    std::vector<Point> out;
    out.reserve(path.size());

    for (size_t i = 0; i + 1 < path.size(); ++i) {
        const Point& p0 = path[i];
        const Point& p1 = path[i + 1];
        const double dx = p1[0] - p0[0];
        const double dy = p1[1] - p0[1];
        const double dist = std::hypot(dx, dy);
        if (dist < 1e-12) {
            continue;
        }

        const int num_segments = std::max(1, static_cast<int>(std::ceil(dist / max_step)));
        if (out.empty()) {
            out.push_back(p0);
        }
        for (int k = 1; k <= num_segments; ++k) {
            const double t = static_cast<double>(k) / static_cast<double>(num_segments);
            out.push_back({p0[0] + t * dx, p0[1] + t * dy});
        }
    }

    if (out.empty() && !path.empty()) {
        out.push_back(path.front());
    }
    return out;
}

void PushPathFollower::build_prefix_s() {
    path_s_.clear();
    if (path_.empty()) {
        return;
    }

    path_s_.reserve(path_.size());
    path_s_.push_back(0.0);
    for (size_t i = 0; i + 1 < path_.size(); ++i) {
        const double dx = path_[i + 1][0] - path_[i][0];
        const double dy = path_[i + 1][1] - path_[i][1];
        path_s_.push_back(path_s_.back() + std::hypot(dx, dy));
    }
}

int PushPathFollower::max_reachable_index(int start_idx) const {
    if (path_s_.empty()) {
        return start_idx;
    }

    start_idx = static_cast<int>(clamp(
        static_cast<double>(start_idx), 0.0, static_cast<double>(path_s_.size() - 1)));
    const double s0 = path_s_[static_cast<size_t>(start_idx)];
    const double s_max = s0 + std::max(0.1, params_.no_skip_ratio) * car_size_m_;

    int j = start_idx;
    while ((j + 1) < static_cast<int>(path_s_.size()) &&
           path_s_[static_cast<size_t>(j + 1)] <= s_max) {
        ++j;
    }
    return j;
}

PushPathFollower::Point PushPathFollower::find_target_point(double robot_x, double robot_y) {
    const int start = path_index_;
    const int max_idx = max_reachable_index(start);

    int best_idx = start;
    double best_dist = std::numeric_limits<double>::infinity();
    for (int i = start; i <= max_idx; ++i) {
        const Point& point = path_[static_cast<size_t>(i)];
        const double dist = std::hypot(robot_x - point[0], robot_y - point[1]);
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = i;
        }
    }
    path_index_ = std::max(path_index_, best_idx);

    const int gated_start = path_index_;
    const int gated_max_idx = max_reachable_index(gated_start);
    for (int i = gated_start; i <= gated_max_idx; ++i) {
        const Point& point = path_[static_cast<size_t>(i)];
        const double dist = std::hypot(robot_x - point[0], robot_y - point[1]);
        if (dist >= lookahead_distance_m_) {
            path_index_ = i;
            return point;
        }
    }

    return path_[static_cast<size_t>(std::min(gated_max_idx, static_cast<int>(path_.size() - 1)))];
}

PushPathFollower::CurvatureResult PushPathFollower::pure_pursuit_curvature(
    double robot_x,
    double robot_y,
    double robot_theta_rad,
    const Point& target) const {
    const double dx = target[0] - robot_x;
    const double dy = target[1] - robot_y;
    const double distance = std::hypot(dx, dy);
    if (distance < 1e-9) {
        return {};
    }

    const double angle_to_target = std::atan2(dy, dx);
    const double heading_error_rad = wrap_to_pi(angle_to_target - robot_theta_rad);
    const double lookahead = std::max(distance, lookahead_distance_m_);
    CurvatureResult result;
    result.curvature = 2.0 * std::sin(heading_error_rad) / lookahead;
    result.heading_error_rad = heading_error_rad;
    result.distance_to_target = distance;
    return result;
}

PushPathFollower::ReferencePoint PushPathFollower::find_closest_reference(const Point& position) {
    ReferencePoint result;
    const size_t n = path_.size();
    if (n == 0) {
        return result;
    }
    if (n == 1) {
        const double dx = position[0] - path_[0][0];
        const double dy = position[1] - path_[0][1];
        result.projection = path_[0];
        result.cte = std::hypot(dx, dy);
        result.segment_index = 0;
        return result;
    }

    const int start_i = std::min(path_index_, static_cast<int>(n - 2));
    const int end_i = std::min(max_reachable_index(start_i), static_cast<int>(n - 1));
    const int seg_end = std::max(start_i, std::min(end_i - 1, static_cast<int>(n - 2)));

    double best_dist2 = std::numeric_limits<double>::infinity();
    result.projection = path_[static_cast<size_t>(start_i)];
    result.segment_index = start_i;

    for (int i = start_i; i <= seg_end; ++i) {
        const Point& a = path_[static_cast<size_t>(i)];
        const Point& b = path_[static_cast<size_t>(i + 1)];
        const auto [projection, _t] = project_point_to_segment(position, a, b);
        const double dx = position[0] - projection[0];
        const double dy = position[1] - projection[1];
        const double dist2 = dx * dx + dy * dy;

        if (dist2 < best_dist2) {
            best_dist2 = dist2;
            result.projection = projection;
            result.segment_index = i;

            const double seg_dx = b[0] - a[0];
            const double seg_dy = b[1] - a[1];
            const double theta_ref = std::atan2(seg_dy, seg_dx);
            result.theta_ref_rad = theta_ref;

            const double nx = -std::sin(theta_ref);
            const double ny = std::cos(theta_ref);
            result.cte = nx * dx + ny * dy;
        }
    }

    return result;
}

double PushPathFollower::clamp(double value, double lo, double hi) {
    return (value < lo) ? lo : ((value > hi) ? hi : value);
}

double PushPathFollower::wrap_to_pi(double angle_rad) {
    while (angle_rad > M_PI) {
        angle_rad -= 2.0 * M_PI;
    }
    while (angle_rad <= -M_PI) {
        angle_rad += 2.0 * M_PI;
    }
    return angle_rad;
}

std::pair<PushPathFollower::Point, double> PushPathFollower::project_point_to_segment(
    const Point& p,
    const Point& a,
    const Point& b) {
    const double vx = b[0] - a[0];
    const double vy = b[1] - a[1];
    const double denom = vx * vx + vy * vy;
    if (denom <= 1e-12) {
        return {a, 0.0};
    }

    double t = ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / denom;
    t = clamp(t, 0.0, 1.0);
    return {{{a[0] + t * vx, a[1] + t * vy}}, t};
}

std::pair<double, double> PushPathFollower::enforce_deadband_scale(
    double left_speed,
    double right_speed) const {
    const double max_abs = std::max(std::abs(left_speed), std::abs(right_speed));
    if (max_abs < 1e-6) {
        return {0.0, 0.0};
    }
    if (max_abs >= params_.wheel_deadband) {
        return {left_speed, right_speed};
    }

    const double scale = params_.wheel_deadband / max_abs;
    double scaled_left = left_speed * scale;
    double scaled_right = right_speed * scale;
    const double max_scaled = std::max(std::abs(scaled_left), std::abs(scaled_right));
    if (max_scaled > 1.0) {
        const double shrink = 1.0 / max_scaled;
        scaled_left *= shrink;
        scaled_right *= shrink;
    }
    return {scaled_left, scaled_right};
}

}  // namespace namo
