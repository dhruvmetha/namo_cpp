#pragma once

#include <array>
#include <limits>
#include <string>
#include <vector>

namespace namo {

class PushPathFollower {
public:
    using Point = std::array<double, 2>;

    struct Params {
        double robot_width_m = 0.07;
        double robot_height_m = 0.07;
        double wheel_base_m = 0.075;
        double lookahead_distance_m = -1.0;
        double goal_tolerance_m = -1.0;
        double max_speed = 0.3;
        double max_point_gap_ratio = 0.01;
        double no_skip_ratio = 0.5;
        double wheel_deadband = 0.05;
    };

    struct Pose {
        double x_m = 0.0;
        double y_m = 0.0;
        double theta_rad = 0.0;
    };

    enum class Mode {
        IDLE,
        TRACK,
        FINISHED,
    };

    struct StepOutput {
        double left_speed = 0.0;
        double right_speed = 0.0;
        Mode mode = Mode::IDLE;
        bool done = false;
        Point target_point = {0.0, 0.0};
        int path_index = 0;
        double heading_error_rad = 0.0;
        double curvature_command = 0.0;
        double cte = 0.0;
    };

    explicit PushPathFollower(const Params& params);

    void set_path(const std::vector<Point>& path);
    void clear_path();
    void reset();
    void set_speed(double speed);

    StepOutput step(
        const Pose& pose,
        double timestamp_s = std::numeric_limits<double>::quiet_NaN());

    bool is_done() const { return mode_ == Mode::FINISHED; }
    Mode mode() const { return mode_; }
    int path_index() const { return path_index_; }
    const std::vector<Point>& path() const { return path_; }
    double car_size_m() const { return car_size_m_; }
    double max_speed() const { return max_speed_; }

    static const char* mode_name(Mode mode);

private:
    struct ReferencePoint {
        Point projection = {0.0, 0.0};
        double theta_ref_rad = 0.0;
        double cte = 0.0;
        int segment_index = 0;
    };

    struct CurvatureResult {
        double curvature = 0.0;
        double heading_error_rad = 0.0;
        double distance_to_target = 0.0;
    };

    std::vector<Point> resample_path(const std::vector<Point>& path, double max_step) const;
    void build_prefix_s();
    int max_reachable_index(int start_idx) const;
    Point find_target_point(double robot_x, double robot_y);
    CurvatureResult pure_pursuit_curvature(
        double robot_x,
        double robot_y,
        double robot_theta_rad,
        const Point& target) const;
    ReferencePoint find_closest_reference(const Point& position);

    static double clamp(double value, double lo, double hi);
    static double wrap_to_pi(double angle_rad);
    static std::pair<Point, double> project_point_to_segment(
        const Point& p,
        const Point& a,
        const Point& b);
    std::pair<double, double> enforce_deadband_scale(
        double left_speed,
        double right_speed) const;

    Params params_;
    double car_size_m_ = 0.0;
    double lookahead_distance_m_ = 0.0;
    double goal_tolerance_m_ = 0.0;
    double max_speed_ = 0.3;

    double cte_dot_alpha_ = 0.25;
    double min_turn_factor_ = 0.25;

    std::vector<Point> path_raw_;
    std::vector<Point> path_;
    std::vector<double> path_s_;
    int path_index_ = 0;

    double prev_cte_ = 0.0;
    double prev_cte_time_ = 0.0;
    bool has_prev_cte_ = false;
    double cte_dot_filt_ = 0.0;

    Mode mode_ = Mode::IDLE;
};

}  // namespace namo
