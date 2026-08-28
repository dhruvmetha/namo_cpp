#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "python/namo/cpp_bindings/rl_env.hpp"

namespace py = pybind11;

PYBIND11_MODULE(namo_rl, m) {
    m.doc() = "Python bindings for the NAMO RL environment";

    py::class_<namo::RegionGoalSample>(m, "RegionGoalSample")
        .def(py::init<>())
        .def_readwrite("x", &namo::RegionGoalSample::x)
        .def_readwrite("y", &namo::RegionGoalSample::y)
        .def_readwrite("theta", &namo::RegionGoalSample::theta);

    py::class_<namo::RegionGoalBundle>(m, "RegionGoalBundle")
        .def(py::init<>())
        .def_readwrite("goals", &namo::RegionGoalBundle::goals)
        .def_readwrite("blocking_objects", &namo::RegionGoalBundle::blocking_objects);

    py::class_<namo::RLState>(m, "RLState")
        .def(py::init<>())
        .def_readwrite("qpos", &namo::RLState::qpos)
        .def_readwrite("qvel", &namo::RLState::qvel)
        .def("__repr__",
            [](const namo::RLState &s) {
                return "<RLState with " + std::to_string(s.qpos.size()) + " qpos and " + std::to_string(s.qvel.size()) + " qvel values>";
            }
        );

    py::class_<namo::RLEnvironment::Action>(m, "Action")
        .def(py::init<>())
        .def_readwrite("object_id", &namo::RLEnvironment::Action::object_id)
        .def_readwrite("x", &namo::RLEnvironment::Action::x)
        .def_readwrite("y", &namo::RLEnvironment::Action::y)
        .def_readwrite("theta", &namo::RLEnvironment::Action::theta)
        .def_readwrite("edge_idx", &namo::RLEnvironment::Action::edge_idx)
        .def_readwrite("depth", &namo::RLEnvironment::Action::depth);

    py::class_<namo::RLEnvironment::StepResult>(m, "StepResult")
        .def(py::init<>())
        .def_readwrite("done", &namo::RLEnvironment::StepResult::done)
        .def_readwrite("reward", &namo::RLEnvironment::StepResult::reward)
        .def_readwrite("info", &namo::RLEnvironment::StepResult::info);

    // NavigateResult py::class_ removed alongside navigate_to (see comment
    // on the .def("navigate_to") removal below).

    py::class_<namo::RLEnvironment::ActionConstraints>(m, "ActionConstraints")
        .def(py::init<>())
        .def_readwrite("min_distance", &namo::RLEnvironment::ActionConstraints::min_distance)
        .def_readwrite("max_distance", &namo::RLEnvironment::ActionConstraints::max_distance)
        .def_readwrite("theta_min", &namo::RLEnvironment::ActionConstraints::theta_min)
        .def_readwrite("theta_max", &namo::RLEnvironment::ActionConstraints::theta_max);

    py::class_<namo::RLEnvironment>(m, "RLEnvironment")
        .def(py::init<const std::string&, const std::string&, bool, bool>(),
             py::arg("xml_path"), py::arg("config_path"),
             py::arg("visualize") = false, py::arg("skip_warmup") = false,
             "skip_warmup=true skips the post-load 3-tick physics warm-up. "
             "Use this when the XML may load the robot in a state that "
             "overlaps obstacles (e.g. car planning, where the included "
             "little_car.xml fixes the freejoint spawn at the origin). "
             "After teleporting the robot to a safe pose with "
             "set_robot_pose(), call warm_up() explicitly to settle physics "
             "and establish the state that later reset() calls restore.")
        .def("warm_up", &namo::RLEnvironment::warm_up,
             "Run the post-load 3-tick physics warm-up. Only needed when the "
             "env was constructed with skip_warmup=True. The first explicit "
             "call also establishes the initialized reset baseline.")
        .def("reset", &namo::RLEnvironment::reset,
             "Reset to the initialized baseline state. For skip_warmup=True "
             "this returns to the first post-teleport warm_up() state, not "
             "the raw XML spawn.")
        .def("step", &namo::RLEnvironment::step, py::arg("action"))
        // navigate_to binding removed: the C++ impl was deleted in commit 254e5c7
        // ("Unify wavefront semantics..."), 2026-04-14, but the header decl + this
        // binding were left orphaned. Nothing in Python calls env.navigate_to() —
        // all callers go through robot_control's NavigationController instead.
        .def("get_observation", &namo::RLEnvironment::get_observation, "Returns a map of object names to their SE(2) poses.")
        .def("get_full_state", &namo::RLEnvironment::get_full_state, "Returns a full snapshot of the simulation state (qpos, qvel).")
        .def("set_full_state", &namo::RLEnvironment::set_full_state, py::arg("state"), "Sets the simulation to a specific state snapshot.")
        .def("render", &namo::RLEnvironment::render, "Renders the current simulation state (requires visualization=True).")
        .def("set_camera_position", &namo::RLEnvironment::set_camera_position,
             py::arg("distance"), py::arg("azimuth"), py::arg("elevation"),
             "Set camera position: distance from lookat, azimuth (horizontal angle), elevation (vertical angle, -90=top-down)")
        .def("set_camera_lookat", &namo::RLEnvironment::set_camera_lookat,
             py::arg("x"), py::arg("y"), py::arg("z"),
             "Set camera lookat point (where camera looks at)")
        .def("get_reachable_objects", &namo::RLEnvironment::get_reachable_objects, "Returns a list of object names that are reachable through push actions.")
        .def("is_object_reachable", &namo::RLEnvironment::is_object_reachable, py::arg("object_name"), "Returns true if the specified object is reachable through push actions.")
        .def("get_reachable_edges", &namo::RLEnvironment::get_reachable_edges, py::arg("object_name"), "Returns list of reachable edge indices (0-59) for the specified object using wavefront analysis.")
        .def("get_reachability_summary",
             [](const namo::RLEnvironment& env, bool analysis_mode) {
                 auto summary = env.get_reachability_summary(analysis_mode);
                 py::dict output;
                 output["goal_reachable"] = summary.goal_reachable;
                 output["analysis_mode"] = analysis_mode;

                 py::dict objects;
                 for (const auto& [name, obj] : summary.objects) {
                     py::dict entry;
                     entry["reachable"] = obj.reachable;
                     entry["reachable_edges"] = obj.reachable_edges;
                     entry["total_edges"] = obj.total_edges;
                     entry["reachable_primitives"] = obj.reachable_primitives;
                     entry["total_primitives"] = obj.total_primitives;
                     if (analysis_mode) {
                         entry["reachable_edge_indices"] = obj.reachable_edge_indices;
                     }
                     objects[py::str(name)] = std::move(entry);
                 }
                 output["objects"] = std::move(objects);
                 return output;
             },
             py::arg("analysis_mode") = false,
             "Return unified reachability from one C++ wavefront snapshot. "
             "Includes goal reachability plus per-object edge/primitive reachability stats.")
        .def("get_object_info", &namo::RLEnvironment::get_object_info, "Returns object geometry information (sizes, positions, orientations) for all objects including static walls.")
        .def("get_world_bounds", &namo::RLEnvironment::get_world_bounds, "Returns world bounds [x_min, x_max, y_min, y_max] calculated from all objects.")
        .def("set_robot_pose", &namo::RLEnvironment::set_robot_pose, py::arg("x"), py::arg("y"), py::arg("theta"),
             "Override the robot's pose loaded from the XML. Used by the "
             "robot_control bridge for car (diff-drive) planning, where the "
             "freejoint spawn pose lives inside the included little_car.xml "
             "and can't be parameterized through a top-level <include>. The "
             "bridge calls this once with the live observation pose right "
             "after env construction so the planner searches from the "
             "correct starting state. Sphere XMLs bake the pose into the "
             "geom and don't need to call this.")
        .def("set_robot_goal", &namo::RLEnvironment::set_robot_goal, py::arg("x"), py::arg("y"), py::arg("theta") = 0.0, "Set robot goal for MCTS planning.")
        .def("set_robot_goal_silent", &namo::RLEnvironment::set_robot_goal_silent, py::arg("x"), py::arg("y"), py::arg("theta") = 0.0,
             "Set robot goal without updating the visualization marker (useful for repeated reachability checks).")
        .def("is_robot_goal_reachable", &namo::RLEnvironment::is_robot_goal_reachable, "Check if robot goal is reachable from current state.")
        .def("count_reachable_points", &namo::RLEnvironment::count_reachable_points, py::arg("points"),
             "Count how many of the given (x, y) points are reachable from the robot's current position. "
             "Does NOT mutate the robot goal — use for subgoal probes (e.g. region opening). "
             "Returns (count, first_reachable_index); first index is -1 if none reachable.")
        .def("get_robot_goal", &namo::RLEnvironment::get_robot_goal, "Get current robot goal.")
        .def("clear_robot_goal", &namo::RLEnvironment::clear_robot_goal, "Clear the skill's robot goal and hide the visualization marker.")
        .def("set_goal_site_visible", &namo::RLEnvironment::set_goal_site_visible, py::arg("visible"),
             "Show/hide the XML `<site name='goal'>` marker when present (visualization only).")
        .def("set_robot_trajectory_collision_checking", &namo::RLEnvironment::set_robot_trajectory_collision_checking, py::arg("enable"), "Enable or disable robot-body collision checking during push trajectory.")
        .def("evaluate_primitive_priorities", &namo::RLEnvironment::evaluate_primitive_priorities,
             py::arg("object_name"), py::arg("target_poses"), py::arg("robot_goal"),
             "Evaluate geometric transport priorities for primitive targets. Returns priorities 1-6 (1=best, 6=worst).")
        .def("evaluate_primitive_region_scores", &namo::RLEnvironment::evaluate_primitive_region_scores,
             py::arg("object_name"), py::arg("target_poses"), py::arg("region_samples"),
             "Virtually place the pushed object at each primitive target and return the reachable target-region fraction.")
        .def("get_last_priority_profile", &namo::RLEnvironment::get_last_priority_profile,
             "Get timing breakdown for the most recent evaluate_primitive_priorities() call.")
        .def("get_action_constraints", &namo::RLEnvironment::get_action_constraints, "Get action space constraints for MCTS.")
        // Video recording interface
        .def("start_recording", &namo::RLEnvironment::start_recording,
             py::arg("width") = 640, py::arg("height") = 480,
             py::arg("capture_frequency") = 100, py::arg("max_frames") = 10000,
             "Start recording frames during physics execution. "
             "capture_frequency=N means capture every N physics steps. "
             "Requires visualize=True for OpenGL context.")
        .def("stop_recording", &namo::RLEnvironment::stop_recording,
             "Stop recording frames.")
        .def("is_recording", &namo::RLEnvironment::is_recording,
             "Check if recording is active.")
        .def("get_frame_count", &namo::RLEnvironment::get_frame_count,
             "Get number of captured frames.")
        .def("get_frames", [](const namo::RLEnvironment& env) {
            auto frames = env.get_frames();
            auto dims = env.get_recording_dimensions();
            int width = std::get<0>(dims);
            int height = std::get<1>(dims);
            size_t n_frames = frames.size();

            if (n_frames == 0) {
                return py::array_t<unsigned char>(std::vector<ssize_t>{0, height, width, 3});
            }

            // Create numpy array with shape (n_frames, height, width, 3)
            py::array_t<unsigned char> result({(ssize_t)n_frames, (ssize_t)height, (ssize_t)width, (ssize_t)3});
            auto buf = result.mutable_unchecked<4>();

            for (size_t f = 0; f < n_frames; f++) {
                const auto& frame = frames[f];
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        for (int c = 0; c < 3; c++) {
                            buf(f, y, x, c) = frame[(y * width + x) * 3 + c];
                        }
                    }
                }
            }
            return result;
        }, "Get captured frames as numpy array with shape (n_frames, height, width, 3).")
        .def("get_frame_bytes", [](const namo::RLEnvironment& env, size_t idx) {
            const auto& frame = env.get_frame_ref(idx);
            if (frame.empty()) {
                return py::bytes();
            }
            return py::bytes(reinterpret_cast<const char*>(frame.data()), frame.size());
        }, py::arg("idx"),
        "Get a single captured frame as raw RGB bytes (width*height*3). "
        "Useful for streaming to encoders without materializing a full (N,H,W,3) array.")
        .def("clear_frames", &namo::RLEnvironment::clear_frames,
             "Clear captured frames to free memory.")
       .def("get_recording_dimensions", &namo::RLEnvironment::get_recording_dimensions,
             "Get recording dimensions as (width, height) tuple.")
       .def("get_region_snapshot",
            [](const namo::RLEnvironment& env,
               int goals_per_region,
               double goal_radius,
               bool local_info_only,
               unsigned int seed,
               bool use_xml_goal) {
                auto snapshot = env.get_region_snapshot(
                    goals_per_region,
                    goal_radius,
                    local_info_only,
                    seed,
                    use_xml_goal
                );
                py::dict out;
                out["adjacency"] = snapshot.adjacency;
                out["edge_objects"] = snapshot.edge_objects;
                out["region_labels"] = snapshot.region_labels;
                out["region_goals"] = snapshot.region_goals;
                out["robot_label"] = snapshot.robot_label;
                out["goal_label"] = snapshot.goal_label;
                out["goal_reachable"] = snapshot.goal_reachable;
                out["goal_in_free_space"] = snapshot.goal_in_free_space;
                return out;
            },
            py::arg("goals_per_region") = 0,
            py::arg("goal_radius") = -1.0,
            py::arg("local_info_only") = false,
            py::arg("seed") = 42,
            py::arg("use_xml_goal") = true,
            "Return one unified C++ wavefront snapshot: region connectivity, sampled goals, "
            "robot/goal labels, and goal reachability flags.")
       .def("get_region_connectivity", &namo::RLEnvironment::get_region_connectivity,
           "Return region adjacency, boundary objects, and region labels from the wavefront grid.")
       .def("sample_region_goals", &namo::RLEnvironment::sample_region_goals,
            py::arg("goals_per_region"),
            "Sample random goal poses for each region, including blocking objects shared with the robot region.")
       .def("get_xml_path", &namo::RLEnvironment::get_xml_path,
           py::return_value_policy::reference_internal,
           "Return the XML scene path used to create this environment.")
       .def("get_config_path", &namo::RLEnvironment::get_config_path,
           py::return_value_policy::reference_internal,
           "Return the NAMO configuration path used to create this environment.");
}
