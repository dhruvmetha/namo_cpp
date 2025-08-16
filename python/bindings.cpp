#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "python/rl_env.hpp"

namespace py = pybind11;

PYBIND11_MODULE(namo_rl, m) {
    m.doc() = "Python bindings for the NAMO RL environment";

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
        .def_readwrite("theta", &namo::RLEnvironment::Action::theta);

    py::class_<namo::RLEnvironment::StepResult>(m, "StepResult")
        .def(py::init<>())
        .def_readwrite("done", &namo::RLEnvironment::StepResult::done)
        .def_readwrite("reward", &namo::RLEnvironment::StepResult::reward)
        .def_readwrite("info", &namo::RLEnvironment::StepResult::info);

    py::class_<namo::RLEnvironment>(m, "RLEnvironment")
        .def(py::init<const std::string&, const std::string&, bool>(), 
             py::arg("xml_path"), py::arg("config_path"), py::arg("visualize") = false)
        .def("reset", &namo::RLEnvironment::reset)
        .def("step", &namo::RLEnvironment::step, py::arg("action"))
        .def("get_observation", &namo::RLEnvironment::get_observation, "Returns a map of object names to their SE(2) poses.")
        .def("get_full_state", &namo::RLEnvironment::get_full_state, "Returns a full snapshot of the simulation state (qpos, qvel).")
        .def("set_full_state", &namo::RLEnvironment::set_full_state, py::arg("state"), "Sets the simulation to a specific state snapshot.")
        .def("render", &namo::RLEnvironment::render, "Renders the current simulation state (requires visualization=True).")
        .def("get_reachable_objects", &namo::RLEnvironment::get_reachable_objects, "Returns a list of object names that are reachable through push actions.")
        .def("is_object_reachable", &namo::RLEnvironment::is_object_reachable, py::arg("object_name"), "Returns true if the specified object is reachable through push actions.");
}
