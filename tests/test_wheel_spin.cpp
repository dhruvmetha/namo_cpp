/**
 * @file test_wheel_spin.cpp
 * @brief Minimal wheel-actuator sanity test for the diff-drive car.
 *
 * Loads a diff-drive NAMO scene, grabs the DiffDriveAdapter, and drives
 * the left/right wheel velocity actuators directly via apply_wheel_control.
 * No wavefront, no nav state machine, no skill.
 *
 * Two cases:
 *   1. Straight-line: both wheels +omega; expect forward chassis motion
 *      and both wheel-joint angles advance by roughly the same amount.
 *   2. In-place spin: wheels at +omega, -omega; expect chassis yaw to
 *      change and wheel angles diverge.
 *
 * Exits 0 on success; non-zero on any check failure.
 *
 * Usage:
 *   test_wheel_spin [xml_path] [config_path]
 *   Defaults:
 *     xml_path     = test_xml/little-car-modeling-package/artifacts/nav_env.xml
 *     config_path  = config/namo_config_car.yaml
 */

#include "environment/namo_environment.hpp"
#include "robot/robot_adapter.hpp"
#include "config/config_manager.hpp"

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

extern "C" {
#include "mujoco/mujoco.h"
}

namespace {

// Test parameters — chosen for clear, easy-to-interpret motion in 2 s
// of sim time at the car's velocity-actuator limits (ctrlrange ±25 rad/s).
constexpr double WHEEL_OMEGA_RAD_PER_S = 10.0;
constexpr int    SETTLE_STEPS          = 200;   // contacts stabilize at zero ctrl
constexpr int    DRIVE_STEPS           = 1000;  // 2 s at dt=0.002

// Minimum wheel rotation we expect after DRIVE_STEPS at WHEEL_OMEGA_RAD_PER_S.
// Pure kinematic ideal = 1000 * 0.002 * 10 = 20 rad. Friction / slip / actuator
// kv lag eat into this; threshold leaves generous margin.
constexpr double MIN_WHEEL_ROT_RAD     = 5.0;

// Forward chassis motion under straight-line drive. At r=0.015 m, ideal
// distance = 20 rad * 0.015 m/rad = 0.30 m. We expect >5 cm even with slip.
constexpr double MIN_FORWARD_DISP_M    = 0.05;

// In-place spin yaw change. At wheelbase ~0.075 m, omega_chassis ≈ 2 *
// (wheel_omega * r) / wheelbase = 2 * 0.15 / 0.075 = 4.0 rad/s ideal,
// giving ~8 rad in 2 s. We just want to see clear rotation.
constexpr double MIN_YAW_CHANGE_RAD    = 0.5;

double wrap_angle(double a) {
    while (a >  M_PI) a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
}

int wheel_qpos_adr(const mjModel* m, const char* joint_name) {
    int jid = mj_name2id(m, mjOBJ_JOINT, joint_name);
    if (jid < 0) {
        std::cerr << "Joint '" << joint_name << "' not found in model.\n";
        return -1;
    }
    return m->jnt_qposadr[jid];
}

}  // namespace

int main(int argc, char** argv) {
    const std::string xml_path =
        (argc > 1) ? argv[1]
                   : "test_xml/little-car-modeling-package/artifacts/nav_env.xml";
    const std::string config_path =
        (argc > 2) ? argv[2] : "config/namo_config_car.yaml";

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Loading scene : " << xml_path << "\n";
    std::cout << "Loading config: " << config_path << "\n";

    auto config = namo::ConfigManager::create_from_file(config_path);
    if (config->get_robot_type() != "diff_drive") {
        std::cerr << "Config robot_type is '" << config->get_robot_type()
                  << "', expected 'diff_drive'. Pick a car config.\n";
        return 1;
    }

    std::shared_ptr<namo::ConfigManager> shared_config(std::move(config));
    namo::NAMOEnvironment env(xml_path, shared_config,
                              /*visualize=*/false, /*enable_logging=*/false);

    auto* sim = env.get_mujoco_wrapper();
    auto* model = sim->model();
    auto* data = sim->data();
    auto* adapter = env.get_robot_adapter();

    if (!adapter || !adapter->is_diff_drive()) {
        std::cerr << "Adapter is not diff-drive. Got: "
                  << (adapter ? adapter->get_body_name() : "nullptr") << "\n";
        return 2;
    }

    const int left_adr  = wheel_qpos_adr(model, "left_wheel_joint");
    const int right_adr = wheel_qpos_adr(model, "right_wheel_joint");
    if (left_adr < 0 || right_adr < 0) return 3;

    // Settle at zero control so contacts stabilize.
    adapter->zero_control(model, data);
    for (int i = 0; i < SETTLE_STEPS; ++i) sim->step();
    env.update_object_states();

    // ─── Case 1: straight-line, both wheels forward ─────────────────────
    {
        const auto xy0   = adapter->get_xy(model, data);
        const double yaw0 = adapter->get_theta(model, data);
        const double wl0 = data->qpos[left_adr];
        const double wr0 = data->qpos[right_adr];

        std::cout << "\n[Case 1] Straight-line drive, both wheels +"
                  << WHEEL_OMEGA_RAD_PER_S << " rad/s for "
                  << DRIVE_STEPS << " steps\n";
        std::cout << "  start: xy=(" << xy0[0] << ", " << xy0[1]
                  << "), yaw=" << yaw0 * 180.0 / M_PI << " deg\n";

        for (int i = 0; i < DRIVE_STEPS; ++i) {
            adapter->apply_wheel_control(model, data,
                                         WHEEL_OMEGA_RAD_PER_S,
                                         WHEEL_OMEGA_RAD_PER_S);
            sim->step();
        }
        adapter->zero_control(model, data);
        for (int i = 0; i < 50; ++i) sim->step();  // brief coast
        env.update_object_states();

        const auto xy1   = adapter->get_xy(model, data);
        const double yaw1 = adapter->get_theta(model, data);
        const double dwl = data->qpos[left_adr]  - wl0;
        const double dwr = data->qpos[right_adr] - wr0;
        const double disp = std::hypot(xy1[0] - xy0[0], xy1[1] - xy0[1]);

        std::cout << "  end  : xy=(" << xy1[0] << ", " << xy1[1]
                  << "), yaw=" << yaw1 * 180.0 / M_PI << " deg\n";
        std::cout << "  wheels rotated: left=" << dwl << " rad, right="
                  << dwr << " rad\n";
        std::cout << "  chassis displacement: " << disp * 1000.0 << " mm\n";

        bool ok = true;
        if (std::abs(dwl) < MIN_WHEEL_ROT_RAD ||
            std::abs(dwr) < MIN_WHEEL_ROT_RAD) {
            std::cerr << "  FAIL: wheel rotation below threshold ("
                      << MIN_WHEEL_ROT_RAD << " rad)\n";
            ok = false;
        }
        if (disp < MIN_FORWARD_DISP_M) {
            std::cerr << "  FAIL: chassis displacement below threshold ("
                      << MIN_FORWARD_DISP_M * 1000.0 << " mm)\n";
            ok = false;
        }
        if (!ok) return 4;
        std::cout << "  PASS\n";
    }

    // ─── Case 2: in-place spin, wheels opposite ─────────────────────────
    {
        const auto xy0   = adapter->get_xy(model, data);
        const double yaw0 = adapter->get_theta(model, data);
        const double wl0 = data->qpos[left_adr];
        const double wr0 = data->qpos[right_adr];

        std::cout << "\n[Case 2] In-place spin, left=+"
                  << WHEEL_OMEGA_RAD_PER_S << " rad/s, right=-"
                  << WHEEL_OMEGA_RAD_PER_S << " rad/s\n";
        std::cout << "  start: xy=(" << xy0[0] << ", " << xy0[1]
                  << "), yaw=" << yaw0 * 180.0 / M_PI << " deg\n";

        for (int i = 0; i < DRIVE_STEPS; ++i) {
            adapter->apply_wheel_control(model, data,
                                         WHEEL_OMEGA_RAD_PER_S,
                                         -WHEEL_OMEGA_RAD_PER_S);
            sim->step();
        }
        adapter->zero_control(model, data);
        for (int i = 0; i < 50; ++i) sim->step();
        env.update_object_states();

        const double yaw1 = adapter->get_theta(model, data);
        const double dyaw = std::abs(wrap_angle(yaw1 - yaw0));
        const double dwl = data->qpos[left_adr]  - wl0;
        const double dwr = data->qpos[right_adr] - wr0;

        std::cout << "  end  : yaw=" << yaw1 * 180.0 / M_PI << " deg\n";
        std::cout << "  wheels rotated: left=" << dwl << " rad, right="
                  << dwr << " rad\n";
        std::cout << "  |yaw change|: " << dyaw * 180.0 / M_PI << " deg\n";

        bool ok = true;
        // Wheels should rotate in opposite directions.
        if (dwl * dwr >= 0.0) {
            std::cerr << "  FAIL: wheels did not rotate in opposite directions\n";
            ok = false;
        }
        if (dyaw < MIN_YAW_CHANGE_RAD) {
            std::cerr << "  FAIL: chassis yaw change below threshold ("
                      << MIN_YAW_CHANGE_RAD * 180.0 / M_PI << " deg)\n";
            ok = false;
        }
        if (!ok) return 5;
        std::cout << "  PASS\n";
    }

    std::cout << "\nAll wheel-spin checks passed.\n";
    return 0;
}
