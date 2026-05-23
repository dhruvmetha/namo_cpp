#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <string>
#include <filesystem>
#include <cstdint>
#include <memory>
#include <algorithm>

#include "../include/core/parameter_loader.hpp"
#include "../include/config/config_manager.hpp"
#include "../include/environment/namo_environment.hpp"
#include "../include/planning/namo_push_controller.hpp"
#include "../include/wavefront/wavefront_planner.hpp"

using namespace namo;

// Scene configuration for primitive generation
struct SceneConfig {
    std::string name;
    std::string xml_path;
    std::string description;
};

// Primitive data structure for binary storage
struct __attribute__((packed)) NominalPrimitive {
    float delta_x;        // Position change in x
    float delta_y;        // Position change in y  
    float delta_theta;    // Rotation change (yaw)
    uint8_t edge_idx;     // Push direction (0-63)
    uint8_t push_steps;   // Push step number (1-10)
};

// Verify struct size
static_assert(sizeof(NominalPrimitive) == 14, "NominalPrimitive must be 14 bytes");

// Helper function to add suffix to filename
std::string add_suffix_to_filename(const std::string& base_path, const std::string& suffix) {
    auto dot_pos = base_path.find_last_of('.');
    if (dot_pos == std::string::npos) {
        return base_path + "_" + suffix;
    }
    return base_path.substr(0, dot_pos) + "_" + suffix + base_path.substr(dot_pos);
}

// Generate primitives for a single scene
std::vector<NominalPrimitive> generate_primitives_for_scene(
    const SceneConfig& scene_config,
    bool visualize,
    double resolution,
    int points_per_face,
    int control_steps,
    int max_push_steps,
    double push_velocity,
    bool dynamic_direction,
    double push_offset_margin,
    std::shared_ptr<ConfigManager> config,
    int min_push_steps_override = 0,
    int settle_ticks_override = 0,
    int single_edge_override = -1
) {
    std::cout << "\n=== Generating primitives for " << scene_config.name << " ===" << std::endl;
    std::cout << "XML: " << scene_config.xml_path << std::endl;
    std::cout << "Description: " << scene_config.description << std::endl;

    // Create NAMO environment for this scene. Pass the ConfigManager so the
    // env can select the right RobotAdapter (holonomic vs diff_drive). Without
    // this, the env defaults to holonomic and looks for a body named "robot",
    // which the car scene XMLs don't have — robot_info ends up with garbage
    // size, WavefrontPlanner uses it as inflation radius, and the obstacle
    // footprint blows out GridFootprint::MAX_CELLS (200k), segfaulting.
    NAMOEnvironment env(scene_config.xml_path, config, visualize, false);

    // The car scene XMLs <include> little_car.xml, which spawns the car body
    // at (0, 0, 0.01). The obstacle is at (0, 0, 0.05) — so without an offset
    // the car overlaps the obstacle and wavefront BFS from the robot finds
    // no reachable movables. Teleport to the same starting offset the sphere
    // scenes use (-0.3333 m in x). For sphere scenes, this is a no-op write
    // back to roughly where the geom already sits — harmless.
    env.set_robot_position(std::array<double, 2>{-0.3333, 0.0});
    
    // Robot footprint half-extents used to compute (a) wavefront inflation
    // and (b) edge-point standoff from the obstacle face. Prefer the
    // authoritative `planning.robot_size` from the config because reading
    // from robot_info.size grabs ONLY the first geom of the robot body
    // (e.g. the car's front_chassis_collision, half-X = 0.0175), which
    // under-represents the full car footprint (the two chassis halves
    // span ±0.035). With a too-small robot_size, edge points land
    // 1.75 cm closer to the obstacle than they should — the car
    // teleports INSIDE the obstacle, contact resolution pushes it out,
    // and "drive forward" produces zero displacement.
    std::vector<double> robot_size;
    if (config && !config->planning().robot_size.empty()) {
        robot_size = {config->planning().robot_size[0],
                      config->planning().robot_size[1]};
    } else {
        const auto& robot_info = env.get_robot_info();
        robot_size = {robot_info.size[0], robot_info.size[1]};
    }
    
    // Create wavefront planner (heap allocation to avoid 32MB stack array)
    auto wavefront_planner = std::make_unique<WavefrontPlanner>(resolution, env, robot_size);
    
    // Set robot goal (fixed for nominal primitive generation)
    std::array<double, 2> robot_goal = {0.0, 0.0};
    env.set_robot_goal(robot_goal);
    
    // Create push controller
    NAMOPushController push_controller(env, *wavefront_planner, max_push_steps, control_steps, push_velocity, points_per_face, dynamic_direction);

    // Apply config-driven stuck-check parameters. NAMOPushSkill does this in
    // its own ctor (src/skills/namo_push_skill.cpp:93-96) but the generator
    // bypasses the skill and drives the controller directly, so without
    // this block the controller falls back to its header defaults
    // (stride=20, threshold=3, min_pos=0.001 m, min_angle=0.05 rad). At the
    // car's 0.002 s timestep that aborts the push at tick 60 — before the
    // car has even traversed the 1 cm standoff to make contact, so every
    // primitive records Δ=0.
    if (config) {
        push_controller.set_stuck_check_stride(config->skill().stuck_check_stride);
        push_controller.set_stuck_threshold(config->skill().controller_stuck_threshold);
        push_controller.set_min_position_change(config->skill().controller_min_position_change);
        push_controller.set_min_angle_change(config->skill().controller_min_angle_change);
        // Wire the calibrated push-tracker fraction cap into the follower.
        // NAMOPushSkill does this in src/skills/namo_push_skill.cpp:53, but
        // the generator bypasses the skill and drives the controller
        // directly — without this, the follower stays at
        // PushPathFollower::Params::max_speed (0.3), making sim push ~5.6×
        // faster than the calibrated value (0.05381 per 2026-05-22 chassis
        // calibration). Symptom: wheel ctrl logs show ~20 rad/s instead of
        // the expected ~3.6 rad/s for diff-drive runs.
        push_controller.set_push_tracker_max_speed(config->skill().push_tracker_max_speed);
    }
    if (settle_ticks_override > 0) {
        push_controller.set_settle_steps(settle_ticks_override);
    }

    // Apply config-driven push_offset_margin so generator-side edge points
    // match what the skill runtime uses. Without this the controller falls
    // back to its hardcoded default (0.02 m) — invisible at 6× scale (~3 mm
    // real) but a 2 cm real gap at 1× scale, which makes the robot land
    // far from the object face.
    push_controller.set_push_offset_margin(push_offset_margin);

    // Get movable objects (should be our nominal object)
    std::array<std::string, 20> reachable_objects;
    size_t reachable_count;
    size_t num_reachable = push_controller.get_reachable_objects(reachable_objects, reachable_count);
    
    if (num_reachable == 0) {
        throw std::runtime_error("No reachable objects found in scene: " + scene_config.xml_path);
    }
    
    std::string target_object = reachable_objects[0];
    std::cout << "Using object: " << target_object << std::endl;
    
    // Get edge points for this object
    std::array<std::array<double, 2>, 64> edge_points;
    std::array<std::array<double, 2>, 64> mid_points;
    size_t edge_count, mid_count;
    size_t num_edges = push_controller.generate_edge_points(target_object, edge_points, mid_points, edge_count, mid_count);
    
    std::cout << "Generated " << num_edges << " edge points" << std::endl;
    
    // Position camera for good view if visualizing
    if (visualize) {
        auto obj_state = env.get_object_state(target_object);
        if (obj_state) {
            std::array<double, 3> focus_point = {obj_state->position[0], obj_state->position[1], 0.0};
            env.set_camera_lookat(focus_point);
            env.set_camera_position(6.0, 0.0, -45.0);
        }
    }
    
    // Get initial object state
    auto initial_obj_state = env.get_object_state(target_object);
    if (!initial_obj_state) {
        throw std::runtime_error("Failed to get initial object state!");
    }
    
    std::array<double, 3> initial_pos = initial_obj_state->position;
    std::array<double, 4> initial_quat = initial_obj_state->quaternion;
    
    std::cout << "Initial object position: [" << initial_pos[0] << ", " << initial_pos[1] << ", " << initial_pos[2] << "]" << std::endl;
    
    // Generate primitives for each edge
    std::vector<NominalPrimitive> all_primitives;
    all_primitives.reserve(num_edges * max_push_steps);
    
    // Initial spawn offset for the robot. Sphere scenes encode this on the
    // sphere geom's `pos="-0.3333 …"`, so the holonomic adapter sees the
    // robot 33 cm from the obstacle at origin. Car scenes include little_car.xml
    // which spawns the car body at (0, 0, 0.01) — overlapping the obstacle at
    // origin. This tool uses a plain post-construction teleport rather than
    // the deferred-warmup API, so env.reset() correctly returns to the ctor
    // baseline (the XML spawn). Re-teleport after every reset.
    const std::array<double, 2> robot_spawn_offset = {-0.3333, 0.0};

    // After teleport, the car body's z stays at the XML's spawn z (= 0.01 m
    // in little_car.xml). The wheel centres are at z = 0.025 m with radius
    // 0.015 m, so the wheel bottoms sit at z = 0.010 m — 1 cm above the
    // floor. Wheels spin in the air and the car never moves the obstacle.
    // Step the simulation a few times after each teleport so gravity drops
    // the chassis onto the wheels and onto the floor. Settle for ~1 s of
    // wall-clock at the scene's timestep — sphere scenes use 0.01 s/tick
    // so 100 ticks = 1 s, car scenes use 0.002 s/tick so 500 ticks = 1 s.
    // The 1 s window matches the Python car-primitive generator.
    const double scene_timestep = env.get_mujoco_wrapper()->model()->opt.timestep;
    const int kSettleStepsAfterReset = settle_ticks_override > 0
        ? settle_ticks_override
        : std::max(100, static_cast<int>(1.0 / scene_timestep));
    auto reset_and_settle = [&]() {
        env.reset();
        env.step_simulation();
        env.set_robot_position(robot_spawn_offset);
        for (int s = 0; s < kSettleStepsAfterReset; s++) {
            env.step_simulation();
        }
    };

    for (size_t edge_idx = 0; edge_idx < num_edges; edge_idx++) {
        if (single_edge_override >= 0 && static_cast<int>(edge_idx) != single_edge_override) {
            continue;
        }
        std::cout << "Generating primitives for edge " << edge_idx << " / " << num_edges << std::endl;

        // Reset environment to initial state
        reset_and_settle();

        // Generate primitives for all step counts. By default starts at 1
        // (pyramid). --min-push-steps lets viz runs skip the small depths
        // so each window-tick is informative motion.
        const int first_ps = std::max(1, min_push_steps_override > 0 ? min_push_steps_override : 1);
        for (int push_steps = first_ps; push_steps <= max_push_steps; push_steps++) {
            std::cout << "  push_steps=" << push_steps << std::endl;

            // Reset environment to initial state for each primitive
            reset_and_settle();

            // Execute push primitive for this number of steps
            bool success = push_controller.execute_push_primitive(target_object, edge_idx, push_steps);
            
            // Get object state after push sequence
            auto final_obj_state = env.get_object_state(target_object);
            if (!final_obj_state) {
                continue;
            }
            
            // Calculate displacement from initial position
            NominalPrimitive primitive;
            primitive.delta_x = final_obj_state->position[0] - initial_pos[0];
            primitive.delta_y = final_obj_state->position[1] - initial_pos[1];
            
            // Calculate rotation change (simple yaw extraction)
            auto quat_to_yaw = [](const std::array<double, 4>& q) -> double {
                return std::atan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 
                                1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]));
            };
            
            primitive.delta_theta = quat_to_yaw(final_obj_state->quaternion) - quat_to_yaw(initial_quat);
            primitive.edge_idx = edge_idx;
            primitive.push_steps = push_steps;
            
            all_primitives.push_back(primitive);
            
            // Render final state if visualizing (only for last step to avoid too much output)
            if (visualize && push_steps == max_push_steps) {
                env.render();
            }
        }
        
        // No interactive pause between edges — viewer just streams through.
        // (Was: prompt + std::cin.get() to step through manually; removed
        // because it blocks unattended runs.)
    }
    
    std::cout << "Generated " << all_primitives.size() << " primitives for " << scene_config.name << std::endl;
    return all_primitives;
}

// Save primitives to binary file
void save_primitives_to_file(const std::string& output_file, const std::vector<NominalPrimitive>& primitives) {
    std::ofstream file(output_file, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to create output file: " + output_file);
    }
    
    std::cout << "Saving primitives to: " << output_file << std::endl;
    
    // Write header
    uint32_t count = primitives.size();
    file.write(reinterpret_cast<const char*>(&count), sizeof(count));
    
    // Write primitives
    file.write(reinterpret_cast<const char*>(primitives.data()), 
              count * sizeof(NominalPrimitive));
    
    file.close();
    
    std::cout << "Saved " << count << " primitives to: " << output_file << std::endl;
    std::cout << "File size: " << std::filesystem::file_size(output_file) << " bytes" << std::endl;
}

int main(int argc, char** argv) {
    // CLI overrides:
    //   --output <path>         -- replaces system.motion_primitives_file from
    //                              config. Used to regenerate to a fresh
    //                              filename without touching the .dat files
    //                              the planner is currently wired to.
    //   --scenes-suffix <text>  -- appended to each scene XML filename before
    //                              .xml (e.g. "_1x" makes the generator look
    //                              for nominal_primitive_scene_square_1x.xml).
    //                              Defaults to "" (current scene set).
    //                              Lets us host a 1×-scaled scene set
    //                              alongside the existing 6× scenes without
    //                              destructively overwriting either.
    //   --config <path>         -- overrides the hardcoded config path. Use
    //                              the 1× config when generating 1× primitives
    //                              so push_velocity / grid resolutions match
    //                              the scene scale. Without this, the generator
    //                              runs with the 6× config and produces
    //                              primitives whose magnitudes don't match the
    //                              scaled-down scenes.
    std::string output_override;
    std::string scenes_suffix;
    std::string config_override;
    // Optional viz-speedup knobs: skip lower depths, and shrink settle.
    int min_push_steps_override = 0;     // 0 = no override (start at 1)
    int settle_ticks_override = 0;       // 0 = use scene-derived defaults
    int single_edge_override = -1;       // -1 = run all edges
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--output") {
            output_override = argv[i + 1];
        } else if (std::string(argv[i]) == "--scenes-suffix") {
            scenes_suffix = argv[i + 1];
        } else if (std::string(argv[i]) == "--config") {
            config_override = argv[i + 1];
        } else if (std::string(argv[i]) == "--min-push-steps") {
            min_push_steps_override = std::atoi(argv[i + 1]);
        } else if (std::string(argv[i]) == "--settle-ticks") {
            settle_ticks_override = std::atoi(argv[i + 1]);
        } else if (std::string(argv[i]) == "--single-edge") {
            single_edge_override = std::atoi(argv[i + 1]);
        }
    }

    std::cout << "=== Multi-Scene Nominal Motion Primitive Generator ===" << std::endl;
    std::cout << "Generating primitives for multiple object shapes" << std::endl;
    
    try {
        // Prefer unified config if present, fallback to minimal local config.
        // --config overrides the hardcoded default (e.g. _1x.yaml when
        // generating 1×-scale primitives).
        std::string config_path = config_override.empty()
            ? std::string("config/namo_config_complete_skill15.yaml")
            : config_override;
        if (!std::filesystem::exists(config_path)) {
            throw std::runtime_error(
                "Config file not found: " + config_path +
                " (override via --config <path>)");
        }

        // Single source of truth for every generation parameter — same
        // ConfigManager the runtime skill uses. All defaults live in the
        // struct definitions in include/config/config_manager.hpp; we don't
        // duplicate them here. Legacy key fallbacks (wavefront_planner.resolution,
        // skill.num_edge_points, motion_primitives.max_push_steps, top-level
        // `visualize`) are gone — every active config uses the unified names
        // after commit d9e7d6b ("Unify wavefront semantics ...").
        auto config = std::make_shared<ConfigManager>(config_path);
        std::cout << "Configuration loaded from: " << config_path << std::endl;
        std::cout << "  Robot type: " << config->get_robot_type() << std::endl;
        
        // Define the three scenes to generate primitives for. When
        // --scenes-suffix is provided, append it before .xml so we can host
        // multiple scale variants (e.g. "_1x") side-by-side.
        auto with_suffix = [&scenes_suffix](const std::string& base) {
            if (scenes_suffix.empty()) return base;
            const std::string dot_xml = ".xml";
            if (base.size() > dot_xml.size() &&
                base.compare(base.size() - dot_xml.size(), dot_xml.size(), dot_xml) == 0) {
                return base.substr(0, base.size() - dot_xml.size()) + scenes_suffix + dot_xml;
            }
            return base + scenes_suffix;
        };
        std::vector<SceneConfig> scenes = {
            {"square", with_suffix("data/nominal_primitive_scene_square.xml"), "Square object"},
            {"wide", with_suffix("data/nominal_primitive_scene_wide.xml"), "Wide object"},
            {"tall", with_suffix("data/nominal_primitive_scene_tall.xml"), "Tall object"}
        };
        
        // Filter to only existing files, with fallback to legacy
        std::vector<SceneConfig> existing_scenes;
        for (const auto& scene : scenes) {
            if (std::filesystem::exists(scene.xml_path)) {
                existing_scenes.push_back(scene);
                std::cout << "Found scene: " << scene.name << " -> " << scene.xml_path << std::endl;
            } else {
                std::cout << "Scene XML not found, skipping: " << scene.xml_path << std::endl;
            }
        }
        
        // Fallback to legacy file if no variants found
        if (existing_scenes.empty()) {
            std::string legacy_xml = "data/nominal_primitive_scene.xml";
            if (std::filesystem::exists(legacy_xml)) {
                std::cout << "Using legacy single scene: " << legacy_xml << std::endl;
                existing_scenes.push_back({"square", legacy_xml, "Legacy single scene"});
            } else {
                throw std::runtime_error("No valid scene XML files found!");
            }
        }
        
        std::cout << "Found " << existing_scenes.size() << " scene(s) to process" << std::endl;
        
        // All generation parameters come from the ConfigManager — same struct
        // defaults the runtime uses. Clamp points_per_face to respect the
        // MAX_EDGE_POINTS capacity (4 faces × 16 = 64 max).
        const bool visualize         = config->system().enable_visualization;
        const double resolution      = config->planning().skill_level_resolution;
        const int points_per_face    = std::clamp(config->skill().points_per_face, 1, 16);
        const int control_steps      = config->skill().control_steps_per_push;
        const double push_offset_margin = config->planning().wavefront_edge_offset_margin;
        const int max_push_steps     = config->skill().max_push_steps;
        const double push_velocity   = config->skill().push_velocity;
        const bool dynamic_direction = config->skill().dynamic_direction;

        // Determine base output file. CLI --output wins over the config
        // setting; without either, ConfigManager's default kicks in.
        std::string base_output = config->system().motion_primitives_file;
        if (!output_override.empty()) {
            base_output = output_override;
            std::cout << "Output overridden via --output: " << base_output << std::endl;
        }

        std::cout << "Generation parameters:" << std::endl;
        std::cout << "  Visualize: " << (visualize ? "true" : "false") << std::endl;
        std::cout << "  Resolution: " << resolution << std::endl;
        std::cout << "  Points per face: " << points_per_face << std::endl;
        std::cout << "  Control steps: " << control_steps << std::endl;
        std::cout << "  Max push steps: " << max_push_steps << std::endl;
        std::cout << "  Push velocity (m/s): " << push_velocity << std::endl;
        std::cout << "  Dynamic direction: " << (dynamic_direction ? "true" : "false") << std::endl;
        std::cout << "  Base output: " << base_output << std::endl;

        // Generate primitives for each scene
        for (const auto& scene : existing_scenes) {
            try {
                auto primitives = generate_primitives_for_scene(
                    scene, visualize, resolution, points_per_face,
                    control_steps, max_push_steps, push_velocity, dynamic_direction,
                    push_offset_margin, config,
                    min_push_steps_override, settle_ticks_override,
                    single_edge_override
                );
                
                // Save to suffixed output file. The base path is used as a
                // PREFIX — only the shape-suffixed files are written, never
                // a duplicate unsuffixed base file. The skill loads only
                // the suffixed siblings; an unsuffixed base file would
                // be a byte-duplicate of _square and a maintenance hazard.
                std::string output_file = add_suffix_to_filename(base_output, scene.name);
                save_primitives_to_file(output_file, primitives);

            } catch (const std::exception& e) {
                std::cerr << "Failed to generate primitives for scene " << scene.name << ": " << e.what() << std::endl;
                // Continue with other scenes
            }
        }
        
        std::cout << "\n=== Generation Complete ===" << std::endl;
        std::cout << "Processed " << existing_scenes.size() << " scene(s)" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
