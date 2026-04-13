from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from car_model.mesh_utils import box_mesh, wheel_prism_mesh, write_obj
from car_model.parameters import CarParameters, default_parameters



def _format_xyz(*values: float) -> str:
    return " ".join(f"{value:.6f}" for value in values)



def render_urdf(params: CarParameters) -> str:
    body_ixx, body_iyy, body_izz = params.body_inertia
    front_body_ixx, front_body_iyy, front_body_izz = params.front_body_inertia
    rear_body_ixx, rear_body_iyy, rear_body_izz = params.rear_body_inertia
    wheel_ixx, wheel_iyy, wheel_izz = params.wheel_inertia
    support_ixx, support_iyy, support_izz = params.support_inertia
    wheel_z = params.wheel_center_z_m - params.body_center_z_m
    support_z = params.support_center_z_m - params.body_center_z_m
    support_geom_offset_x = params.support_geom_offset_x_m

    return dedent(
        f'''\
        <?xml version="1.0"?>
        <robot name="little_car">
          <material name="body_gray"><color rgba="0.4 0.4 0.4 1.0"/></material>
          <material name="wheel_black"><color rgba="0.1 0.1 0.1 1.0"/></material>
          <material name="support_red"><color rgba="0.7 0.1 0.1 1.0"/></material>

          <link name="base_footprint"/>

          <link name="chassis">
            <visual>
              <origin xyz="0 0 0" rpy="0 0 0"/>
              <geometry><mesh filename="../meshes/chassis_top.obj"/></geometry>
              <material name="body_gray"/>
            </visual>
            <visual>
              <origin xyz="{_format_xyz(params.front_body_center_x_m, 0.0, 0.0)}" rpy="0 0 0"/>
              <geometry><box size="{_format_xyz(params.body_half_length_m, params.body_width_m, params.body_height_m)}"/></geometry>
              <material name="body_gray"/>
            </visual>
            <visual>
              <origin xyz="{_format_xyz(params.rear_body_center_x_m, 0.0, 0.0)}" rpy="0 0 0"/>
              <geometry><box size="{_format_xyz(params.body_half_length_m, params.body_width_m, params.body_height_m)}"/></geometry>
              <material name="body_gray"/>
            </visual>
            <collision>
              <origin xyz="{_format_xyz(params.front_body_center_x_m, 0.0, 0.0)}" rpy="0 0 0"/>
              <geometry><box size="{_format_xyz(params.body_half_length_m, params.body_width_m, params.body_height_m)}"/></geometry>
            </collision>
            <collision>
              <origin xyz="{_format_xyz(params.rear_body_center_x_m, 0.0, 0.0)}" rpy="0 0 0"/>
              <geometry><box size="{_format_xyz(params.body_half_length_m, params.body_width_m, params.body_height_m)}"/></geometry>
            </collision>
            <inertial>
              <origin xyz="{_format_xyz(params.body_com_x_m, 0.0, 0.0)}" rpy="0 0 0"/>
              <mass value="{params.total_body_mass_kg:.6f}"/>
              <inertia ixx="{body_ixx:.9f}" ixy="0" ixz="0" iyy="{body_iyy:.9f}" iyz="0" izz="{body_izz:.9f}"/>
            </inertial>
          </link>

          <link name="left_wheel">
            <visual>
              <origin xyz="0 0 0" rpy="0 1.57079632679 0"/>
              <geometry><mesh filename="../meshes/wheel.obj"/></geometry>
              <material name="wheel_black"/>
            </visual>
            <collision>
              <origin xyz="0 0 0" rpy="1.57079632679 0 0"/>
              <geometry><cylinder radius="{params.wheel_radius_m:.6f}" length="{params.wheel_thickness_m:.6f}"/></geometry>
            </collision>
            <inertial>
              <origin xyz="0 0 0" rpy="0 0 0"/>
              <mass value="{params.wheel_mass_kg:.6f}"/>
              <inertia ixx="{wheel_ixx:.9f}" ixy="0" ixz="0" iyy="{wheel_iyy:.9f}" iyz="0" izz="{wheel_izz:.9f}"/>
            </inertial>
          </link>

          <link name="right_wheel">
            <visual>
              <origin xyz="0 0 0" rpy="0 1.57079632679 0"/>
              <geometry><mesh filename="../meshes/wheel.obj"/></geometry>
              <material name="wheel_black"/>
            </visual>
            <collision>
              <origin xyz="0 0 0" rpy="1.57079632679 0 0"/>
              <geometry><cylinder radius="{params.wheel_radius_m:.6f}" length="{params.wheel_thickness_m:.6f}"/></geometry>
            </collision>
            <inertial>
              <origin xyz="0 0 0" rpy="0 0 0"/>
              <mass value="{params.wheel_mass_kg:.6f}"/>
              <inertia ixx="{wheel_ixx:.9f}" ixy="0" ixz="0" iyy="{wheel_iyy:.9f}" iyz="0" izz="{wheel_izz:.9f}"/>
            </inertial>
          </link>

          <link name="rear_support">
            <visual>
              <origin xyz="{_format_xyz(support_geom_offset_x, 0.0, 0.0)}" rpy="0 0 0"/>
              <geometry><box size="{_format_xyz(params.support_length_m, params.support_width_m, params.support_height_m)}"/></geometry>
              <material name="support_red"/>
            </visual>
            <collision>
              <origin xyz="{_format_xyz(support_geom_offset_x, 0.0, 0.0)}" rpy="0 0 0"/>
              <geometry><box size="{_format_xyz(params.support_length_m, params.support_width_m, params.support_height_m)}"/></geometry>
            </collision>
            <inertial>
              <origin xyz="{_format_xyz(support_geom_offset_x, 0.0, 0.0)}" rpy="0 0 0"/>
              <mass value="{params.support_mass_kg:.6f}"/>
              <inertia ixx="{support_ixx:.9f}" ixy="0" ixz="0" iyy="{support_iyy:.9f}" iyz="0" izz="{support_izz:.9f}"/>
            </inertial>
          </link>

          <joint name="base_to_chassis" type="fixed">
            <parent link="base_footprint"/>
            <child link="chassis"/>
            <origin xyz="{_format_xyz(0.0, 0.0, params.body_center_z_m)}" rpy="0 0 0"/>
          </joint>

          <joint name="left_wheel_joint" type="continuous">
            <parent link="chassis"/>
            <child link="left_wheel"/>
            <origin xyz="{_format_xyz(0.0, params.wheel_offset_y_m, wheel_z)}" rpy="0 0 0"/>
            <axis xyz="0 1 0"/>
          </joint>

          <joint name="right_wheel_joint" type="continuous">
            <parent link="chassis"/>
            <child link="right_wheel"/>
            <origin xyz="{_format_xyz(0.0, -params.wheel_offset_y_m, wheel_z)}" rpy="0 0 0"/>
            <axis xyz="0 1 0"/>
          </joint>

          <joint name="rear_support_joint" type="fixed">
            <parent link="chassis"/>
            <child link="rear_support"/>
            <origin xyz="{_format_xyz(params.support_offset_x_m, params.support_offset_y_m, support_z)}" rpy="0 0 0"/>
          </joint>
        </robot>
        '''
    )


def render_mjcf_car(params: CarParameters) -> str:
    wheel_z = params.wheel_center_z_m - params.body_center_z_m
    support_geom_offset_x = params.support_geom_offset_x_m
    rear_support_friction = _format_xyz(
        params.rear_support_friction_slide,
        params.rear_support_friction_torsion,
        params.rear_support_friction_roll,
    )

    return dedent(
        f'''\
        <mujoco model="little_car">
          <compiler angle="radian" autolimits="true"/>
          <default>
            <geom condim="4" solref="0.004 1" solimp="0.9 0.95 0.001" friction="1.2 0.04 0.01"/>
            <joint damping="0.01" armature="0.0001"/>
          </default>
          <worldbody>
            <body name="car" pos="0 0 {params.scene_spawn_height_m:.6f}">
              <freejoint name="car_freejoint"/>
              <inertial pos="{params.body_com_x_m:.6f} 0 {params.body_center_z_m:.6f}" mass="{params.total_body_mass_kg:.6f}"
                        diaginertia="{_format_xyz(*params.body_inertia)}"/>
              <geom name="front_chassis_collision" type="box" pos="{params.front_body_center_x_m:.6f} 0 {params.body_center_z_m:.6f}"
                    size="{_format_xyz(params.body_half_length_m / 2.0, params.body_width_m / 2.0, params.body_height_m / 2.0)}"
                    rgba="0.3 0.3 0.7 1"/>
              <geom name="rear_chassis_collision" type="box" pos="{params.rear_body_center_x_m:.6f} 0 {params.body_center_z_m:.6f}"
                    size="{_format_xyz(params.body_half_length_m / 2.0, params.body_width_m / 2.0, params.body_height_m / 2.0)}"
                    rgba="0.25 0.25 0.6 1"/>
              <geom name="front_marker" type="box" pos="{params.body_half_length_m - 0.001:.6f} 0 {params.body_center_z_m + params.body_height_m * 0.2:.6f}"
                    size="0.002000 0.015000 0.010000"
                    rgba="1.0 0.2 0.2 1" contype="0" conaffinity="0"/>

              <body name="rear_support_body" pos="{_format_xyz(params.support_offset_x_m, params.support_offset_y_m, params.support_center_z_m)}">
                <joint name="rear_caster_joint" type="ball" damping="0.0001"/>
                <inertial pos="0 0 0" mass="{params.support_mass_kg / 2.0:.6f}"
                          diaginertia="0.000001 0.000001 0.000001"/>
                <geom name="rear_support" type="sphere"
                      pos="0 0 0"
                      size="{params.support_height_m / 2.0:.6f}"
                      friction="{rear_support_friction}"
                      rgba="0.7 0.1 0.1 1"/>
              </body>

              <body name="front_support_body" pos="{_format_xyz(-params.support_offset_x_m, params.support_offset_y_m, params.support_center_z_m)}">
                <joint name="front_caster_joint" type="ball" damping="0.0001"/>
                <inertial pos="0 0 0" mass="{params.support_mass_kg / 2.0:.6f}"
                          diaginertia="0.000001 0.000001 0.000001"/>
                <geom name="front_support" type="sphere"
                      pos="0 0 0"
                      size="{params.support_height_m / 2.0:.6f}"
                      friction="{rear_support_friction}"
                      rgba="0.1 0.7 0.1 1"/>
              </body>

              <body name="left_wheel" pos="{_format_xyz(0.0, params.wheel_offset_y_m, params.body_center_z_m + wheel_z)}">
                <inertial pos="0 0 0" mass="{params.wheel_mass_kg:.6f}"
                          diaginertia="{_format_xyz(*params.wheel_inertia)}"/>
                <joint name="left_wheel_joint" type="hinge" axis="0 1 0"/>
                <geom name="left_wheel_collision" type="cylinder" size="{params.wheel_radius_m:.6f} {params.wheel_thickness_m / 2.0:.6f}"
                      euler="1.57079632679 0 0" rgba="0.1 0.1 0.1 1"/>
              </body>

              <body name="right_wheel" pos="{_format_xyz(0.0, -params.wheel_offset_y_m, params.body_center_z_m + wheel_z)}">
                <inertial pos="0 0 0" mass="{params.wheel_mass_kg:.6f}"
                          diaginertia="{_format_xyz(*params.wheel_inertia)}"/>
                <joint name="right_wheel_joint" type="hinge" axis="0 1 0"/>
                <geom name="right_wheel_collision" type="cylinder" size="{params.wheel_radius_m:.6f} {params.wheel_thickness_m / 2.0:.6f}"
                      euler="1.57079632679 0 0" rgba="0.1 0.1 0.1 1"/>
              </body>
            </body>
          </worldbody>
          <actuator>
            <velocity name="left_wheel_drive" joint="left_wheel_joint"
                      ctrlrange="{-params.drive_ctrl_limit_rad_s:.6f} {params.drive_ctrl_limit_rad_s:.6f}"
                      kv="{params.drive_kv:.6f}" forcerange="{-params.drive_force_limit_n_m:.6f} {params.drive_force_limit_n_m:.6f}"/>
            <velocity name="right_wheel_drive" joint="right_wheel_joint"
                      ctrlrange="{-params.drive_ctrl_limit_rad_s:.6f} {params.drive_ctrl_limit_rad_s:.6f}"
                      kv="{params.drive_kv:.6f}" forcerange="{-params.drive_force_limit_n_m:.6f} {params.drive_force_limit_n_m:.6f}"/>
          </actuator>
        </mujoco>
        '''
    )


def render_mjcf_scene() -> str:
    return dedent(
        '''\
        <mujoco model="little_car_scene">
          <compiler angle="radian"/>
          <include file="little_car.xml"/>
          <option gravity="0 0 -9.81" timestep="0.002" integrator="implicitfast" iterations="100"/>
          <visual>
            <global offwidth="960" offheight="720"/>
            <headlight ambient="0.5 0.5 0.5" diffuse="0.6 0.6 0.6" specular="0.1 0.1 0.1"/>
          </visual>
          <worldbody>
            <light name="sun" pos="0 0 2" dir="0 0 -1" directional="true"/>
            <geom name="ground" type="plane" size="2 2 0.1" rgba="0.85 0.85 0.85 1" friction="1.3 0.05 0.01"/>
            <camera name="overview" pos="0.35 -0.35 0.22" xyaxes="0.707107 0.707107 0 -0.3 0.3 0.905539"/>
            <camera name="forward_is_positive_x" pos="-0.20 -0.22 0.18" xyaxes="0.739940 -0.672673 0 0.380157 0.418173 0.825336"/>
            <camera name="square_path_capture" pos="0.22 -0.26 0.30" xyaxes="0.766044 0.642788 0 -0.454519 0.541675 0.707107"/>
          </worldbody>
        </mujoco>
        '''
    )



def generate_all(root: Path, params: CarParameters | None = None) -> dict[str, Path]:
    params = params or default_parameters()
    mesh_dir = root / "meshes"
    urdf_dir = root / "urdf"
    mjcf_dir = root / "mjcf"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    urdf_dir.mkdir(parents=True, exist_ok=True)
    mjcf_dir.mkdir(parents=True, exist_ok=True)

    chassis_path = mesh_dir / "chassis_top.obj"
    wheel_path = mesh_dir / "wheel.obj"
    urdf_path = urdf_dir / "little_car.urdf"
    mjcf_car_path = mjcf_dir / "little_car.xml"
    mjcf_scene_path = mjcf_dir / "little_car_scene.xml"

    write_obj(chassis_path, *box_mesh(params.body_length_m, params.body_width_m, params.body_height_m))
    write_obj(wheel_path, *wheel_prism_mesh(params.wheel_radius_m, params.wheel_thickness_m))
    urdf_path.write_text(render_urdf(params), encoding="utf-8")
    mjcf_car_path.write_text(render_mjcf_car(params), encoding="utf-8")
    mjcf_scene_path.write_text(render_mjcf_scene(), encoding="utf-8")

    return {
        "chassis_mesh": chassis_path,
        "wheel_mesh": wheel_path,
        "urdf": urdf_path,
        "mjcf_car": mjcf_car_path,
        "mjcf_scene": mjcf_scene_path,
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    outputs = generate_all(project_root / "assets")
    for name, path in outputs.items():
        print(f"{name}: {path}")
