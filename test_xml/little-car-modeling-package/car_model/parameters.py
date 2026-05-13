from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CarParameters:
    body_size_m: float = 0.07
    body_height_m: float = 0.065       # 6.5cm: total 7cm minus 0.5cm ground clearance
    wheel_diameter_m: float = 0.03     # 3cm diameter (1.5cm radius)
    wheel_thickness_m: float = 0.001   # 0.1cm thin disc wheels
    wheel_clearance_m: float = 0.002
    body_bottom_clearance_m: float = 0.005
    support_height_m: float = 0.005
    support_length_m: float = 0.008
    support_width_m: float = 0.016
    support_offset_x_m: float = -0.030  # inward from chassis edge (half-length=0.035)
    support_offset_y_m: float = 0.0
    body_mass_kg: float = 0.35
    wheel_mass_kg: float = 0.05
    support_mass_kg: float = 0.05
    scene_spawn_height_m: float = 0.01     # smaller drop for 1.5cm radius wheels
    drive_ctrl_limit_rad_s: float = 25.0
    drive_kv: float = 0.75
    drive_force_limit_n_m: float = 0.5
    rear_support_friction_slide: float = 0.0
    rear_support_friction_torsion: float = 0.0
    rear_support_friction_roll: float = 0.0

    @property
    def body_length_m(self) -> float:
        return self.body_size_m

    @property
    def body_width_m(self) -> float:
        return self.body_size_m

    @property
    def wheel_radius_m(self) -> float:
        return self.wheel_diameter_m / 2.0

    @property
    def wheel_offset_y_m(self) -> float:
        return self.body_width_m / 2.0 + self.wheel_clearance_m + self.wheel_thickness_m / 2.0

    @property
    def wheel_center_z_m(self) -> float:
        return self.wheel_radius_m

    @property
    def body_center_z_m(self) -> float:
        return self.body_bottom_clearance_m + self.body_height_m / 2.0

    @property
    def body_top_z_m(self) -> float:
        return self.body_bottom_clearance_m + self.body_height_m

    @property
    def body_half_length_m(self) -> float:
        return self.body_length_m / 2.0

    @property
    def front_body_center_x_m(self) -> float:
        return self.body_length_m / 4.0

    @property
    def rear_body_center_x_m(self) -> float:
        return -self.body_length_m / 4.0

    @property
    def front_body_mass_kg(self) -> float:
        return self.body_mass_kg / 2.0

    @property
    def rear_body_mass_kg(self) -> float:
        return self.body_mass_kg / 2.0

    @property
    def total_body_mass_kg(self) -> float:
        return self.front_body_mass_kg + self.rear_body_mass_kg

    @property
    def body_com_x_m(self) -> float:
        return (
            self.front_body_mass_kg * self.front_body_center_x_m
            + self.rear_body_mass_kg * self.rear_body_center_x_m
        ) / self.total_body_mass_kg

    @property
    def support_center_z_m(self) -> float:
        return self.support_height_m / 2.0

    @property
    def support_top_z_m(self) -> float:
        return self.support_height_m

    @property
    def body_rear_x_m(self) -> float:
        return -self.body_half_length_m

    @property
    def support_geom_offset_x_m(self) -> float:
        return self.body_rear_x_m - self.support_offset_x_m + self.support_length_m / 2.0

    @property
    def support_rear_face_x_m(self) -> float:
        return self.support_offset_x_m + self.support_geom_offset_x_m - self.support_length_m / 2.0

    @property
    def support_front_face_x_m(self) -> float:
        return self.support_offset_x_m + self.support_geom_offset_x_m + self.support_length_m / 2.0

    @property
    def support_within_body_footprint(self) -> bool:
        return self.support_rear_face_x_m >= self.body_rear_x_m and self.support_front_face_x_m <= self.body_half_length_m

    @property
    def body_inertia(self) -> tuple[float, float, float]:
        front_ixx, front_iyy, front_izz = self.front_body_inertia
        rear_ixx, rear_iyy, rear_izz = self.rear_body_inertia
        front_dx = self.front_body_center_x_m - self.body_com_x_m
        rear_dx = self.rear_body_center_x_m - self.body_com_x_m
        return (
            front_ixx + rear_ixx,
            front_iyy + self.front_body_mass_kg * front_dx * front_dx + rear_iyy + self.rear_body_mass_kg * rear_dx * rear_dx,
            front_izz + self.front_body_mass_kg * front_dx * front_dx + rear_izz + self.rear_body_mass_kg * rear_dx * rear_dx,
        )

    def _box_inertia(self, x: float, y: float, z: float, mass: float) -> tuple[float, float, float]:
        return (
            mass * (y * y + z * z) / 12.0,
            mass * (x * x + z * z) / 12.0,
            mass * (x * x + y * y) / 12.0,
        )

    @property
    def front_body_inertia(self) -> tuple[float, float, float]:
        return self._box_inertia(
            self.body_half_length_m,
            self.body_width_m,
            self.body_height_m,
            self.front_body_mass_kg,
        )

    @property
    def rear_body_inertia(self) -> tuple[float, float, float]:
        return self._box_inertia(
            self.body_half_length_m,
            self.body_width_m,
            self.body_height_m,
            self.rear_body_mass_kg,
        )

    @property
    def support_inertia(self) -> tuple[float, float, float]:
        return self._box_inertia(
            self.support_length_m,
            self.support_width_m,
            self.support_height_m,
            self.support_mass_kg,
        )

    @property
    def wheel_inertia(self) -> tuple[float, float, float]:
        radius = self.wheel_radius_m
        length = self.wheel_thickness_m
        mass = self.wheel_mass_kg
        i_axial = 0.5 * mass * radius * radius
        i_radial = mass * (3.0 * radius * radius + length * length) / 12.0
        return (i_radial, i_axial, i_radial)



def default_parameters() -> CarParameters:
    return CarParameters()
