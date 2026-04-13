from __future__ import annotations

from pathlib import Path

Vertex = tuple[float, float, float]
Face = tuple[int, int, int]



def write_obj(path: Path, vertices: list[Vertex], faces: list[Face]) -> None:
    lines: list[str] = []
    for x, y, z in vertices:
        lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
    for a, b, c in faces:
        lines.append(f"f {a} {b} {c}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def box_mesh(size_x: float, size_y: float, size_z: float) -> tuple[list[Vertex], list[Face]]:
    hx, hy, hz = size_x / 2.0, size_y / 2.0, size_z / 2.0
    vertices = [
        (-hx, -hy, -hz),
        (hx, -hy, -hz),
        (hx, hy, -hz),
        (-hx, hy, -hz),
        (-hx, -hy, hz),
        (hx, -hy, hz),
        (hx, hy, hz),
        (-hx, hy, hz),
    ]
    faces = [
        (1, 2, 3), (1, 3, 4),
        (5, 8, 7), (5, 7, 6),
        (1, 5, 6), (1, 6, 2),
        (2, 6, 7), (2, 7, 3),
        (3, 7, 8), (3, 8, 4),
        (4, 8, 5), (4, 5, 1),
    ]
    return vertices, faces



def wheel_prism_mesh(radius: float, thickness: float, segments: int = 32) -> tuple[list[Vertex], list[Face]]:
    import math
    half_t = thickness / 2.0

    # Generate circle vertices on left and right faces
    vertices: list[Vertex] = []
    for side_x in (-half_t, half_t):
        for i in range(segments):
            angle = 2.0 * math.pi * i / segments
            vertices.append((side_x, radius * math.sin(angle), radius * math.cos(angle)))
    # Center vertices for end caps (fan triangulation)
    left_center_idx = len(vertices) + 1   # 1-indexed for OBJ
    vertices.append((-half_t, 0.0, 0.0))
    right_center_idx = len(vertices) + 1
    vertices.append((half_t, 0.0, 0.0))

    faces: list[Face] = []
    # Left cap (vertices 1..segments, center = left_center_idx)
    for i in range(segments):
        v0 = i + 1
        v1 = (i + 1) % segments + 1
        faces.append((left_center_idx, v1, v0))
    # Right cap (vertices segments+1..2*segments, center = right_center_idx)
    for i in range(segments):
        v0 = segments + i + 1
        v1 = segments + (i + 1) % segments + 1
        faces.append((right_center_idx, v0, v1))
    # Side quads (two triangles each)
    for i in range(segments):
        l0 = i + 1
        l1 = (i + 1) % segments + 1
        r0 = segments + i + 1
        r1 = segments + (i + 1) % segments + 1
        faces.append((l0, r0, r1))
        faces.append((l0, r1, l1))

    return vertices, faces
