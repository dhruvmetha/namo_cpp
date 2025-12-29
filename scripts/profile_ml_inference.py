#!/usr/bin/env python3
"""Profile ML inference pipeline to identify actual bottlenecks.

Run with:
    PYTHONPATH=./build_python_mjxrl_arrakis:./python:$PYTHONPATH python scripts/profile_ml_inference.py
"""

import time
import sys
import os

# Add paths
sys.path.insert(0, "/common/home/dm1487/robotics_research/ktamp/sage_learning")

import numpy as np
from collections import defaultdict


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name, timings):
        self.name = name
        self.timings = timings

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = (time.perf_counter() - self.start) * 1000  # ms
        self.timings[self.name].append(elapsed)


def profile_mask_generation():
    """Profile the mask generation pipeline."""
    from namo.visualization.mask_generation.visualizer import NAMODataVisualizer
    from collections import deque
    import cv2

    timings = defaultdict(list)

    # Create a representative episode_data
    episode_data = {
        'state_observations': [{
            'robot_pose': [0.0, 0.0, 0.0],
            'movable_box_0_pose': [2.0, 1.0, 0.1],
            'movable_box_1_pose': [3.0, -1.0, 0.2],
            'movable_box_2_pose': [-2.0, 2.0, -0.1],
        }],
        'static_object_info': {
            'robot': {'size_x': 0.15, 'size_y': 0.15},
            'movable_box_0': {'size_x': 0.3, 'size_y': 0.3},
            'movable_box_1': {'size_x': 0.25, 'size_y': 0.4},
            'movable_box_2': {'size_x': 0.35, 'size_y': 0.35},
            'wall_north': {'size_x': 7.0, 'size_y': 0.1, 'pos_x': 0.0, 'pos_y': 7.0, 'pos_z': 0.0,
                          'quat_w': 1.0, 'quat_x': 0.0, 'quat_y': 0.0, 'quat_z': 0.0},
            'wall_south': {'size_x': 7.0, 'size_y': 0.1, 'pos_x': 0.0, 'pos_y': -7.0, 'pos_z': 0.0,
                          'quat_w': 1.0, 'quat_x': 0.0, 'quat_y': 0.0, 'quat_z': 0.0},
            'wall_east': {'size_x': 0.1, 'size_y': 7.0, 'pos_x': 7.0, 'pos_y': 0.0, 'pos_z': 0.0,
                         'quat_w': 1.0, 'quat_x': 0.0, 'quat_y': 0.0, 'quat_z': 0.0},
            'wall_west': {'size_x': 0.1, 'size_y': 7.0, 'pos_x': -7.0, 'pos_y': 0.0, 'pos_z': 0.0,
                         'quat_w': 1.0, 'quat_x': 0.0, 'quat_y': 0.0, 'quat_z': 0.0},
        },
        'action_sequence': [{'object_id': 'movable_box_0', 'target': (3.0, 2.0, 0.0)}],
        'robot_goal': (5.0, 5.0, 0.0),
        'world_bounds': (-7.5, 7.5, -7.5, 7.5),
        'algorithm_stats': {'region_goals_sampled': [(5.0, 5.0, 0.0)]},
    }

    print("=" * 60)
    print("PROFILING MASK GENERATION PIPELINE")
    print("=" * 60)

    n_runs = 5

    for run in range(n_runs):
        with Timer("total", timings):
            with Timer("visualizer_init", timings):
                visualizer = NAMODataVisualizer()

            with Timer("generate_all_masks_highres", timings):
                result = visualizer.generate_all_masks_highres(
                    episode_data,
                    highres_size=1024,
                    global_output_size=224,
                    local_output_size=224,
                    local_crop_size_meters=5.0,
                )

    # Print results
    print("\nTiming Results (averaged over {} runs):".format(n_runs))
    print("-" * 50)
    for name, times in sorted(timings.items(), key=lambda x: -np.mean(x[1])):
        mean_ms = np.mean(times)
        std_ms = np.std(times)
        print(f"  {name:40s}: {mean_ms:8.2f} ms ± {std_ms:.2f}")

    return timings


def profile_bfs_variants():
    """Compare Python BFS vs potential optimizations."""
    from collections import deque
    import numpy as np

    print("\n" + "=" * 60)
    print("PROFILING BFS IMPLEMENTATIONS")
    print("=" * 60)

    # Create test obstacle grid (1024x1024)
    size = 1024
    obstacles = np.zeros((size, size), dtype=np.uint8)

    # Add some obstacles
    obstacles[100:200, 100:200] = 1
    obstacles[400:600, 300:500] = 1
    obstacles[700:800, 600:900] = 1

    # BFS directions
    DIRECTIONS = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]

    def python_bfs(obstacles, start_x, start_y):
        """Standard Python BFS."""
        visited = np.zeros_like(obstacles, dtype=np.uint8)
        queue = deque([(start_x, start_y)])
        visited[start_y, start_x] = 1
        cells_visited = 0

        while queue:
            cx, cy = queue.popleft()
            cells_visited += 1
            for dx, dy in DIRECTIONS:
                nx, ny = cx + dx, cy + dy
                if (0 <= nx < size and 0 <= ny < size and
                    visited[ny, nx] == 0 and obstacles[ny, nx] == 0):
                    visited[ny, nx] = 1
                    queue.append((nx, ny))

        return visited, cells_visited

    def scipy_bfs(obstacles, start_x, start_y):
        """Use scipy.ndimage.label for connected components."""
        from scipy import ndimage

        # Create free space mask
        free_space = (obstacles == 0).astype(np.int32)

        # Label connected components
        labeled, num_features = ndimage.label(free_space, structure=np.ones((3, 3)))

        # Find which label contains the start point
        start_label = labeled[start_y, start_x]

        # Create visited mask for that component
        visited = (labeled == start_label).astype(np.uint8)
        cells_visited = np.sum(visited)

        return visited, cells_visited

    # Warmup
    for _ in range(2):
        python_bfs(obstacles, 50, 50)
        scipy_bfs(obstacles, 50, 50)

    # Profile Python BFS
    n_runs = 10
    python_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        visited, cells = python_bfs(obstacles, 50, 50)
        python_times.append((time.perf_counter() - start) * 1000)

    print(f"\nPython deque BFS:")
    print(f"  Time: {np.mean(python_times):.2f} ms ± {np.std(python_times):.2f}")
    print(f"  Cells visited: {cells:,}")

    # Profile scipy
    scipy_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        visited, cells = scipy_bfs(obstacles, 50, 50)
        scipy_times.append((time.perf_counter() - start) * 1000)

    print(f"\nScipy ndimage.label:")
    print(f"  Time: {np.mean(scipy_times):.2f} ms ± {np.std(scipy_times):.2f}")
    print(f"  Cells visited: {cells:,}")

    # Try numba if available
    try:
        import numba
        from numba import njit

        @njit(cache=True)
        def numba_bfs(obstacles, start_x, start_y):
            """Numba-accelerated BFS."""
            size = obstacles.shape[0]
            visited = np.zeros_like(obstacles, dtype=np.uint8)

            # Use fixed-size queue (pre-allocated)
            queue_x = np.zeros(size * size, dtype=np.int32)
            queue_y = np.zeros(size * size, dtype=np.int32)
            front, back = 0, 0

            # Enqueue start
            queue_x[back] = start_x
            queue_y[back] = start_y
            back += 1
            visited[start_y, start_x] = 1
            cells_visited = 0

            # 8-connected directions
            dx = np.array([1, -1, 0, 0, 1, 1, -1, -1], dtype=np.int32)
            dy = np.array([0, 0, 1, -1, 1, -1, 1, -1], dtype=np.int32)

            while front < back:
                cx = queue_x[front]
                cy = queue_y[front]
                front += 1
                cells_visited += 1

                for i in range(8):
                    nx = cx + dx[i]
                    ny = cy + dy[i]
                    if (0 <= nx < size and 0 <= ny < size and
                        visited[ny, nx] == 0 and obstacles[ny, nx] == 0):
                        visited[ny, nx] = 1
                        queue_x[back] = nx
                        queue_y[back] = ny
                        back += 1

            return visited, cells_visited

        # Compile
        numba_bfs(obstacles, 50, 50)

        # Profile
        numba_times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            visited, cells = numba_bfs(obstacles, 50, 50)
            numba_times.append((time.perf_counter() - start) * 1000)

        print(f"\nNumba JIT BFS:")
        print(f"  Time: {np.mean(numba_times):.2f} ms ± {np.std(numba_times):.2f}")
        print(f"  Cells visited: {cells:,}")
        print(f"\nSpeedup vs Python: {np.mean(python_times) / np.mean(numba_times):.1f}x")

    except ImportError:
        print("\nNumba not available - install with: pip install numba")

    print("\n" + "=" * 60)


def profile_cv2_operations():
    """Profile cv2 drawing operations."""
    import cv2
    import numpy as np

    print("\n" + "=" * 60)
    print("PROFILING CV2 OPERATIONS")
    print("=" * 60)

    n_runs = 100
    size = 1024

    # Profile fillPoly
    mask = np.zeros((size, size), dtype=np.float32)
    box = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.int32)

    times = []
    for _ in range(n_runs):
        mask.fill(0)
        start = time.perf_counter()
        cv2.fillPoly(mask, [box], 1.0)
        times.append((time.perf_counter() - start) * 1000)

    print(f"\ncv2.fillPoly (single box):")
    print(f"  Time: {np.mean(times)*1000:.3f} μs ± {np.std(times)*1000:.3f}")

    # Profile multiple boxes (typical scene has 20-40 objects)
    n_objects = 30
    boxes = []
    for i in range(n_objects):
        x = (i * 30) % 900 + 50
        y = (i * 40) % 900 + 50
        boxes.append(np.array([[x, y], [x+30, y], [x+30, y+30], [x, y+30]], dtype=np.int32))

    times = []
    for _ in range(n_runs):
        mask.fill(0)
        start = time.perf_counter()
        for box in boxes:
            cv2.fillPoly(mask, [box], 1.0)
        times.append((time.perf_counter() - start) * 1000)

    print(f"\ncv2.fillPoly ({n_objects} boxes):")
    print(f"  Time: {np.mean(times):.3f} ms ± {np.std(times):.3f}")

    # Profile batch fillPoly
    times = []
    for _ in range(n_runs):
        mask.fill(0)
        start = time.perf_counter()
        cv2.fillPoly(mask, boxes, 1.0)  # All at once
        times.append((time.perf_counter() - start) * 1000)

    print(f"\ncv2.fillPoly (batched {n_objects} boxes):")
    print(f"  Time: {np.mean(times):.3f} ms ± {np.std(times):.3f}")


if __name__ == "__main__":
    profile_mask_generation()
    profile_bfs_variants()
    profile_cv2_operations()
