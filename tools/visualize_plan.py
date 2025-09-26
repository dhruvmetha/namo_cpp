#!/usr/bin/env python3
"""
Visualize greedy planner output showing how primitives get transformed and chained
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
import argparse
import json
import math
import sys
import struct
import os

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

def normalize_angle(angle):
    """Normalize angle to [-π, π]"""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle

def transform_primitive(primitive, current_theta):
    """Transform a primitive vector by current object orientation"""
    dx, dy, dtheta = primitive
    cos_theta = math.cos(current_theta)
    sin_theta = math.sin(current_theta)

    transformed_dx = dx * cos_theta - dy * sin_theta
    transformed_dy = dx * sin_theta + dy * cos_theta

    return transformed_dx, transformed_dy, dtheta

def read_primitives_dat(filename):
    """Read motion primitives from binary .dat file (same as visualize_primitives.py)"""
    primitives = []

    if not os.path.exists(filename):
        print(f"Warning: Primitive file {filename} not found")
        return primitives

    with open(filename, 'rb') as f:
        # Read header (number of primitives)
        count_bytes = f.read(4)
        if len(count_bytes) < 4:
            raise ValueError("File too short to contain valid header")

        count = struct.unpack('I', count_bytes)[0]  # uint32_t
        print(f"Loading {count} primitives from {filename}")

        # Read primitives (each is 14 bytes: float, float, float, uint8, uint8)
        for i in range(count):
            data = f.read(14)
            if len(data) < 14:
                print(f"Warning: Incomplete primitive {i}, only got {len(data)} bytes")
                break

            # Unpack: delta_x, delta_y, delta_theta, edge_idx, push_steps
            delta_x, delta_y, delta_theta, edge_idx, push_steps = struct.unpack('fffBB', data)

            primitives.append({
                'delta_x': delta_x,
                'delta_y': delta_y,
                'delta_theta': delta_theta,
                'edge_idx': edge_idx,
                'push_steps': push_steps
            })

    return primitives

def plot_plan_visualization_interactive(start_state, goal_state, selected_primitives, all_primitives, output_file="interactive_plan.html"):
    """
    Create interactive 3D visualization using plotly
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available, falling back to matplotlib")
        return plot_plan_visualization(start_state, goal_state, selected_primitives, all_primitives, output_file)

    n_steps = len(selected_primitives) + 1
    cols = min(n_steps, 4)
    rows = (n_steps + cols - 1) // cols

    # Create subplot titles
    subplot_titles = [f'Step {i}: Action Space Evolution' for i in range(n_steps)]

    fig = make_subplots(
        rows=rows, cols=cols,
        specs=[[{'type': 'scatter3d'} for _ in range(cols)] for _ in range(rows)],
        subplot_titles=subplot_titles,
        horizontal_spacing=0.05,
        vertical_spacing=0.1
    )

    step_colors = ['green', 'blue', 'red', 'black']

    for plot_step in range(n_steps):
        row = plot_step // cols + 1
        col = plot_step % cols + 1

        # Start and goal points
        fig.add_trace(go.Scatter3d(
            x=[start_state[0]], y=[start_state[1]], z=[start_state[2]],
            mode='markers',
            marker=dict(size=8, color='green', symbol='circle'),
            name='Start' if plot_step == 0 else None,
            showlegend=(plot_step == 0),
            legendgroup='start'
        ), row=row, col=col)

        fig.add_trace(go.Scatter3d(
            x=[goal_state[0]], y=[goal_state[1]], z=[goal_state[2]],
            mode='markers',
            marker=dict(size=8, color='red', symbol='circle'),
            name='Goal' if plot_step == 0 else None,
            showlegend=(plot_step == 0),
            legendgroup='goal'
        ), row=row, col=col)

        # Plot primitive clouds and path
        current_pose = list(start_state)
        path_x = [current_pose[0]]
        path_y = [current_pose[1]]
        path_z = [current_pose[2]]

        for step in range(plot_step + 1):
            color = step_colors[step % len(step_colors)]
            opacity = 0.7 if step == plot_step else 0.3
            size = 3 if step == plot_step else 2

            if all_primitives:
                endpoints_x, endpoints_y, endpoints_z = [], [], []

                for prim in all_primitives:
                    tdx, tdy, tdtheta = transform_primitive([prim['delta_x'], prim['delta_y'], prim['delta_theta']], current_pose[2])
                    endpoints_x.append(current_pose[0] + tdx)
                    endpoints_y.append(current_pose[1] + tdy)
                    endpoints_z.append(normalize_angle(current_pose[2] + prim['delta_theta']))

                fig.add_trace(go.Scatter3d(
                    x=endpoints_x, y=endpoints_y, z=endpoints_z,
                    mode='markers',
                    marker=dict(size=size, color=color, opacity=opacity),
                    name=f'Step {step} Actions' if plot_step == n_steps-1 and step < 4 else None,
                    showlegend=(plot_step == n_steps-1 and step < 4),
                    legendgroup=f'step{step}'
                ), row=row, col=col)

            # Update pose for next iteration
            if step < len(selected_primitives):
                prim_data = selected_primitives[step]
                primitive = prim_data['primitive']
                tdx, tdy, tdtheta = transform_primitive(primitive, current_pose[2])
                current_pose[0] += tdx
                current_pose[1] += tdy
                current_pose[2] = normalize_angle(current_pose[2] + primitive[2])
                path_x.append(current_pose[0])
                path_y.append(current_pose[1])
                path_z.append(current_pose[2])

        # Plot path
        if len(path_x) > 1:
            fig.add_trace(go.Scatter3d(
                x=path_x, y=path_y, z=path_z,
                mode='lines+markers',
                line=dict(color='black', width=4),
                marker=dict(size=3, color='black'),
                name='Path' if plot_step == 0 else None,
                showlegend=(plot_step == 0),
                legendgroup='path'
            ), row=row, col=col)

    fig.update_layout(
        title_text="Greedy Planner: Interactive 3D Action Space Evolution",
        height=400*rows,
        width=400*cols
    )

    # Update axis labels for all subplots
    for i in range(1, rows*cols + 1):
        fig.update_layout(**{f'scene{i}': dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Theta (rad)'
        )})

    if output_file:
        pyo.plot(fig, filename=output_file, auto_open=False)
        print(f"Interactive visualization saved to {output_file}")
        return output_file
    else:
        pyo.plot(fig)

def plot_plan_visualization(start_state, goal_state, selected_primitives, all_primitives, output_file=None):
    """
    Create visualization of the plan showing:
    1. All primitives from database with selected ones highlighted
    2. Step-by-step path evolution with transformations
    3. Object poses at each step in 3D (x, y, theta)
    """

    # Create subplot grid for step-by-step evolution
    n_steps = len(selected_primitives) + 1  # +1 for initial state
    cols = min(n_steps, 4)  # Max 4 columns
    rows = (n_steps + cols - 1) // cols  # Ceiling division

    fig = plt.figure(figsize=(5*cols, 5*rows))
    fig.suptitle('Greedy Planner: Action Space Evolution Step by Step (3D)', fontsize=16)

    # Colors for primitive clouds at each step
    step_colors = ['green', 'blue', 'red', 'black']

    # Create each subplot showing cumulative evolution
    for plot_step in range(n_steps):
        subplot_idx = plot_step + 1
        ax = fig.add_subplot(rows, cols, subplot_idx, projection='3d')

        ax.set_title(f'Step {plot_step}: Action Space Evolution')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Theta (rad)')
        ax.grid(True, alpha=0.3)

        # Plot start and goal in 3D
        ax.scatter([start_state[0]], [start_state[1]], [start_state[2]], c='green', s=100, marker='o',
                  label='Start' if plot_step == 0 else None, alpha=1.0, edgecolors='black')
        ax.scatter([goal_state[0]], [goal_state[1]], [goal_state[2]], c='red', s=100, marker='o',
                  label='Goal' if plot_step == 0 else None, alpha=1.0, edgecolors='black')

        # Show cumulative action space evolution up to current plot_step
        current_pose = list(start_state)
        path_x = [current_pose[0]]
        path_y = [current_pose[1]]
        path_z = [current_pose[2]]

        # Plot primitive clouds for each step up to plot_step
        for step in range(plot_step + 1):
            # Use different colors for each step's primitives
            color = step_colors[step % len(step_colors)]
            alpha = 0.4 if step == plot_step else 0.2  # Highlight current step
            size = 10 if step == plot_step else 6

            if all_primitives:
                # Plot all primitive endpoints when applied from current_pose
                endpoints_x = []
                endpoints_y = []
                endpoints_z = []

                for prim in all_primitives:
                    # Transform primitive based on current orientation
                    tdx, tdy, tdtheta = transform_primitive([prim['delta_x'], prim['delta_y'], prim['delta_theta']], current_pose[2])

                    # Calculate endpoint
                    end_x = current_pose[0] + tdx
                    end_y = current_pose[1] + tdy
                    end_z = normalize_angle(current_pose[2] + prim['delta_theta'])

                    endpoints_x.append(end_x)
                    endpoints_y.append(end_y)
                    endpoints_z.append(end_z)

                # Plot all endpoints in 3D
                ax.scatter(endpoints_x, endpoints_y, endpoints_z, c=color, alpha=alpha, s=size,
                          label=f'Step {step} Actions' if plot_step == n_steps-1 and step < 4 else None, edgecolors='none')

            # Update pose for next iteration (if not the last step)
            if step < len(selected_primitives):
                prim_data = selected_primitives[step]
                primitive = prim_data['primitive']
                tdx, tdy, tdtheta = transform_primitive(primitive, current_pose[2])

                current_pose[0] += tdx
                current_pose[1] += tdy
                current_pose[2] = normalize_angle(current_pose[2] + primitive[2])

                # Add to path
                path_x.append(current_pose[0])
                path_y.append(current_pose[1])
                path_z.append(current_pose[2])

        # Plot the path up to current step (thin line) in 3D
        if len(path_x) > 1:
            ax.plot(path_x, path_y, path_z, 'k-', linewidth=2, alpha=0.6, label='Path' if plot_step == 0 else None)

        # Plot position markers in 3D
        temp_pose = list(start_state)
        for i in range(min(plot_step, len(selected_primitives))):
            ax.scatter([temp_pose[0]], [temp_pose[1]], [temp_pose[2]], c='black', s=36, alpha=0.8, marker='o')

            prim_data = selected_primitives[i]
            primitive = prim_data['primitive']
            tdx, tdy, tdtheta = transform_primitive(primitive, temp_pose[2])
            temp_pose[0] += tdx
            temp_pose[1] += tdy
            temp_pose[2] = normalize_angle(temp_pose[2] + primitive[2])

        # Final position marker if we've reached the end
        if plot_step == len(selected_primitives):
            ax.scatter([temp_pose[0]], [temp_pose[1]], [temp_pose[2]], c='black', s=64, marker='s',
                   alpha=1.0, edgecolors='black', linewidths=1,
                   label='Final' if plot_step == n_steps-1 else None)

        # Add legend only to the last subplot
        if plot_step == n_steps - 1:
            ax.legend(loc='upper right', fontsize=8)

    # Hide unused subplots
    for i in range(n_steps, rows * cols):
        subplot_idx = i + 1
        ax = fig.add_subplot(rows, cols, subplot_idx)
        ax.axis('off')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {output_file}")
    else:
        plt.show()

def parse_json_from_text(text):
    """Extract JSON data from mixed text output"""
    lines = text.split('\n')
    json_start = -1
    json_end = -1

    for i, line in enumerate(lines):
        if "=== PLAN DATA (JSON FORMAT) ===" in line:
            json_start = i + 1
        elif "=== END PLAN DATA ===" in line:
            json_end = i
            break

    if json_start == -1 or json_end == -1:
        raise ValueError("Could not find JSON data in input")

    json_lines = lines[json_start:json_end]
    json_text = '\n'.join(json_lines)

    return json.loads(json_text)

def load_plan_from_json(filename, primitives_file="data/motion_primitives_15_square.dat"):
    """Load plan data from JSON file and primitives from .dat file"""
    with open(filename, 'r') as f:
        data = json.load(f)

    start_state = data['start_state']
    goal_state = data['goal_state']
    selected_primitives = data['selected_primitives']
    primitive_steps = data['primitive_steps']

    # Load all primitives from .dat file
    all_primitives = read_primitives_dat(primitives_file)

    return start_state, goal_state, selected_primitives, all_primitives

def load_plan_from_stdin():
    """Load plan data from stdin (mixed text with JSON)"""
    text = sys.stdin.read()
    data = parse_json_from_text(text)

    start_state = data['start_state']
    goal_state = data['goal_state']
    primitive_curves = data['primitive_curves']
    primitive_steps = data['primitive_steps']

    return start_state, goal_state, primitive_curves, primitive_steps

def create_sample_plan():
    """Create a sample plan for testing"""
    start_state = [0.0, 0.0, 0.0]
    goal_state = [0.05, 0.05, 0.1]

    # Sample selected primitives
    selected_primitives = [
        {
            "edge_idx": 7,
            "push_steps": 1,
            "primitive": [0.184643, 0.0503864, 0.21501]
        },
        {
            "edge_idx": 6,
            "push_steps": 1,
            "primitive": [-0.184643, 0.0503864, -0.21501]
        }
    ]

    # Try to load all primitives, but provide empty list if file not found
    all_primitives = read_primitives_dat("data/motion_primitives_15_square.dat")

    return start_state, goal_state, selected_primitives, all_primitives

def main():
    parser = argparse.ArgumentParser(description='Visualize greedy planner path')
    parser.add_argument('--input', '-i',
                       help='Input JSON file with plan data')
    parser.add_argument('--output', '-o',
                       help='Output image file (if not specified, show interactive plot)')
    parser.add_argument('--interactive', action='store_true',
                       help='Create interactive HTML visualization')
    parser.add_argument('--stdin', action='store_true',
                       help='Read plan data from stdin (output of test_planner_output)')
    parser.add_argument('--sample', action='store_true',
                       help='Use sample plan data for testing')

    args = parser.parse_args()

    try:
        if args.stdin:
            print("Reading plan data from stdin...")
            start_state, goal_state, selected_primitives, primitive_steps = load_plan_from_stdin()
            all_primitives = read_primitives_dat("data/motion_primitives_15_square.dat")
        elif args.input:
            start_state, goal_state, selected_primitives, all_primitives = load_plan_from_json(args.input)
        elif args.sample:
            print("Using sample plan data")
            start_state, goal_state, selected_primitives, all_primitives = create_sample_plan()
        else:
            print("Error: Must specify --input, --stdin, or --sample")
            return 1

        print(f"Visualizing plan with {len(selected_primitives)} steps")
        print(f"Loaded {len(all_primitives) if all_primitives else 0} total primitives")
        print(f"Start: {start_state}")
        print(f"Goal: {goal_state}")

        if args.interactive:
            output_file = args.output if args.output else "interactive_plan.html"
            plot_plan_visualization_interactive(start_state, goal_state, selected_primitives, all_primitives, output_file)
        else:
            plot_plan_visualization(start_state, goal_state, selected_primitives, all_primitives, args.output)

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0

if __name__ == '__main__':
    exit(main())