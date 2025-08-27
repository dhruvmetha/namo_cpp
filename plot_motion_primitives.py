#!/usr/bin/env python3
"""
Plot motion primitives from motion_primitives.dat file.

The binary format is:
- Header: uint32_t count (4 bytes)
- Primitives: Array of NominalPrimitive structs (14 bytes each):
  - float delta_x (4 bytes) - Position change in x
  - float delta_y (4 bytes) - Position change in y  
  - float delta_theta (4 bytes) - Rotation change (yaw)
  - uint8_t edge_idx (1 byte) - Push direction (0-11)
  - uint8_t push_steps (1 byte) - Push step number (1-10)
"""

import struct
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def read_motion_primitives(filename):
    """Read motion primitives from binary file."""
    primitives = []
    
    with open(filename, 'rb') as f:
        # Read header (count)
        count_data = f.read(4)
        if len(count_data) < 4:
            raise ValueError("File too short to contain header")
        count = struct.unpack('<I', count_data)[0]  # Little-endian uint32
        
        print(f"Reading {count} primitives from {filename}")
        
        # Read primitives
        for i in range(count):
            # Read one primitive (14 bytes total)
            primitive_data = f.read(14)
            if len(primitive_data) < 14:
                raise ValueError(f"Incomplete primitive {i}, got {len(primitive_data)} bytes")
            
            # Unpack: 3 floats (little-endian) + 2 uint8
            delta_x, delta_y, delta_theta, edge_idx, push_steps = struct.unpack('<fffBB', primitive_data)
            
            primitives.append({
                'delta_x': delta_x,
                'delta_y': delta_y, 
                'delta_theta': delta_theta,
                'edge_idx': edge_idx,
                'push_steps': push_steps
            })
    
    return primitives

def plot_primitives_overview(primitives):
    """Plot overview of all primitives."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract data
    delta_x = [p['delta_x'] for p in primitives]
    delta_y = [p['delta_y'] for p in primitives]
    delta_theta = [p['delta_theta'] for p in primitives]
    edge_idx = [p['edge_idx'] for p in primitives]
    push_steps = [p['push_steps'] for p in primitives]
    
    # 1. Displacement vectors
    ax1.scatter(delta_x, delta_y, c=edge_idx, cmap='tab10', alpha=0.7, s=50)
    ax1.set_xlabel('Delta X (m)')
    ax1.set_ylabel('Delta Y (m)')
    ax1.set_title('Displacement Vectors (colored by edge direction)')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Add origin
    ax1.plot(0, 0, 'ko', markersize=8, label='Origin')
    ax1.legend()
    
    # 2. Displacement magnitude vs push steps
    displacement_mag = [np.sqrt(dx**2 + dy**2) for dx, dy in zip(delta_x, delta_y)]
    colors = plt.cm.tab10(np.array(edge_idx) / 11)
    
    ax2.scatter(push_steps, displacement_mag, c=colors, alpha=0.7, s=50)
    ax2.set_xlabel('Push Steps')
    ax2.set_ylabel('Displacement Magnitude (m)')
    ax2.set_title('Displacement vs Push Steps')
    ax2.grid(True, alpha=0.3)
    
    # 3. Rotation vs displacement
    ax3.scatter(displacement_mag, np.degrees(delta_theta), c=edge_idx, cmap='tab10', alpha=0.7, s=50)
    ax3.set_xlabel('Displacement Magnitude (m)')
    ax3.set_ylabel('Rotation Change (degrees)')
    ax3.set_title('Rotation vs Displacement')
    ax3.grid(True, alpha=0.3)
    
    # 4. Edge direction distribution
    edge_counts = np.bincount(edge_idx, minlength=12)
    ax4.bar(range(12), edge_counts, alpha=0.7, color='skyblue')
    ax4.set_xlabel('Edge Direction Index')
    ax4.set_ylabel('Number of Primitives')
    ax4.set_title('Primitives per Edge Direction')
    ax4.set_xticks(range(12))
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_primitives_by_direction(primitives):
    """Plot primitives grouped by edge direction."""
    # Group by edge direction
    directions = {}
    for p in primitives:
        edge = p['edge_idx']
        if edge not in directions:
            directions[edge] = []
        directions[edge].append(p)
    
    # Create subplot grid
    n_dirs = len(directions)
    cols = 4
    rows = (n_dirs + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, (edge_idx, prims) in enumerate(sorted(directions.items())):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Plot displacement vectors for this direction
        delta_x = [p['delta_x'] for p in prims]
        delta_y = [p['delta_y'] for p in prims]
        push_steps = [p['push_steps'] for p in prims]
        
        # Color by push steps
        scatter = ax.scatter(delta_x, delta_y, c=push_steps, cmap='viridis', 
                           s=60, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        # Add arrows from origin
        for dx, dy in zip(delta_x, delta_y):
            ax.arrow(0, 0, dx, dy, head_width=0.01, head_length=0.01, 
                    fc='red', ec='red', alpha=0.3, linewidth=1)
        
        ax.plot(0, 0, 'ko', markersize=8)
        ax.set_xlabel('Delta X (m)')
        ax.set_ylabel('Delta Y (m)')
        ax.set_title(f'Edge Direction {edge_idx}')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Push Steps')
    
    # Hide unused subplots
    for i in range(len(directions), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig

def plot_primitives_summary_stats(primitives):
    """Plot summary statistics."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Extract data
    delta_x = np.array([p['delta_x'] for p in primitives])
    delta_y = np.array([p['delta_y'] for p in primitives])
    delta_theta = np.array([p['delta_theta'] for p in primitives])
    edge_idx = np.array([p['edge_idx'] for p in primitives])
    push_steps = np.array([p['push_steps'] for p in primitives])
    
    # 1. Displacement magnitude histogram
    displacement_mag = np.sqrt(delta_x**2 + delta_y**2)
    ax1.hist(displacement_mag, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Displacement Magnitude (m)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Displacement Magnitudes')
    ax1.grid(True, alpha=0.3)
    
    # 2. Rotation histogram
    ax2.hist(np.degrees(delta_theta), bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.set_xlabel('Rotation Change (degrees)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Rotation Changes')
    ax2.grid(True, alpha=0.3)
    
    # 3. Push steps vs average displacement
    step_stats = {}
    for step in range(1, 11):
        mask = push_steps == step
        if np.any(mask):
            avg_disp = np.mean(displacement_mag[mask])
            std_disp = np.std(displacement_mag[mask])
            step_stats[step] = (avg_disp, std_disp)
    
    steps = list(step_stats.keys())
    avg_disps = [step_stats[s][0] for s in steps]
    std_disps = [step_stats[s][1] for s in steps]
    
    ax3.errorbar(steps, avg_disps, yerr=std_disps, marker='o', capsize=5, capthick=2)
    ax3.set_xlabel('Push Steps')
    ax3.set_ylabel('Average Displacement (m)')
    ax3.set_title('Displacement vs Push Steps (with std dev)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary table
    ax4.axis('off')
    stats_text = f"""
Motion Primitives Summary:
    
Total Primitives: {len(primitives)}
Edge Directions: {len(set(edge_idx))} (0-{max(edge_idx)})
Push Steps Range: {min(push_steps)}-{max(push_steps)}

Displacement Statistics:
  Min: {np.min(displacement_mag):.4f} m
  Max: {np.max(displacement_mag):.4f} m
  Mean: {np.mean(displacement_mag):.4f} m
  Std: {np.std(displacement_mag):.4f} m

Rotation Statistics:
  Min: {np.degrees(np.min(delta_theta)):.2f}°
  Max: {np.degrees(np.max(delta_theta)):.2f}°
  Mean: {np.degrees(np.mean(delta_theta)):.2f}°
  Std: {np.degrees(np.std(delta_theta)):.2f}°
    """
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    return fig

def main():
    """Main function."""
    # File path
    data_file = Path('data/motion_primitives.dat')
    
    if not data_file.exists():
        print(f"Error: {data_file} not found!")
        print("Run ./build/generate_motion_primitives_db first")
        return
    
    try:
        # Read primitives
        primitives = read_motion_primitives(data_file)
        print(f"Successfully loaded {len(primitives)} motion primitives")
        
        # Create plots
        print("Creating overview plot...")
        fig1 = plot_primitives_overview(primitives)
        fig1.savefig('motion_primitives_overview.png', dpi=150, bbox_inches='tight')
        
        print("Creating direction-specific plots...")
        fig2 = plot_primitives_by_direction(primitives)
        fig2.savefig('motion_primitives_by_direction.png', dpi=150, bbox_inches='tight')
        
        print("Creating summary statistics...")
        fig3 = plot_primitives_summary_stats(primitives)
        fig3.savefig('motion_primitives_stats.png', dpi=150, bbox_inches='tight')
        
        print("\nPlots saved:")
        print("  - motion_primitives_overview.png")
        print("  - motion_primitives_by_direction.png") 
        print("  - motion_primitives_stats.png")
        
        # Show plots
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()