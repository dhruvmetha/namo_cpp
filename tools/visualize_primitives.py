#!/usr/bin/env python3
"""
Visualize motion primitives from the binary .dat file
"""

import struct
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def read_primitives_dat(filename):
    """Read motion primitives from binary .dat file"""
    primitives = []
    
    with open(filename, 'rb') as f:
        # Read header (number of primitives)
        count_bytes = f.read(4)
        if len(count_bytes) < 4:
            raise ValueError("File too short to contain valid header")
        
        count = struct.unpack('I', count_bytes)[0]  # uint32_t
        print(f"Reading {count} primitives from {filename}")
        
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

def plot_primitives(primitives, output_file=None, stats_file=None):
    """Create visualizations of the motion primitives"""
    
    # Convert to numpy arrays for easier manipulation
    data = np.array([(p['delta_x'], p['delta_y'], p['delta_theta'], 
                      p['edge_idx'], p['push_steps']) for p in primitives])
    
    delta_x = data[:, 0]
    delta_y = data[:, 1]
    delta_theta = data[:, 2]
    edge_idx = data[:, 3].astype(int)
    push_steps = data[:, 4].astype(int)
    
    # Create figure with subplots (2x3 grid, but we'll only use 5 plots)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Motion Primitives Visualization ({len(primitives)} primitives)', fontsize=16)
    
    # 1. Displacement vectors colored by edge index
    ax1 = axes[0, 0]
    scatter = ax1.scatter(delta_x, delta_y, c=edge_idx, cmap='tab20', alpha=0.7, s=30)
    ax1.set_xlabel('Delta X (m)')
    ax1.set_ylabel('Delta Y (m)')
    ax1.set_title('Displacement Vectors by Edge Index')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    plt.colorbar(scatter, ax=ax1, label='Edge Index')
    
    # 2. Displacement vectors colored by push steps
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(delta_x, delta_y, c=push_steps, cmap='viridis', alpha=0.7, s=30)
    ax2.set_xlabel('Delta X (m)')
    ax2.set_ylabel('Delta Y (m)')
    ax2.set_title('Displacement Vectors by Push Steps')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    plt.colorbar(scatter2, ax=ax2, label='Push Steps')
    
    # 3. Rotation vs translation magnitude
    ax3 = axes[0, 2]
    translation_mag = np.sqrt(delta_x**2 + delta_y**2)
    scatter3 = ax3.scatter(translation_mag, np.abs(delta_theta), c=edge_idx, cmap='tab20', alpha=0.7, s=30)
    ax3.set_xlabel('Translation Magnitude (m)')
    ax3.set_ylabel('|Rotation| (rad)')
    ax3.set_title('Translation vs Rotation')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter3, ax=ax3, label='Edge Index')
    
    # 4. Displacement magnitude by edge and steps
    ax4 = axes[1, 0]
    unique_edges = np.unique(edge_idx)
    unique_steps = np.unique(push_steps)
    
    for edge in unique_edges[:min(12, len(unique_edges))]:  # Limit to first 12 for readability
        mask = edge_idx == edge
        if np.any(mask):
            steps_for_edge = push_steps[mask]
            mag_for_edge = translation_mag[mask]
            ax4.plot(steps_for_edge, mag_for_edge, 'o-', label=f'Edge {edge}', alpha=0.7)
    
    ax4.set_xlabel('Push Steps')
    ax4.set_ylabel('Translation Magnitude (m)')
    ax4.set_title('Translation Magnitude vs Push Steps')
    ax4.grid(True, alpha=0.3)
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 5. Displacement distribution histograms
    ax5 = axes[1, 1]
    ax5.hist2d(delta_x, delta_y, bins=20, cmap='Blues', alpha=0.7)
    ax5.set_xlabel('Delta X (m)')
    ax5.set_ylabel('Delta Y (m)')
    ax5.set_title('Displacement Distribution (2D Histogram)')
    ax5.axis('equal')
    
    # Hide the unused subplot
    axes[1, 2].axis('off')
    
    # Calculate and save statistics to file
    if stats_file:
        stats_text = f"""Motion Primitives Statistics Summary
============================================

Total Primitives: {len(primitives)}
Unique Edges: {len(unique_edges)}
Unique Step Counts: {len(unique_steps)}

Translation Statistics:
  Mean magnitude: {np.mean(translation_mag):.6f} m
  Max magnitude: {np.max(translation_mag):.6f} m
  Min magnitude: {np.min(translation_mag):.6f} m
  Std deviation: {np.std(translation_mag):.6f} m

Rotation Statistics:
  Mean |rotation|: {np.mean(np.abs(delta_theta)):.6f} rad ({np.mean(np.abs(delta_theta)) * 180/np.pi:.3f} deg)
  Max |rotation|: {np.max(np.abs(delta_theta)):.6f} rad ({np.max(np.abs(delta_theta)) * 180/np.pi:.3f} deg)
  Min |rotation|: {np.min(np.abs(delta_theta)):.6f} rad ({np.min(np.abs(delta_theta)) * 180/np.pi:.3f} deg)
  Std deviation: {np.std(np.abs(delta_theta)):.6f} rad ({np.std(np.abs(delta_theta)) * 180/np.pi:.3f} deg)

Edge Distribution (primitives per edge):
"""
        edge_counts = dict(zip(*np.unique(edge_idx, return_counts=True)))
        for edge, count in sorted(edge_counts.items()):
            stats_text += f"  Edge {edge:2d}: {count:3d} primitives\n"
        
        stats_text += f"\nStep Distribution (primitives per step count):\n"
        step_counts = dict(zip(*np.unique(push_steps, return_counts=True)))
        for steps, count in sorted(step_counts.items()):
            stats_text += f"  {steps:2d} steps: {count:3d} primitives\n"
        
        stats_text += f"\nDetailed Edge-Step Matrix:\n"
        for edge in sorted(unique_edges):
            edge_mask = edge_idx == edge
            edge_steps = push_steps[edge_mask]
            step_counts_for_edge = dict(zip(*np.unique(edge_steps, return_counts=True)))
            stats_text += f"  Edge {edge:2d}: {step_counts_for_edge}\n"
        
        with open(stats_file, 'w') as f:
            f.write(stats_text)
        print(f"Statistics saved to {stats_file}")
    
    # Print basic info to console
    print(f"Generated visualization with {len(primitives)} primitives")
    print(f"Edges: {min(unique_edges)}-{max(unique_edges)} ({len(unique_edges)} total)")
    print(f"Steps: {min(unique_steps)}-{max(unique_steps)} ({len(unique_steps)} total)")
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {output_file}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize motion primitives from .dat file')
    parser.add_argument('--input', '-i', default='data/motion_primitives.dat',
                       help='Input .dat file (default: data/motion_primitives.dat)')
    parser.add_argument('--output', '-o', 
                       help='Output image file (if not specified, show interactive plot)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found")
        return 1
    
    try:
        primitives = read_primitives_dat(args.input)
        
        if not primitives:
            print("No primitives found in file")
            return 1
            
        # Generate stats filename if output is specified
        stats_file = None
        if args.output:
            base_name = os.path.splitext(args.output)[0]
            stats_file = f"{base_name}_stats.txt"
        else:
            stats_file = "primitives_stats.txt"
            
        plot_primitives(primitives, args.output, stats_file)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
