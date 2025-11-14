# NAMO Data Collection Guide

Complete guide for collecting region opening data and generating training masks.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Step 1: Configure Collection Parameters](#step-1-configure-collection-parameters)
3. [Step 2: Run Data Collection](#step-2-run-data-collection)
4. [Step 3: Generate Training Masks](#step-3-generate-training-masks)
5. [Step 4: Load and Use Data](#step-4-load-and-use-data)
6. [Understanding the Output](#understanding-the-output)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Environment Setup
```bash
# Activate your conda/virtual environment
source /common/users/dm1487/envs/mjxrl/bin/activate

# Verify you're in the NAMO directory
cd /common/home/dm1487/robotics_research/ktamp/namo
```

### Check Available Environments
```bash
# Data is collected from XML environment files
ls ../ml4kp_ktamp/resources/models/custom_walled_envs/aug9/medium/set1/benchmark_1/*.xml | wc -l
```

---

## Step 1: Configure Collection Parameters

### Edit YAML Configuration

File: `python/namo/data_collection/region_opening_collection.yaml`

```yaml
# Core execution
output_dir: /common/users/dm1487/namo_data/oct26/2_push
start_idx: 0          # First environment to process
end_idx: 100          # Last environment (exclusive)
algorithm: region_opening
workers: 8            # Parallel processes (set to CPU count - 2)
episodes_per_env: 1   # Episodes per environment (usually 1 for region opening)

# Generic planner limits (not used by region opening)
max_depth: 5
max_goals_per_object: 5
max_terminal_checks: 5000
search_timeout: 300.0

# Region opening specific parameters (IMPORTANT!)
region_allow_collisions: false        # false = only physically valid pushes
region_max_chain_depth: 2             # 1=single push, 2=two pushes, 3=three pushes
region_max_solutions_per_neighbor: 5  # Max solutions to FIND per neighbor
region_frontier_beam_width: null      # null = complete search, 200 = beam search
region_max_recorded_solutions_per_neighbor: 2  # Max solutions to SAVE per neighbor

# Paths
xml_dir: ../ml4kp_ktamp/resources/models/custom_walled_envs/aug9
config_file: config/namo_config_complete_skill15.yaml

# Optional pipeline features
verbose: false
filter_minimum_length: false
smooth_solutions: false
max_smooth_actions: 20
refine_actions: false
validate_refinement: false
```

### Key Parameters to Adjust

**For Fast Collection:**
- `region_max_chain_depth: 1` (single pushes only)
- `region_max_solutions_per_neighbor: 3`
- `region_frontier_beam_width: 100`

**For Comprehensive Collection:**
- `region_max_chain_depth: 2` (2-push chains)
- `region_max_solutions_per_neighbor: 10`
- `region_frontier_beam_width: null` (complete search)

**For Maximum Diversity:**
- `region_max_chain_depth: 3`
- `region_max_solutions_per_neighbor: 20`
- `region_max_recorded_solutions_per_neighbor: 10`

---

## Step 2: Run Data Collection

### Basic Usage

```bash
python python/namo/data_collection/modular_parallel_collection.py \
  --config-yaml python/namo/data_collection/region_opening_collection.yaml
```

### With CLI Overrides

```bash
# Override specific parameters
python python/namo/data_collection/modular_parallel_collection.py \
  --config-yaml python/namo/data_collection/region_opening_collection.yaml \
  --start-idx 0 \
  --end-idx 200 \
  --workers 16 \
  --verbose
```

### Common Usage Patterns

**Quick Test (5 environments):**
```bash
python python/namo/data_collection/modular_parallel_collection.py \
  --config-yaml python/namo/data_collection/region_opening_collection.yaml \
  --end-idx 5 \
  --verbose
```

**Production Run (1000 environments):**
```bash
python python/namo/data_collection/modular_parallel_collection.py \
  --config-yaml python/namo/data_collection/region_opening_collection.yaml \
  --start-idx 0 \
  --end-idx 1000 \
  --workers 16
```

**Custom Range:**
```bash
python python/namo/data_collection/modular_parallel_collection.py \
  --config-yaml python/namo/data_collection/region_opening_collection.yaml \
  --start-idx 100 \
  --end-idx 200 \
  --output-dir /path/to/different/output
```

### Expected Output

```
🚀 Starting modular parallel data collection
📊 Algorithm: region_opening
🔢 Processing 100 environments with 8 workers
Collecting data: 100%|████████████| 100/100 [10:23<00:00, episodes: 650, failed: 0]

🎉 Collection complete!
📊 Episodes: 650 total
🎯 Task success rate: 100.0% (10.4m)
```

### Output Location

```
/common/users/dm1487/namo_data/oct26/2_push/
  └── modular_data_<hostname>/
      ├── <hostname>_env_000000_results.pkl
      ├── <hostname>_env_000001_results.pkl
      ├── ...
      ├── collection_summary_<hostname>.pkl
      └── summary_<hostname>.txt
```

---

## Step 3: Generate Training Masks

### Basic Mask Generation

```bash
python python/namo/visualization/run_mask_generation.py batch \
  --input-dir /common/users/dm1487/namo_data/oct26/2_push/modular_data_<hostname> \
  --output-dir /common/users/dm1487/namo_data/oct26/2_push_masks \
  --workers 16
```

**Note:** Replace `<hostname>` with your actual hostname (e.g., `ilab1`, `ilab2`)

### With Options

```bash
# Serial processing (for debugging)
python python/namo/visualization/run_mask_generation.py batch \
  --input-dir /path/to/pkl/files \
  --output-dir /path/to/output \
  --serial

# With minimum length filtering
python python/namo/visualization/run_mask_generation.py batch \
  --input-dir /path/to/pkl/files \
  --output-dir /path/to/output \
  --workers 16 \
  --filter-minimum-length
```

### Expected Output

```
Processing pickle files...
100%|████████████████| 100/100 [05:32<00:00,  3.32s/file]

✓ Processed 650 episodes from 100 environments
✓ Generated 1,200 training examples (after trajectory suffix splitting)
✓ Output saved to /common/users/dm1487/namo_data/oct26/2_push_masks
```

### Output Structure

```
output_dir/
  ├── <hostname>_env_000000/
  │   ├── <hostname>_env_000000_episode_0_neighbour_0_R1_step_0.npz
  │   ├── <hostname>_env_000000_episode_0_neighbour_0_R1_step_1.npz
  │   └── ...
  ├── <hostname>_env_000001/
  │   └── ...
  └── batch_summary.txt
```

---

## Step 4: Load and Use Data

### Loading a Single Example

```python
import numpy as np

# Load .npz file
data = np.load('example_step_0.npz')

# Access base masks (9 channels, 224×224 each)
robot_mask = data['robot']              # Robot position
movable_mask = data['movable']          # All movable objects
static_mask = data['static']            # Static walls
reachable_mask = data['reachable']      # Reachable objects
target_object = data['target_object']   # Object at current position
target_goal = data['target_goal']       # Object at immediate next target
robot_distance = data['robot_distance'] # Distance field from robot
goal_distance = data['goal_distance']   # Distance field from goal
goal_mask = data['goal']                # Robot goal position

# Access multi-horizon masks
num_horizons = data['num_goal_horizons'][0]
print(f"Number of action horizons: {num_horizons}")

for i in range(1, num_horizons + 1):
    goal_mask_ai = data[f'goal_mask_a{i}']
    print(f"Goal mask a{i} shape: {goal_mask_ai.shape}")

# Access metadata
episode_id = str(data['episode_id'][0])
task_id = str(data['task_id'][0])
algorithm = str(data['algorithm'][0])
solution_depth = int(data['solution_depth'][0])

print(f"Episode: {episode_id}")
print(f"Solution depth: {solution_depth}")
```

### Batch Loading

```python
import numpy as np
from pathlib import Path

def load_dataset(data_dir):
    """Load all .npz files from directory."""
    data_dir = Path(data_dir)
    episodes = []

    for npz_file in data_dir.rglob('*.npz'):
        data = np.load(npz_file)
        episodes.append({
            'file': str(npz_file),
            'masks': {key: data[key] for key in data.keys() if key.startswith('goal_mask_a') or key in ['robot', 'movable', 'static', 'reachable', 'target_object', 'target_goal', 'robot_distance', 'goal_distance', 'goal']},
            'num_horizons': int(data['num_goal_horizons'][0]),
            'episode_id': str(data['episode_id'][0]),
            'solution_depth': int(data['solution_depth'][0])
        })

    return episodes

# Load all data
dataset = load_dataset('/path/to/masks/')
print(f"Loaded {len(dataset)} training examples")
```

---

## Understanding the Output

### What Each Mask Contains

For a **2-push chain** from state S0:

#### Base Masks (Same for all examples)
1. **robot**: Circle showing robot position at S0
2. **movable**: All movable objects at S0
3. **static**: Static walls (same across all states)
4. **reachable**: Only reachable objects at S0
5. **target_object**: Object to push (at current S0 position)
6. **target_goal**: Same object at immediate next action target
7. **robot_distance**: Wavefront distance field from robot
8. **goal_distance**: Wavefront distance field from robot goal
9. **goal**: Robot goal position circle

#### Multi-Horizon Masks (Variable number)
- **goal_mask_a1**: Target object at action 0's target position
- **goal_mask_a2**: Target object at action 1's target position
- **goal_mask_a3**: Target object at action 2's target position (if 3-push chain)
- ... and so on

### Example: 2-Push Chain

**Episode collected from environment 42, neighbor R2:**
- Action 0: Push obj_3 to (0.5, 0.3, 0.0)
- Action 1: Push obj_3 to (0.7, 0.5, 0.0)

**Generates 2 training examples:**

**Example 0 (from S0):**
- 9 base masks + 2 goal masks = **11 channels**
- `goal_mask_a1`: obj_3 at (0.5, 0.3, 0.0)
- `goal_mask_a2`: obj_3 at (0.7, 0.5, 0.0)

**Example 1 (from S1):**
- 9 base masks + 1 goal mask = **10 channels**
- `goal_mask_a1`: obj_3 at (0.7, 0.5, 0.0)

### Data Statistics

With `region_max_chain_depth: 2` and 100 environments:
- **Collected episodes**: ~300-500 (3-5 per environment × 100)
- **Training examples**: ~600-1000 (after trajectory suffix splitting)
- **Disk usage**: ~50-100 MB (compressed .npz files)

---

## Troubleshooting

### Issue: "No module named namo_rl"

**Solution:**
```bash
# Check if you're in the correct directory
pwd  # Should be .../namo

# Check if C++ bindings are built
ls build_python_mjxrl_man/namo_rl*.so

# If not built, compile:
cmake -B build_python_mjxrl_man -DCMAKE_BUILD_TYPE=Release
cmake --build build_python_mjxrl_man --parallel 8
```

### Issue: "Permission denied" for output directory

**Solution:**
```bash
# Create output directory with correct permissions
mkdir -p /common/users/dm1487/namo_data/oct26/2_push
chmod 755 /common/users/dm1487/namo_data/oct26/2_push
```

### Issue: Workers hanging or slow

**Solution:**
```bash
# Reduce number of workers
--workers 4

# Or use serial mode for debugging
--serial
```

### Issue: "XML file not found"

**Solution:**
```bash
# Verify XML directory exists
ls ../ml4kp_ktamp/resources/models/custom_walled_envs/aug9/medium/set1/benchmark_1/

# Update xml_dir in YAML if path is different
```

### Issue: No goal masks in .npz files

**Solution:**
Check that:
1. Episodes have action sequences (`solution_found: true`)
2. `all_future_states` is populated (trajectory suffix splitting)
3. Action sequence contains valid targets

### Issue: Memory errors during mask generation

**Solution:**
```bash
# Reduce number of workers
--workers 4

# Process in smaller batches (split start/end indices)
--start-idx 0 --end-idx 50
--start-idx 50 --end-idx 100
```

---

## Advanced Usage

### Custom Region Opening Parameters

```bash
# Single pushes only (fastest)
python python/namo/data_collection/modular_parallel_collection.py \
  --config-yaml python/namo/data_collection/region_opening_collection.yaml \
  --region-max-chain-depth 1 \
  --region-max-solutions-per-neighbor 3

# Complex 3-push chains
python python/namo/data_collection/modular_parallel_collection.py \
  --config-yaml python/namo/data_collection/region_opening_collection.yaml \
  --region-max-chain-depth 3 \
  --region-max-solutions-per-neighbor 10 \
  --region-frontier-beam-width null
```

### Combining Multiple Collections

```bash
# Collect different ranges on different machines
# Machine 1:
python ... --start-idx 0 --end-idx 500

# Machine 2:
python ... --start-idx 500 --end-idx 1000

# Then combine during mask generation
python python/namo/visualization/run_mask_generation.py batch \
  --input-dir /path/to/all_collections \
  --output-dir /path/to/combined_masks
```

---

## Quick Reference

### Data Collection
```bash
python python/namo/data_collection/modular_parallel_collection.py \
  --config-yaml python/namo/data_collection/region_opening_collection.yaml \
  --start-idx 0 --end-idx 100 --workers 16
```

### Mask Generation
```bash
python python/namo/visualization/run_mask_generation.py batch \
  --input-dir <collection_output>/modular_data_<hostname> \
  --output-dir <mask_output> \
  --workers 16
```

### Check Results
```bash
# Count episodes collected
ls <collection_output>/modular_data_<hostname>/*_results.pkl | wc -l

# Count training examples generated
find <mask_output> -name "*.npz" | wc -l

# View summary
cat <collection_output>/modular_data_<hostname>/summary_<hostname>.txt
```

---

## Contact & Support

For issues or questions:
1. Check this guide first
2. Review error messages in terminal output
3. Check log files in output directories
4. Consult CLAUDE.md for algorithm details

---

**Last Updated:** 2025-01-13
