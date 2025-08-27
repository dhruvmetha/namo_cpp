# Sequential ML Data Collection

This script provides efficient single-process data collection optimized for ML inference, eliminating the overhead of repeated model loading.

## Key Advantages over Parallel Collection

1. **Model Reuse**: ML models loaded once and reused across all episodes
2. **Memory Efficiency**: Single model instance in GPU memory  
3. **No Inter-Process Overhead**: Direct execution without multiprocessing
4. **Consistent Performance**: Predictable ML inference latency

## Usage Examples

### Basic ML Goal Selection
```bash
PYTHONPATH=/path/to/build_python /path/to/python \
python/idfs/sequential_ml_collection.py \
    --algorithm idfs \
    --goal-strategy ml \
    --ml-goal-model /path/to/goal/model \
    --output-dir ./results \
    --start-idx 0 --end-idx 100 \
    --episodes-per-env 5
```

### Dual ML Strategies (Object + Goal)
```bash
PYTHONPATH=/path/to/build_python /path/to/python \
python/idfs/sequential_ml_collection.py \
    --algorithm idfs \
    --object-strategy ml --goal-strategy ml \
    --ml-object-model /path/to/object/model \
    --ml-goal-model /path/to/goal/model \
    --output-dir ./results \
    --start-idx 0 --end-idx 100 \
    --episodes-per-env 3 \
    --max-depth 5
```

### Full Parameter Example  
```bash
PYTHONPATH=/common/home/dm1487/robotics_research/ktamp/namo/build_python_mjxrl_westeros \
/common/users/dm1487/envs/mjxrl/bin/python \
python/idfs/sequential_ml_collection.py \
    --algorithm idfs \
    --object-strategy ml \
    --goal-strategy ml \
    --ml-object-model /path/to/object/model \
    --ml-goal-model /path/to/goal/model \
    --ml-samples 32 \
    --ml-device cuda \
    --output-dir /tmp/sequential_results \
    --start-idx 0 --end-idx 50 \
    --episodes-per-env 3 \
    --max-depth 5 \
    --max-goals-per-object 5 \
    --search-timeout 300.0 \
    --verbose \
    --filter-minimum-length
```

## Performance Comparison

### Parallel Collection (Typical)
- Model loading: ~20s per worker process  
- Memory usage: N × model_size (N workers)
- Overhead: Inter-process communication + repeated loading

### Sequential Collection (This Implementation)
- Model loading: ~20s once at startup
- Memory usage: 1 × model_size 
- Overhead: Minimal (direct method calls)

**Result**: For ML inference workloads, sequential is often faster overall due to eliminated overhead.

## Output Format

The script generates the same output format as `modular_parallel_collection.py`:

- `{hostname}_env_{index}_results.pkl`: Per-environment episode data
- `collection_summary_{hostname}.pkl`: Complete run statistics
- `summary_{hostname}.txt`: Human-readable summary

## Command Line Arguments

All arguments from the parallel version are supported:

**Required:**
- `--output-dir`: Output directory path
- `--start-idx`: Starting environment index  
- `--end-idx`: Ending environment index (exclusive)

**Algorithm:**
- `--algorithm`: Planning algorithm (idfs, tree_idfs, etc.)
- `--object-strategy`: Object selection strategy (no_heuristic, nearest_first, ml, etc.)
- `--goal-strategy`: Goal selection strategy (random, grid, ml, etc.)

**ML-specific:**
- `--ml-object-model`: Path to object inference model (required for ML object strategy)
- `--ml-goal-model`: Path to goal inference model (required for ML goal strategy)
- `--ml-samples`: Number of ML inference samples (default: 32)
- `--ml-device`: ML inference device (cuda/cpu, default: cuda)

**Optional:**
- `--episodes-per-env`: Episodes per environment (default: 1)
- `--max-depth`: Maximum search depth (default: 5)
- `--max-goals-per-object`: Max goals per object (default: 5)  
- `--search-timeout`: Search timeout in seconds (default: 300)
- `--verbose`: Enable verbose output
- `--filter-minimum-length`: Filter to minimum action sequence length

## When to Use Sequential vs Parallel

**Use Sequential Collection When:**
- Using ML strategies (object or goal selection)
- Model loading time is significant
- Memory is constrained  
- Small to medium dataset sizes
- Debugging ML inference behavior

**Use Parallel Collection When:**  
- Using non-ML strategies only
- Very large dataset sizes (>1000 environments)
- Model loading overhead is minimal
- Maximum throughput is critical