# Modular IDFS Planning System

A clean, modular implementation of IDFS planning algorithms with pluggable architecture.

## Available Algorithms

### 1. **IDFS** (`standard_idfs.py`)
- **Name**: `"idfs"` or `"standard_idfs"`  
- **Type**: Restart-based iterative deepening
- **Memory**: O(depth) - minimal memory usage
- **Performance**: Standard IDFS with re-exploration overhead

### 2. **Tree-IDFS** (`tree_idfs.py`)
- **Name**: `"tree_idfs"`
- **Type**: Tree-maintained iterative deepening  
- **Memory**: O(branching_factor^depth) - caches tree structure
- **Performance**: Avoids re-sampling goals and re-executing actions

## Usage Examples

### Basic Algorithm Usage
```bash
# Use standard IDFS
python modular_parallel_collection.py --algorithm idfs --output-dir ./data --start-idx 10 --end-idx 12

# Use Tree-IDFS  
python modular_parallel_collection.py --algorithm tree_idfs --output-dir ./data --start-idx 10 --end-idx 12
```


### Testing
```bash
# Run comprehensive tests
python test_tree_idfs.py
```

## Architecture

### Core Components
- **`base_planner.py`**: Abstract planner interface and factory
- **`standard_idfs.py`**: Standard restart-based IDFS implementation
- **`tree_idfs.py`**: Tree-maintained IDFS implementation  
- **`modular_parallel_collection.py`**: Algorithm-agnostic parallel data collector
- **`test_tree_idfs.py`**: Testing and benchmarking framework

### Key Benefits
1. **Clean Architecture**: No legacy code, purpose-built for modularity
2. **Algorithm Swapping**: Easy to switch between algorithms  
3. **Consistent Metrics**: Standardized performance tracking
4. **Performance Optimization**: Tree-IDFS eliminates redundant computations
5. **Extensibility**: Easy to add new algorithms (A*, MCTS, etc.)

## Performance Expectations

**Tree-IDFS vs Standard IDFS:**
- **Node Expansions**: Tree-IDFS should use ~75% fewer nodes (no re-exploration)
- **Memory Usage**: Tree-IDFS uses more memory for tree caching
- **Search Time**: Tree-IDFS should be faster for deeper searches
- **Determinism**: Same results with same random seed

## File Structure
```
python/idfs/
├── __init__.py              # Package exports and available algorithms
├── base_planner.py          # Abstract interface & factory
├── standard_idfs.py         # Standard IDFS (restart-based)
├── tree_idfs.py            # Tree-IDFS (tree-maintained) 
├── modular_parallel_collection.py  # Parallel data collector
├── test_tree_idfs.py       # Testing framework
└── README.md              # This file
```

## Adding New Algorithms

To add a new planning algorithm:

1. **Inherit from `BasePlanner`**:
```python
class MyNewPlanner(BasePlanner):
    def search(self, robot_goal):
        # Implement your algorithm
        return PlannerResult(...)
```

2. **Register with factory**:
```python
PlannerFactory.register_planner("my_algorithm", MyNewPlanner)
```

3. **Use immediately**:
```bash
python modular_parallel_collection.py --algorithm my_algorithm
```

## No Legacy Dependencies

This system is completely independent:
- ✅ No dependencies on old IDFS code
- ✅ Clean, purpose-built implementations  
- ✅ Consistent interface across all algorithms
- ✅ Modern Python practices and type hints