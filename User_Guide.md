# Problems-Kit User Guide

*A comprehensive system for implementing and benchmarking algorithms across Python, Triton, and CUDA platforms*

## Table of Contents

- [Introduction](#introduction)
- [System Requirements](#system-requirements)
- [Getting Started](#getting-started)
- [Implementing Solutions](#implementing-solutions)
  - [Solution Structure](#solution-structure)
  - [Implementation Variants](#implementation-variants)
  - [Metadata System](#metadata-system)
- [Running and Testing Implementations](#running-and-testing-implementations)
  - [Using the CLI](#using-the-cli)
  - [Direct Testing](#direct-testing)
  - [Custom Testing Scripts](#custom-testing-scripts)
- [Benchmarking System](#benchmarking-system)
  - [Unified Benchmarking Framework](#unified-benchmarking-framework)
  - [Configuration-Driven Benchmarks](#configuration-driven-benchmarks)
  - [Input Generation System](#input-generation-system)
  - [Advanced Visualization Suite](#advanced-visualization-suite)
  - [Data Export Options](#data-export-options)
- [Directory Structure](#directory-structure)
- [Routine Problem-Solving Workflow](#routine-problem-solving-workflow)
- [Debugging the System](#debugging-the-system)
  - [Common Issues](#common-issues)
  - [Debugging Implementations](#debugging-implementations)
  - [Debugging Benchmarks](#debugging-benchmarks)
  - [Logging System](#logging-system)
- [System Maintenance](#system-maintenance)
- [Extending the System](#extending-the-system)
- [FAQ](#faq)

## Introduction

The Problems-Kit is a comprehensive system designed to facilitate the implementation, testing, and benchmarking of algorithms across different platforms:

- **Python implementations**: Pure Python and NumPy-based solutions
- **Triton implementations**: GPU-accelerated solutions using OpenAI's Triton
- **CUDA implementations**: NVIDIA CUDA-based solutions

Each problem can have multiple implementation variants per platform, allowing you to explore different optimization strategies and compare their performance through a sophisticated benchmarking system.

## System Requirements

### Basic Setup (Python Only)
- Python 3.7+
- NumPy
- Matplotlib
- Scipy

### Full System (with GPU Acceleration)
- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.0+
- CuPy package
- Triton package (optional for Triton implementations)
- NVCC compiler (for CUDA implementations)

## Getting Started

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Problems-kit
   ```

2. Set up a Python virtual environment (recommended):
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Unix/MacOS
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install numpy matplotlib scipy
   # If you have a GPU and want to use Triton/CUDA
   pip install cupy-cuda11x triton
   ```

### Using the CLI Interface

The central entry point is the `problems_kit.py` script:

```bash
python problems_kit.py
```

This opens the main menu with these options:
1. **Browse Problems by Group**: View and select problems organized by category
2. **Run Implementation**: Execute a specific implementation
3. **Run Benchmark**: Benchmark and compare implementations
4. **Initialize New Problem**: Set up a new problem directory structure
5. **View Available Implementations**: Check which implementations exist on your system

## Implementing Solutions

### Solution Structure

Each solution should follow this standard structure:

```python
"""
Problem X: [Problem Title]
[Implementation Type] Implementation - [Variant Description]

[Additional notes or explanation]
"""

import numpy as np
from typing import [appropriate types]

def solution([appropriate parameters]):
    """
    [Function description]
    
    Args:
        [Parameter descriptions]
        
    Returns:
        [Return value description]
    """
    # Your implementation code here
    
    return result

# Metadata for the implementation registry
IMPLEMENTATION_METADATA = {
    "name": "[Readable Name]",
    "version": "v1",  # or v2, v3, etc.
    "description": "[Brief description]",
    "date": "[Current date]",
    "author": "[Your name]",
    "optimization_techniques": [
        "[List optimization techniques used]"
    ],
    "expected_performance": "[Performance expectations]"
}
```

### Implementation Variants

For each problem and platform, you can create multiple variants:

1. **Base Implementation (v1)**: Simple, straightforward implementation focusing on correctness
2. **Optimized Variants**: Different optimization approaches, for example:
   - `v2_optimized`: General optimizations like algorithm improvements
   - `v3_vectorized`: Vectorization-focused optimizations
   - `v4_cache_friendly`: Cache-friendly memory access patterns
   - `v5_multithreaded`: Parallelization using multiple threads

### Metadata System

Each implementation should include metadata to help with documentation and analysis:

```python
IMPLEMENTATION_METADATA = {
    "name": "Optimized NumPy",           # Human-readable name
    "version": "v2",                      # Version identifier
    "description": "Matrix-vector dot product using NumPy's BLAS backend",
    "date": "2025-04-05",                 # Implementation date
    "author": "Problems-Kit Team",        # Author information
    "optimization_techniques": [          # List of techniques used
        "Contiguous memory layout",
        "BLAS backend utilization",
        "Efficient type conversion"
    ],
    "expected_performance": "Best performance for Python implementations without manual loops"
}
```

This metadata is used in the benchmarking reports and CLI interface.

## Running and Testing Implementations

### Using the CLI

The simplest way to run implementations is through the CLI:

```bash
python problems_kit.py
# Select option 2 (Run Implementation)
# Follow the prompts to select problem, type, and variant
```

You can choose to use default inputs or provide custom inputs.

### Direct Testing

For quick Python implementation testing, use the provided script:

```bash
python run_python_implementations.py
```

You can modify the script to test specific implementations or input sizes.

### Custom Testing Scripts

Create custom test scripts for more control:

```python
# test_my_solution.py
import numpy as np
from solutions.group_XX_category.pXXX_problem_name.python.solution_vY import solution

# Define test cases
test_cases = [
    {
        "input": [np.array([[1,2],[3,4]]), np.array([5,6])],
        "expected": np.array([17, 39])
    },
    # Add more test cases...
]

# Run tests
for i, test in enumerate(test_cases):
    result = solution(*test["input"])
    expected = test["expected"]
    if np.allclose(result, expected):
        print(f"Test {i+1}: PASSED")
    else:
        print(f"Test {i+1}: FAILED")
        print(f"Expected: {expected}")
        print(f"Got: {result}")
```

## Benchmarking System

### Unified Benchmarking Framework

The Problems-Kit now features a unified benchmarking system that standardizes benchmarking across different problem types. Run benchmarks using the command-line interface:

```bash
python run_final_benchmark.py --problem-id p001_matrix_vector_dot
```

The system supports various command-line options:

```bash
python run_final_benchmark.py --problem-id p002_sorting --sizes 1000,5000,10000 --config configs/custom_config.json
```

Or programmatically from Python:

```python
from utils.benchmark_unified import run_problem_benchmark

results = run_problem_benchmark(
    problem_id='p001_matrix_vector_dot',
    config_path='configs/benchmarks/p001_matrix_vector_dot_benchmark_config.json'
)
```

### Configuration-Driven Benchmarks

Benchmarks are now defined through JSON configuration files, making it easy to standardize and customize benchmark parameters:

```json
{
    "problem_id": "p001_matrix_vector_dot",
    "name": "Matrix-Vector Dot Product",
    "description": "Benchmark for matrix-vector dot product implementations",
    "implementations": [
        ["python", "v1"],
        ["python", "v2_optimized"],
        ["cuda", "v1"],
        ["triton", "v1"],
        ["triton", "v2_optimized"]
    ],
    "input_sizes": [128, 256, 512, 1024, 2048, 4096],
    "num_runs": 10,
    "warmup_runs": 3,
    "error_thresholds": {
        "default": 0.0001,
        "1024": 0.0002,
        "2048": 0.0002,
        "4096": 0.0005
    },
    "input_generator": "generate_matrix_vector_inputs",
    "reference_impl": ["python", "v1"]
}
```

### Input Generation System

The system uses a modular input generation approach, allowing for standardized inputs across different problem types:

```python
from utils.benchmark_generators import generate_matrix_vector_inputs, generate_sorting_inputs, generate_graph_inputs

# Generate matrix-vector inputs of size 1024
matrix, vector = generate_matrix_vector_inputs(1024)

# Generate sorting inputs with different distributions
array_to_sort = generate_sorting_inputs(10000, distribution='nearly_sorted')

# Generate graph inputs with custom parameters
graph_data = generate_graph_inputs(1000, edge_density=0.1, weighted=True, directed=True)
```

### Advanced Visualization Suite

The enhanced visualization system provides comprehensive performance insights:

```python
from utils.enhanced_visualizations import generate_complete_visualization_suite

# Generate complete visualization suite
viz_files = generate_complete_visualization_suite(
    results=benchmark_results,
    output_dir="benchmarks/visualizations",
    problem_id="p001_matrix_vector_dot",
    generate_html=True  # Create interactive HTML dashboard
)
```

Visualization capabilities include:
- Performance comparison plots with error bars
- Scaling analysis comparing to theoretical complexity bounds (O(n), O(n log n), O(n²))
- Numerical accuracy verification and error reporting
- Interactive HTML dashboards (when Plotly is available)

### Data Export Options

Benchmark results are automatically exported in multiple formats:

```
# CSV format for tabular analysis
benchmarks/csv/p001_matrix_vector_dot_benchmark_20250407_181517.csv

# JSON format for programmatic processing
benchmarks/json/p001_matrix_vector_dot_benchmark_20250407_181517.json

# Visualization files
benchmarks/visualizations/p001_matrix_vector_dot_performance_20250407_181517.png
benchmarks/visualizations/p001_matrix_vector_dot_scaling_20250407_181517.png
benchmarks/visualizations/p001_matrix_vector_dot_accuracy_20250407_181517.png
benchmarks/visualizations/html/p001_matrix_vector_dot_dashboard_20250407_181517.html
```

The exports include comprehensive data:
- Implementation details and input sizes
- Timing statistics (mean, median, min, max, std)
- Numerical accuracy and error thresholds
- Scaling factors relative to baseline sizes
- Standard deviation and confidence intervals
- Memory usage metrics (if tracked)
- Throughput measurements (if calculated)

### Visualization Options

Benchmarks generate several visualization types:

1. **Bar charts**: Comparing mean execution times with error bars
2. **Box plots**: Showing distribution of execution times
3. **Line charts**: Displaying scaling behavior across input sizes
4. **Interactive Plotly visualizations**: HTML files with hover details and zooming

## Directory Structure

The Problems-Kit is organized with the following directory structure:

```
Problems-kit/
│
├── problems_kit.py                  # Main CLI interface
├── problems.md                      # Problem definitions and checklist
├── User_Guide.md                    # This comprehensive user guide
├── run_benchmark_example.py         # Example benchmark script
├── run_python_implementations.py    # Python implementation test script
├── run_setup.py                     # Setup script for initial configuration
├── test_implementation.py           # Test script for implementations
├── verify_python_implementation.py  # Verification script for Python setup
│
├── solutions/                       # Solutions organized by category and problem
│   ├── categories.py                # Categories definitions
│   ├── group_01_linear_algebra/     # Group 1: Linear Algebra problems
│   │   ├── p001_matrix_vector_dot/  # Problem 1: Matrix-Vector Dot Product
│   │   │   ├── __init__.py          # Problem registration & input generation
│   │   │   ├── python/              # Python implementations
│   │   │   │   ├── __init__.py      # Python implementation registry
│   │   │   │   ├── solution_v1.py   # Baseline implementation
│   │   │   │   └── solution_v2_optimized.py # Optimized implementation
│   │   │   ├── triton/              # Triton implementations (optional)
│   │   │   └── cuda/                # CUDA implementations (optional)
│   │   │
│   │   ├── p002_transpose_matrix/   # Problem 2
│   │   │   ├── ...
│   │   └── ...
│   │
│   ├── group_02_data_transformations/ # Group 2
│   │   ├── ...
│   └── ...
│
├── utils/                           # Utility modules
│   ├── __init__.py                  # Package initialization
│   ├── benchmark_enhanced.py        # Enhanced benchmarking system with detailed metrics
│   ├── bench_runner_enhanced.py     # Enhanced benchmark runner with multi-platform support
│   ├── benchmark_suite.py           # Benchmark suite for organizing and running tests
│   ├── benchmark_visualization.py   # Visualization utilities for benchmark results
│   ├── path_manager.py              # Centralized path management
│   ├── problem_parser.py            # Problem definition parsing and code generation
│   ├── setup_project.py             # Project setup utilities
│   └── solution_template.py         # Templates for solution implementations
│
├── benchmarks/                      # Benchmark results (automatically created)
│   ├── csv/                         # CSV exports of benchmark results
│   └── visualizations/              # Generated plots and visualization files
│
└── tests/                           # Test cases for solutions
    └── test_implementations.py      # Test suite for verifying implementations
```

## Routine Problem-Solving Workflow

Follow this routine workflow for solving problems efficiently:

1. **Problem Selection**
   - Review `problems.md` to find a problem to solve
   - Note the prerequisites and difficulty level

2. **Environment Setup**
   - Ensure your virtual environment is activated
   - Check that all dependencies are installed

3. **Problem Initialization**
   - If the problem directory doesn't exist:
     ```bash
     python problems_kit.py
     # Select option 4 (Initialize New Problem)
     # Follow the prompts
     ```

4. **Base Implementation**
   - Create `solution_v1.py` in the appropriate directory
   - Implement a simple, correct solution
   - Add the necessary metadata

5. **Testing**
   - Run your implementation through the CLI or test script
   - Verify correctness against known examples
   - Fix any bugs

6. **Optimization**
   - Create `solution_v2_optimized.py` with improvements
   - Focus on performance bottlenecks
   - Use different optimization techniques

7. **Benchmarking**
   - Run benchmarks to compare all variants
   - Analyze the results to understand performance
   - Add performance insights to implementation metadata

8. **Documentation**
   - Update implementation comments with insights
   - Note any challenges or interesting findings

9. **Update Checklist**
   - Mark the problem as completed in `problems.md`
   - Update the checkboxes for implemented platforms

10. **Review and Refine**
    - Review code for potential improvements
    - Consider additional optimization variants
    - Prepare for the next problem

## Debugging the System

### Common Issues

#### Import Errors

If you see `ImportError` or `ModuleNotFoundError`:

1. **Check the directory structure**:
   ```bash
   find solutions -name "__init__.py" | sort
   ```

2. **Verify Python Path**:
   ```python
   import sys
   print(sys.path)
   ```

3. **Use explicit imports**:
   ```python
   # Instead of relative imports
   from .solution_v1 import solution
   
   # Try absolute imports
   from solutions.group_01_linear_algebra.p001_matrix_vector_dot.python.solution_v1 import solution
   ```

4. **Fix circular imports**:
   If you have modules importing each other, restructure your code to avoid this pattern.

5. **Check module load order**:
   Add print statements to your __init__.py files to verify they're being loaded:
   ```python
   print(f"Loading module {__name__}")
   ```

#### Function Signature Mismatches

If you get errors like `solution() missing required positional arguments`:

1. **Check the function signature** in your implementation against what's expected:
   ```python
   # Expected signature
   def solution(matrix, vector):
       # Implementation
   
   # Actual signature may be different
   def solution(matrix, vector, axis=None):  # Extra parameter
       # Implementation
   ```

2. **Verify input generation** is returning the correct structure:
   ```python
   def generate_inputs(size=1000):
       # Add debugging
       args = [np.random.rand(size, size), np.random.rand(size)]
       kwargs = {}
       print(f"Generated args: {[a.shape for a in args]}")
       print(f"Generated kwargs: {kwargs}")
       return args, kwargs
   ```

3. **Check implementation registry** in __init__.py files:
   ```python
   print(f"Registered implementations: {IMPLEMENTATIONS.keys()}")
   for key, impl in IMPLEMENTATIONS.items():
       print(f"Implementation {key}: {impl['function']}")
   ```

#### Memory Issues

For out-of-memory errors:

1. **Reduce input sizes** for testing:
   ```python
   # Change from
   input_sizes = [1000, 2000, 4000, 8000]
   # To
   input_sizes = [100, 200, 400, 800]  # Smaller sizes for testing
   ```

2. **Monitor memory usage**:
   ```python
   import psutil
   import os
   
   def print_memory_usage():
       process = psutil.Process(os.getpid())
       print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
   
   # Call before and after large operations
   print_memory_usage()
   result = large_computation()
   print_memory_usage()
   ```

3. **Use memory profiling**:
   ```python
   import tracemalloc
   
   tracemalloc.start()
   # Run your code
   current, peak = tracemalloc.get_traced_memory()
   print(f"Current memory: {current / 10**6}MB; Peak: {peak / 10**6}MB")
   tracemalloc.stop()
   ```

4. **Free memory** when possible:
   ```python
   import gc
   
   # Delete large objects when done
   del large_matrix
   # Force garbage collection
   gc.collect()
   ```

#### File Path Issues

If you encounter file not found errors:

1. **Use absolute paths** with Path from pathlib:
   ```python
   from pathlib import Path
   
   # Get the project root
   ROOT_DIR = Path(__file__).parent.parent.absolute()
   # Build paths relative to root
   file_path = ROOT_DIR / "benchmarks" / "results.json"
   ```

2. **Enable path debugging**:
   ```python
   def find_file(filename, search_path="."):
       """Search for a file and print all locations checked."""
       search_path = Path(search_path)
       print(f"Searching for {filename} in {search_path}")
       
       for path in search_path.glob("**/*"):
           print(f"Checking {path}")
           if path.is_file() and path.name == filename:
               print(f"Found at {path}")
               return path
       
       print(f"File {filename} not found")
       return None
   ```

3. **Verify directory creation**:
   ```python
   def ensure_dir(directory):
       """Ensure a directory exists and print status."""
       directory = Path(directory)
       if not directory.exists():
           print(f"Creating directory {directory}")
           directory.mkdir(parents=True, exist_ok=True)
           return True
       else:
           print(f"Directory {directory} already exists")
           return False
   ```

### Debugging Implementations

1. **Add Debug Mode**:
   ```python
   def debug_solution(matrix, vector, debug=False):
       if debug:
           print(f"Input matrix shape: {matrix.shape}, dtype: {matrix.dtype}")
           print(f"Input vector shape: {vector.shape}, dtype: {vector.dtype}")
           print(f"Matrix memory layout: {'C' if matrix.flags.c_contiguous else 'F'}-contiguous")
           
           # Check for problematic values
           print(f"Matrix contains NaN: {np.isnan(matrix).any()}")
           print(f"Matrix contains Inf: {np.isinf(matrix).any()}")
           
           # Show a small preview
           print(f"Matrix preview:\n{matrix[:3, :3]}")
           print(f"Vector preview:\n{vector[:3]}")
       
       # Your implementation
       result = np.dot(matrix, vector)
       
       if debug:
           print(f"Result shape: {result.shape}, dtype: {result.dtype}")
           print(f"Result preview:\n{result[:3]}")
       
       return result
   ```

2. **Step-by-Step Execution**:
   ```python
   def solution(matrix, vector):
       print("Step 1: Validating inputs")
       # Input validation
       assert matrix.ndim == 2, f"Expected 2D matrix, got {matrix.ndim}D"
       assert vector.ndim == 1, f"Expected 1D vector, got {vector.ndim}D"
       assert matrix.shape[1] == vector.shape[0], f"Dimension mismatch: matrix.shape[1]={matrix.shape[1]}, vector.shape[0]={vector.shape[0]}"
       
       print("Step 2: Converting data types")
       # Type conversion
       matrix = matrix.astype(np.float32, copy=False)
       vector = vector.astype(np.float32, copy=False)
       
       print("Step 3: Memory optimization")
       # Memory layout optimization
       if not matrix.flags.c_contiguous:
           print("  Converting matrix to C-contiguous")
           matrix = np.ascontiguousarray(matrix)
       
       print("Step 4: Performing computation")
       # Core computation
       result = np.dot(matrix, vector)
       
       print("Step 5: Validating result")
       # Result validation
       assert result.shape == (matrix.shape[0],), f"Unexpected result shape: {result.shape}"
       
       return result
   ```

3. **Performance Tracking**:
   ```python
   import time
   
   def solution(matrix, vector, profile=False):
       if profile:
           times = {}
           
           start = time.perf_counter()
           # Prepare inputs
           if not isinstance(matrix, np.ndarray):
               matrix = np.array(matrix, dtype=np.float32)
           times['input_prep'] = time.perf_counter() - start
           
           start = time.perf_counter()
           # Core computation
           result = np.dot(matrix, vector)
           times['computation'] = time.perf_counter() - start
           
           print("Performance breakdown:")
           for step, duration in times.items():
               print(f"  {step}: {duration*1000:.3f} ms")
           
           return result
       else:
           # Regular implementation without profiling
           if not isinstance(matrix, np.ndarray):
               matrix = np.array(matrix, dtype=np.float32)
           return np.dot(matrix, vector)
   ```

4. **Intermediate Results Visualization**:
   ```python
   import matplotlib.pyplot as plt
   
   def debug_visualize(data, title, filename=None):
       """Visualize intermediate results for debugging."""
       plt.figure(figsize=(10, 8))
       
       if data.ndim == 1:
           plt.plot(data)
           plt.title(f"{title} - Shape: {data.shape}")
       elif data.ndim == 2:
           plt.imshow(data, cmap='viridis')
           plt.colorbar()
           plt.title(f"{title} - Shape: {data.shape}")
       
       if filename:
           plt.savefig(f"debug_{filename}.png")
           print(f"Saved visualization to debug_{filename}.png")
       
       plt.close()
   ```

### Debugging Benchmarks

1. **Single Run Mode**:
   ```python
   from utils.bench_runner_enhanced import run_benchmark
   
   # Run with minimal configuration for quick debugging
   run_benchmark(
       problem_id='p001_matrix_vector_dot',
       implementations=[('python', 'v1')],
       num_runs=1,
       input_sizes=[100],  # Small size for quick test
       show_plots=False,
       save_plots=False,
       verbose=True  # Enable detailed logging
   )
   ```

2. **Verbose Logging**:
   Add a `verbose` parameter to key functions in the benchmarking system:
   ```python
   def run_benchmark(..., verbose=False):
       if verbose:
           print(f"Starting benchmark for {problem_id}")
           print(f"Implementations: {implementations}")
           print(f"Input sizes: {input_sizes}")
       
       # Implementation...
   ```

3. **Step-by-Step Benchmark**:
   Create a manual benchmark function to test each component individually:
   ```python
   def manual_benchmark(problem_id, impl_type, variant):
       """Run a manual step-by-step benchmark for debugging."""
       print(f"Manual benchmark for {problem_id}, {impl_type}, {variant}")
       
       # Step 1: Import the solution
       print("\nStep 1: Importing solution")
       try:
           module_path = f"solutions.{problem_id.replace('-', '_')}.{impl_type}.solution_{variant}"
           print(f"Importing from {module_path}")
           module = importlib.import_module(module_path)
           solution_func = getattr(module, "solution")
           print(f"Solution function: {solution_func}")
       except Exception as e:
           print(f"Import failed: {e}")
           traceback.print_exc()
           return
       
       # Step 2: Generate inputs
       print("\nStep 2: Generating inputs")
       try:
           problem_module = importlib.import_module(f"solutions.{problem_id.replace('-', '_')}")
           generator = getattr(problem_module, "generate_inputs", None)
           if generator:
               print(f"Using generator from {problem_module}")
               size = 100  # Small size for testing
               args, kwargs = generator(size)
               print(f"Generated args: {[type(a).__name__ for a in args]}")
               print(f"Generated kwargs: {kwargs}")
           else:
               print("No input generator found")
               return
       except Exception as e:
           print(f"Input generation failed: {e}")
           traceback.print_exc()
           return
       
       # Step 3: Run the solution
       print("\nStep 3: Running solution")
       try:
           start = time.perf_counter()
           result = solution_func(*args, **kwargs)
           end = time.perf_counter()
           print(f"Execution time: {(end - start) * 1000:.3f} ms")
           print(f"Result type: {type(result).__name__}")
           if hasattr(result, 'shape'):
               print(f"Result shape: {result.shape}")
           return result
       except Exception as e:
           print(f"Execution failed: {e}")
           traceback.print_exc()
           return None
   ```

4. **Debug Export Paths**:
   Verify that all paths for saving results exist:
   ```python
   from utils.path_manager import ensure_directories_exist
   
   # Make sure all required directories exist
   ensure_directories_exist()
   ```

### Logging System

Set up a comprehensive logging system for systematic debugging:

```python
import logging
import sys
from pathlib import Path

def setup_logger(name, log_file=None, level=logging.DEBUG):
    """Set up a logger with console and file output."""
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if specified
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Usage in your modules:
# logger = setup_logger(__name__, "logs/benchmark.log")
# logger.debug("Loading module")
# logger.info("Starting benchmark")
# logger.warning("Potential issue detected")
# logger.error("An error occurred", exc_info=True)
```

### Script to Verify System Components

Create a verification script to check all system components are working correctly:

```python
#!/usr/bin/env python
"""
Verification script to check the Problems-Kit system components.
"""

import os
import sys
import importlib
from pathlib import Path

def check_system():
    """Check all system components and dependencies."""
    print("========== Problems-Kit System Check ==========")
    
    # Check Python version
    print(f"\nPython version: {sys.version}")
    
    # Check required packages
    print("\nChecking required packages:")
    packages = ['numpy', 'matplotlib', 'scipy']
    for package in packages:
        try:
            module = importlib.import_module(package)
            print(f"  ✓ {package} {module.__version__}")
        except (ImportError, AttributeError):
            print(f"  ✗ {package} not found or version unknown")
    
    # Check optional GPU packages
    print("\nChecking optional GPU packages:")
    gpu_packages = ['cupy', 'triton']
    for package in gpu_packages:
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"  ✓ {package} {version}")
        except ImportError:
            print(f"  ○ {package} not installed (optional)")
    
    # Check directory structure
    print("\nChecking directory structure:")
    required_dirs = [
        'solutions',
        'utils',
        'benchmarks',
        'benchmarks/csv',
        'benchmarks/visualizations'
    ]
    for directory in required_dirs:
        path = Path(directory)
        if path.exists() and path.is_dir():
            print(f"  ✓ {directory} exists")
        else:
            print(f"  ✗ {directory} missing")
            path.mkdir(parents=True, exist_ok=True)
            print(f"    Created {directory}")
    
    # Check critical files
    print("\nChecking critical files:")
    critical_files = [
        'problems_kit.py',
        'problems.md',
        'utils/bench_runner_enhanced.py',
        'utils/benchmark_suite.py',
        'utils/benchmark_visualization.py'
    ]
    for file in critical_files:
        path = Path(file)
        if path.exists() and path.is_file():
            print(f"  ✓ {file} exists")
        else:
            print(f"  ✗ {file} missing")
    
    # Check implementation structure
    print("\nChecking implementation structure:")
    try:
        # Find all problem directories
        solution_dir = Path('solutions')
        problem_dirs = []
        for group_dir in solution_dir.glob('group_*'):
            for problem_dir in group_dir.glob('p*'):
                if problem_dir.is_dir():
                    problem_dirs.append(problem_dir)
        
        print(f"  Found {len(problem_dirs)} problem directories")
        
        # Check implementation structure for each problem
        for problem_dir in problem_dirs:
            problem_id = problem_dir.name
            print(f"  Checking {problem_id}:")
            
            # Check if __init__.py exists
            init_file = problem_dir / '__init__.py'
            if init_file.exists():
                print(f"    ✓ {problem_id}/__init__.py")
            else:
                print(f"    ✗ {problem_id}/__init__.py missing")
            
            # Check implementation directories
            impl_types = ['python', 'triton', 'cuda']
            for impl_type in impl_types:
                impl_dir = problem_dir / impl_type
                if impl_dir.exists() and impl_dir.is_dir():
                    impl_init = impl_dir / '__init__.py'
                    if impl_init.exists():
                        print(f"    ✓ {problem_id}/{impl_type}/__init__.py")
                    else:
                        print(f"    ✗ {problem_id}/{impl_type}/__init__.py missing")
                    
                    # Check for solutions
                    solutions = list(impl_dir.glob('solution_*.py'))
                    if solutions:
                        print(f"    ✓ {problem_id}/{impl_type} has {len(solutions)} solution(s)")
                    else:
                        print(f"    ✗ {problem_id}/{impl_type} has no solutions")
    except Exception as e:
        print(f"  Error checking implementations: {e}")
    
    print("\n============ System Check Complete ============")

if __name__ == "__main__":
    check_system()
```

## Common Error Messages and Solutions

| Error Message | Likely Cause | Solution |
|---------------|--------------|----------|
| `ModuleNotFoundError: No module named 'solutions.group_01_linear_algebra'` | Python path issue or missing __init__.py file | Add the root directory to the Python path or create missing __init__.py files |
| `AttributeError: module 'solutions.group_01_linear_algebra.p001_matrix_vector_dot.python.solution_v1' has no attribute 'solution'` | Function not defined in module | Ensure function is named 'solution' or update the registry to use the correct name |
| `TypeError: solution() missing 1 required positional argument: 'vector'` | Input generation issue | Check that generate_inputs returns the correct number of arguments with the right shapes |
| `MemoryError` | Input sizes too large | Reduce input sizes or process in chunks |
| `KeyError: 'v2_optimized'` | Implementation variant not registered | Check implementation registry in __init__.py |
| `FileNotFoundError: [Errno 2] No such file or directory: 'benchmarks/csv/...'` | Missing directory | Use `from utils.path_manager import ensure_directories_exist` and call `ensure_directories_exist()` before operations that save files |

---

This User Guide covers all aspects of using the Problems-Kit system. For any questions not covered here, refer to the source code documentation or create custom debugging scripts to address specific issues.

Remember that the system is designed to be extensible - feel free to modify it to suit your specific needs!

## GPU Implementations

This section provides detailed guidance for implementing solutions using GPU acceleration with Triton and CUDA.

### GPU Requirements

Before working with GPU implementations, ensure your system meets these requirements:

1. **Hardware Requirements**:
   - NVIDIA GPU with compute capability 3.5 or higher for CUDA
   - At least 4GB of GPU memory (8GB+ recommended for larger problems)

2. **Software Requirements**:
   - CUDA Toolkit (11.x or 12.x recommended)
   - cuDNN for deep learning implementations
   - Python packages:
     - For Triton: `triton`
     - For CUDA: `cupy` and/or `pycuda`

3. **Verification**:
   Before implementing, verify your GPU setup:
   ```bash
   # Check NVIDIA driver and CUDA version
   nvidia-smi
   
   # Verify Python packages
   python -c "import triton; print(f'Triton version: {triton.__version__}')"
   python -c "import cupy; print(f'CuPy version: {cupy.__version__}'); print(f'CUDA version: {cupy.cuda.runtime.runtimeGetVersion()}')"
   ```

### Triton Implementation Guide

[OpenAI's Triton](https://github.com/openai/triton) is a language and compiler for writing highly efficient GPU kernels. It provides a Python-like syntax for writing custom GPU operations.

#### Basic Triton Template

Create a basic Triton implementation in `triton/solution_v1.py`:

```python
import numpy as np
import triton
import triton.language as tl

# Define your Triton kernel
@triton.jit
def my_kernel(
    # Pointers to inputs and outputs
    input_ptr, output_ptr,
    # Problem dimensions
    n_elements,
    # Meta-parameters (optimizations)
    BLOCK_SIZE: tl.constexpr,
):
    # Compute program ID
    pid = tl.program_id(0)
    # Compute block offset
    block_start = pid * BLOCK_SIZE
    # Generate offsets for block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create mask for bounds checking
    mask = offsets < n_elements
    # Load data from memory
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    # Perform computation (example: multiply by 2)
    y = x * 2.0
    # Store results back to memory
    tl.store(output_ptr + offsets, y, mask=mask)

def solution(input_data):
    """
    Triton implementation of the algorithm.
    
    Args:
        input_data: NumPy array input
        
    Returns:
        NumPy array output
    """
    # Get input dimensions and validate
    n_elements = input_data.size
    
    # Convert to the right data type if needed
    if input_data.dtype != np.float32:
        input_data = input_data.astype(np.float32)
    
    # Allocate output array
    output = np.empty_like(input_data)
    
    # Configure grid and block dimensions
    BLOCK_SIZE = 1024  # Typical GPU block size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)  # Ceiling division
    
    # Launch kernel
    my_kernel[grid](
        input_data, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Add implementation metadata
IMPLEMENTATION_METADATA = {
    "name": "Basic Triton Implementation",
    "version": "v1",
    "description": "Basic implementation using Triton",
    "date": "2025-04-05",
    "author": "Your Name",
    "optimization_techniques": ["GPU parallelization"],
    "expected_performance": "Significant speedup over CPU for large inputs",
    "hardware_requirements": "NVIDIA GPU with Triton support"
}
```

#### Common Triton Optimization Techniques

1. **Tiling**: Divide the computation into small blocks that fit in shared memory or registers.

```python
@triton.jit
def optimized_kernel(
    input_ptr, output_ptr,
    n_rows, n_cols,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    # 2D tiling example for matrix operations
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute block offsets
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    
    # Generate offset ranges
    m_offsets = m_start + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = n_start + tl.arange(0, BLOCK_SIZE_N)
    
    # Bounds checking masks
    m_mask = m_offsets < n_rows
    n_mask = n_offsets < n_cols
    
    # Load matrix block into shared memory
    # Process block
    # Store results
```

2. **Auto-tuning**: Let Triton find the optimal configuration for your kernel.

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
    ],
    key=['n_elements'],
)
@triton.jit
def autotuned_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Kernel implementation
    pass
```

3. **Memory optimizations**: Use shared memory for frequently accessed data.

```python
@triton.jit
def kernel_with_shared_memory(
    matrix_ptr, vector_ptr, output_ptr,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID (row index)
    row_idx = tl.program_id(0)
    
    # Allocate shared memory for vector
    vec_shared = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Load vector chunk to shared memory
    for i in range(0, n_cols, BLOCK_SIZE):
        # Check bounds
        mask = i + tl.arange(0, BLOCK_SIZE) < n_cols
        # Load to shared memory
        vec_shared = tl.load(vector_ptr + i + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
        # Use the shared data in computations
        # ...
```

#### Triton Debugging Tips

1. **Shape and Type Validation**: Always verify input shapes and types:
   ```python
   assert input_data.dtype == np.float32, "Input must be float32"
   assert len(input_data.shape) == 2, "Input must be 2D (matrix)"
   ```

2. **Error Handling**: Catch and report Triton-specific errors:
   ```python
   try:
       result = triton_solution(matrix, vector)
   except Exception as e:
       print(f"Triton error: {e}")
       # Fallback to CPU implementation
       result = numpy_solution(matrix, vector)
   ```

3. **Debug Printing**: Add debug prints for GPU configurations:
   ```python
   print(f"Grid dimensions: {grid}")
   print(f"Input shape: {input_data.shape}")
   ```

### CUDA Implementation Guide

CUDA implementations in this project use CuPy, a NumPy-compatible array library for GPU-accelerated computing.

#### Basic CUDA Template

Create a basic CUDA implementation in `cuda/solution_v1.py`:

```python
import numpy as np
import cupy as cp

def solution(input_data):
    """
    CUDA implementation using CuPy.
    
    Args:
        input_data: NumPy array input
        
    Returns:
        NumPy array output
    """
    # Convert to GPU
    gpu_data = cp.asarray(input_data)
    
    # Perform operations (example: element-wise multiplication)
    gpu_result = gpu_data * 2.0
    
    # Transfer back to CPU
    cpu_result = cp.asnumpy(gpu_result)
    
    return cpu_result

# Add implementation metadata
IMPLEMENTATION_METADATA = {
    "name": "Basic CUDA Implementation",
    "version": "v1",
    "description": "Basic implementation using CuPy (CUDA)",
    "date": "2025-04-05",
    "author": "Your Name",
    "optimization_techniques": ["GPU acceleration"],
    "expected_performance": "Significant speedup over CPU for large inputs",
    "hardware_requirements": "NVIDIA GPU with CUDA support"
}
```

#### Advanced CUDA Techniques

1. **Custom CUDA Kernels** with CuPy:
   ```python
   kernel_code = r'''
   extern "C" __global__
   void my_custom_kernel(float* input, float* output, int n) {
       int idx = blockDim.x * blockIdx.x + threadIdx.x;
       if (idx < n) {
           output[idx] = input[idx] * 2.0f;
       }
   }
   '''
   
   # Compile the kernel
   my_kernel = cp.RawKernel(kernel_code, 'my_custom_kernel')
   
   def solution(input_data):
       # Ensure float32 type
       if input_data.dtype != np.float32:
           input_data = input_data.astype(np.float32)
       
       # Transfer to GPU
       gpu_input = cp.asarray(input_data)
       gpu_output = cp.empty_like(gpu_input)
       
       # Configure kernel launch
       threads_per_block = 256
       blocks_per_grid = (input_data.size + threads_per_block - 1) // threads_per_block
       
       # Launch kernel
       my_kernel((blocks_per_grid,), (threads_per_block,), (gpu_input, gpu_output, input_data.size))
       
       # Transfer result back to CPU
       return cp.asnumpy(gpu_output)
   ```

2. **Stream Processing** for concurrent operations:
   ```python
   def solution(matrix_a, matrix_b, matrix_c):
       """Matrix-vector multiplication distributed across multiple GPUs."""
       num_gpus = cp.cuda.runtime.getDeviceCount()
       
       # Split matrix by rows
       rows_per_gpu = matrix_a.shape[0] // num_gpus
       results = []
       
       for i in range(num_gpus):
           with cp.cuda.Device(i):
               # Calculate start and end row
               start_row = i * rows_per_gpu
               end_row = (i + 1) * rows_per_gpu if i < num_gpus - 1 else matrix_a.shape[0]
               
               # Get matrix slice for this GPU
               matrix_slice = matrix_a[start_row:end_row]
               
               # Copy data to current GPU
               gpu_matrix = cp.asarray(matrix_slice)
               gpu_vector = cp.asarray(matrix_b)
               
               # Compute result for this partition
               gpu_result = cp.dot(gpu_matrix, gpu_vector)
               results.append((start_row, end_row, cp.asnumpy(gpu_result)))
       
       # Combine results
       final_result = np.zeros(matrix_a.shape[0], dtype=np.float32)
       for start, end, result in results:
           final_result[start:end] = result
           
       return final_result
   ```

3. **Memory Management** for large-scale operations:
   ```python
   def solution(large_matrix):
       # Clear GPU memory cache before operation
       cp.get_default_memory_pool().free_all_blocks()
       
       # Check available memory
       mem_info = cp.cuda.Device().mem_info
       available_memory = mem_info[0]
       required_memory = large_matrix.nbytes
       
       if required_memory > available_memory * 0.9:  # 90% threshold
           # Process in chunks if matrix is too large
           chunk_size = int(available_memory * 0.5 / large_matrix.shape[1] / 4)  # 50% of available memory, 4 bytes per float32
           result = np.empty_like(large_matrix)
           
           for i in range(0, large_matrix.shape[0], chunk_size):
               end = min(i + chunk_size, large_matrix.shape[0])
               # Process chunk
               chunk = large_matrix[i:end]
               gpu_chunk = cp.asarray(chunk)
               # Process on GPU
               result_chunk = cp.asnumpy(gpu_chunk * 2.0)  # Example operation
               result[i:end] = result_chunk
               # Free memory
               del gpu_chunk
               cp.get_default_memory_pool().free_all_blocks()
           
           return result
       else:
           # Process the entire matrix at once
           gpu_matrix = cp.asarray(large_matrix)
           result = cp.asnumpy(gpu_matrix * 2.0)  # Example operation
           return result
   ```

#### Matrix-Vector Dot Product CUDA Example

Here's a specific example for implementing matrix-vector dot product using CUDA/CuPy:

```python
import numpy as np
import cupy as cp

def solution(matrix, vector):
    """
    CUDA implementation of matrix-vector dot product using CuPy.
    
    Args:
        matrix: 2D NumPy array of shape (M, N)
        vector: 1D NumPy array of shape (N,)
        
    Returns:
        1D NumPy array of shape (M,)
    """
    # Validate input types
    assert matrix.dtype == np.float32, "Matrix must be float32"
    assert vector.dtype == np.float32, "Vector must be float32"
    
    # Validate input dimensions
    assert len(matrix.shape) == 2, "Matrix must be 2D"
    assert len(vector.shape) == 1, "Vector must be 1D"
    assert matrix.shape[1] == vector.shape[0], f"Matrix columns ({matrix.shape[1]}) must match vector length ({vector.shape[0]})"
    
    try:
        # Convert to CuPy arrays
        matrix_gpu = cp.asarray(matrix)
        vector_gpu = cp.asarray(vector)
        
        # Use CuPy's dot product (calls optimized cuBLAS)
        result_gpu = cp.dot(matrix_gpu, vector_gpu)
        
        # Transfer back to CPU
        return cp.asnumpy(result_gpu)
    except Exception as e:
        print(f"CUDA error: {e}")
        print("Falling back to NumPy implementation")
        return np.dot(matrix, vector)

# Add implementation metadata
IMPLEMENTATION_METADATA = {
    "name": "Basic CUDA Implementation",
    "version": "v1",
    "description": "Matrix-vector dot product using CuPy/cuBLAS",
    "date": "2025-04-05",
    "author": "Your Name",
    "optimization_techniques": ["cuBLAS acceleration"],
    "expected_performance": "5-20x speedup over NumPy for large matrices",
    "hardware_requirements": "NVIDIA GPU with CUDA support"
}
```

## Transitioning from Python to GPU

To effectively transition from Python to GPU implementations:

1. **Start with a working Python implementation** as a reference.
2. **Identify parallelizable components** in your algorithm.
3. **Create a naive GPU port** first, focusing on correctness.
4. **Compare results** with the Python version using small test cases.
5. **Optimize iteratively**, measuring performance at each step.
6. **Add robust error handling** for GPU-specific issues.

## Debugging GPU Implementations

When debugging GPU implementations, add the following to your toolkit:

1. **GPU Memory Monitoring**
   ```python
   # Add this to your solution function for debugging
   def debug_solution(matrix, vector, debug=True):
       if debug:
           # Before GPU operations
           if 'cp' in globals():  # If CuPy is available
               print(f"GPU memory before: {cp.cuda.Device().mem_info[0]/1e9:.2f}GB free")
       
       # Your regular solution code here
       # ...
       
       if debug:
           # After GPU operations
           if 'cp' in globals():
               print(f"GPU memory after: {cp.cuda.Device().mem_info[0]/1e9:.2f}GB free")
       
       return result
   ```

2. **Gradual Problem Size Scaling**
   ```python
   # Create a script that gradually increases problem size to find limits
   import numpy as np
   
   try:
       import cupy as cp
       has_gpu = True
   except ImportError:
       has_gpu = False
   
   def find_gpu_limits():
       if not has_gpu:
           print("GPU support not available")
           return
       
       # Try increasing sizes until failure
       for size in [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
           try:
               print(f"Testing size {size}...")
               matrix = np.random.rand(size, size).astype(np.float32)
               vector = np.random.rand(size).astype(np.float32)
               
               # Convert to GPU
               matrix_gpu = cp.asarray(matrix)
               vector_gpu = cp.asarray(vector)
               
               # Run computation
               result = cp.dot(matrix_gpu, vector_gpu)
               
               # Force synchronization
               cp.cuda.Stream.null.synchronize()
               
               # Get memory stats
               mem_free = cp.cuda.Device().mem_info[0]/1e9
               print(f"Size {size} succeeded. Free memory: {mem_free:.2f}GB")
               
               # Clean up
               del matrix_gpu, vector_gpu, result
               cp.get_default_memory_pool().free_all_blocks()
           except Exception as e:
               print(f"Failed at size {size}: {e}")
               break
   ```

3. **Verification Using Small Data**
   ```python
   def verify_implementations():
       # Create small test case where results are easily verified
       matrix = np.array([[1, 2], [3, 4]], dtype=np.float32)
       vector = np.array([1, 2], dtype=np.float32)
       
       # Expected result
       expected = np.dot(matrix, vector)  # [5, 11]
       
       # Test Python implementation
       from solutions.group_XX_category.pXXX_problem_name.python.solution_vY import solution as py_solution
       result_py = py_solution(matrix, vector)
       print(f"Python result: {result_py}")
       print(f"Python matches: {np.allclose(result_py, expected)}")
       
       # Test CUDA implementation if available
       try:
           from solutions.group_XX_category.pXXX_problem_name.cuda.solution_v1 import solution as cuda_solution
           result_cuda = cuda_solution(matrix, vector)
           print(f"CUDA result: {result_cuda}")
           print(f"CUDA matches: {np.allclose(result_cuda, expected)}")
       except ImportError:
           print("CUDA implementation not available")
   ```

## Best Practices for GPU Implementations

1. **Data Preparation**:
   - Use `np.float32` instead of `np.float64` when possible.
   - Align memory to improve transfer efficiency.
   - Minimize host-device transfers (they're expensive).

2. **Algorithm Design**:
   - Maximize parallelism in your algorithms.
   - Minimize divergent branching within kernels.
   - Coalesce memory access patterns.

3. **Performance Measurement**:
   - Always include GPU initialization and memory transfers in benchmarks.
   - Use event timers for accurate kernel timing:
     ```python
     # CUDA timing example
     start_event = cp.cuda.Event()
     end_event = cp.cuda.Event()
     
     start_event.record()
     # GPU operation
     gpu_result = cp.dot(gpu_a, gpu_b)
     end_event.record()
     end_event.synchronize()
     
     elapsed_time = cp.cuda.get_elapsed_time(start_event, end_event)
     print(f"Kernel execution time: {elapsed_time} milliseconds")
     ```

4. **Multi-GPU Support** (for advanced implementations):
   ```python
   def solution_multi_gpu(matrix, vector):
       """Matrix-vector multiplication distributed across multiple GPUs."""
       num_gpus = cp.cuda.runtime.getDeviceCount()
       
       # Split matrix by rows
       rows_per_gpu = matrix.shape[0] // num_gpus
       results = []
       
       for i in range(num_gpus):
           with cp.cuda.Device(i):
               # Calculate start and end row
               start_row = i * rows_per_gpu
               end_row = (i + 1) * rows_per_gpu if i < num_gpus - 1 else matrix.shape[0]
               
               # Get matrix slice for this GPU
               matrix_slice = matrix[start_row:end_row]
               
               # Copy data to current GPU
               gpu_matrix = cp.asarray(matrix_slice)
               gpu_vector = cp.asarray(vector)
               
               # Compute result for this partition
               gpu_result = cp.dot(gpu_matrix, gpu_vector)
               results.append((start_row, end_row, cp.asnumpy(gpu_result)))
       
       # Combine results
       final_result = np.zeros(matrix.shape[0], dtype=np.float32)
       for start, end, result in results:
           final_result[start:end] = result
           
       return final_result
   ```

## Common GPU Issues and Solutions

| Issue | Description | Solution |
|-------|-------------|----------|
| `ImportError: libcuda.so.1: cannot open shared object file` | CUDA driver not found | Install NVIDIA GPU drivers |
| `ImportError: No module named 'cupy'` | CuPy not installed | `pip install cupy-cuda12x` (match your CUDA version) |
| `ImportError: No module named 'triton'` | Triton not installed | `pip install triton` |
| `OutOfMemoryError: out of memory` | GPU memory exceeded | Reduce batch/problem size or process in chunks |
| `Illegal Memory Access` | Invalid memory address | Check array bounds and synchronization points |
| `Device-side assert triggered` | Assertion failed in kernel | Add bounds checking in kernel code |
| `All CUDA-capable devices are busy or unavailable` | GPUs in use or driver issues | Check `nvidia-smi`, restart driver |

## Transitioning from Python to GPU

To effectively transition from Python to GPU implementations:

1. **Start with a working Python implementation** as a reference.
2. **Identify parallelizable components** in your algorithm.
3. **Create a naive GPU port** first, focusing on correctness.
4. **Compare results** with the Python version using small test cases.
5. **Optimize iteratively**, measuring performance at each step.
6. **Add robust error handling** for GPU-specific issues.

## Debugging GPU Implementations

When debugging GPU implementations, add the following to your toolkit:

1. **GPU Memory Monitoring**
   ```python
   # Add this to your solution function for debugging
   def debug_solution(matrix, vector, debug=True):
       if debug:
           # Before GPU operations
           if 'cp' in globals():  # If CuPy is available
               print(f"GPU memory before: {cp.cuda.Device().mem_info[0]/1e9:.2f}GB free")
       
       # Your regular solution code here
       # ...
       
       if debug:
           # After GPU operations
           if 'cp' in globals():
               print(f"GPU memory after: {cp.cuda.Device().mem_info[0]/1e9:.2f}GB free")
       
       return result
   ```

2. **Gradual Problem Size Scaling**
   ```python
   # Create a script that gradually increases problem size to find limits
   import numpy as np
   
   try:
       import cupy as cp
       has_gpu = True
   except ImportError:
       has_gpu = False
   
   def find_gpu_limits():
       if not has_gpu:
           print("GPU support not available")
           return
       
       # Try increasing sizes until failure
       for size in [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
           try:
               print(f"Testing size {size}...")
               matrix = np.random.rand(size, size).astype(np.float32)
               vector = np.random.rand(size).astype(np.float32)
               
               # Convert to GPU
               matrix_gpu = cp.asarray(matrix)
               vector_gpu = cp.asarray(vector)
               
               # Run computation
               result = cp.dot(matrix_gpu, vector_gpu)
               
               # Force synchronization
               cp.cuda.Stream.null.synchronize()
               
               # Get memory stats
               mem_free = cp.cuda.Device().mem_info[0]/1e9
               print(f"Size {size} succeeded. Free memory: {mem_free:.2f}GB")
               
               # Clean up
               del matrix_gpu, vector_gpu, result
               cp.get_default_memory_pool().free_all_blocks()
           except Exception as e:
               print(f"Failed at size {size}: {e}")
               break
   ```

3. **Verification Using Small Data**
   ```python
   def verify_implementations():
       # Create small test case where results are easily verified
       matrix = np.array([[1, 2], [3, 4]], dtype=np.float32)
       vector = np.array([1, 2], dtype=np.float32)
       
       # Expected result
       expected = np.dot(matrix, vector)  # [5, 11]
       
       # Test Python implementation
       from solutions.group_XX_category.pXXX_problem_name.python.solution_vY import solution as py_solution
       result_py = py_solution(matrix, vector)
       print(f"Python result: {result_py}")
       print(f"Python matches: {np.allclose(result_py, expected)}")
       
       # Test CUDA implementation if available
       try:
           from solutions.group_XX_category.pXXX_problem_name.cuda.solution_v1 import solution as cuda_solution
           result_cuda = cuda_solution(matrix, vector)
           print(f"CUDA result: {result_cuda}")
           print(f"CUDA matches: {np.allclose(result_cuda, expected)}")
       except ImportError:
           print("CUDA implementation not available")
   ```

## Best Practices for GPU Implementations

1. **Data Preparation**:
   - Use `np.float32` instead of `np.float64` when possible.
   - Align memory to improve transfer efficiency.
   - Minimize host-device transfers (they're expensive).

2. **Algorithm Design**:
   - Maximize parallelism in your algorithms.
   - Minimize divergent branching within kernels.
   - Coalesce memory access patterns.

3. **Performance Measurement**:
   - Always include GPU initialization and memory transfers in benchmarks.
   - Use event timers for accurate kernel timing:
     ```python
     # CUDA timing example
     start_event = cp.cuda.Event()
     end_event = cp.cuda.Event()
     
     start_event.record()
     # GPU operation
     gpu_result = cp.dot(gpu_a, gpu_b)
     end_event.record()
     end_event.synchronize()
     
     elapsed_time = cp.cuda.get_elapsed_time(start_event, end_event)
     print(f"Kernel execution time: {elapsed_time} milliseconds")
     ```

4. **Multi-GPU Support** (for advanced implementations):
   ```python
   def solution_multi_gpu(matrix, vector):
       """Matrix-vector multiplication distributed across multiple GPUs."""
       num_gpus = cp.cuda.runtime.getDeviceCount()
       
       # Split matrix by rows
       rows_per_gpu = matrix.shape[0] // num_gpus
       results = []
       
       for i in range(num_gpus):
           with cp.cuda.Device(i):
               # Calculate start and end row
               start_row = i * rows_per_gpu
               end_row = (i + 1) * rows_per_gpu if i < num_gpus - 1 else matrix.shape[0]
               
               # Get matrix slice for this GPU
               matrix_slice = matrix[start_row:end_row]
               
               # Copy data to current GPU
               gpu_matrix = cp.asarray(matrix_slice)
               gpu_vector = cp.asarray(vector)
               
               # Compute result for this partition
               gpu_result = cp.dot(gpu_matrix, gpu_vector)
               results.append((start_row, end_row, cp.asnumpy(gpu_result)))
       
       # Combine results
       final_result = np.zeros(matrix.shape[0], dtype=np.float32)
       for start, end, result in results:
           final_result[start:end] = result
           
       return final_result
   ```
