# Problems-Kit

A comprehensive implementation and benchmarking system for algorithm problems with solutions in Python, Triton, and CUDA.

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.7%2B-blue)

## Overview

Problems-Kit is a structured framework for implementing, testing, and benchmarking algorithms across different platforms. The system allows you to:

- Implement solutions in Python, Triton, and CUDA
- Create multiple optimization variants for each implementation
- Run comprehensive benchmarks with detailed metrics
- Generate visualization charts and CSV exports
- Compare performance across platforms and implementation strategies

The repository is organized to facilitate algorithmic problem-solving, performance optimization, and knowledge transfer between different computing paradigms.

## Key Features

- **Multi-Platform Support**: Implement solutions in pure Python, Triton (GPU), and CUDA
- **Multiple Implementation Variants**: Create and compare different optimization approaches
- **Enhanced Benchmarking System**: Detailed performance metrics including:
  - Execution timing (mean, median, min, max, std)
  - Memory usage tracking
  - Performance scaling with input size
  - Outlier detection and filtering
- **Rich Visualization**: Generate charts for performance comparison
- **CSV Export**: Export benchmark results for further analysis
- **Command-Line Interface**: User-friendly menu system for navigation
- **Comprehensive Test Suite**: Verify implementation correctness

## Getting Started

### Prerequisites

- Python 3.7+
- NumPy
- Matplotlib
- For GPU acceleration (optional):
  - NVIDIA GPU with CUDA support
  - CUDA Toolkit
  - Triton package (for Triton implementations)
  - CuPy (for CUDA implementations)

### Quick Start

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Problems-kit
   ```

2. Run the CLI interface:
   ```bash
   python problems_kit.py
   ```

3. For quick testing of Python implementations:
   ```bash
   python run_python_implementations.py
   ```

4. For benchmarking with the enhanced system:
   ```bash
   python run_benchmark_example.py
   ```

5. For verifying Python implementation setup:
   ```bash
   python verify_python_implementation.py
   ```

## Project Structure

```
Problems-kit/
│
├── problems_kit.py                  # Main CLI interface
├── problems.md                      # Problem definitions and checklist
├── User_Guide.md                    # Comprehensive user guide
├── run_benchmark_example.py         # Example benchmark script
├── run_python_implementations.py    # Python implementation test script
├── run_setup.py                     # Setup script for initial configuration
├── test_implementation.py           # Test script for implementations
├── verify_python_implementation.py  # Verification script for Python setup
│
├── solutions/                       # Solutions organized by category and problem
│   ├── categories.py                # Categories definitions
│   ├── group_01_linear_algebra/     # Group 1: Fundamental Linear Algebra & Basic ML
│   │   ├── p001_matrix_vector_dot/  # Problem 1: Matrix-Vector Dot Product
│   │   │   ├── __init__.py          # Problem registration & input generation
│   │   │   ├── python/              # Python implementations
│   │   │   │   ├── __init__.py      # Python implementation registry
│   │   │   │   ├── solution_v1.py   # Baseline implementation
│   │   │   │   └── solution_v2_optimized.py # Optimized variant
│   │   │   ├── triton/              # Triton implementations
│   │   │   └── cuda/                # CUDA implementations
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

## Usage Guide

### Running Benchmarks

The Problems-Kit provides several ways to run benchmarks:

1. **Using the CLI Interface**:
   ```bash
   python problems_kit.py
   ```
   Navigate to a problem and select the benchmark option.

2. **Direct Benchmark Example**:
   ```bash
   python run_benchmark_example.py
   ```
   This runs a predefined benchmark for the matrix-vector dot product.

3. **Python-Only Comparison**:
   ```bash
   python run_python_implementations.py
   ```
   This focuses on comparing Python implementations without CUDA or Triton.

### Benchmarking Options

The enhanced benchmarking system supports:

- Multiple input sizes for scaling analysis
- Memory usage tracking
- Customizable number of runs and warmup iterations
- Interactive and static visualization
- CSV export for further analysis
- Implementation metadata tracking

### Implementing New Solutions

To add a new implementation for an existing problem:

1. Navigate to the appropriate problem directory
2. Create a new solution file with the required function signature
3. Register it in the `__init__.py` file
4. Run tests and benchmarks to compare performance

For a new problem implementation see the complete User_Guide.md.

## Contributing

Contributions are welcome! Please see our contribution guidelines for details.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NumPy and SciPy communities
- NVIDIA & OpenAI for CUDA and Triton
- Contributors and reviewers

For more detailed information, please refer to the [User_Guide.md](User_Guide.md) file.
