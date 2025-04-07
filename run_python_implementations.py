#!/usr/bin/env python
"""
Script to directly run the Python implementations of matrix-vector dot product
and compare their performance.

This script doesn't require Triton or CUDA, just focuses on the Python implementations.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Ensure project root is in path
ROOT_DIR = Path(__file__).parent
import sys
sys.path.insert(0, str(ROOT_DIR))

# Import path manager to ensure directories exist
from utils.path_manager import ensure_directories_exist

# Direct imports of the solution functions
from solutions.group_01_linear_algebra.p001_matrix_vector_dot.python.solution_v1 import solution as solution_v1
from solutions.group_01_linear_algebra.p001_matrix_vector_dot.python.solution_v2_optimized import solution as solution_v2

def benchmark_implementation(solution_func, matrix, vector, name, num_runs=5):
    """
    Benchmark a specific implementation.
    
    Args:
        solution_func: Solution function to benchmark
        matrix: Input matrix
        vector: Input vector
        name: Name of the implementation
        num_runs: Number of benchmark runs
        
    Returns:
        Average execution time
    """
    print(f"Benchmarking {name}...")
    
    # Warmup run (not included in timing)
    solution_func(matrix, vector)
    
    # Time multiple runs
    times = []
    for i in range(num_runs):
        start_time = time.time()
        result = solution_func(matrix, vector)
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000  # Convert to ms
        times.append(execution_time)
        print(f"  Run {i+1}/{num_runs}: {execution_time:.3f} ms")
    
    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"  Average: {avg_time:.3f} ms")
    print(f"  Min: {min_time:.3f} ms")
    print(f"  Max: {max_time:.3f} ms")
    
    return avg_time, result

def run_size_benchmark(sizes):
    """
    Run benchmark for multiple input sizes.
    
    Args:
        sizes: List of input sizes to benchmark
    """
    # Ensure directories exist for saving plots
    ensure_directories_exist()
    
    v1_times = []
    v2_times = []
    
    for size in sizes:
        print(f"\nBenchmarking with size {size}x{size}")
        
        # Generate random matrix and vector
        matrix = np.random.rand(size, size).astype(np.float32)
        vector = np.random.rand(size).astype(np.float32)
        
        # Benchmark both implementations
        avg_v1, result_v1 = benchmark_implementation(solution_v1, matrix, vector, "solution_v1")
        avg_v2, result_v2 = benchmark_implementation(solution_v2, matrix, vector, "solution_v2_optimized")
        
        # Verify results match
        match = np.allclose(result_v1, result_v2, rtol=1e-5, atol=1e-5)
        print(f"Results match: {match}")
        
        # Store times
        v1_times.append(avg_v1)
        v2_times.append(avg_v2)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, v1_times, 'o-', label='v1 (Basic NumPy)')
    plt.plot(sizes, v2_times, 's-', label='v2 (Optimized)')
    plt.xlabel('Matrix Size')
    plt.ylabel('Execution Time (ms)')
    plt.title('Python Implementation Performance Comparison')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig(ROOT_DIR / 'python_implementation_comparison.png')
    print(f"\nPlot saved to python_implementation_comparison.png")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    print("=" * 80)
    print("Matrix-Vector Dot Product Python Implementation Benchmark")
    print("=" * 80)
    
    # Run benchmark with multiple sizes
    sizes = [100, 500, 1000, 2000]
    run_size_benchmark(sizes)
