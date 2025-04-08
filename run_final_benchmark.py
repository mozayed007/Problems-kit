#!/usr/bin/env python
"""
Final benchmark script with proper imports to run all implementations.
This version uses the unified benchmarking system for better organization and consistency.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import torch
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Add the root directory to the path to ensure imports work
sys.path.append(str(Path(__file__).parent))

# Import the unified benchmarking system
from utils.benchmark_unified import (
    run_problem_benchmark,
    save_benchmark_results,
    get_adaptive_error_threshold,
    BenchmarkResult
)

# Import enhanced visualization tools
from utils.enhanced_visualizations import (
    create_performance_plot,
    create_scaling_analysis_plot,
    create_accuracy_comparison_plot,
    generate_complete_visualization_suite,
    create_html_dashboard
)

# Ensure the output directories exist
os.makedirs("benchmarks/csv", exist_ok=True)
os.makedirs("benchmarks/json", exist_ok=True)
os.makedirs("benchmarks/visualizations", exist_ok=True)

def generate_matrix_vector_inputs(size):
    """Generate random matrix and vector for testing
    
    Args:
        size: Size of the matrix (size x size) and vector (size)
        
    Returns:
        Tuple of (matrix, vector) as numpy arrays
    """
    np.random.seed(42)  # For reproducibility
    matrix = np.random.randn(size, size).astype(np.float32)
    vector = np.random.randn(size).astype(np.float32)
    return matrix, vector

def plot_benchmark_summary(results: List[BenchmarkResult], save_path: Optional[str] = None):
    """Generate and save benchmark summary plots
    
    Args:
        results: List of benchmark results
        save_path: Path to save the plot, or None to display only
    """
    # Group results by size
    by_size = {}
    for result in results:
        size = result.input_size
        if size not in by_size:
            by_size[size] = []
        by_size[size].append(result)
    
    # Extract unique implementations and sizes
    implementations = set()
    sizes = sorted(by_size.keys())
    
    for result in results:
        implementations.add(f"{result.implementation_type} ({result.variant})")
    implementations = sorted(implementations)
    
    # Create performance plot
    plt.figure(figsize=(12, 8))
    
    # Set up colors and markers
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    markers = ['o', 's', '^', 'D', 'x']
    
    # Plot performance by size for each implementation
    for i, impl in enumerate(implementations):
        impl_type, variant = impl.split(' ')[0], impl.split(' ')[1].strip('()')
        
        mean_times = []
        for size in unique_sizes:
            # Find matching result
            matching = [r for r in by_size[size] 
                      if r.implementation_type == impl_type and r.variant == variant]
            
            if matching:
                # Convert to milliseconds
                mean_times.append(matching[0].stats['mean'] * 1000)
            else:
                mean_times.append(float('nan'))
        
        # Plot this implementation
        plt.plot(sizes, mean_times, 
                 marker=markers[i % len(markers)],
                 color=colors[i % len(colors)],
                 label=impl)
    
    # Set up plot appearance
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.xlabel('Matrix Size')
    plt.ylabel('Mean Execution Time (ms)')
    plt.title('Performance Comparison of Different Implementations')
    plt.legend()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def main():
    print("=" * 80)
    print("Final Matrix-Vector Dot Product Benchmark")
    print("Comparing Python, CUDA, and Triton Implementations")
    print("Using the Unified Benchmarking System")
    print("=" * 80)
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU detected: {gpu_name}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("No GPU detected. CUDA and Triton implementations might not work.")
    
    # Define problem ID - can be overridden by command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run benchmarks for a specific problem')
    parser.add_argument('--problem-id', type=str, default='p001_matrix_vector_dot',
                        help='ID of the problem to benchmark')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to custom benchmark configuration file')
    parser.add_argument('--sizes', type=str, default=None,
                        help='Comma-separated list of input sizes to test')
    args = parser.parse_args()
    
    problem_id = args.problem_id
    
    # Check if configuration exists, create if needed
    config_path = None
    if args.config:
        config_path = args.config
    else:
        # Load or create default configuration
        from utils.benchmark_config import ensure_config_exists
        config = ensure_config_exists(problem_id)
        config_path = f"configs/benchmarks/{problem_id}_benchmark_config.json"
        print(f"Using configuration: {config_path}")
    
    # Override sizes if specified in command line
    input_sizes = None
    if args.sizes:
        try:
            input_sizes = [int(size) for size in args.sizes.split(',')]
            print(f"Using custom sizes: {input_sizes}")
        except ValueError:
            print(f"Invalid size format. Expected comma-separated integers, got: {args.sizes}")
            input_sizes = None
    
    # Run benchmark using configuration
    print("\nRunning benchmarks...")
    results = run_problem_benchmark(
        problem_id=problem_id,
        input_sizes=input_sizes,  # Will use config if None
        config_path=config_path
    )
    
    # Save results to files
    output_files = save_benchmark_results(results)
    
    # Generate comprehensive visualizations
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    visualization_dir = f"benchmarks/visualizations"
    
    # Create visualization suite
    viz_files = generate_complete_visualization_suite(
        results=results,
        output_dir=visualization_dir,
        problem_id=problem_id,
        timestamp=timestamp,
        generate_html=True  # Set to True if plotly is available
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"Benchmark complete! Results saved to:")
    print("\nData files:")
    for format_name, file_path in output_files.items():
        print(f"- {format_name.upper()}: {file_path}")
    
    print("\nVisualizations:")
    for viz_type, viz_path in viz_files.items():
        if viz_path:  # Only print if file was successfully created
            print(f"- {viz_type.capitalize()}: {viz_path}")
    print("=" * 80 + "\n")
    
    # Create performance summary by implementation
    # Extract all unique sizes from results
    unique_sizes = sorted(list(set(result.input_size for result in results)))
    # Use the smallest size to get the list of implementations
    implementations_list = [f"{result.implementation_type} ({result.variant})" for result in results 
                           if result.input_size == unique_sizes[0]]
    implementations_list = sorted(set(implementations_list))
    
    print("Performance Summary (mean execution time in milliseconds):")
    print("-" * 80)
    headers = ["Size"] + implementations_list
    print(f"{headers[0]:<10}", end="")
    for header in headers[1:]:
        print(f"{header:<15}", end="")
    print()
    print("-" * 80)
    
    for size in unique_sizes:
        print(f"{size:<10}", end="")
        
        size_results = [r for r in results if r.input_size == size]
        for impl_name in implementations_list:
            impl_type, variant = impl_name.split(' ')[0], impl_name.split(' ')[1].strip('()')
            
            # Find matching result
            matching = [r for r in size_results 
                      if r.implementation_type == impl_type and r.variant == variant]
            
            if matching and hasattr(matching[0], 'stats'):
                print(f"{matching[0].stats['mean']*1000:.2f} ms", end="       ")
            else:
                print("ERROR", end="          ")
        print()
    
    # Create performance scaling table
    print("\nPerformance Scaling (relative to size 128):")
    print("-" * 80)
    print(f"{headers[0]:<10}", end="")
    for header in headers[1:]:
        print(f"{header:<15}", end="")
    print()
    print("-" * 80)
    
    # Get baseline times for each implementation at the smallest size
    baseline_size = unique_sizes[0]
    baseline_results = [r for r in results if r.input_size == baseline_size]
    baseline_times = {}
    
    for result in baseline_results:
        impl_key = f"{result.implementation_type} ({result.variant})"
        if hasattr(result, 'stats'):
            baseline_times[impl_key] = result.stats['mean']
    
    for size in unique_sizes:
        print(f"{size:<10}", end="")
        
        size_results = [r for r in results if r.input_size == size]
        for impl_name in implementations_list:
            # Find matching result
            matching = [r for r in size_results 
                      if f"{r.implementation_type} ({r.variant})" == impl_name]
            
            if matching and impl_name in baseline_times and baseline_times[impl_name] > 0:
                scaling = matching[0].stats['mean'] / baseline_times[impl_name]
                print(f"{scaling:.2f}x", end="          ")
            else:
                print("N/A", end="            ")
        print()
    
    # Create accuracy summary table
    print("\nAccuracy Summary (max difference from NumPy):")
    print("-" * 80)
    print(f"{headers[0]:<10}", end="")
    for header in headers[1:]:
        print(f"{header:<15}", end="")
    print()
    print("-" * 80)
    
    for size in unique_sizes:
        print(f"{size:<10}", end="")
        
        size_results = [r for r in results if r.input_size == size]
        for impl_name in implementations_list:
            impl_type, variant = impl_name.split(' ')[0], impl_name.split(' ')[1].strip('()')
            
            # Find matching result
            matching = [r for r in size_results 
                      if r.implementation_type == impl_type and r.variant == variant]
            
            if matching and hasattr(matching[0], 'error') and matching[0].error is not None:
                print(f"{matching[0].error:.8f}", end="    ")
            else:
                print("ERROR", end="          ")
        print()
    
    # Print files generated
    print("\nFiles to check:")
    i = 1
    print("Data files:")
    for format_name, file_path in output_files.items():
        print(f"{i}. {format_name.upper()} results: {file_path}")
        i += 1
    
    print("\nVisualizations:")
    for viz_type, viz_path in viz_files.items():
        if viz_path:  # Only print if file was successfully created
            print(f"{i}. {viz_type.capitalize()}: {viz_path}")
            i += 1

if __name__ == "__main__":
    main()
