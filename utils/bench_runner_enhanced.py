"""
Enhanced benchmark runner for Problems-Kit.

This module provides high-level functions to run benchmarks for problems using
the enhanced benchmarking system with CSV output and additional metrics.
"""

import os
import importlib
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import sys
import traceback

# Add the project root to Python path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Import benchmark components
from utils.benchmark_suite import BenchmarkSuite
from utils.benchmark_visualization import (
    plot_execution_times,
    plot_size_comparison,
    export_to_csv,
    save_benchmark_results
)

def run_benchmark(problem_id: str, 
                 function_name: str = 'solution',
                 implementations: Optional[List[Tuple[str, str]]] = None,
                 num_runs: int = 10,
                 warmup_runs: int = 3,
                 input_sizes: Optional[List[int]] = None,
                 show_plots: bool = True,
                 save_plots: bool = True,
                 export_csv: bool = True,
                 track_memory: bool = True,
                 filter_outliers: bool = False):
    """
    Run benchmarks for a specific problem with multiple implementations and variants.
    
    Args:
        problem_id: ID of the problem (e.g., 'p001_matrix_vector_dot')
        function_name: Name of the function to benchmark
        implementations: List of (impl_type, variant) tuples to benchmark
                        If None, benchmarks all available implementations
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        input_sizes: List of input sizes to benchmark
                    If None, uses the problem's default input sizes
        show_plots: Whether to display the plots
        save_plots: Whether to save the plots to file
        export_csv: Whether to export results to CSV
        track_memory: Whether to track memory usage
        filter_outliers: Whether to filter outliers from results
        
    Returns:
        Dictionary mapping (impl_type, variant) to BenchmarkResult
    """
    print(f"\n{'='*80}\nBenchmarking problem {problem_id}\n{'='*80}")
    
    # Create benchmark suite
    benchmark_suite = BenchmarkSuite(problem_id)
    
    # If no implementations specified, use all available ones
    if implementations is None:
        available_impls = benchmark_suite.list_implementations()
        implementations = []
        for impl_type, variants in available_impls.items():
            for variant in variants:
                implementations.append((impl_type, variant))
    
    # Check if any implementations found
    if not implementations:
        print(f"No implementations found for problem {problem_id}")
        return {}
    
    print(f"Found {len(implementations)} implementation variants to benchmark")
    for impl_type, variant in implementations:
        print(f"  - {impl_type} ({variant})")
    
    results = {}
    
    # If benchmarking across multiple input sizes
    if input_sizes:
        print(f"\nBenchmarking across {len(input_sizes)} input sizes: {input_sizes}")
        size_results = {}
        
        for size in input_sizes:
            print(f"\nGenerating inputs for size {size}...")
            
            # Try to import the problem module for generating inputs
            try:
                module_path = f"solutions.{benchmark_suite.problem_path.parent.name}.{problem_id}"
                problem_module = importlib.import_module(module_path)
                
                # Check if the module has a generate_inputs function
                if hasattr(problem_module, 'generate_inputs'):
                    input_args, input_kwargs = problem_module.generate_inputs(size)
                    print(f"Generated inputs for size {size}")
                else:
                    # Fall back to a default implementation based on the problem type
                    # For matrix-vector, we could assume inputs are a matrix and vector of appropriate sizes
                    if 'matrix_vector' in problem_id:
                        input_args = [np.random.rand(size, size).astype(np.float32), 
                                     np.random.rand(size).astype(np.float32)]
                        input_kwargs = {}
                        print(f"Generated default matrix-vector inputs for size {size}")
                    else:
                        input_args = [size]
                        input_kwargs = {}
                        print(f"Using size {size} as a direct input parameter")
            except ImportError:
                # Fall back to simple inputs
                input_args = [size]
                input_kwargs = {}
                print(f"Using size {size} as a direct input parameter (no problem module found)")
            
            # Benchmark each implementation with this input size
            for impl_type, variant in implementations:
                try:
                    print(f"\nBenchmarking {impl_type} ({variant}) with size {size}...")
                    
                    # Run the benchmark
                    result = benchmark_suite.benchmark_implementation(
                        impl_type=impl_type,
                        variant=variant,
                        function_name=function_name,
                        args=input_args,
                        kwargs=input_kwargs,
                        num_runs=num_runs,
                        warmup_runs=warmup_runs,
                        track_memory=track_memory
                    )
                    
                    # Filter outliers if requested
                    if filter_outliers:
                        num_outliers = result.filter_outliers()
                        if num_outliers > 0:
                            print(f"Filtered {num_outliers} outliers from {impl_type} ({variant}) results")
                    
                    # Format time for readability
                    if result.mean_time < 0.001:
                        time_str = f"{result.mean_time*1e6:.2f} µs"
                    elif result.mean_time < 1.0:
                        time_str = f"{result.mean_time*1e3:.2f} ms"
                    else:
                        time_str = f"{result.mean_time:.4f} s"
                    
                    print(f"Mean execution time: {time_str}")
                    print(f"Standard deviation: {result.std_time:.6f} s")
                    
                    if result.memory_usage:
                        print(f"Mean memory usage: {result.mean_memory:.2f} MB")
                    
                    # Store result
                    size_results[(impl_type, variant, size)] = result
                except Exception as e:
                    print(f"Error benchmarking {impl_type} ({variant}) for size {size}: {e}")
                    traceback.print_exc()
        
        # Plot results for each size
        plot_size_comparison(size_results, problem_id, show_plots=show_plots, save_plots=save_plots)
        
        # Group results by size for comparison
        for size in input_sizes:
            size_group = {}
            for (impl_type, variant, s), result in size_results.items():
                if s == size:
                    results[(impl_type, variant)] = result
            
            # Only create plots if we have results for this size
            if size_group:
                print(f"\nPlotting results for size {size}...")
                plot_execution_times(size_group, f"{problem_id}_size_{size}", 
                                    show_plots=show_plots, save_plots=save_plots)
                
                if export_csv:
                    export_to_csv(size_group, f"{problem_id}_size_{size}")
    else:
        # Benchmark with default inputs
        print("\nGenerating default inputs...")
        
        # Try to import the problem module for generating inputs
        try:
            module_path = f"solutions.{benchmark_suite.problem_path.parent.name}.{problem_id}"
            problem_module = importlib.import_module(module_path)
            
            # Check if the module has a generate_inputs function
            if hasattr(problem_module, 'generate_inputs'):
                input_args, input_kwargs = problem_module.generate_inputs()
                print("Generated default inputs")
            else:
                # Fall back to empty inputs (implementation should have default params)
                input_args = []
                input_kwargs = {}
                print("No input generation function found, using empty inputs")
        except ImportError:
            # Fall back to empty inputs
            input_args = []
            input_kwargs = {}
            print("No problem module found, using empty inputs")
        
        # Benchmark each implementation
        for impl_type, variant in implementations:
            try:
                print(f"\nBenchmarking {impl_type} ({variant})...")
                
                # Run the benchmark
                result = benchmark_suite.benchmark_implementation(
                    impl_type=impl_type,
                    variant=variant,
                    function_name=function_name,
                    args=input_args,
                    kwargs=input_kwargs,
                    num_runs=num_runs,
                    warmup_runs=warmup_runs,
                    track_memory=track_memory
                )
                
                # Filter outliers if requested
                if filter_outliers:
                    num_outliers = result.filter_outliers()
                    if num_outliers > 0:
                        print(f"Filtered {num_outliers} outliers from {impl_type} ({variant}) results")
                
                # Format time for readability
                if result.mean_time < 0.001:
                    time_str = f"{result.mean_time*1e6:.2f} µs"
                elif result.mean_time < 1.0:
                    time_str = f"{result.mean_time*1e3:.2f} ms"
                else:
                    time_str = f"{result.mean_time:.4f} s"
                
                print(f"Mean execution time: {time_str}")
                print(f"Standard deviation: {result.std_time:.6f} s")
                
                if result.memory_usage:
                    print(f"Mean memory usage: {result.mean_memory:.2f} MB")
                
                # Store result
                results[(impl_type, variant)] = result
            except Exception as e:
                print(f"Error benchmarking {impl_type} ({variant}): {e}")
                traceback.print_exc()
    
    # Create summary plots and export results
    if results:
        print("\nCreating summary visualizations...")
        plot_execution_times(results, problem_id, show_plots=show_plots, save_plots=save_plots)
        
        if export_csv:
            print("\nExporting results to CSV...")
            csv_path = export_to_csv(results, problem_id)
            print(f"Results exported to {csv_path}")
        
        # Save results to JSON for future reference
        save_benchmark_results(results, problem_id)
    
    return results


def check_implementation(problem_id: str, impl_type: str, variant: str = 'v1') -> bool:
    """
    Check if a specific implementation variant exists for a problem.
    
    Args:
        problem_id: ID of the problem
        impl_type: Type of implementation ('python', 'triton', 'cuda')
        variant: Variant of the implementation
        
    Returns:
        True if the implementation exists, False otherwise
    """
    try:
        # Create benchmark suite to use its utilities
        benchmark_suite = BenchmarkSuite(problem_id)
        
        # Get available implementations
        available_impls = benchmark_suite.list_implementations()
        
        # Check if the implementation exists
        if impl_type in available_impls and variant in available_impls[impl_type]:
            return True
        
        # Try to import it directly as a fallback
        benchmark_suite._import_implementation(impl_type, variant)
        return True
    except Exception:
        return False


def list_implementations(problem_id: str) -> Dict[str, List[str]]:
    """
    List all available implementations for a problem.
    
    Args:
        problem_id: ID of the problem
        
    Returns:
        Dictionary mapping implementation types to lists of variants
    """
    try:
        # Create benchmark suite
        benchmark_suite = BenchmarkSuite(problem_id)
        
        # Get available implementations
        return benchmark_suite.list_implementations()
    except Exception as e:
        print(f"Error listing implementations for {problem_id}: {e}")
        return {}
