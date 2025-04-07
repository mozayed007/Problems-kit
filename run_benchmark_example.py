#!/usr/bin/env python
"""
Example script to run enhanced benchmarking with the matrix-vector dot product.
This resolves the import issues and properly generates test data.
"""

import numpy as np
from utils.bench_runner_enhanced import run_benchmark
from utils.path_manager import ensure_directories_exist

def main():
    # Ensure all needed directories exist
    ensure_directories_exist()
    
    # Run benchmark comparing v1 and v2_optimized Python implementations
    print("Running benchmark for matrix-vector dot product...")
    
    # Run with multiple input sizes to show scaling performance
    input_sizes = [128, 512, 1024, 2048]
    
    run_benchmark(
        problem_id='p001_matrix_vector_dot',
        implementations=[('python', 'v1'), ('python', 'v2_optimized')], 
        num_runs=5,
        warmup_runs=2,
        input_sizes=input_sizes,
        show_plots=True,
        export_csv=True,
        track_memory=True
    )
    
    print("Benchmark complete!")

if __name__ == "__main__":
    main()
