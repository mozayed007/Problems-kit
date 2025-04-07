#!/usr/bin/env python
"""
Verification script for Python implementations in Problems-Kit.

This script checks all Python components of the system to ensure they're working
correctly, focusing on the Python implementations since CUDA and Triton
aren't available.
"""

import os
import sys
import importlib
import numpy as np
from pathlib import Path
import traceback

# Add the project root to Python path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

# Import path management utilities to ensure directories exist
from utils.path_manager import ensure_directories_exist

# Ensure all required directories exist
ensure_directories_exist()

def check_imports():
    """Check that all necessary modules can be imported."""
    print("\nChecking imports...")
    
    required_modules = [
        "utils.bench_runner_enhanced",
        "utils.benchmark_suite",
        "utils.benchmark_visualization",
        "utils.benchmark_enhanced",
        "utils.path_manager"
    ]
    
    for module_name in required_modules:
        try:
            module = importlib.import_module(module_name)
            print(f"✓ Successfully imported {module_name}")
        except Exception as e:
            print(f"✗ Failed to import {module_name}: {e}")
            traceback.print_exc()
            return False
    
    return True

def check_python_implementation():
    """Check that Python implementations can be loaded and executed."""
    print("\nChecking Python implementations...")
    
    # Try to import the matrix-vector dot product implementations
    try:
        from solutions.group_01_linear_algebra.p001_matrix_vector_dot.python.solution_v1 import solution as solution_v1
        print("✓ Successfully imported solution_v1")
        
        # Test with a small example
        matrix = np.random.rand(10, 10).astype(np.float32)
        vector = np.random.rand(10).astype(np.float32)
        
        result = solution_v1(matrix, vector)
        numpy_result = np.dot(matrix, vector)
        
        if np.allclose(result, numpy_result):
            print("✓ solution_v1 produces correct results")
        else:
            print("✗ solution_v1 produces incorrect results")
            return False
        
        # Try to import the optimized version
        try:
            from solutions.group_01_linear_algebra.p001_matrix_vector_dot.python.solution_v2_optimized import solution as solution_v2
            print("✓ Successfully imported solution_v2_optimized")
            
            result_v2 = solution_v2(matrix, vector)
            
            if np.allclose(result_v2, numpy_result):
                print("✓ solution_v2_optimized produces correct results")
            else:
                print("✗ solution_v2_optimized produces incorrect results")
                return False
        except ImportError:
            print("○ solution_v2_optimized not found (optional)")
        
        return True
        
    except Exception as e:
        print(f"✗ Error checking Python implementation: {e}")
        traceback.print_exc()
        return False

def check_input_generation():
    """Check that input generation works correctly."""
    print("\nChecking input generation...")
    
    try:
        from solutions.group_01_linear_algebra.p001_matrix_vector_dot import generate_inputs
        
        # Test with default size
        args, kwargs = generate_inputs()
        print(f"✓ generate_inputs() returned args with shapes: {[a.shape for a in args]}")
        
        # Test with custom size
        args, kwargs = generate_inputs(size=50)
        if args[0].shape == (50, 50) and args[1].shape == (50,):
            print(f"✓ generate_inputs(size=50) returned correct shapes: {[a.shape for a in args]}")
        else:
            print(f"✗ generate_inputs(size=50) returned incorrect shapes: {[a.shape for a in args]}")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Error checking input generation: {e}")
        traceback.print_exc()
        return False

def check_benchmark_runner():
    """Check that the benchmark runner can be initialized."""
    print("\nChecking benchmark runner...")
    
    try:
        from utils.bench_runner_enhanced import check_implementation, list_implementations
        
        # Check if implementations are properly registered
        implementations = list_implementations('p001_matrix_vector_dot')
        print(f"✓ Found implementations: {implementations}")
        
        # Check specific implementations
        if check_implementation('p001_matrix_vector_dot', 'python', 'v1'):
            print("✓ Python v1 implementation exists")
        else:
            print("✗ Python v1 implementation not found")
            
        return True
    except Exception as e:
        print(f"✗ Error checking benchmark runner: {e}")
        traceback.print_exc()
        return False

def run_micro_benchmark():
    """Run a small benchmark to test the system."""
    print("\nRunning micro benchmark...")
    
    try:
        from utils.bench_runner_enhanced import run_benchmark
        
        # Run a very small benchmark
        run_benchmark(
            problem_id='p001_matrix_vector_dot',
            implementations=[('python', 'v1')],
            num_runs=2,
            warmup_runs=1,
            input_sizes=[10],
            show_plots=False,
            save_plots=True,
            export_csv=True,
            track_memory=True
        )
        
        print("✓ Micro benchmark completed successfully")
        return True
    except Exception as e:
        print(f"✗ Error running micro benchmark: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all verification checks."""
    print("=" * 80)
    print(" Problems-Kit Python Implementation Verification ")
    print("=" * 80)
    
    # Run all checks
    checks = [
        check_imports,
        check_python_implementation,
        check_input_generation,
        check_benchmark_runner,
        run_micro_benchmark
    ]
    
    results = []
    for check in checks:
        results.append(check())
    
    # Summarize results
    print("\n" + "=" * 80)
    print(" Verification Summary ")
    print("=" * 80)
    
    all_passed = all(results)
    
    if all_passed:
        print("\n✓ All checks PASSED! The Python implementation is working correctly.")
        print("  You can now run the full benchmark with:")
        print("  python run_benchmark_example.py")
    else:
        print("\n✗ Some checks FAILED. Please fix the issues above before proceeding.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
