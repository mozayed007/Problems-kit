#!/usr/bin/env python
"""
Test all implementations of matrix-vector dot product.
This script tests Python, CUDA, and Triton implementations if available.
"""

import numpy as np
import os
import sys
from pathlib import Path
import time

# Add the project root to Python path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

# Test matrices of different sizes
test_sizes = [32, 64, 128, 256, 512, 1024]

def generate_test_data(size):
    """Generate random test matrix and vector"""
    matrix = np.random.rand(size, size).astype(np.float32)
    vector = np.random.rand(size).astype(np.float32)
    return matrix, vector

def run_test(implementation_name, solution_func, matrix, vector):
    """Run a test for a specific implementation"""
    start_time = time.time()
    result = solution_func(matrix, vector)
    end_time = time.time()
    
    # Verify with NumPy
    expected = np.dot(matrix, vector)
    is_correct = np.allclose(result, expected, rtol=1e-5)
    
    return {
        "implementation": implementation_name,
        "size": matrix.shape[0],
        "time": end_time - start_time,
        "correct": is_correct,
        "max_diff": np.max(np.abs(result - expected)) if is_correct else float('inf')
    }

def print_result(result):
    """Print a test result in a formatted way"""
    status = "✅ PASS" if result["correct"] else "❌ FAIL"
    print(f"{result['implementation']} ({result['size']}x{result['size']}): {status} - {result['time']:.6f}s - Max diff: {result['max_diff']:.8f}")

def main():
    print("\n== Matrix-Vector Dot Product Implementation Tests ==\n")
    
    # Import Python implementation (should always be available)
    try:
        from solutions.group_01_linear_algebra.p001_matrix_vector_dot.python.solution_v1 import solution as py_solution
        has_python = True
        print("✅ Python implementation found")
    except ImportError:
        has_python = False
        print("❌ Python implementation not found")
    
    # Try to import CUDA implementation
    try:
        from solutions.group_01_linear_algebra.p001_matrix_vector_dot.cuda.solution_v1 import solution as cuda_solution
        has_cuda = True
        print("✅ CUDA implementation found")
    except ImportError as e:
        has_cuda = False
        print(f"❌ CUDA implementation not available: {e}")
    
    # Try to import Triton implementation
    try:
        from solutions.group_01_linear_algebra.p001_matrix_vector_dot.triton.solution_v1 import solution as triton_solution
        
        # Test with a small example to verify it actually works
        test_m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        test_v = np.array([2, 3, 4], dtype=np.float32)
        result = triton_solution(test_m, test_v)
        expected = np.dot(test_m, test_v)
        
        if np.allclose(result, expected, rtol=1e-5):
            has_triton = True
            print("✅ Triton implementation found and works")
        else:
            has_triton = False
            print("❌ Triton implementation found but produces incorrect results")
    except Exception as e:
        has_triton = False
        print(f"❌ Triton implementation not available or has errors: {e}")
    
    print("\n== Running Performance Tests ==\n")
    
    all_results = []
    
    # Run tests for each implementation and size
    for size in test_sizes:
        matrix, vector = generate_test_data(size)
        
        print(f"\nTesting with matrix size {size}x{size}:")
        
        if has_python:
            result = run_test("Python", py_solution, matrix, vector)
            all_results.append(result)
            print_result(result)
        
        if has_cuda:
            try:
                result = run_test("CUDA", cuda_solution, matrix, vector)
                all_results.append(result)
                print_result(result)
            except Exception as e:
                print(f"CUDA implementation failed for size {size}: {e}")
        
        if has_triton:
            try:
                result = run_test("Triton", triton_solution, matrix, vector)
                all_results.append(result)
                print_result(result)
            except Exception as e:
                print(f"Triton implementation failed for size {size}: {e}")
    
    print("\n== Summary ==\n")
    
    if all(r["correct"] for r in all_results if r["implementation"] == "Python"):
        print("✅ Python implementation: All tests PASSED")
    else:
        print("❌ Python implementation: Some tests FAILED")
    
    if has_cuda:
        if all(r["correct"] for r in all_results if r["implementation"] == "CUDA"):
            print("✅ CUDA implementation: All tests PASSED")
        else:
            print("❌ CUDA implementation: Some tests FAILED")
    
    if has_triton:
        if all(r["correct"] for r in all_results if r["implementation"] == "Triton"):
            print("✅ Triton implementation: All tests PASSED")
        else:
            print("❌ Triton implementation: Some tests FAILED")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
