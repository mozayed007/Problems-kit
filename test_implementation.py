#!/usr/bin/env python
"""
Test script for the enhanced implementation system.
This script demonstrates how to select and run specific implementation variants.
"""

import sys
from pathlib import Path

# Add project root to Python path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

def test_python_implementation():
    """Test both Python implementation variants for matrix-vector dot product."""
    import numpy as np
    
    # Create test data
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    vector = np.array([2, 3, 4], dtype=np.float32)
    
    print("\nTesting matrix-vector dot product implementations...")
    print(f"Matrix:\n{matrix}")
    print(f"Vector:\n{vector}")
    
    # Test the v1 implementation
    try:
        from solutions.group_01_linear_algebra.p001_matrix_vector_dot.python.solution_v1 import solution as v1_solution
        print("\nRunning Python v1 implementation:")
        result_v1 = v1_solution(matrix, vector)
        print(f"Result v1:\n{result_v1}")
    except ImportError:
        print("Python v1 implementation not found.")
    
    # Test the v2_optimized implementation
    try:
        from solutions.group_01_linear_algebra.p001_matrix_vector_dot.python.solution_v2_optimized import solution as v2_solution
        print("\nRunning Python v2_optimized implementation:")
        result_v2 = v2_solution(matrix, vector)
        print(f"Result v2:\n{result_v2}")
    except ImportError:
        print("Python v2_optimized implementation not found.")
    
    # Compare with NumPy reference implementation
    numpy_result = np.dot(matrix, vector)
    print(f"\nNumPy reference result:\n{numpy_result}")

def test_implementation_registry():
    """Test the implementation registry system."""
    print("\nTesting implementation registry system...")
    
    try:
        # First, test direct import
        from solutions.group_01_linear_algebra.p001_matrix_vector_dot import list_all_implementations
        
        print("Available implementations:")
        implementations = list_all_implementations()
        
        for impl_type, variants in implementations.items():
            print(f"\n{impl_type.capitalize()} implementations:")
            for variant, metadata in variants.items():
                print(f"  - {variant}: {metadata.get('name', 'Unknown')}")
                if 'description' in metadata:
                    print(f"    {metadata['description']}")
        
        # Test getting a specific implementation
        from solutions.group_01_linear_algebra.p001_matrix_vector_dot import get_implementation
        
        print("\nTesting specific implementation retrieval:")
        py_impl = get_implementation('python', 'v2_optimized')
        if py_impl:
            print("Successfully retrieved Python v2_optimized implementation!")
        else:
            print("Failed to retrieve Python v2_optimized implementation.")
    
    except ImportError as e:
        print(f"Error accessing implementation registry: {e}")

if __name__ == "__main__":
    print("=" * 80)
    print("Testing enhanced implementation system")
    print("=" * 80)
    
    test_python_implementation()
    print("\n" + "=" * 80)
    test_implementation_registry()
