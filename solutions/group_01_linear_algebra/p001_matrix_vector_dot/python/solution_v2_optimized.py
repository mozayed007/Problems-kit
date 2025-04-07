"""
Problem 1: Matrix-Vector Dot Product
Implementation of a function to compute the dot product between a matrix and a vector.

Python Implementation - Optimized version using NumPy's optimized functions and more efficient memory handling
"""

import numpy as np
from typing import Union, List, Tuple


def solution(matrix: Union[np.ndarray, List[List[float]]], vector: Union[np.ndarray, List[float]]) -> np.ndarray:
    """
    Compute the dot product between a matrix and a vector using optimized NumPy operations.
    
    This variant focuses on efficient memory usage and leveraging NumPy's optimized BLAS backend.
    
    Args:
        matrix: Input matrix of shape (m, n)
        vector: Input vector of shape (n,)
        
    Returns:
        Result vector of shape (m,)
    """
    # Convert inputs to contiguous numpy arrays with correct dtype for performance
    if not isinstance(matrix, np.ndarray):
        matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    elif not matrix.flags.c_contiguous:
        matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    else:
        matrix = matrix.astype(np.float32, copy=False)
    
    if not isinstance(vector, np.ndarray):
        vector = np.ascontiguousarray(vector, dtype=np.float32)
    elif not vector.flags.c_contiguous:
        vector = np.ascontiguousarray(vector, dtype=np.float32)
    else:
        vector = vector.astype(np.float32, copy=False)
    
    # Use NumPy's optimized dot product
    # This will use BLAS backend when available, which is highly optimized
    result = np.dot(matrix, vector)
    
    return result


# Metadata for the implementation registry
IMPLEMENTATION_METADATA = {
    "name": "Optimized NumPy",
    "version": "v2",
    "description": "Optimized matrix-vector dot product using NumPy's BLAS backend",
    "date": "2025-04-05",
    "author": "Problems-Kit Team",
    "optimization_techniques": [
        "Contiguous memory layout",
        "BLAS backend utilization",
        "Efficient type conversion"
    ],
    "expected_performance": "Best performance for Python implementations without manual loops"
}


if __name__ == "__main__":
    # Example usage
    m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    v = np.array([2, 3, 4], dtype=np.float32)
    
    result = solution(m, v)
    print("Matrix:")
    print(m)
    print("\nVector:")
    print(v)
    print("\nMatrix-Vector Dot Product (Optimized):")
    print(result)
