"""
Problem 1: Matrix-Vector Dot Product
Implementation of a function to compute the dot product between a matrix and a vector.

Python Implementation
"""

import numpy as np
from typing import Union, List, Tuple


def solution(matrix: Union[np.ndarray, List[List[float]]], vector: Union[np.ndarray, List[float]]) -> np.ndarray:
    """
    Compute the dot product between a matrix and a vector.
    
    Args:
        matrix: Input matrix of shape (m, n)
        vector: Input vector of shape (n,)
        
    Returns:
        Result vector of shape (m,)
    """
    # Convert inputs to numpy arrays if they're not already
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix, dtype=np.float32)
    
    if not isinstance(vector, np.ndarray):
        vector = np.array(vector, dtype=np.float32)
    
    # Perform the matrix-vector dot product
    result = np.dot(matrix, vector)
    
    return result


# Metadata for the implementation registry
IMPLEMENTATION_METADATA = {
    "name": "Basic NumPy",
    "version": "v1",
    "description": "Basic matrix-vector dot product using NumPy",
    "date": "2025-04-05",
    "author": "Problems-Kit Team",
    "optimization_techniques": [],
    "expected_performance": "Baseline performance"
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
    print("\nMatrix-Vector Dot Product:")
    print(result)
