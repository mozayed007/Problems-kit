"""
Test cases for Problem 1: Matrix-Vector Dot Product
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import Python implementation
from solutions.group_01_linear_algebra.p001_matrix_vector_dot.solution_py import solution as py_solution

# Constants
SAMPLE_MATRIX = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
SAMPLE_VECTOR = np.array([2, 3, 4], dtype=np.float32)
EXPECTED_RESULT = np.array([20, 47, 74], dtype=np.float32)  # Known result for the sample inputs


def test_python_implementation():
    """Test Python implementation of matrix-vector dot product."""
    result = py_solution(SAMPLE_MATRIX, SAMPLE_VECTOR)
    np.testing.assert_allclose(result, EXPECTED_RESULT, rtol=1e-5)


@pytest.mark.skip(reason="Triton not available on this machine")
def test_triton_implementation():
    """Test Triton implementation of matrix-vector dot product."""
    pass


@pytest.mark.skip(reason="CUDA not available on this machine")
def test_cuda_implementation():
    """Test CUDA implementation of matrix-vector dot product."""
    pass


def test_larger_inputs():
    """Test with larger inputs to ensure scalability."""
    # Create a larger test case
    m, n = 1000, 500
    matrix = np.random.rand(m, n).astype(np.float32)
    vector = np.random.rand(n).astype(np.float32)
    
    # Calculate expected result using NumPy's dot product
    expected = np.dot(matrix, vector)
    
    # Test Python implementation
    result = py_solution(matrix, vector)
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_edge_cases():
    """Test edge cases."""
    # Test with a 1x1 matrix
    assert py_solution(np.array([[5]], dtype=np.float32), np.array([2], dtype=np.float32))[0] == 10
    
    # Test with a matrix with zeros
    matrix = np.zeros((3, 3), dtype=np.float32)
    vector = np.ones(3, dtype=np.float32)
    result = py_solution(matrix, vector)
    np.testing.assert_allclose(result, np.zeros(3, dtype=np.float32))
