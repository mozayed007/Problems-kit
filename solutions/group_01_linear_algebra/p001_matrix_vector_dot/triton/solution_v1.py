"""
Problem 1: Matrix-Vector Dot Product
Implementation of a function to compute the dot product between a matrix and a vector.

Triton Implementation
"""

import numpy as np
import triton
import triton.language as tl
from typing import Union, List


@triton.jit
def matrix_vector_dot_kernel(
    matrix_ptr, vector_ptr, output_ptr,
    m, n, 
    matrix_stride_row, matrix_stride_col,
    BLOCK_SIZE: tl.constexpr
):
    """
    Triton kernel for matrix-vector dot product.
    
    Args:
        matrix_ptr: Pointer to the matrix
        vector_ptr: Pointer to the vector
        output_ptr: Pointer to the output vector
        m: Number of rows in the matrix
        n: Number of columns in the matrix
        matrix_stride_row: Row stride of the matrix
        matrix_stride_col: Column stride of the matrix
        BLOCK_SIZE: Block size for parallelization
    """
    # Get the program ID
    pid = tl.program_id(axis=0)
    
    # Check if this thread is responsible for a valid row
    if pid < m:
        # Initialize accumulator for dot product
        acc = 0.0
        
        # Load row of the matrix and perform dot product with the vector
        for j in range(0, n, BLOCK_SIZE):
            # Create a mask to handle the case where n is not a multiple of BLOCK_SIZE
            mask = tl.arange(0, BLOCK_SIZE) < n - j
            
            # Load a block of the vector
            v_block = tl.load(vector_ptr + j, mask=mask, other=0.0)
            
            # Load a block of the matrix row
            m_block = tl.load(matrix_ptr + pid * matrix_stride_row + j * matrix_stride_col, mask=mask, other=0.0)
            
            # Multiply and accumulate
            acc += tl.sum(m_block * v_block, axis=0)
        
        # Store the result
        tl.store(output_ptr + pid, acc)


def solution(matrix: Union[np.ndarray, List[List[float]]], vector: Union[np.ndarray, List[float]]) -> np.ndarray:
    """
    Compute the dot product between a matrix and a vector using Triton.
    
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
    
    # Ensure inputs are float32
    matrix = matrix.astype(np.float32)
    vector = vector.astype(np.float32)
    
    # Get dimensions
    m, n = matrix.shape
    
    # Allocate output
    output = np.empty(m, dtype=np.float32)
    
    # Launch the kernel
    grid = lambda meta: (triton.cdiv(m, meta['BLOCK_SIZE']),)
    
    matrix_vector_dot_kernel[grid](
        matrix, vector, output,
        m, n, 
        matrix.strides[0] // 4, matrix.strides[1] // 4,  # Convert byte strides to element strides
        BLOCK_SIZE=min(128, n)  # Choose a reasonable block size
    )
    
    return output


if __name__ == "__main__":
    # Example usage
    m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    v = np.array([2, 3, 4], dtype=np.float32)
    
    try:
        result = solution(m, v)
        print("Matrix:")
        print(m)
        print("\nVector:")
        print(v)
        print("\nMatrix-Vector Dot Product (Triton):")
        print(result)
        
        # Verify with NumPy
        numpy_result = np.dot(m, v)
        print("\nNumPy Result:")
        print(numpy_result)
        print("Difference:", np.max(np.abs(result - numpy_result)))
    except ImportError:
        print("Triton not available. Make sure to install it with: pip install triton")
