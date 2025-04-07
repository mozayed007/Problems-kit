"""
Problem 1: Matrix-Vector Dot Product
Implementation of a function to compute the dot product between a matrix and a vector.

CUDA Implementation (Python bindings)
"""

import numpy as np
from typing import Union, List

try:
    import cupy as cp
except ImportError:
    cp = None


def solution(matrix: Union[np.ndarray, List[List[float]]], vector: Union[np.ndarray, List[float]]) -> np.ndarray:
    """
    Compute the dot product between a matrix and a vector using CUDA.
    
    Args:
        matrix: Input matrix of shape (m, n)
        vector: Input vector of shape (n,)
        
    Returns:
        Result vector of shape (m,)
    """
    if cp is None:
        raise ImportError("CuPy is not installed. Install it with: pip install cupy-cudaXX")
    
    # Convert inputs to numpy arrays if they're not already
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix, dtype=np.float32)
    
    if not isinstance(vector, np.ndarray):
        vector = np.array(vector, dtype=np.float32)
    
    # Ensure inputs are float32
    matrix = matrix.astype(np.float32)
    vector = vector.astype(np.float32)
    
    # Transfer to GPU
    matrix_gpu = cp.asarray(matrix)
    vector_gpu = cp.asarray(vector)
    
    # Perform matrix-vector dot product on GPU
    result_gpu = cp.dot(matrix_gpu, vector_gpu)
    
    # Transfer result back to CPU
    result = cp.asnumpy(result_gpu)
    
    return result


# Define CUDA kernel as a string if custom implementation is needed
_cuda_kernel = r"""
extern "C" __global__ void matrix_vector_dot_kernel(
    const float* matrix, const float* vector, float* output,
    int m, int n, int matrix_stride_row, int matrix_stride_col) 
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += matrix[row * matrix_stride_row + j * matrix_stride_col] * vector[j];
        }
        output[row] = sum;
    }
}
"""

# Compile and get the kernel function if custom implementation is needed
# If CuPy is available, this can be used instead of the built-in cp.dot
def _custom_matrix_vector_dot(matrix, vector):
    if cp is None:
        raise ImportError("CuPy is not installed")
    
    # Compile the kernel
    matrix_vector_dot_kernel = cp.RawKernel(_cuda_kernel, 'matrix_vector_dot_kernel')
    
    # Get dimensions
    m, n = matrix.shape
    
    # Allocate output
    output = cp.zeros(m, dtype=cp.float32)
    
    # Calculate grid and block dimensions
    block_size = 256
    grid_size = (m + block_size - 1) // block_size
    
    # Calculate strides
    matrix_stride_row = matrix.strides[0] // 4  # Convert byte strides to element strides
    matrix_stride_col = matrix.strides[1] // 4
    
    # Launch the kernel
    matrix_vector_dot_kernel((grid_size,), (block_size,), (
        matrix, vector, output, m, n, matrix_stride_row, matrix_stride_col
    ))
    
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
        print("\nMatrix-Vector Dot Product (CUDA):")
        print(result)
        
        # Verify with NumPy
        numpy_result = np.dot(m, v)
        print("\nNumPy Result:")
        print(numpy_result)
        print("Difference:", np.max(np.abs(result - numpy_result)))
        
        # Try custom implementation
        if cp is not None:
            print("\nCustom CUDA Kernel Implementation:")
            m_gpu = cp.asarray(m)
            v_gpu = cp.asarray(v)
            custom_result = _custom_matrix_vector_dot(m_gpu, v_gpu)
            print(cp.asnumpy(custom_result))
            print("Difference from NumPy:", np.max(np.abs(cp.asnumpy(custom_result) - numpy_result)))
    except ImportError as e:
        print(f"CUDA not available: {e}")
