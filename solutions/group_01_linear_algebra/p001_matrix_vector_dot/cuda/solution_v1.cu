/*
Problem 1: Matrix-Vector Dot Product
Implementation of a function to compute the dot product between a matrix and a vector.

Raw CUDA Implementation
*/

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matrix_vector_dot_kernel(
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

// Host function to launch the kernel
extern "C" void solution_launcher(float* h_matrix, float* h_vector, float* h_output, int m, int n) {
    // Allocate device memory
    float *d_matrix, *d_vector, *d_output;
    cudaMalloc(&d_matrix, m * n * sizeof(float));
    cudaMalloc(&d_vector, n * sizeof(float));
    cudaMalloc(&d_output, m * sizeof(float));
    
    // Copy inputs to device
    cudaMemcpy(d_matrix, h_matrix, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, h_vector, n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int block_size = 256;
    int grid_size = (m + block_size - 1) / block_size;
    
    // Matrix strides (assuming row-major layout)
    int matrix_stride_row = n;
    int matrix_stride_col = 1;
    
    matrix_vector_dot_kernel<<<grid_size, block_size>>>(
        d_matrix, d_vector, d_output, m, n, matrix_stride_row, matrix_stride_col
    );
    
    // Copy result back to host
    cudaMemcpy(h_output, d_output, m * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_matrix);
    cudaFree(d_vector);
    cudaFree(d_output);
}

/*
To compile this file:
nvcc -o solution.o -c solution_cuda.cu -arch=sm_XX
where XX is your GPU's compute capability (e.g., 60 for Pascal, 70 for Volta)

To create a shared library:
nvcc -shared -o libsolution.so solution.o
*/
