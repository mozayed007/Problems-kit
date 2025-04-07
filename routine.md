# Problems-Kit Routine Guide

This document provides a step-by-step routine for solving problems using the Problems-Kit, focusing first on Python implementations with guidance for later GPU implementations.

## Getting Started

### Step 1: Select a Problem

1. Open `problems.md` and browse the available problems from the implementation plan
2. Choose a problem based on:
   - Categories in Group 1-4 (Fundamental Linear Algebra, Data Transformations, etc.)
   - Current day's focus from the plan
   - Your learning priorities
3. Note the problem ID and group (e.g., `p001_matrix_vector_dot` in `group_01_linear_algebra`)

### Step 2: Set Up Problem Structure

1. Ensure the problem directory exists:
   ```
   solutions/group_XX_category/pXXX_problem_name/
   ```
2. If not, create it with the proper structure:
   - `__init__.py` - Problem registration
   - `python/` folder for Python implementations
   - `python/__init__.py` - Python implementation registry
   - (Future) `triton/` and `cuda/` folders for GPU implementations

3. Create or update the problem's `__init__.py` with:
   ```python
   import numpy as np
   
   # Import implementation modules
   try:
       from . import python
       PYTHON_AVAILABLE = True
   except ImportError:
       PYTHON_AVAILABLE = False
   
   try:
       from . import triton
       TRITON_AVAILABLE = True
   except ImportError:
       TRITON_AVAILABLE = False
   
   try:
       from . import cuda
       CUDA_AVAILABLE = True
   except ImportError:
       CUDA_AVAILABLE = False
   
   # The input generation can now be handled by utils.benchmark_generators
   # This function should match the name specified in your benchmark config
   def generate_problem_specific_inputs(size=1000):
       """
       Generate input data for benchmarking.
       
       Args:
           size: Size parameter for generated inputs
           
       Returns:
           tuple: (args, kwargs) to pass to solution functions
       """
       # Example for matrix-vector dot product:
       matrix = np.random.rand(size, size).astype(np.float32)
       vector = np.random.rand(size).astype(np.float32)
       return [matrix, vector], {}
   ```

### Step 3: Understand the Problem

1. Define the problem clearly in terms of:
   - Input format and constraints
   - Expected output
   - Performance considerations

## Python Implementation Phase

### Step 4: Create a Baseline Python Implementation (v1)

1. Create `python/solution_v1.py` with a basic implementation:
   ```python
   import numpy as np
   
   def solution(arg1, arg2, ...):
       """
       Basic implementation of the algorithm.
       
       Args:
           arg1: First input argument
           arg2: Second input argument
           
       Returns:
           Expected output for the problem
       """
       # Your basic implementation here
       # Focus on correctness first, not performance
       
       return result
   
   # Add implementation metadata
   IMPLEMENTATION_METADATA = {
       "name": "Basic Implementation",
       "version": "v1",
       "description": "Basic implementation using standard approaches",
       "date": "2025-04-05",  # Today's date
       "author": "Your Name",
       "optimization_techniques": [],
       "expected_performance": "Baseline performance"
   }
   ```

2. Register your implementation in `python/__init__.py`:
   ```python
   from .solution_v1 import solution as solution_v1
   
   # You can add more implementations later
   ```

### Step 5: Test Your Baseline Implementation

1. Run the verification script to check your environment:
   ```bash
   python verify_python_implementation.py
   ```

2. Test your specific implementation:
   ```bash
   python test_implementation.py
   ```

3. Or create a test helper in your solution file:
   ```python
   # Add this to your solution_v1.py file
   if __name__ == "__main__":
       # Create small test inputs
       # For matrix-vector dot product example:
       import numpy as np
       
       matrix = np.array([[1, 2], [3, 4]], dtype=np.float32)
       vector = np.array([1, 2], dtype=np.float32)
       
       # Run your solution
       result = solution(matrix, vector)
       print(f"Result: {result}")
       
       # Compare with expected result
       expected = np.dot(matrix, vector)
       print(f"Expected: {expected}")
       print(f"Match: {np.allclose(result, expected)}")
   ```

### Step 6: Benchmark Your Baseline Solution

1. Create a benchmark configuration file in `configs/benchmarks/pXXX_problem_name_benchmark_config.json`:
   ```json
   {
       "problem_id": "pXXX_problem_name",
       "name": "Problem Description",
       "description": "Benchmark for baseline implementation",
       "implementations": [
           ["python", "v1"]
       ],
       "input_sizes": [128, 256, 512, 1024, 2048],
       "num_runs": 10,
       "warmup_runs": 3,
       "error_thresholds": {
           "default": 0.0001
       }
   }
   ```

2. Run the unified benchmarking system:
   ```bash
   python run_final_benchmark.py --problem-id pXXX_problem_name
   ```

3. Review the generated performance metrics and visualizations in the `benchmarks/` directory

## Optimization Phase

### Step 7: Create an Optimized Python Implementation (v2)

1. Create `python/solution_v2_optimized.py` with improvements:
   ```python
   import numpy as np
   
   def solution(arg1, arg2, ...):
       """
       Optimized implementation of the algorithm.
       
       Args:
           arg1: First input argument
           arg2: Second input argument
           
       Returns:
           Expected output for the problem
       """
       # Your optimized implementation
       # Focus on Python-specific optimizations:
       # - Vectorization with NumPy
       # - Efficient memory usage
       # - Algorithm improvements
       
       return result
   
   # Add implementation metadata
   IMPLEMENTATION_METADATA = {
       "name": "Optimized Implementation",
       "version": "v2_optimized",
       "description": "Optimized implementation using [specific techniques]",
       "date": "2025-04-05",  # Today's date
       "author": "Your Name",
       "optimization_techniques": ["vectorization", "technique2"],
       "expected_performance": "XX% improvement over baseline"
   }
   ```

2. Update the registry in `python/__init__.py`:
   ```python
   from .solution_v1 import solution as solution_v1
   from .solution_v2_optimized import solution as solution_v2_optimized
   ```

### Step 8: Test and Benchmark the Optimized Solution

1. Verify correctness:
   ```bash
   python test_implementation.py --problem_id pXXX_problem_name --implementation python.v2_optimized
   ```

2. Update your benchmark configuration to include both implementations:
   ```json
   {
       "problem_id": "pXXX_problem_name",
       "name": "Problem Description",
       "description": "Benchmark comparing baseline and optimized implementations",
       "implementations": [
           ["python", "v1"],
           ["python", "v2_optimized"]
       ],
       "input_sizes": [128, 256, 512, 1024, 2048, 4096],
       "num_runs": 10,
       "warmup_runs": 3,
       "error_thresholds": {
           "default": 0.0001
       }
   }
   ```

3. Run the unified benchmarking system:
   ```bash
   python run_final_benchmark.py --problem-id pXXX_problem_name
   ```

4. Alternatively, use the API directly in a custom script:
   ```python
   # Example custom benchmark script using unified system
   import os
   from utils.benchmark_unified import run_problem_benchmark
   from utils.enhanced_visualizations import generate_complete_visualization_suite
   
   def main():
       # Define problem ID
       problem_id = "pXXX_problem_name"
       
       # Run benchmark using configuration file
       print("Running benchmark...")
       results = run_problem_benchmark(
           problem_id=problem_id,
           config_path=f"configs/benchmarks/{problem_id}_benchmark_config.json"
       )
       
       # Generate enhanced visualizations
       viz_files = generate_complete_visualization_suite(
           results=results,
           output_dir="benchmarks/visualizations",
           problem_id=problem_id,
           generate_html=True
       )
       
   if __name__ == "__main__":
       main()
   ```

4. Analyze the performance improvements

## Analysis and Documentation

### Step 9: Analyze Results and Document Findings

1. Document key findings in your code:
   - Performance bottlenecks identified
   - Optimization techniques used
   - Percentage improvements 

2. Use the visualization outputs from the unified benchmarking system:
   - Review the performance comparison plots (`problem_id_performance_timestamp.png`)
   - Analyze the scaling behavior plots (`problem_id_scaling_timestamp.png`)
   - Check the numerical accuracy plots if applicable (`problem_id_accuracy_timestamp.png`)
   - Explore the interactive HTML dashboard for detailed insights (`problem_id_dashboard_timestamp.html`)

3. Update IMPLEMENTATION_METADATA with actual performance data

### Step 10: Update Problems Tracking

1. Open `problems.md`
2. Mark your progress on the problem (Python implementation completed)
3. Note any key insights for future reference

## Future GPU Implementations

### Step 11: Prepare for GPU Implementations (When Ready)

When you're ready to implement GPU versions on your main PC, follow these structured steps:

#### Setting Up Triton Implementation

1. Create the Triton implementation directory and files:
   ```
   solutions/group_XX_category/pXXX_problem_name/triton/
   solutions/group_XX_category/pXXX_problem_name/triton/__init__.py
   solutions/group_XX_category/pXXX_problem_name/triton/solution_v1.py
   ```

2. Create a basic Triton `__init__.py`:
   ```python
   # Import implementations - add more as you develop them
   try:
       from .solution_v1 import solution as solution_v1
       VARIANTS = ['v1']
   except ImportError:
       VARIANTS = []
   ```

3. Implement a basic Triton solution in `solution_v1.py`:
   ```python
   import numpy as np
   import triton
   import triton.language as tl

   # Define Triton kernel
   @triton.jit
   def matrix_vector_kernel(
       # Pointers to matrices
       matrix_ptr, vector_ptr, output_ptr,
       # Matrix dimensions
       n_rows, n_cols,
       # Meta-parameters
       BLOCK_SIZE: tl.constexpr,
   ):
       # Example for matrix-vector dot product
       row_idx = tl.program_id(0)
       if row_idx >= n_rows:
           return
       
       # Compute dot product for this row
       row_start = row_idx * n_cols
       dot_product = 0.0
       for col in range(0, n_cols, BLOCK_SIZE):
           # Bounds check
           mask = col + tl.arange(0, BLOCK_SIZE) < n_cols
           # Load matrix row elements
           row_elements = tl.load(matrix_ptr + row_start + col + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
           # Load vector elements
           vector_elements = tl.load(vector_ptr + col + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
           # Update dot product
           dot_product += tl.sum(row_elements * vector_elements, axis=0)
       
       # Store result
       tl.store(output_ptr + row_idx, dot_product)
   
   def solution(matrix, vector):
       """
       Triton implementation of matrix-vector dot product.
       
       Args:
           matrix: 2D NumPy array of shape (M, N)
           vector: 1D NumPy array of shape (N,)
           
       Returns:
           1D NumPy array of shape (M,)
       """
       # Get matrix dimensions
       n_rows, n_cols = matrix.shape
       assert vector.shape[0] == n_cols, "Matrix and vector dimensions mismatch"
       
       # Allocate output
       output = np.empty(n_rows, dtype=matrix.dtype)
       
       # Determine block size
       BLOCK_SIZE = 32  # Can be tuned for performance
       
       # Launch kernel
       grid = (n_rows,)
       matrix_vector_kernel[grid](
           matrix, vector, output,
           n_rows, n_cols,
           BLOCK_SIZE=BLOCK_SIZE,
       )
       
       return output
   
   # Add implementation metadata
   IMPLEMENTATION_METADATA = {
       "name": "Basic Triton Implementation",
       "version": "v1",
       "description": "Basic matrix-vector dot product using Triton",
       "date": "2025-04-05",
       "author": "Your Name",
       "optimization_techniques": ["GPU parallelization"],
       "expected_performance": "Significant speedup over CPU for large matrices",
       "hardware_requirements": "NVIDIA GPU with Triton support"
   }
   ```

#### Setting Up CUDA Implementation

1. Create the CUDA implementation directory and files:
   ```
   solutions/group_XX_category/pXXX_problem_name/cuda/
   solutions/group_XX_category/pXXX_problem_name/cuda/__init__.py
   solutions/group_XX_category/pXXX_problem_name/cuda/solution_v1.py
   ```

2. Create a basic CUDA `__init__.py`:
   ```python
   # Import implementations - add more as you develop them
   try:
       from .solution_v1 import solution as solution_v1
       VARIANTS = ['v1']
   except ImportError:
       VARIANTS = []
   ```

3. Implement a basic CUDA solution in `solution_v1.py`:
   ```python
   import numpy as np
   import cupy as cp

   def solution(matrix, vector):
       """
       CUDA implementation of matrix-vector dot product using CuPy.
       
       Args:
           matrix: 2D NumPy array of shape (M, N)
           vector: 1D NumPy array of shape (N,)
           
       Returns:
           1D NumPy array of shape (M,)
       """
       # Convert inputs to CuPy arrays
       matrix_gpu = cp.asarray(matrix)
       vector_gpu = cp.asarray(vector)
       
       # Perform matrix-vector multiplication on GPU
       result_gpu = cp.dot(matrix_gpu, vector_gpu)
       
       # Transfer result back to CPU and return
       return cp.asnumpy(result_gpu)
   
   # Add implementation metadata
   IMPLEMENTATION_METADATA = {
       "name": "Basic CUDA Implementation",
       "version": "v1",
       "description": "Basic matrix-vector dot product using CuPy (CUDA)",
       "date": "2025-04-05",
       "author": "Your Name",
       "optimization_techniques": ["GPU BLAS acceleration"],
       "expected_performance": "Significant speedup over CPU for large matrices",
       "hardware_requirements": "NVIDIA GPU with CUDA support"
   }
   ```

### Step 12: Test GPU Implementations

1. Install required dependencies for GPU implementations:
   ```bash
   # For Triton
   pip install triton

   # For CUDA via CuPy
   pip install cupy-cuda12x  # Replace with your CUDA version
   ```

2. Test your implementations using the same verification framework:
   ```bash
   python verify_python_implementation.py
   ```

3. Create a GPU-specific test script:
   ```python
   # gpu_test.py
   import numpy as np
   import sys
   from pathlib import Path
   
   # Add project root to Python path
   ROOT_DIR = Path(__file__).parent
   sys.path.insert(0, str(ROOT_DIR))
   
   def test_gpu_implementations():
       """Test Triton and CUDA implementations for matrix-vector dot product."""
       # Create test data
       matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
       vector = np.array([2, 3, 4], dtype=np.float32)
       expected = np.dot(matrix, vector)
       
       # Test Triton implementation
       try:
           from solutions.group_XX_category.pXXX_problem_name.triton.solution_v1 import solution as triton_solution
           print("\nTesting Triton implementation...")
           result_triton = triton_solution(matrix, vector)
           match = np.allclose(result_triton, expected)
           print(f"Triton result matches NumPy: {match}")
       except ImportError as e:
           print(f"Triton implementation not available: {e}")
       
       # Test CUDA implementation
       try:
           from solutions.group_XX_category.pXXX_problem_name.cuda.solution_v1 import solution as cuda_solution
           print("\nTesting CUDA implementation...")
           result_cuda = cuda_solution(matrix, vector)
           match = np.allclose(result_cuda, expected)
           print(f"CUDA result matches NumPy: {match}")
       except ImportError as e:
           print(f"CUDA implementation not available: {e}")
   
   if __name__ == "__main__":
       test_gpu_implementations()
   ```

### Step 13: Optimize GPU Implementations

#### Triton Optimization Techniques

1. Create an optimized Triton implementation in `triton/solution_v2_optimized.py`:
   ```python
   import numpy as np
   import triton
   import triton.language as tl
   
   # Optimized kernel with tiling and other optimizations
   @triton.jit
   def optimized_matrix_vector_kernel(
       matrix_ptr, vector_ptr, output_ptr,
       n_rows, n_cols, stride_row, stride_col,
       BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
   ):
       # Advanced implementation with:
       # - Tiling for better memory access
       # - Shared memory usage
       # - Efficient parallel reduction
       # Your optimized kernel implementation here
       pass
   
   def solution(matrix, vector):
       """
       Optimized Triton implementation with advanced techniques.
       
       Args:
           matrix: 2D NumPy array
           vector: 1D NumPy array
           
       Returns:
           1D NumPy array
       """
       # Implementation using optimized kernel
       # Include auto-tuning for block sizes and other parameters
       pass
   
   # Add implementation metadata
   IMPLEMENTATION_METADATA = {
       "name": "Optimized Triton Implementation",
       "version": "v2_optimized",
       "description": "Optimized matrix-vector dot product using advanced Triton techniques",
       "date": "2025-04-05",
       "author": "Your Name",
       "optimization_techniques": [
           "Tiling", 
           "Shared memory usage", 
           "Memory coalescing",
           "Auto-tuning"
       ],
       "expected_performance": "Further improved over basic Triton implementation",
       "hardware_requirements": "NVIDIA GPU with Triton support"
   }
   ```

#### CUDA Optimization Techniques

1. Create an optimized CUDA implementation in `cuda/solution_v2_optimized.py`:
   ```python
   import numpy as np
   import cupy as cp
   
   def solution(matrix, vector):
       """
       Optimized CUDA implementation of matrix-vector dot product.
       
       Args:
           matrix: 2D NumPy array
           vector: 1D NumPy array
           
       Returns:
           1D NumPy array
       """
       # Advanced implementation with:
       # - Custom CUDA kernels
       # - Memory optimization
       # - Stream management
       # Your optimized implementation here
       
       # Example with streams for async computation
       with cp.cuda.Stream() as stream:
           matrix_gpu = cp.asarray(matrix, stream=stream)
           vector_gpu = cp.asarray(vector, stream=stream)
           result_gpu = cp.dot(matrix_gpu, vector_gpu)
           result = cp.asnumpy(result_gpu)
       
       return result
   
   # You could also use custom CUDA kernels via CuPy's RawKernel
   
   # Add implementation metadata
   IMPLEMENTATION_METADATA = {
       "name": "Optimized CUDA Implementation",
       "version": "v2_optimized",
       "description": "Optimized matrix-vector dot product using advanced CUDA techniques",
       "date": "2025-04-05",
       "author": "Your Name",
       "optimization_techniques": [
           "Custom kernels", 
           "Efficient memory transfers", 
           "Stream management",
           "Shared memory optimization"
       ],
       "expected_performance": "Maximum GPU performance for matrix-vector operations",
       "hardware_requirements": "NVIDIA GPU with CUDA support"
   }
   ```

### Step 14: Comprehensive Benchmarking

1. Create a final benchmark configuration that includes all implementations:
   ```json
   {
       "problem_id": "pXXX_problem_name",
       "name": "Problem Description",
       "description": "Comprehensive benchmark comparing all implementations",
       "implementations": [
           ["python", "v1"],
           ["python", "v2_optimized"],
           ["cuda", "v1"],
           ["cuda", "v2_optimized"],
           ["triton", "v1"],
           ["triton", "v2_optimized"]
       ],
       "input_sizes": [128, 256, 512, 1024, 2048, 4096, 8192],
       "num_runs": 10,
       "warmup_runs": 3,
       "error_thresholds": {
           "default": 0.0001,
           "1024": 0.0002,
           "2048": 0.0002,
           "4096": 0.0005,
           "8192": 0.001
       }
   }
   ```

2. Run the unified benchmarking system:
   ```bash
   python run_final_benchmark.py --problem-id pXXX_problem_name
   ```

3. For advanced GPU benchmarking scenarios, create a custom script:
   ```python
   # Example: advanced_gpu_benchmark.py
   import os
   from datetime import datetime
   from utils.benchmark_unified import run_problem_benchmark
   from utils.enhanced_visualizations import generate_complete_visualization_suite
   
   def main():
       # Define problem ID
       problem_id = "pXXX_problem_name"
       config_path = f"configs/benchmarks/{problem_id}_benchmark_config.json"
       
       print("Running preliminary validation test...")
       # First run on small sizes to verify correctness
       validation_results = run_problem_benchmark(
           problem_id=problem_id,
           config_path=config_path,
           input_sizes=[16, 32, 64]  # Small sizes for validation
       )
       
       # Check if all implementations passed validation
       all_passed = all(result.passed for result in validation_results)
       if not all_passed:
           print("Warning: Some implementations failed validation!")
           for result in validation_results:
               if not result.passed:
                   print(f"  Failed: {result.implementation_type} ({result.variant}) - Error: {result.error}")
       
       print("\nRunning comprehensive GPU benchmark...")
       # Run full benchmark with specified sizes
       timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
       results = run_problem_benchmark(
           problem_id=problem_id,
           config_path=config_path
       )
       
       # Generate enhanced visualizations
       viz_dir = f"benchmarks/visualizations/{problem_id}_comprehensive_{timestamp}"
       os.makedirs(viz_dir, exist_ok=True)
       
       viz_files = generate_complete_visualization_suite(
           results=results,
           output_dir=viz_dir,
           problem_id=problem_id,
           generate_html=True
       )
       
       print("\nComprehensive GPU benchmark complete!")
       print(f"Results saved to: {viz_dir}")
   
   if __name__ == "__main__":
       main()
   ```

4. Examine the visualization outputs to gain insights:
   - Performance comparison plots show relative speeds across implementations
   - Scaling analysis reveals how each implementation scales with input size
   - Accuracy comparison helps verify numerical stability of GPU implementations
   - Interactive HTML dashboards provide detailed views of all metrics

## Troubleshooting

If you encounter issues:

1. Check the `User_Guide.md` for detailed debugging information
2. Run `verify_python_implementation.py` to check your system setup
3. For Python implementations focus on:
   - Correct NumPy usage
   - Proper data type handling (e.g., np.float32)
   - Memory efficiency for large inputs

4. For Triton and CUDA implementations:
   - Verify CUDA/GPU drivers are properly installed
   - Check Triton/CuPy version compatibility with your CUDA version
   - Use `nvidia-smi` to check GPU status before running
   - Add memory management to avoid GPU OOM errors:
     ```python
     # In your CUDA implementation
     import cupy as cp
     cp.get_default_memory_pool().free_all_blocks()  # Clear memory cache
     ```
   - Add proper error handling for GPU-specific issues:
     ```python
     try:
         # GPU operation
         result = cuda_solution(matrix, vector)
     except cp.cuda.runtime.CUDARuntimeError as e:
         if "out of memory" in str(e):
             print("GPU out of memory, try reducing input size")
         else:
             print(f"CUDA error: {e}")
     ```

5. Start with smaller input sizes to isolate issues:
   ```python
   # In generate_inputs
   def generate_inputs(size=100):  # Use smaller size for debugging
   ```

6. Use path management to avoid file access issues:
   ```python
   from utils.path_manager import ensure_directories_exist
   ensure_directories_exist()  # Before saving any files
   ```

7. For GPU-specific debugging:
   ```python
   # In your GPU implementation
   print(f"GPU memory usage before: {cp.cuda.Device().mem_info[0]/1e9:.2f} GB free")
   # Your GPU operation
   print(f"GPU memory usage after: {cp.cuda.Device().mem_info[0]/1e9:.2f} GB free")
   ```
