"""
Problem 1: Matrix-Vector Dot Product

This module provides implementations for the matrix-vector dot product in different languages:
- Python (multiple variants)
- Triton (multiple variants)
- CUDA (multiple variants)

Each implementation type has multiple variants with different optimization strategies.
"""

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

# For backward compatibility
try:
    from .python import get_implementation as get_python_implementation
    py_solution = get_python_implementation()
except (ImportError, AttributeError):
    py_solution = None

try:
    from .triton import get_implementation as get_triton_implementation
    triton_solution = get_triton_implementation()
except (ImportError, AttributeError):
    triton_solution = None

try:
    from .cuda import get_implementation as get_cuda_implementation
    cuda_solution = get_cuda_implementation()
except (ImportError, AttributeError):
    cuda_solution = None

# Dictionary mapping implementation types to their default functions (for backward compatibility)
implementations = {
    'python': py_solution,
    'triton': triton_solution,
    'cuda': cuda_solution
}

def generate_inputs(size=1000):
    """
    Generate input data for benchmarking.
    
    Args:
        size: Size of the matrix (size x size) and vector (size)
        
    Returns:
        Tuple of (args, kwargs) to be passed to the solution function
    """
    matrix = np.random.rand(size, size).astype(np.float32)
    vector = np.random.rand(size).astype(np.float32)
    
    return [matrix, vector], {}

def get_implementation(impl_type='python', variant=None):
    """
    Get the implementation function for the specified type and variant.
    
    Args:
        impl_type: Type of implementation ('python', 'triton', 'cuda')
        variant: Specific variant of the implementation (e.g., 'v1', 'v2_optimized')
               If None, uses the default variant for that implementation type.
        
    Returns:
        The implementation function or None if not available
    """
    if variant is None:
        # Use default implementation from module
        return implementations.get(impl_type)
    
    # Get specific variant
    if impl_type == 'python' and PYTHON_AVAILABLE:
        return python.get_implementation(variant)
    elif impl_type == 'triton' and TRITON_AVAILABLE:
        return triton.get_implementation(variant)
    elif impl_type == 'cuda' and CUDA_AVAILABLE:
        return cuda.get_implementation(variant)
    
    return None

def list_all_implementations():
    """
    List all available implementations with their metadata.
    
    Returns:
        Dictionary mapping implementation types to available variants and their metadata
    """
    all_impls = {}
    
    if PYTHON_AVAILABLE:
        try:
            all_impls['python'] = python.list_implementations()
        except AttributeError:
            pass
    
    if TRITON_AVAILABLE:
        try:
            all_impls['triton'] = triton.list_implementations()
        except AttributeError:
            pass
    
    if CUDA_AVAILABLE:
        try:
            all_impls['cuda'] = cuda.list_implementations()
        except AttributeError:
            pass
    
    return all_impls
