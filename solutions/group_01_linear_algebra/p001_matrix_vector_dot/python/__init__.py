"""
Python implementations for Matrix-Vector Dot Product.

This module provides multiple Python implementations for the matrix-vector dot product operation.
"""

# Import all available solutions
try:
    from .solution_v1 import solution as solution_v1
    # Also import the module for direct access
    import solutions.group_01_linear_algebra.p001_matrix_vector_dot.python.solution_v1 as solution_v1_module
except ImportError:
    solution_v1 = None
    solution_v1_module = None

try:
    from .solution_v2_optimized import solution as solution_v2_optimized
    from .solution_v2_optimized import IMPLEMENTATION_METADATA as v2_metadata
    # Also import the module for direct access
    import solutions.group_01_linear_algebra.p001_matrix_vector_dot.python.solution_v2_optimized as solution_v2_optimized_module
except ImportError:
    solution_v2_optimized = None
    v2_metadata = None
    solution_v2_optimized_module = None

# Registry of all implementations
IMPLEMENTATIONS = {
    "v1": {
        "function": solution_v1,
        "module": solution_v1_module,
        "metadata": {
            "name": "Basic NumPy",
            "version": "v1",
            "description": "Basic matrix-vector dot product using NumPy",
            "date": "2025-04-05",
            "author": "Problems-Kit Team",
            "optimization_techniques": [],
            "expected_performance": "Baseline performance"
        }
    }
}

# Add v2 if available
if solution_v2_optimized is not None:
    IMPLEMENTATIONS["v2_optimized"] = {
        "function": solution_v2_optimized,
        "module": solution_v2_optimized_module,
        "metadata": v2_metadata
    }

# Default implementation (the most optimized one)
if "v2_optimized" in IMPLEMENTATIONS:
    default = "v2_optimized"
else:
    default = "v1"

DEFAULT_IMPLEMENTATION = default

def get_implementation(variant=DEFAULT_IMPLEMENTATION):
    """
    Get a specific implementation variant.
    
    Args:
        variant: The variant name (e.g., 'v1', 'v2_optimized')
        
    Returns:
        The implementation function or None if not available
    """
    return IMPLEMENTATIONS.get(variant, {}).get("function")

def get_metadata(variant=DEFAULT_IMPLEMENTATION):
    """
    Get metadata for a specific implementation variant.
    
    Args:
        variant: The variant name (e.g., 'v1', 'v2_optimized')
        
    Returns:
        The implementation metadata or None if not available
    """
    return IMPLEMENTATIONS.get(variant, {}).get("metadata")

def list_implementations():
    """
    List all available implementations.
    
    Returns:
        Dictionary of implementation variants and their metadata
    """
    return {k: v["metadata"] for k, v in IMPLEMENTATIONS.items() if v["function"] is not None}
