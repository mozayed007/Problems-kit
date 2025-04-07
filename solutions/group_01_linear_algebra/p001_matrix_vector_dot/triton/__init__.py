"""
Triton implementations for Matrix-Vector Dot Product.

This module provides multiple Triton implementations for the matrix-vector dot product operation.
"""

# Import all available solutions
try:
    from .solution_v1 import solution as solution_v1
except ImportError:
    solution_v1 = None

# Registry of all implementations
IMPLEMENTATIONS = {
    "v1": {
        "function": solution_v1,
        "metadata": {
            "name": "Basic Triton",
            "version": "v1",
            "description": "Basic matrix-vector dot product using Triton",
            "date": "2025-04-05",
            "author": "Problems-Kit Team",
            "optimization_techniques": [
                "GPU parallelization",
                "Blocked computation"
            ],
            "expected_performance": "Baseline GPU performance"
        }
    }
}

# Default implementation
DEFAULT_IMPLEMENTATION = "v1"

def get_implementation(variant=DEFAULT_IMPLEMENTATION):
    """
    Get a specific implementation variant.
    
    Args:
        variant: The variant name (e.g., 'v1')
        
    Returns:
        The implementation function or None if not available
    """
    return IMPLEMENTATIONS.get(variant, {}).get("function")

def get_metadata(variant=DEFAULT_IMPLEMENTATION):
    """
    Get metadata for a specific implementation variant.
    
    Args:
        variant: The variant name (e.g., 'v1')
        
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
