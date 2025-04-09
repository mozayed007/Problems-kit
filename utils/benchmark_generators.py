"""
Benchmark Input Generators for Problems-Kit

This module contains input generators for various problem types.
Each generator creates appropriate test inputs based on problem requirements.
"""

import numpy as np
import torch
from typing import Any, Tuple, List, Dict, Optional, Union

# Set random seeds for reproducibility
np.random.seed(42)
if torch.cuda.is_available():
    torch.manual_seed(42)


def generate_matrix_vector_inputs(size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a random matrix and vector for matrix-vector dot product testing.
    
    Args:
        size: Size parameter (creates a size×size matrix and size-length vector)
        
    Returns:
        Tuple of (matrix, vector) as numpy arrays
    """
    matrix = np.random.randn(size, size).astype(np.float32)
    vector = np.random.randn(size).astype(np.float32)
    return matrix, vector


def generate_matrix_matrix_inputs(size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate random matrices for matrix-matrix multiplication testing.
    
    Args:
        size: Size parameter (creates size×size matrices)
        
    Returns:
        Tuple of (matrix_a, matrix_b) as numpy arrays
    """
    matrix_a = np.random.randn(size, size).astype(np.float32)
    matrix_b = np.random.randn(size, size).astype(np.float32)
    return matrix_a, matrix_b


def generate_sorting_inputs(size: int, distribution: str = 'random') -> np.ndarray:
    """
    Generate a random array for sorting algorithm testing.
    
    Args:
        size: Number of elements in the array
        distribution: Distribution type ('random', 'nearly_sorted', 'reversed', 'few_unique')
        
    Returns:
        Numpy array with elements to be sorted
    """
    # Generate array based on distribution type
    if distribution == 'random':
        # Completely random data
        return np.random.randn(size).astype(np.float32)
    
    elif distribution == 'nearly_sorted':
        # Create a sorted array and swap a few elements
        arr = np.arange(size).astype(np.float32)
        # Swap about 5% of elements
        num_swaps = max(int(size * 0.05), 1)
        for _ in range(num_swaps):
            i, j = np.random.randint(0, size, 2)
            arr[i], arr[j] = arr[j], arr[i]
        return arr
    
    elif distribution == 'reversed':
        # Create a reversed sorted array
        return np.arange(size-1, -1, -1).astype(np.float32)
    
    elif distribution == 'few_unique':
        # Create an array with few unique values (high repetition)
        num_unique = min(int(size * 0.1), 100)
        unique_values = np.random.randn(num_unique).astype(np.float32)
        return np.random.choice(unique_values, size=size)
    
    else:
        # Default to random if distribution not recognized
        return np.random.randn(size).astype(np.float32)


def generate_convolution_inputs(size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate random inputs for convolution testing.
    
    Args:
        size: Size parameter (creates size×size input and kernel_size×kernel_size kernel)
        
    Returns:
        Tuple of (input_data, kernel) as numpy arrays
    """
    # Set kernel size based on input size
    if size <= 128:
        kernel_size = 3
    elif size <= 512:
        kernel_size = 5
    else:
        kernel_size = 7
    
    input_data = np.random.randn(size, size).astype(np.float32)
    kernel = np.random.randn(kernel_size, kernel_size).astype(np.float32)
    return input_data, kernel


def generate_graph_inputs(size: int, edge_density: float = 0.1, weighted: bool = True, directed: bool = True) -> Dict[str, Any]:
    """
    Generate a random graph for graph algorithm testing.
    
    Args:
        size: Number of nodes in the graph
        edge_density: Probability of edge between any two nodes (0.0 to 1.0)
        weighted: Whether to assign random weights to edges
        directed: Whether the graph is directed
        
    Returns:
        Dictionary with graph representation (adjacency matrix and optional metadata)
    """
    # Create a random adjacency matrix based on edge density
    adj_matrix = np.random.rand(size, size) < edge_density
    np.fill_diagonal(adj_matrix, 0)  # No self-loops
    
    # For undirected graphs, make the adjacency matrix symmetric
    if not directed:
        adj_matrix = np.logical_or(adj_matrix, adj_matrix.T)
    
    # Add weights
    weights = np.zeros((size, size), dtype=np.float32)
    if weighted:
        # Generate random weights between 1 and 10
        weights[adj_matrix] = 1 + np.random.rand(np.sum(adj_matrix)) * 9
    else:
        # Unweighted graph (all edges have weight 1)
        weights[adj_matrix] = 1.0
    
    # Ensure the graph is connected (create a path through all nodes)
    for i in range(size-1):
        # Add an edge from i to i+1 (create a guaranteed path)
        adj_matrix[i, i+1] = True
        weights[i, i+1] = 1 + np.random.rand() * 4  # Weight between 1 and 5
        
        # For undirected, add the reverse edge too
        if not directed:
            adj_matrix[i+1, i] = True
            weights[i+1, i] = weights[i, i+1]  # Same weight in both directions
    
    # Generate random node features
    node_features = np.random.randn(size, 10).astype(np.float32)
    
    # For shortest path problems, select source and target nodes
    source = 0
    target = size - 1
    
    return {
        'adjacency_matrix': adj_matrix,
        'weights': weights,
        'node_features': node_features,
        'num_nodes': size,
        'source': source,
        'target': target,
        'directed': directed,
        'weighted': weighted
    }


def generate_differential_equation_inputs(size: int) -> Dict[str, Any]:
    """
    Generate inputs for differential equation solver testing.
    
    Args:
        size: Number of points in the grid/discretization
        
    Returns:
        Dictionary with initial conditions and parameters
    """
    # Generate initial conditions
    initial_values = np.random.randn(size).astype(np.float32)
    
    # Time points (evenly spaced)
    t_start = 0.0
    t_end = 10.0
    t_points = np.linspace(t_start, t_end, size).astype(np.float32)
    
    # System parameters
    parameters = {
        'alpha': np.random.uniform(0.1, 1.0),
        'beta': np.random.uniform(0.1, 1.0),
        'gamma': np.random.uniform(0.1, 1.0)
    }
    
    return {
        'initial_values': initial_values,
        't_points': t_points,
        'parameters': parameters,
        'grid_size': size
    }


def generate_optimization_inputs(size: int) -> Dict[str, Any]:
    """
    Generate inputs for optimization algorithm testing.
    
    Args:
        size: Dimensionality of the optimization problem
        
    Returns:
        Dictionary with objective function parameters and constraints
    """
    # Generate a random positive semi-definite matrix for quadratic objectives
    A = np.random.randn(size, size).astype(np.float32)
    A = A.T @ A  # Make it symmetric positive semi-definite
    
    # Generate a random linear term
    b = np.random.randn(size).astype(np.float32)
    
    # Generate random constraints (Ax <= b form)
    num_constraints = max(1, size // 2)
    constraint_matrix = np.random.randn(num_constraints, size).astype(np.float32)
    constraint_vector = np.random.randn(num_constraints).astype(np.float32)
    
    return {
        'objective_matrix': A,
        'objective_vector': b,
        'constraint_matrix': constraint_matrix,
        'constraint_vector': constraint_vector,
        'dimension': size
    }


def generate_custom_inputs(size: int, problem_type: str) -> Any:
    """
    Generate inputs based on a custom problem type string.
    
    Args:
        size: Size parameter
        problem_type: String identifier for the problem type
        
    Returns:
        Generated inputs based on the problem type
    """
    problem_type = problem_type.lower()
    
    if "matrix" in problem_type and "vector" in problem_type:
        return generate_matrix_vector_inputs(size)
    elif "matrix" in problem_type and "matrix" in problem_type:
        return generate_matrix_matrix_inputs(size)
    elif "sort" in problem_type:
        return generate_sorting_inputs(size)
    elif "conv" in problem_type:
        return generate_convolution_inputs(size)
    elif "graph" in problem_type:
        return generate_graph_inputs(size)
    elif "diff" in problem_type or "ode" in problem_type:
        return generate_differential_equation_inputs(size)
    elif "optim" in problem_type:
        return generate_optimization_inputs(size)
    else:
        # Default to simple array
        return np.random.randn(size).astype(np.float32)
