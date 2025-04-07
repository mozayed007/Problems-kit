"""
Unified Benchmarking System for Problems-Kit

This module provides a centralized benchmark interface with standardized result formats,
adaptive error thresholds, and consistent measurement methods across all implementations.
"""

import os
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime
import importlib
from pathlib import Path
import torch
import sys
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

# Add the project root to the path to ensure imports work correctly
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

class BenchmarkResult:
    """
    Standard container for benchmark results with unified output format.
    """
    def __init__(self, 
                problem_id: str,
                implementation_type: str,
                variant: str,
                input_size: int,
                execution_times: List[float],
                result_data: Any,
                reference_data: Any = None,
                error_threshold: float = 1e-4,
                gpu_info: Optional[Dict[str, Any]] = None,
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize benchmark result with comprehensive information.
        
        Args:
            problem_id: Identifier for the problem being benchmarked
            implementation_type: Type of implementation (e.g., 'python', 'cuda', 'triton')
            variant: Implementation variant (e.g., 'v1', 'v2_optimized')
            input_size: Size of the input data
            execution_times: List of execution times in seconds
            result_data: The actual output from the implementation
            reference_data: Reference output for correctness validation
            error_threshold: Maximum allowed error between result and reference
            gpu_info: GPU-related information if applicable
            metadata: Additional information about the benchmark
        """
        self.problem_id = problem_id
        self.implementation_type = implementation_type
        self.variant = variant
        self.input_size = input_size
        self.execution_times = np.array(execution_times)
        self.result_data = result_data
        self.reference_data = reference_data
        self.error_threshold = error_threshold
        self.gpu_info = gpu_info or {}
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()
        
        # Calculate statistics
        self.stats = {
            'min': float(np.min(self.execution_times)),
            'max': float(np.max(self.execution_times)),
            'mean': float(np.mean(self.execution_times)),
            'median': float(np.median(self.execution_times)),
            'std': float(np.std(self.execution_times))
        }
        
        # Validate result if reference is provided
        self.error = None
        self.passed = None
        if reference_data is not None:
            self._validate_result()
    
    def _validate_result(self):
        """Validate result against reference data."""
        try:
            if isinstance(self.result_data, np.ndarray) and isinstance(self.reference_data, np.ndarray):
                # For numpy arrays, compute the maximum absolute difference
                self.error = float(np.max(np.abs(self.result_data - self.reference_data)))
                self.passed = self.error <= self.error_threshold
            else:
                # For other types, use direct equality
                self.error = 0.0 if self.result_data == self.reference_data else float('inf')
                self.passed = self.error == 0.0
        except Exception as e:
            self.error = float('inf')
            self.passed = False
            self.metadata['validation_error'] = str(e)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        result_dict = {
            'problem_id': self.problem_id,
            'implementation_type': self.implementation_type,
            'variant': self.variant,
            'input_size': self.input_size,
            'timestamp': self.timestamp,
            'stats': self.stats
        }
        
        # Include validation results if available
        if self.error is not None:
            result_dict['validation'] = {
                'error': self.error,
                'threshold': self.error_threshold,
                'passed': self.passed
            }
        
        # Include GPU info if available
        if self.gpu_info:
            result_dict['gpu_info'] = self.gpu_info
            
        # Include additional metadata
        if self.metadata:
            result_dict['metadata'] = self.metadata
            
        return result_dict
    
    def to_json(self) -> str:
        """Convert result to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def to_series(self) -> pd.Series:
        """Convert to pandas Series for CSV export."""
        base_dict = {
            'problem_id': self.problem_id,
            'implementation': f"{self.implementation_type}_{self.variant}",
            'input_size': self.input_size,
            'min_time_ms': self.stats['min'] * 1000,
            'max_time_ms': self.stats['max'] * 1000,
            'mean_time_ms': self.stats['mean'] * 1000,
            'median_time_ms': self.stats['median'] * 1000,
            'std_time_ms': self.stats['std'] * 1000,
        }
        
        # Add validation data if available
        if self.error is not None:
            base_dict['error'] = self.error
            base_dict['threshold'] = self.error_threshold
            base_dict['passed'] = self.passed
            
        return pd.Series(base_dict)


def get_adaptive_error_threshold(size: int) -> float:
    """
    Return an appropriate error threshold based on input size.
    
    Args:
        size: Input size (e.g., matrix dimension)
        
    Returns:
        Appropriate error threshold
    """
    if size <= 512:
        return 1e-4  # Strict for small inputs
    elif size <= 2048:
        return 2e-4  # Moderate for medium inputs
    else:
        return 5e-4  # Relaxed for large inputs


def import_solution(problem_id: str, implementation_type: str, variant: str = 'v1') -> Optional[Callable]:
    """
    Import a solution function from the appropriate module.
    
    Args:
        problem_id: ID of the problem
        implementation_type: Type of implementation ('python', 'triton', 'cuda')
        variant: Variant of the implementation
        
    Returns:
        Solution function if found, None otherwise
    """
    # First approach - direct import with full path
    try:
        if implementation_type == 'python':
            if variant == 'v1':
                from solutions.group_01_linear_algebra.p001_matrix_vector_dot.python.solution_v1 import solution
            else:
                from solutions.group_01_linear_algebra.p001_matrix_vector_dot.python.solution_v2_optimized import solution
        elif implementation_type == 'cuda':
            from solutions.group_01_linear_algebra.p001_matrix_vector_dot.cuda.solution_v1 import solution
        elif implementation_type == 'triton':
            if variant == 'v1':
                from solutions.group_01_linear_algebra.p001_matrix_vector_dot.triton.solution_v1 import solution
            else:
                from solutions.group_01_linear_algebra.p001_matrix_vector_dot.triton.solution_v2_optimized import solution
        else:
            return None
        return solution
    except ImportError:
        # If direct import fails, try the dynamic approach
        pass
    
    # Alternative approach - dynamic import
    # Extract group ID from problem ID (e.g., 'group_01' from 'p001_matrix_vector_dot')
    parts = problem_id.split('_')
    group_id = f"group_{parts[0][1:3]}"
    
    # Handle the special case for optimized variants
    variant_path = variant
    if variant == 'v2_optimized':
        variant_path = "solution_v2_optimized"
    else:
        variant_path = f"solution_{variant}"
    
    # Construct module path
    module_path = f"solutions.{group_id}.{problem_id}.{implementation_type}.{variant_path}"
    
    try:
        module = importlib.import_module(module_path)
        return getattr(module, 'solution')
    except (ImportError, AttributeError) as e:
        print(f"Debug - Import error: {e} for path {module_path}")
        return None


def benchmark_implementation(
    problem_id: str,
    implementation_type: str,
    variant: str,
    input_generator: Callable,
    input_size: int,
    num_runs: int = 10,
    warmup_runs: int = 3,
    error_threshold: Optional[float] = None
) -> Optional[BenchmarkResult]:
    """
    Benchmark a specific implementation with given input size.
    
    Args:
        problem_id: ID of the problem
        implementation_type: Type of implementation ('python', 'triton', 'cuda')
        variant: Variant of the implementation
        input_generator: Function to generate test inputs based on size
        input_size: Size parameter for input generation
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        error_threshold: Custom error threshold (if None, uses adaptive thresholds)
        
    Returns:
        BenchmarkResult object if successful, None if implementation not found
    """
    # Import the solution function
    solution_func = import_solution(problem_id, implementation_type, variant)
    if solution_func is None:
        return None
    
    # Generate input data
    input_data = input_generator(input_size)
    
    # For gpu implementations, precompute reference result and set up GPU info
    gpu_info = None
    if implementation_type in ['cuda', 'triton']:
        if torch.cuda.is_available():
            gpu_info = {
                'device_name': torch.cuda.get_device_name(0),
                'cuda_version': torch.version.cuda,
                'device_count': torch.cuda.device_count()
            }
        else:
            # Cannot benchmark GPU implementations without a GPU
            return None
    
    # Perform warmup runs
    for _ in range(warmup_runs):
        try:
            if isinstance(input_data, tuple):
                solution_func(*input_data)
            else:
                solution_func(input_data)
                
            # Ensure GPU operations are complete
            if implementation_type in ['cuda', 'triton'] and torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as e:
            # Return failed result if warmup fails
            return BenchmarkResult(
                problem_id=problem_id,
                implementation_type=implementation_type,
                variant=variant,
                input_size=input_size,
                execution_times=[],
                result_data=None,
                reference_data=None,
                error_threshold=error_threshold or get_adaptive_error_threshold(input_size),
                gpu_info=gpu_info,
                metadata={'error': str(e), 'traceback': f"{type(e).__name__}: {str(e)}"}
            )
    
    # Perform actual benchmark runs
    execution_times = []
    result = None
    
    for i in range(num_runs):
        try:
            # Start timing
            start_time = time.time()
            
            # Execute solution
            if isinstance(input_data, tuple):
                result = solution_func(*input_data)
            else:
                result = solution_func(input_data)
                
            # Ensure GPU operations are complete
            if implementation_type in ['cuda', 'triton'] and torch.cuda.is_available():
                torch.cuda.synchronize()
                
            # Stop timing and record
            execution_times.append(time.time() - start_time)
        except Exception as e:
            # Return failed result if any run fails
            return BenchmarkResult(
                problem_id=problem_id,
                implementation_type=implementation_type,
                variant=variant,
                input_size=input_size,
                execution_times=execution_times[:i] if i > 0 else [],
                result_data=None,
                reference_data=None,
                error_threshold=error_threshold or get_adaptive_error_threshold(input_size),
                gpu_info=gpu_info,
                metadata={'error': str(e), 'traceback': f"{type(e).__name__}: {str(e)}"}
            )
    
    # Calculate reference result using numpy implementation
    reference_func = import_solution(problem_id, 'python', 'v1')
    reference_result = None
    
    if reference_func is not None:
        try:
            if isinstance(input_data, tuple):
                reference_result = reference_func(*input_data)
            else:
                reference_result = reference_func(input_data)
        except Exception as e:
            # If reference calculation fails, log it but continue
            metadata = {'reference_error': str(e)}
    else:
        metadata = {'reference_warning': 'No python reference implementation found'}
    
    # Use adaptive error threshold if not provided
    if error_threshold is None:
        error_threshold = get_adaptive_error_threshold(input_size)
    
    # Create and return benchmark result
    return BenchmarkResult(
        problem_id=problem_id,
        implementation_type=implementation_type,
        variant=variant,
        input_size=input_size,
        execution_times=execution_times,
        result_data=result,
        reference_data=reference_result,
        error_threshold=error_threshold,
        gpu_info=gpu_info,
        metadata={}  # No errors to report
    )


def run_problem_benchmark(
    problem_id: str,
    input_generator: Optional[Callable] = None,
    input_sizes: Optional[List[int]] = None,
    implementations: Optional[List[Tuple[str, str]]] = None,
    num_runs: int = 10,
    warmup_runs: int = 3,
    custom_error_thresholds: Optional[Dict[int, float]] = None,
    config_path: Optional[str] = None
) -> List[BenchmarkResult]:
    """
    Run benchmarks for multiple implementations of a problem with different input sizes.
    
    Args:
        problem_id: ID of the problem
        input_generator: Function to generate inputs based on size
        input_sizes: List of input sizes to test
        implementations: List of (impl_type, variant) tuples to benchmark
                         If None, attempts to find all available implementations
        num_runs: Number of benchmark runs per implementation
        warmup_runs: Number of warmup runs per implementation
        custom_error_thresholds: Dictionary mapping input size to error threshold
        
    Returns:
        List of BenchmarkResult objects
    """
    # Load configuration if not provided explicitly
    if config_path is not None:
        try:
            from utils.benchmark_config import BenchmarkConfig
            with open(config_path, 'r') as f:
                import json
                config_dict = json.load(f)
                # Override parameters with config values if not provided
                if input_sizes is None and 'input_sizes' in config_dict:
                    input_sizes = config_dict['input_sizes']
                if implementations is None and 'implementations' in config_dict:
                    implementations = config_dict['implementations']
                if num_runs == 10 and 'num_runs' in config_dict:  # Default value check
                    num_runs = config_dict['num_runs']
                if warmup_runs == 3 and 'warmup_runs' in config_dict:  # Default value check
                    warmup_runs = config_dict['warmup_runs']
                if custom_error_thresholds is None and 'error_thresholds' in config_dict:
                    custom_error_thresholds = {int(k): float(v) for k, v in config_dict['error_thresholds'].items()}
                if input_generator is None and 'input_generator_module' in config_dict:
                    try:
                        module = importlib.import_module(config_dict['input_generator_module'])
                        input_generator = getattr(module, config_dict['input_generator_function'])
                    except (ImportError, AttributeError) as e:
                        print(f"Warning: Could not import input generator from config: {e}")
        except Exception as e:
            print(f"Warning: Error loading config from {config_path}: {e}")
    
    # Load from default config location if needed
    if input_sizes is None or implementations is None or input_generator is None:
        try:
            from utils.benchmark_config import ensure_config_exists
            config = ensure_config_exists(problem_id)
            
            # Use config values for missing parameters
            if input_sizes is None:
                input_sizes = config.input_sizes
            if implementations is None:
                implementations = config.implementations
            if custom_error_thresholds is None:
                custom_error_thresholds = config.error_thresholds
            if input_generator is None:
                input_generator = config.get_input_generator()
        except Exception as e:
            print(f"Warning: Could not load default config: {e}")
    
    # Final fallbacks for required parameters
    if input_generator is None:
        from utils.benchmark_generators import generate_custom_inputs
        print("Warning: No input generator specified or found in config, using default")
        input_generator = lambda size: generate_custom_inputs(size, problem_id)
    
    if input_sizes is None:
        print("Warning: No input sizes specified or found in config, using default")
        input_sizes = [128, 256, 512, 1024, 2048]
        
    if implementations is None:
        implementations = []
        print("Warning: No implementations specified or found in config, detecting available ones")
        for impl_type in ['python', 'cuda', 'triton']:
            for variant in ['v1', 'v2_optimized']:
                if import_solution(problem_id, impl_type, variant) is not None:
                    implementations.append((impl_type, variant))
    
    results = []
    
    # Run benchmarks for each implementation and input size
    for input_size in input_sizes:
        # Get custom error threshold if specified
        error_threshold = None
        if custom_error_thresholds and (input_size in custom_error_thresholds or str(input_size) in custom_error_thresholds):
            error_threshold = custom_error_thresholds.get(input_size, custom_error_thresholds.get(str(input_size)))
        
        for impl_type, variant in implementations:
            print(f"Benchmarking {problem_id} with {impl_type} ({variant}) for size {input_size}...")
            
            # Run the benchmark
            result = benchmark_implementation(
                problem_id=problem_id,
                implementation_type=impl_type,
                variant=variant,
                input_generator=input_generator,
                input_size=input_size,
                num_runs=num_runs,
                warmup_runs=warmup_runs,
                error_threshold=error_threshold
            )
            
            # Add to results if benchmark was successful
            if result is not None:
                results.append(result)
                
                # Print summary
                if hasattr(result, 'passed'):
                    status = "✅ PASS" if result.passed else "❌ FAIL"
                    if result.error is not None:
                        print(f"  {status} - Mean: {result.stats['mean']*1000:.2f}ms - Max diff: {result.error:.8f}")
                    else:
                        print(f"  ⚠️ ERROR - Mean: {result.stats['mean']*1000:.2f}ms - Could not validate result")
                else:
                    print(f"  ⚠️ ERROR - Could not run benchmark")
            else:
                print(f"  ⚠️ ERROR - Implementation not found or not compatible")
    
    return results


def save_benchmark_results(results: List[BenchmarkResult], output_dir: str = 'benchmarks') -> Dict[str, str]:
    """
    Save benchmark results to various formats.
    
    Args:
        results: List of BenchmarkResult objects
        output_dir: Base directory for output files
        
    Returns:
        Dictionary with paths to saved files
    """
    # Create directories if they don't exist
    json_dir = os.path.join(output_dir, 'json')
    csv_dir = os.path.join(output_dir, 'csv')
    
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Extract problem ID from first result
    if not results:
        return {}
        
    problem_id = results[0].problem_id
    
    # Prepare paths
    json_path = os.path.join(json_dir, f"{problem_id}_benchmark_{timestamp}.json")
    csv_path = os.path.join(csv_dir, f"{problem_id}_benchmark_{timestamp}.csv")
    
    # Convert results to JSON
    json_data = {
        'problem_id': problem_id,
        'timestamp': datetime.now().isoformat(),
        'results': [result.to_dict() for result in results]
    }
    
    # Save JSON
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    # Convert results to DataFrame for CSV
    df = pd.DataFrame([result.to_series() for result in results])
    
    # Save CSV
    df.to_csv(csv_path, index=False)
    
    return {
        'json': json_path,
        'csv': csv_path
    }
