"""
Enhanced benchmark suite for Problems-Kit.

This module provides the BenchmarkSuite class which coordinates benchmarking
for multiple implementations and variants of a problem.
"""

import os
import time as perf_time
import json
import csv
import importlib
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
import sys
import platform
import traceback

# Add the project root to Python path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Import path management utilities
from utils.path_manager import (
    ensure_directories_exist,
    BENCHMARKS_DIR,
    VISUALIZATIONS_DIR,
    CSV_DIR
)

# Import the BenchmarkResult class
from utils.benchmark_enhanced import BenchmarkResult, MemoryTracker, PLOTLY_AVAILABLE

# Try to import GPU monitoring tools if available
try:
    import pynvml
    PYNVML_AVAILABLE = True
    pynvml.nvmlInit()
except (ImportError, Exception):
    PYNVML_AVAILABLE = False

# Ensure required directories exist
ensure_directories_exist()


class BenchmarkSuite:
    """Enhanced class to manage benchmarking for a problem with multiple implementations and variants."""
    
    def __init__(self, problem_id: str):
        """
        Initialize a benchmark suite for a specific problem.
        
        Args:
            problem_id: ID of the problem (e.g., 'p001_matrix_vector_dot')
        """
        self.problem_id = problem_id
        self.results: Dict[Tuple[str, str], BenchmarkResult] = {}  # (impl_type, variant) -> result
        self.problem_path = self._get_problem_path()
        self.memory_tracker = MemoryTracker()
        self.system_info = self._get_system_info()
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get information about the system for benchmark context."""
        info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Try to get GPU info if available
        if PYNVML_AVAILABLE:
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                gpus = []
                
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle)
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    gpus.append({
                        "name": name.decode('utf-8') if isinstance(name, bytes) else name,
                        "total_memory_mb": memory.total / (1024 * 1024)
                    })
                
                info["gpus"] = gpus
            except Exception:
                pass
        
        # Try to get NumPy info
        try:
            info["numpy_version"] = np.__version__
        except Exception:
            pass
        
        return info
        
    def _get_problem_path(self) -> Path:
        """Get the path to the problem directory."""
        # First, try to import the categories module to find the problem's group
        try:
            from solutions.categories import get_problem_info
            problem_info = get_problem_info(self.problem_id)
            if problem_info and 'group' in problem_info:
                group = problem_info['group']
                return ROOT_DIR / "solutions" / group / self.problem_id
        except (ImportError, KeyError):
            pass
        
        # If that fails, search for the problem directory
        for group_dir in (ROOT_DIR / "solutions").glob("group_*"):
            problem_dir = group_dir / self.problem_id
            if problem_dir.exists():
                return problem_dir
        
        # If still not found, create the directory in a default group
        default_dir = ROOT_DIR / "solutions" / "ungrouped" / self.problem_id
        default_dir.mkdir(parents=True, exist_ok=True)
        return default_dir
    
    def list_implementations(self) -> Dict[str, List[str]]:
        """
        List all available implementations and their variants for the problem.
        
        Returns:
            Dictionary mapping implementation types to lists of available variants
        """
        implementations = {}
        
        # Check for Python implementations
        python_dir = self.problem_path / "python"
        if python_dir.exists():
            implementations['python'] = []
            # Try to use the registry first
            try:
                from importlib import import_module
                module_path = f"solutions.{self.problem_path.parent.name}.{self.problem_path.name}.python"
                python_module = import_module(module_path)
                if hasattr(python_module, 'list_implementations'):
                    implementations['python'] = list(python_module.list_implementations().keys())
            except (ImportError, AttributeError):
                # Fall back to scanning the directory
                for file in python_dir.glob('solution_*.py'):
                    variant = file.stem.replace('solution_', '')
                    implementations['python'].append(variant)
        
        # Check for Triton implementations
        triton_dir = self.problem_path / "triton"
        if triton_dir.exists():
            implementations['triton'] = []
            # Try to use the registry first
            try:
                from importlib import import_module
                module_path = f"solutions.{self.problem_path.parent.name}.{self.problem_path.name}.triton"
                triton_module = import_module(module_path)
                if hasattr(triton_module, 'list_implementations'):
                    implementations['triton'] = list(triton_module.list_implementations().keys())
            except (ImportError, AttributeError):
                # Fall back to scanning the directory
                for file in triton_dir.glob('solution_*.py'):
                    variant = file.stem.replace('solution_', '')
                    implementations['triton'].append(variant)
        
        # Check for CUDA implementations
        cuda_dir = self.problem_path / "cuda"
        if cuda_dir.exists():
            implementations['cuda'] = []
            # Try to use the registry first
            try:
                from importlib import import_module
                module_path = f"solutions.{self.problem_path.parent.name}.{self.problem_path.name}.cuda"
                cuda_module = import_module(module_path)
                if hasattr(cuda_module, 'list_implementations'):
                    implementations['cuda'] = list(cuda_module.list_implementations().keys())
            except (ImportError, AttributeError):
                # Fall back to scanning the directory
                for file in cuda_dir.glob('solution_*.py'):
                    variant = file.stem.replace('solution_', '')
                    implementations['cuda'].append(variant)
        
        return implementations
    
    def _import_implementation(self, impl_type: str, variant: str) -> Any:
        """
        Import an implementation function based on the type and variant.
        
        Args:
            impl_type: Type of implementation ('python', 'triton', 'cuda')
            variant: Variant of the implementation (e.g., 'v1', 'v2_optimized')
        
        Returns:
            The imported function
        """
        # Try to use the new structure first
        try:
            from importlib import import_module
            module_path = f"solutions.{self.problem_path.parent.name}.{self.problem_path.name}"
            problem_module = import_module(module_path)
            
            # Try to get implementation via the get_implementation function
            if hasattr(problem_module, 'get_implementation'):
                func = problem_module.get_implementation(impl_type, variant)
                if func is not None:
                    return func
        except (ImportError, AttributeError):
            pass
        
        # Fall back to direct import
        try:
            if variant == 'v1' and impl_type == 'python':
                legacy_path = f"solutions.{self.problem_path.parent.name}.{self.problem_path.name}.solution_py"
                module = import_module(legacy_path)
                return module.solution
            elif variant == 'v1' and impl_type == 'triton':
                legacy_path = f"solutions.{self.problem_path.parent.name}.{self.problem_path.name}.solution_triton"
                module = import_module(legacy_path)
                return module.solution
            elif variant == 'v1' and impl_type == 'cuda':
                legacy_path = f"solutions.{self.problem_path.parent.name}.{self.problem_path.name}.solution_cuda"
                module = import_module(legacy_path)
                return module.solution
                
            # Try the new directory structure
            module_path = f"solutions.{self.problem_path.parent.name}.{self.problem_path.name}.{impl_type}.solution_{variant}"
            module = import_module(module_path)
            return module.solution
        except ImportError as e:
            raise ImportError(f"Could not import {impl_type} implementation variant '{variant}': {e}")
    
    def benchmark_implementation(self, 
                                impl_type: str,
                                variant: str = 'v1',
                                function_name: str = 'solution',
                                args: List[Any] = None,
                                kwargs: Dict[str, Any] = None,
                                num_runs: int = 10,
                                warmup_runs: int = 3,
                                track_memory: bool = True,
                                calculate_throughput: bool = True,
                                operation_count: Optional[int] = None,
                                input_sizes: Optional[List[int]] = None) -> BenchmarkResult:
        """
        Benchmark a specific implementation variant.
        
        Args:
            impl_type: Type of implementation ('python', 'triton', 'cuda')
            variant: Variant of the implementation (e.g., 'v1', 'v2_optimized')
            function_name: Name of the function to benchmark
            args: Positional arguments to pass to the function
            kwargs: Keyword arguments to pass to the function
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs (not included in results)
            track_memory: Whether to track memory usage
            calculate_throughput: Whether to calculate throughput (ops/sec)
            operation_count: Number of operations performed per call (for throughput calculation)
            input_sizes: List of input sizes (if benchmarking multiple sizes)
            
        Returns:
            BenchmarkResult object containing execution times and other metrics
        """
        args = args or []
        kwargs = kwargs or {}
        
        # Import the implementation
        try:
            func = self._import_implementation(impl_type, variant)
        except ImportError as e:
            raise ImportError(f"Could not import {impl_type} implementation variant '{variant}': {e}")
        
        # Try to get implementation metadata
        try:
            from importlib import import_module
            module_path = f"solutions.{self.problem_path.parent.name}.{self.problem_path.name}.{impl_type}"
            module = import_module(module_path)
            
            if hasattr(module, 'get_metadata'):
                metadata = module.get_metadata(variant)
            elif hasattr(module, f"solution_{variant}") and hasattr(getattr(module, f"solution_{variant}"), "IMPLEMENTATION_METADATA"):
                metadata = getattr(module, f"solution_{variant}").IMPLEMENTATION_METADATA
            else:
                metadata = {
                    "name": f"{impl_type.capitalize()} {variant}",
                    "version": variant,
                    "description": f"{impl_type.capitalize()} implementation variant {variant}"
                }
        except (ImportError, AttributeError):
            metadata = {
                "name": f"{impl_type.capitalize()} {variant}",
                "version": variant,
                "description": f"{impl_type.capitalize()} implementation variant {variant}"
            }
        
        # Perform warmup runs
        print(f"Warming up {impl_type} implementation variant '{variant}'...")
        for _ in range(warmup_runs):
            func(*args, **kwargs)
        
        # Reset memory tracker baseline after warmup
        if track_memory:
            self.memory_tracker.reset_baseline()
        
        # Perform benchmark runs
        print(f"Benchmarking {impl_type} implementation variant '{variant}'...")
        execution_times = []
        memory_usage = [] if track_memory else None
        throughput = [] if calculate_throughput else None
        
        for _ in range(num_runs):
            start_time = perf_time.perf_counter()
            func(*args, **kwargs)
            end_time = perf_time.perf_counter()
            execution_time = end_time - start_time
            execution_times.append(execution_time)
            
            # Track memory usage
            if track_memory:
                memory_usage.append(self.memory_tracker.get_memory_usage())
            
            # Calculate throughput
            if calculate_throughput and operation_count is not None:
                throughput.append(operation_count / execution_time)
        
        # Create and store benchmark result
        result = BenchmarkResult(
            implementation=impl_type,
            variant=variant,
            execution_times=execution_times,
            input_sizes=input_sizes,
            memory_usage=memory_usage,
            throughput=throughput,
            metadata={
                'function_name': function_name,
                'num_runs': num_runs,
                'warmup_runs': warmup_runs,
                'timestamp': datetime.now().isoformat(),
                'impl_metadata': metadata
            }
        )
        
        self.results[(impl_type, variant)] = result
        return result
