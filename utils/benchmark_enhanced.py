"""
Enhanced benchmarking system for the Problems-Kit.

This module provides advanced functionality to benchmark multiple implementations
and variants of the same problem (Python, Triton, CUDA) and compare their performance
with extended metrics and output formats.
"""

import os
import time as perf_time
import json
import csv
import argparse
import importlib
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
import sys
import traceback
import platform
import psutil  # For memory usage tracking

# Root directory of the project
ROOT_DIR = Path(__file__).parent.parent

# Add the root directory to Python path to allow imports
sys.path.insert(0, str(ROOT_DIR))

# Import path management utilities
from utils.path_manager import (
    ensure_directories_exist,
    BENCHMARKS_DIR,
    VISUALIZATIONS_DIR,
    CSV_DIR
)

# Ensure required directories exist
ensure_directories_exist()

# Backward compatibility aliases
BENCHMARK_DIR = BENCHMARKS_DIR
VISUALIZATION_DIR = VISUALIZATIONS_DIR

# Try to import plotly for interactive visualizations
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class MemoryTracker:
    """Class to track memory usage during benchmarking."""
    
    def __init__(self):
        """Initialize memory tracker."""
        self.process = psutil.Process(os.getpid())
        self.baseline = self.get_current_memory()
    
    def get_current_memory(self) -> float:
        """Get current memory usage in MB."""
        try:
            return self.process.memory_info().rss / (1024 * 1024)  # Convert to MB
        except (psutil.Error, AttributeError):
            return 0.0
    
    def get_memory_usage(self) -> float:
        """Get memory usage relative to baseline in MB."""
        current = self.get_current_memory()
        return current - self.baseline
    
    def reset_baseline(self):
        """Reset the baseline memory usage."""
        self.baseline = self.get_current_memory()


class BenchmarkResult:
    """Enhanced class to store benchmark results for a single implementation."""
    
    def __init__(self, 
                 implementation: str,
                 variant: str, 
                 execution_times: List[float],
                 input_sizes: Optional[List[int]] = None,
                 memory_usage: Optional[List[float]] = None,
                 throughput: Optional[List[float]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a benchmark result.
        
        Args:
            implementation: Type of implementation (e.g., 'python', 'triton', 'cuda')
            variant: Variant of the implementation (e.g., 'v1', 'v2_optimized')
            execution_times: List of execution times for each run (in seconds)
            input_sizes: Optional list of input sizes corresponding to each execution time
            memory_usage: Optional list of memory usage data (MB)
            throughput: Optional list of throughput measurements (ops/sec)
            metadata: Optional additional metadata about the implementation
        """
        self.implementation = implementation
        self.variant = variant
        self.execution_times = execution_times
        self.input_sizes = input_sizes
        self.memory_usage = memory_usage
        self.throughput = throughput
        self.metadata = metadata or {}
        
        # Calculate statistics
        self._calculate_statistics()
    
    def _calculate_statistics(self):
        """Calculate performance statistics."""
        if not self.execution_times:
            return
            
        # Time statistics
        self.mean_time = float(np.mean(self.execution_times))
        self.median_time = float(np.median(self.execution_times))
        self.min_time = float(np.min(self.execution_times))
        self.max_time = float(np.max(self.execution_times))
        self.std_time = float(np.std(self.execution_times))
        
        # Calculate 95% confidence interval
        n = len(self.execution_times)
        if n > 1:
            # Using t-distribution for small sample sizes
            import scipy.stats as stats
            t_value = stats.t.ppf(0.975, n - 1)  # 95% CI (two-tailed)
            self.ci_95_lower = float(self.mean_time - t_value * self.std_time / np.sqrt(n))
            self.ci_95_upper = float(self.mean_time + t_value * self.std_time / np.sqrt(n))
        else:
            self.ci_95_lower = self.mean_time
            self.ci_95_upper = self.mean_time
        
        # Memory statistics if available
        if self.memory_usage:
            self.mean_memory = float(np.mean(self.memory_usage))
            self.max_memory = float(np.max(self.memory_usage))
        else:
            self.mean_memory = None
            self.max_memory = None
        
        # Throughput statistics if available
        if self.throughput:
            self.mean_throughput = float(np.mean(self.throughput))
        else:
            self.mean_throughput = None
    
    def detect_outliers(self, threshold=1.5):
        """
        Detect outliers in execution times using IQR method.
        
        Args:
            threshold: IQR multiplier for outlier detection
        
        Returns:
            List of indices of outlier runs
        """
        if len(self.execution_times) < 4:  # Need more data for meaningful outlier detection
            return []
            
        q1 = np.percentile(self.execution_times, 25)
        q3 = np.percentile(self.execution_times, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        outliers = [i for i, time in enumerate(self.execution_times) 
                   if time < lower_bound or time > upper_bound]
        return outliers
    
    def filter_outliers(self, threshold=1.5):
        """
        Remove outliers from execution times and recalculate statistics.
        
        Args:
            threshold: IQR multiplier for outlier detection
            
        Returns:
            Number of outliers removed
        """
        outliers = self.detect_outliers(threshold)
        if not outliers:
            return 0
        
        # Create filtered versions of all data
        non_outliers = [i for i in range(len(self.execution_times)) if i not in outliers]
        self.execution_times = [self.execution_times[i] for i in non_outliers]
        
        if self.memory_usage:
            self.memory_usage = [self.memory_usage[i] for i in non_outliers]
        
        if self.throughput:
            self.throughput = [self.throughput[i] for i in non_outliers]
        
        # Recalculate statistics
        self._calculate_statistics()
        
        return len(outliers)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the benchmark result to a dictionary."""
        result = {
            'implementation': self.implementation,
            'variant': self.variant,
            'execution_times': self.execution_times,
            'mean_time': self.mean_time,
            'median_time': self.median_time,
            'min_time': self.min_time,
            'max_time': self.max_time,
            'std_time': self.std_time,
            'ci_95_lower': getattr(self, 'ci_95_lower', None),
            'ci_95_upper': getattr(self, 'ci_95_upper', None),
            'metadata': self.metadata
        }
        
        if self.input_sizes:
            result['input_sizes'] = self.input_sizes
            
        if self.memory_usage:
            result['memory_usage'] = self.memory_usage
            result['mean_memory'] = self.mean_memory
            result['max_memory'] = self.max_memory
            
        if self.throughput:
            result['throughput'] = self.throughput
            result['mean_throughput'] = self.mean_throughput
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkResult':
        """Create a benchmark result from a dictionary."""
        result = cls(
            implementation=data['implementation'],
            variant=data.get('variant', 'v1'),
            execution_times=data['execution_times'],
            input_sizes=data.get('input_sizes'),
            memory_usage=data.get('memory_usage'),
            throughput=data.get('throughput'),
            metadata=data.get('metadata', {})
        )
        return result
