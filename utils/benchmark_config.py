"""
Benchmark Configuration System for Problems-Kit

This module provides utilities for defining and loading benchmark configurations
for different problem types. It supports custom input generators, validation methods,
and problem-specific settings.
"""

import os
import json
import importlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union, Tuple

# Add the project root to the path to ensure imports work correctly
ROOT_DIR = Path(__file__).parent.parent
CONFIG_DIR = ROOT_DIR / "configs" / "benchmarks"

@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking a specific problem."""
    problem_id: str
    name: str
    description: str
    implementations: List[Tuple[str, str]] = field(default_factory=list)
    input_sizes: List[int] = field(default_factory=list)
    num_runs: int = 10
    warmup_runs: int = 3
    error_thresholds: Dict[int, float] = field(default_factory=dict)
    input_generator_module: Optional[str] = None
    input_generator_function: str = "generate_inputs"
    reference_module: Optional[str] = None
    reference_function: str = "solution"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    def save(self, filename: Optional[str] = None) -> str:
        """
        Save configuration to a JSON file.
        
        Args:
            filename: Name of the file to save to, without extension.
                     If None, uses problem_id.
                     
        Returns:
            Path to the saved file
        """
        if filename is None:
            filename = f"{self.problem_id}_benchmark_config"
            
        # Ensure config directory exists
        os.makedirs(CONFIG_DIR, exist_ok=True)
        
        # Save to JSON
        file_path = CONFIG_DIR / f"{filename}.json"
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
        return str(file_path)
    
    @classmethod
    def load(cls, problem_id: str) -> 'BenchmarkConfig':
        """
        Load configuration for a problem from its JSON file.
        
        Args:
            problem_id: ID of the problem
            
        Returns:
            Loaded configuration
            
        Raises:
            FileNotFoundError: If configuration file does not exist
        """
        file_path = CONFIG_DIR / f"{problem_id}_benchmark_config.json"
        
        if not file_path.exists():
            raise FileNotFoundError(f"No benchmark configuration found for problem {problem_id}")
            
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
            
        return cls(**config_dict)
    
    def get_input_generator(self) -> Callable:
        """
        Get the input generator function defined in the configuration.
        
        Returns:
            Input generator function
            
        Raises:
            ImportError: If the module or function cannot be imported
        """
        if self.input_generator_module is None:
            # Use default generator based on problem type
            return self._get_default_input_generator()
        
        try:
            module = importlib.import_module(self.input_generator_module)
            return getattr(module, self.input_generator_function)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not import input generator: {str(e)}")
    
    def get_reference_implementation(self) -> Callable:
        """
        Get the reference implementation function for validation.
        
        Returns:
            Reference implementation function
            
        Raises:
            ImportError: If the module or function cannot be imported
        """
        if self.reference_module is None:
            # Default to Python v1 implementation
            parts = self.problem_id.split('_')
            group_id = f"group_{parts[0][1:3]}"
            module_path = f"solutions.{group_id}.{self.problem_id}.python.solution_v1"
        else:
            module_path = self.reference_module
        
        try:
            module = importlib.import_module(module_path)
            return getattr(module, self.reference_function)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not import reference implementation: {str(e)}")
    
    def _get_default_input_generator(self) -> Callable:
        """Get a default input generator based on the problem ID."""
        # Matrix-vector dot product
        if self.problem_id == "p001_matrix_vector_dot":
            def generate_matrix_vector_inputs(size):
                import numpy as np
                np.random.seed(42)  # For reproducibility
                matrix = np.random.randn(size, size).astype(np.float32)
                vector = np.random.randn(size).astype(np.float32)
                return matrix, vector
            return generate_matrix_vector_inputs
        
        # Example for a sorting problem
        elif self.problem_id.startswith("p00") and "sort" in self.problem_id:
            def generate_sorting_inputs(size):
                import numpy as np
                np.random.seed(42)  # For reproducibility
                return np.random.randn(size).astype(np.float32)
            return generate_sorting_inputs
            
        # Default fallback
        else:
            def default_generator(size):
                import numpy as np
                np.random.seed(42)  # For reproducibility
                return np.random.randn(size).astype(np.float32)
            return default_generator


def create_default_config(problem_id: str, name: Optional[str] = None, 
                         description: Optional[str] = None) -> BenchmarkConfig:
    """
    Create a default benchmark configuration for a problem.
    
    Args:
        problem_id: ID of the problem
        name: Name of the problem (defaults to problem_id if None)
        description: Description of the problem
        
    Returns:
        Default configuration for the problem
    """
    if name is None:
        name = problem_id.replace('_', ' ').title()
        
    if description is None:
        description = f"Benchmark configuration for {name}"
    
    # Set default implementation types to test
    implementations = [
        ('python', 'v1'),
        ('python', 'v2_optimized'),
        ('cuda', 'v1'),
        ('triton', 'v1'),
        ('triton', 'v2_optimized')
    ]
    
    # Set default input sizes based on problem type
    if problem_id == "p001_matrix_vector_dot":
        input_sizes = [128, 256, 512, 1024, 2048, 4096]
        error_thresholds = {
            128: 1e-4,
            256: 1e-4,
            512: 1e-4,
            1024: 2e-4,
            2048: 2e-4,
            4096: 5e-4
        }
    else:
        # Default sizes for other problems
        input_sizes = [100, 1000, 10000]
        error_thresholds = {
            100: 1e-6,
            1000: 1e-5,
            10000: 1e-4
        }
    
    return BenchmarkConfig(
        problem_id=problem_id,
        name=name,
        description=description,
        implementations=implementations,
        input_sizes=input_sizes,
        num_runs=10,
        warmup_runs=3,
        error_thresholds=error_thresholds
    )


def ensure_config_exists(problem_id: str) -> BenchmarkConfig:
    """
    Ensure a benchmark configuration exists for a problem, creating one if needed.
    
    Args:
        problem_id: ID of the problem
        
    Returns:
        Configuration for the problem
    """
    # Create config directory if it doesn't exist
    os.makedirs(CONFIG_DIR, exist_ok=True)
    
    config_path = CONFIG_DIR / f"{problem_id}_benchmark_config.json"
    
    if config_path.exists():
        return BenchmarkConfig.load(problem_id)
    else:
        # Create a new default configuration
        config = create_default_config(problem_id)
        config.save()
        return config


def list_available_configs() -> List[str]:
    """
    List all available benchmark configurations.
    
    Returns:
        List of problem IDs with available configurations
    """
    # Create config directory if it doesn't exist
    os.makedirs(CONFIG_DIR, exist_ok=True)
    
    # Get all JSON files in the config directory
    config_files = list(CONFIG_DIR.glob('*.json'))
    
    # Extract problem IDs from filenames
    return [file.stem.replace('_benchmark_config', '') for file in config_files]
