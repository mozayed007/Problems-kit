"""
Compatibility wrappers for the Problems-Kit.

This module provides compatibility wrappers to ensure legacy code 
works with the new unified benchmarking system.
"""

from typing import List, Tuple, Dict, Any, Optional
import os
import sys
from pathlib import Path

# Add project root to Python path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Import the new unified benchmarking system
from utils.benchmark_unified import run_problem_benchmark
from utils.enhanced_visualizations import generate_complete_visualization_suite
from utils.benchmark_visualization import export_to_csv

def run_benchmark(
    problem_id: str,
    implementations: List[Tuple[str, str]],
    num_runs: int = 5,
    warmup_runs: int = 2,
    input_sizes: List[int] = None,
    show_plots: bool = True,
    save_plots: bool = True,
    export_csv: bool = True,
    track_memory: bool = False,
    **kwargs
) -> None:
    """
    Compatibility wrapper for the old run_benchmark function.
    
    This function adapts the old function signature to work with the new 
    unified benchmarking system.
    
    Args:
        problem_id: ID of the problem to benchmark
        implementations: List of implementation tuples (type, variant)
        num_runs: Number of benchmark runs for each implementation
        warmup_runs: Number of warmup runs (excluded from results)
        input_sizes: List of input sizes to test
        show_plots: Whether to display plots interactively
        save_plots: Whether to save plots to files
        export_csv: Whether to export results to CSV
        track_memory: Whether to track memory usage (ignored in unified system)
    """
    # Set default input sizes if not provided
    if input_sizes is None:
        input_sizes = [128, 512, 1024, 2048]
    
    # Run the benchmark using the unified system
    results = run_problem_benchmark(
        problem_id=problem_id,
        implementations=implementations,
        input_sizes=input_sizes,
        num_runs=num_runs,
        warmup_runs=warmup_runs
    )
    
    # Generate visualizations if requested
    if show_plots or save_plots:
        # Create a timestamped output directory
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"benchmarks/visualizations/{problem_id}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate the visualization suite
        viz_files = generate_complete_visualization_suite(
            results=results,
            output_dir=output_dir,
            problem_id=problem_id,
            generate_html=True,
            show_plots=show_plots
        )
    
    # Export to CSV if requested
    if export_csv:
        # Create the CSV directory if it doesn't exist
        csv_dir = "benchmarks/csv"
        os.makedirs(csv_dir, exist_ok=True)
        
        # Export to CSV using the old format for compatibility
        csv_file = f"{csv_dir}/{problem_id}_{timestamp}.csv"
        export_to_csv(results, csv_file)
    
    return results
