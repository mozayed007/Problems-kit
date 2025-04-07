"""
Path management utilities for Problems-Kit.

This module ensures that all necessary directories exist for 
benchmark results, visualizations, and other system functions.
"""

import os
from pathlib import Path

# Define the project root directory
ROOT_DIR = Path(__file__).parent.parent

# Common directories
BENCHMARKS_DIR = ROOT_DIR / "benchmarks"
CSV_DIR = BENCHMARKS_DIR / "csv"
VISUALIZATIONS_DIR = BENCHMARKS_DIR / "visualizations"


def ensure_directories_exist():
    """
    Ensure that all necessary directories exist for the system to function properly.
    Creates directories if they don't exist.
    """
    directories = [
        BENCHMARKS_DIR,
        CSV_DIR,
        VISUALIZATIONS_DIR
    ]
    
    for directory in directories:
        if not directory.exists():
            print(f"Creating directory: {directory}")
            directory.mkdir(parents=True, exist_ok=True)
    
    return True


def get_benchmark_csv_path(problem_id, timestamp=None):
    """
    Get the path for a benchmark CSV file.
    
    Args:
        problem_id: ID of the problem
        timestamp: Optional timestamp string to append to filename
        
    Returns:
        Path object for the CSV file
    """
    ensure_directories_exist()
    
    if timestamp:
        filename = f"{problem_id}_benchmark_{timestamp}.csv"
    else:
        filename = f"{problem_id}_benchmark.csv"
    
    return CSV_DIR / filename


def get_visualization_path(problem_id, visual_type, timestamp=None):
    """
    Get the path for a visualization file.
    
    Args:
        problem_id: ID of the problem
        visual_type: Type of visualization (e.g., 'bar', 'line', 'box')
        timestamp: Optional timestamp string to append to filename
        
    Returns:
        Path object for the visualization file
    """
    ensure_directories_exist()
    
    if timestamp:
        filename = f"{problem_id}_{visual_type}_{timestamp}.png"
    else:
        filename = f"{problem_id}_{visual_type}.png"
    
    return VISUALIZATIONS_DIR / filename


if __name__ == "__main__":
    # Test directory creation
    ensure_directories_exist()
    print("All directories exist or have been created.")
