"""
Setup script for the Problems-Kit project.

This script initializes the project structure, creates necessary directories,
and sets up an example problem to demonstrate the benchmarking system.
"""

import os
import argparse
from pathlib import Path

# Add the current directory to the Python path
import sys
sys.path.append(str(Path(__file__).parent.parent))

# Import project utilities
from utils.problem_parser import (
    initialize_group_directories,
    generate_categories_file,
    initialize_example_problem
)

# Root directory of the project
ROOT_DIR = Path(__file__).parent.parent

def create_basic_structure():
    """Create the basic directory structure for the project."""
    # Create main directories
    directories = [
        ROOT_DIR / "solutions",
        ROOT_DIR / "benchmarks",
        ROOT_DIR / "benchmarks" / "visualizations",
        ROOT_DIR / "utils",
        ROOT_DIR / "tests"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        
    # Create __init__.py files in each directory
    for directory in directories:
        init_file = directory / "__init__.py"
        if not init_file.exists():
            with open(init_file, 'w') as f:
                f.write(f'"""\n{directory.name} package\n"""\n')

def create_requirements_file():
    """Create a requirements.txt file with necessary dependencies."""
    requirements = [
        "# Core dependencies",
        "numpy>=1.20.0",
        "matplotlib>=3.5.0",
        "pytest>=7.0.0",
        "",
        "# CUDA dependencies (optional)",
        "cupy-cuda12x>=12.0.0  # Adjust version based on your CUDA version",
        "",
        "# Triton dependencies (optional)",
        "triton>=2.0.0",
        "torch>=2.0.0  # Required for Triton"
    ]
    
    with open(ROOT_DIR / "requirements.txt", 'w') as f:
        f.write('\n'.join(requirements))

def create_setup_file():
    """Create setup.py for package installation."""
    setup_content = """
from setuptools import setup, find_packages

setup(
    name="problems-kit",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.5.0",
    ],
    extras_require={
        "cuda": ["cupy-cuda12x>=12.0.0"],
        "triton": ["triton>=2.0.0", "torch>=2.0.0"],
        "dev": ["pytest>=7.0.0"],
    },
    python_requires='>=3.7',
)
"""
    
    with open(ROOT_DIR / "setup.py", 'w') as f:
        f.write(setup_content.strip())

def create_pyproject_file():
    """Create pyproject.toml file."""
    pyproject_content = """
[build-system]
requires = ["setuptools>=42", "wheel"]
build-backend = "setuptools.build_meta"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
"""
    
    with open(ROOT_DIR / "pyproject.toml", 'w') as f:
        f.write(pyproject_content.strip())

def main():
    """Main function to initialize the project."""
    parser = argparse.ArgumentParser(description='Set up the Problems-Kit project')
    parser.add_argument('--example', action='store_true', help='Set up an example problem')
    parser.add_argument('--group', type=str, default="group_01_linear_algebra", 
                        help='Group ID for the example problem')
    parser.add_argument('--problem', type=int, default=1, 
                        help='Problem number for the example problem')
    
    args = parser.parse_args()
    
    print("Setting up Problems-Kit project...")
    
    # Create basic structure
    print("Creating directory structure...")
    create_basic_structure()
    
    # Initialize group directories
    print("Initializing group directories...")
    group_paths = initialize_group_directories()
    print(f"Created {len(group_paths)} group directories")
    
    # Generate categories.py
    print("Generating categories file...")
    categories_path = generate_categories_file()
    print(f"Created categories file: {categories_path}")
    
    # Create requirements.txt
    print("Creating requirements.txt...")
    create_requirements_file()
    
    # Create setup.py
    print("Creating setup.py...")
    create_setup_file()
    
    # Create pyproject.toml
    print("Creating pyproject.toml...")
    create_pyproject_file()
    
    # Initialize example problem
    if args.example:
        print(f"Setting up example problem {args.problem} in group {args.group}...")
        problem_dir = initialize_example_problem(args.group, args.problem)
        print(f"Created example problem: {problem_dir}")
    
    print("\nProject setup complete!")
    print("\nTo get started:")
    print("1. Install dependencies: pip install -e .")
    print("2. Create a new problem: python -m utils.solution_template --problem <problem_id> --group <group_id>")
    print("3. Run benchmarks: python -m utils.benchmark --problem <problem_id>")
    print("4. Visualize results: python -m utils.visualization --problem <problem_id> --plot-type times")

if __name__ == "__main__":
    main()
