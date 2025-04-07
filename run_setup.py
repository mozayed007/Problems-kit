"""
Helper script to run the setup process
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run the setup function
from utils.problem_parser import initialize_group_directories, generate_categories_file, initialize_example_problem
from utils.solution_template import create_problem_structure

# Create basic structure
print("Creating directory structure...")
for directory in ["solutions", "benchmarks", "benchmarks/visualizations", "utils", "tests"]:
    os.makedirs(os.path.join(Path(__file__).parent, directory), exist_ok=True)

# Initialize group directories
print("Initializing group directories...")
try:
    group_paths = initialize_group_directories()
    print(f"Created {len(group_paths)} group directories")
except Exception as e:
    print(f"Error initializing group directories: {e}")

# Generate categories.py
print("Generating categories file...")
try:
    categories_path = generate_categories_file()
    print(f"Created categories file: {categories_path}")
except Exception as e:
    print(f"Error generating categories file: {e}")

# Initialize example problem
print("Setting up example problem...")
try:
    problem_dir = initialize_example_problem("group_01_linear_algebra", 1)
    print(f"Created example problem: {problem_dir}")
except Exception as e:
    print(f"Error initializing example problem: {e}")

print("\nProject setup complete!")
