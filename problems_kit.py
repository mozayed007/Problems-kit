#!/usr/bin/env python
"""
Problems-Kit Main CLI Interface

This script provides a central command-line interface for interacting with the Problems-Kit project.
It allows you to:
- Browse problems by group or category
- Run specific implementations (Python, Triton, CUDA)
- Execute benchmarks with customizable parameters
- Generate visualizations of benchmark results
- Export benchmark results to CSV
- Compare multiple implementation variants per problem
"""

import os
import sys
import argparse
import importlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple

# Add project root to Python path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

# Import project modules
try:
    from solutions.categories import (
        GROUPS, PROBLEMS, 
        get_problem_info, get_problems_by_group, get_problems_by_category
    )
    CATEGORIES_LOADED = True
except ImportError:
    print("Warning: Could not import categories module. Some features may be limited.")
    CATEGORIES_LOADED = False

# Import enhanced benchmark modules - now these are the only ones we use
try:
    from utils.bench_runner_enhanced import run_benchmark, check_implementation, list_implementations
    from utils.benchmark_visualization import export_to_csv, visualize_benchmark_results
    from utils.path_manager import ensure_directories_exist
    
    # Ensure directories exist
    ensure_directories_exist()
    
    ENHANCED_BENCH_LOADED = True
except ImportError:
    print("Warning: Could not import enhanced benchmarking modules. Benchmarking will not be available.")
    ENHANCED_BENCH_LOADED = False

try:
    from utils.problem_parser import initialize_example_problem
    PROBLEM_PARSER_LOADED = True
except ImportError:
    print("Warning: Could not import problem parser module. Problem initialization will not be available.")
    PROBLEM_PARSER_LOADED = False

try:
    from utils.solution_template import create_problem_structure
    SOLUTION_TEMPLATE_LOADED = True
except ImportError:
    print("Warning: Could not import solution template module. Template creation will not be available.")
    SOLUTION_TEMPLATE_LOADED = False


def clear_screen():
    """Clear the terminal screen based on the operating system."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header(title: str):
    """Print a formatted header for a menu section."""
    print("=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)


def print_menu_item(index: int, text: str, indent: int = 0):
    """Print a numbered menu item with optional indentation."""
    print(f"{' ' * indent}{index}. {text}")


def get_user_choice(min_value: int, max_value: int, prompt: str = "Enter your choice: ") -> int:
    """
    Get a validated integer choice from the user within a specified range.
    
    Args:
        min_value: Minimum valid value
        max_value: Maximum valid value
        prompt: Text to display when prompting for input
        
    Returns:
        User's validated choice as an integer
    """
    while True:
        try:
            choice = input(prompt)
            if choice.lower() in ['q', 'quit', 'exit', 'back', 'b']:
                return 0
                
            choice = int(choice)
            if min_value <= choice <= max_value:
                return choice
            else:
                print(f"Please enter a number between {min_value} and {max_value}.")
        except ValueError:
            print("Please enter a valid number.")


def find_problem_by_id(problem_id: str) -> Optional[Dict[str, Any]]:
    """Find a problem by its ID."""
    if not CATEGORIES_LOADED:
        return None
    
    return get_problem_info(problem_id)


def list_groups() -> List[Tuple[str, str]]:
    """
    List all available problem groups.
    
    Returns:
        List of tuples (group_id, group_name)
    """
    if not CATEGORIES_LOADED:
        # Try to find groups by scanning the solutions directory
        groups = []
        solutions_dir = ROOT_DIR / "solutions"
        if solutions_dir.exists():
            for item in solutions_dir.glob("group_*"):
                if item.is_dir():
                    group_id = item.name
                    group_name = " ".join(word.capitalize() for word in group_id.split("_")[2:])
                    groups.append((group_id, group_name))
        return groups
    
    return [(group_id, group_info["name"]) for group_id, group_info in GROUPS.items()]


def list_problems_in_group(group_id: str) -> List[Tuple[str, str]]:
    """
    List all problems in a specific group.
    
    Args:
        group_id: ID of the group
        
    Returns:
        List of tuples (problem_id, problem_title)
    """
    if not CATEGORIES_LOADED:
        # Try to find problems by scanning the group directory
        problems = []
        group_dir = ROOT_DIR / "solutions" / group_id
        if group_dir.exists():
            for item in group_dir.glob("p*"):
                if item.is_dir():
                    problem_id = item.name
                    problem_title = " ".join(word.capitalize() for word in problem_id.split("_")[1:])
                    problems.append((problem_id, problem_title))
        return problems
    
    return [(prob_id, prob_info["title"]) 
            for prob_id, prob_info in PROBLEMS.items() 
            if prob_info.get("group") == group_id]


def check_implementation(problem_id: str, impl_type: str, variant: str = 'v1') -> bool:
    """
    Check if a specific implementation exists for a problem.
    
    Args:
        problem_id: ID of the problem
        impl_type: Type of implementation ('python', 'triton', 'cuda')
        variant: Variant of the implementation
        
    Returns:
        True if the implementation exists, False otherwise
    """
    if ENHANCED_BENCH_LOADED:
        # Use the enhanced benchmarking implementation registry
        try:
            from utils.bench_runner_enhanced import check_implementation
            return check_implementation(problem_id, impl_type, variant)
        except (ImportError, Exception):
            pass
            
    # Try to find the problem directory
    problem_path = None
    for group_dir in (ROOT_DIR / "solutions").glob("*"):
        if group_dir.is_dir():
            potential_path = group_dir / problem_id
            if potential_path.exists() and potential_path.is_dir():
                problem_path = potential_path
                break
    
    if problem_path is None:
        return False
    
    # Check new directory structure first
    if (problem_path / impl_type).exists():
        if variant == 'v1':
            return (problem_path / impl_type / 'solution_v1.py').exists()
        else:
            return (problem_path / impl_type / f'solution_{variant}.py').exists()
    
    # Fall back to legacy structure
    if impl_type == 'python':
        return (problem_path / 'solution_py.py').exists()
    else:
        return (problem_path / f'solution_{impl_type}.py').exists()


def run_implementation(problem_id: str, impl_type: str, variant: str = 'v1', custom_input: bool = False):
    """
    Run a specific implementation of a problem.
    
    Args:
        problem_id: ID of the problem
        impl_type: Type of implementation ('python', 'triton', 'cuda')
        variant: Variant of the implementation
        custom_input: Whether to use custom input from the user
    """
    # Try to find the problem directory
    problem_path = None
    for group_dir in (ROOT_DIR / "solutions").glob("*"):
        if group_dir.is_dir():
            potential_path = group_dir / problem_id
            if potential_path.exists() and potential_path.is_dir():
                problem_path = potential_path
                break
    
    if problem_path is None:
        print(f"Error: Could not find problem directory for {problem_id}")
        return
    
    # Check if implementation exists in the enhanced directory structure
    implementation_found = False
    module_path = None
    
    # Try the new directory structure first (preferred)
    if (problem_path / impl_type).exists():
        if (problem_path / impl_type / f'solution_{variant}.py').exists():
            module_path = f"solutions.{problem_path.parent.name}.{problem_id}.{impl_type}.solution_{variant}"
            implementation_found = True
    
    # Fall back to the legacy directory structure
    if not implementation_found:
        if impl_type == 'python':
            if (problem_path / 'solution_py.py').exists():
                module_path = f"solutions.{problem_path.parent.name}.{problem_id}.solution_py"
                implementation_found = True
        else:
            if (problem_path / f'solution_{impl_type}.py').exists():
                module_path = f"solutions.{problem_path.parent.name}.{problem_id}.solution_{impl_type}"
                implementation_found = True
    
    if not implementation_found:
        print(f"Error: {impl_type.capitalize()} implementation variant '{variant}' not found for problem {problem_id}")
        return
    
    # Import the solution module
    try:
        solution_module = importlib.import_module(module_path)
        solution_function = getattr(solution_module, 'solution')
    except (ImportError, AttributeError) as e:
        print(f"Error: Failed to import solution module: {e}")
        return
    
    # Check if the module has a generate_inputs function
    try:
        if hasattr(solution_module, 'generate_inputs'):
            # Try with custom size if specified
            if custom_input:
                try:
                    size = int(input("\nEnter input size: "))
                    args, kwargs = solution_module.generate_inputs(size)
                except (ValueError, TypeError):
                    print("Invalid input size. Using default inputs.")
                    args, kwargs = solution_module.generate_inputs()
            else:
                args, kwargs = solution_module.generate_inputs()
        else:
            # Try to import the parent module for input generation
            try:
                parent_module_path = f"solutions.{problem_path.parent.name}.{problem_id}"
                parent_module = importlib.import_module(parent_module_path)
                if hasattr(parent_module, 'generate_inputs'):
                    if custom_input:
                        try:
                            size = int(input("\nEnter input size: "))
                            args, kwargs = parent_module.generate_inputs(size)
                        except (ValueError, TypeError):
                            print("Invalid input size. Using default inputs.")
                            args, kwargs = parent_module.generate_inputs()
                    else:
                        args, kwargs = parent_module.generate_inputs()
                else:
                    print("No input generation function found. Using empty inputs.")
                    args, kwargs = [], {}
            except ImportError:
                print("No input generation function found. Using empty inputs.")
                args, kwargs = [], {}
    except Exception as e:
        print(f"Error generating inputs: {e}")
        args, kwargs = [], {}
    
    # Run the solution
    print(f"\nRunning {impl_type.capitalize()} implementation variant '{variant}'...")
    
    try:
        start_time = perf_time.perf_counter()
        result = solution_function(*args, **kwargs)
        end_time = perf_time.perf_counter()
        execution_time = end_time - start_time
        
        # Display results
        print("\nResult:")
        print(result)
        
        # Display execution time
        if execution_time < 0.001:
            time_str = f"{execution_time*1e6:.2f} µs"
        elif execution_time < 1.0:
            time_str = f"{execution_time*1e3:.2f} ms"
        else:
            time_str = f"{execution_time:.4f} s"
            
        print(f"\nExecution time: {time_str}")
    except Exception as e:
        print(f"Error running solution: {e}")
        traceback.print_exc()


def menu_main():
    """Display the main menu and handle user selection."""
    while True:
        clear_screen()
        print_header("PROBLEMS-KIT MAIN MENU")
        print("\nSelect an option:")
        print_menu_item(1, "Browse Problems by Group")
        print_menu_item(2, "Run Implementation")
        print_menu_item(3, "Run Benchmark")
        print_menu_item(4, "Initialize New Problem")
        print_menu_item(5, "View Available Implementations")
        print_menu_item(0, "Exit")
        
        choice = get_user_choice(0, 5)
        
        if choice == 0:
            print("\nExiting Problems-Kit. Goodbye!")
            sys.exit(0)
        elif choice == 1:
            menu_browse_groups()
        elif choice == 2:
            menu_run_implementation()
        elif choice == 3:
            menu_run_benchmark()
        elif choice == 4:
            menu_initialize_problem()
        elif choice == 5:
            check_available_implementations()
        
        input("\nPress Enter to continue...")


def menu_browse_groups():
    """Display the groups menu and handle user selection."""
    groups = list_groups()
    
    if not groups:
        print("\nNo groups found.")
        return
    
    while True:
        clear_screen()
        print_header("BROWSE PROBLEMS BY GROUP")
        print("\nSelect a group:")
        
        for i, (group_id, group_name) in enumerate(groups, 1):
            print_menu_item(i, f"{group_name} ({group_id})")
        
        print_menu_item(0, "Back to Main Menu")
        
        choice = get_user_choice(0, len(groups))
        
        if choice == 0:
            return
        else:
            group_id, group_name = groups[choice - 1]
            menu_browse_problems(group_id, group_name)


def menu_browse_problems(group_id: str, group_name: str):
    """
    Display the problems in a group and handle user selection.
    
    Args:
        group_id: ID of the group
        group_name: Name of the group for display purposes
    """
    problems = list_problems_in_group(group_id)
    
    if not problems:
        print(f"\nNo problems found in group '{group_name}'.")
        return
    
    while True:
        clear_screen()
        print_header(f"PROBLEMS IN {group_name.upper()}")
        print("\nSelect a problem:")
        
        for i, (problem_id, problem_title) in enumerate(problems, 1):
            print_menu_item(i, f"{problem_title} ({problem_id})")
        
        print_menu_item(0, "Back to Groups")
        
        choice = get_user_choice(0, len(problems))
        
        if choice == 0:
            return
        else:
            problem_id, problem_title = problems[choice - 1]
            menu_problem_details(problem_id, problem_title)


def menu_problem_details(problem_id: str, problem_title: str):
    """
    Display details about a problem and available options.
    
    Args:
        problem_id: ID of the problem
        problem_title: Title of the problem for display purposes
    """
    while True:
        clear_screen()
        print_header(f"PROBLEM: {problem_title}")
        
        # Get problem details
        problem_info = find_problem_by_id(problem_id)
        
        if problem_info:
            print(f"\nID: {problem_info['id']}")
            print(f"Title: {problem_info['title']}")
            print(f"Difficulty: {problem_info.get('difficulty', 'unknown')}")
            print(f"Category: {problem_info.get('category', 'uncategorized')}")
            print(f"Platform: {problem_info.get('platform', 'none')}")
            print(f"Description: {problem_info.get('description', 'No description available.')}")
        else:
            print(f"\nProblem ID: {problem_id}")
            print(f"Title: {problem_title}")
            print("No additional details available.")
        
        # Check which implementations are available
        has_python = check_implementation(problem_id, "python")
        has_triton = check_implementation(problem_id, "triton")
        has_cuda = check_implementation(problem_id, "cuda")
        
        print(f"\nAvailable implementations:")
        print(f"  Python: {'✓' if has_python else '✗'}")
        print(f"  Triton: {'✓' if has_triton else '✗'}")
        print(f"  CUDA:   {'✓' if has_cuda else '✗'}")
        
        # Show options
        print("\nOptions:")
        option_index = 1
        
        if has_python:
            print_menu_item(option_index, "Run Python implementation")
            option_index += 1
        else:
            print_menu_item(option_index, "Python implementation not available")
            option_index += 1
            
        if has_triton:
            print_menu_item(option_index, "Run Triton implementation")
            option_index += 1
        else:
            print_menu_item(option_index, "Triton implementation not available")
            option_index += 1
            
        if has_cuda:
            print_menu_item(option_index, "Run CUDA implementation")
            option_index += 1
        else:
            print_menu_item(option_index, "CUDA implementation not available")
            option_index += 1
        
        print_menu_item(option_index, "Run benchmark")
        option_index += 1
        
        print_menu_item(0, "Back to problem list")
        
        choice = get_user_choice(0, option_index - 1)
        
        if choice == 0:
            return
        
        # Calculate actual option index accounting for unavailable implementations
        python_option = 1
        triton_option = 2
        cuda_option = 3
        benchmark_option = 4
        
        if choice == python_option and has_python:
            menu_run_implementation(problem_id, problem_title)
        elif choice == triton_option and has_triton:
            menu_run_implementation(problem_id, problem_title)
        elif choice == cuda_option and has_cuda:
            menu_run_implementation(problem_id, problem_title)
        elif choice == benchmark_option:
            run_benchmark_for_problem(problem_id)
            input("\nPress Enter to continue...")


def menu_run_implementation(problem_id: str, problem_title: str):
    """
    Menu for running a specific implementation.
    
    Args:
        problem_id: ID of the problem
        problem_title: Title of the problem for display
    """
    clear_screen()
    print_header(f"RUN IMPLEMENTATION: {problem_title}")
    
    # Check which implementation types are available
    implementation_types = []
    
    if ENHANCED_BENCH_LOADED:
        # Use enhanced implementation registry
        available_implementations = list_implementations(problem_id)
        
        # Check if any implementations were found
        if not any(available_implementations.values()):
            print("\nNo implementations found for this problem.")
            input("\nPress Enter to continue...")
            return
        
        # Display available implementation types
        print("\nAvailable implementation types:")
        
        # Display available implementation types and variants
        all_impls = []
        for impl_type, variants in available_implementations.items():
            if variants:
                print(f"\n{impl_type.capitalize()}:")
                for i, variant in enumerate(variants, 1):
                    print(f"  {i}. {variant}")
                    all_impls.append((impl_type, variant))
    else:
        # Legacy check
        if check_implementation(problem_id, "python"):
            print_menu_item(1, "Python")
            implementation_types.append("python")
            
        if check_implementation(problem_id, "triton"):
            print_menu_item(len(implementation_types) + 1, "Triton")
            implementation_types.append("triton")
            
        if check_implementation(problem_id, "cuda"):
            print_menu_item(len(implementation_types) + 1, "CUDA")
            implementation_types.append("cuda")
    
    if not implementation_types:
        print("\nNo implementations found for this problem.")
        input("\nPress Enter to continue...")
        return
    
    # Get user's choice of implementation type
    impl_choice = get_user_choice(1, len(implementation_types))
    if impl_choice == 0:
        return
    
    selected_type = implementation_types[impl_choice - 1]
    
    # Get available variants for the selected implementation type
    if ENHANCED_BENCH_LOADED:
        variants = available_implementations.get(selected_type, [])
        
        # If only one variant, run it directly
        if len(variants) == 1:
            variant = variants[0]
        else:
            # Let user select variant
            print(f"\nAvailable {selected_type.capitalize()} variants:")
            for i, variant in enumerate(variants, 1):
                print_menu_item(i, variant)
            
            variant_choice = get_user_choice(1, len(variants))
            if variant_choice == 0:
                return
            
            variant = variants[variant_choice - 1]
    else:
        # Legacy mode - only v1 available
        variant = 'v1'
    
    # Ask if user wants to use custom input
    print("\nInput options:")
    print_menu_item(1, "Use default input")
    print_menu_item(2, "Use custom input")
    
    input_choice = get_user_choice(1, 2)
    if input_choice == 0:
        return
    
    custom_input = input_choice == 2
    
    # Run the implementation
    try:
        run_implementation(problem_id, selected_type, variant, custom_input)
    except Exception as e:
        print(f"\nError running implementation: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to continue...")


def menu_run_implementation():
    """Menu for running a specific implementation."""
    clear_screen()
    print_header("RUN IMPLEMENTATION")
    
    # First, select a group
    groups = list_groups()
    if not groups:
        print("\nNo groups found.")
        return
    
    print("\nSelect a group:")
    for i, (group_id, group_name) in enumerate(groups, 1):
        print_menu_item(i, f"{group_name} ({group_id})")
    
    print_menu_item(0, "Back to Main Menu")
    group_choice = get_user_choice(0, len(groups))
    
    if group_choice == 0:
        return
    
    group_id, group_name = groups[group_choice - 1]
    
    # Then, select a problem
    problems = list_problems_in_group(group_id)
    if not problems:
        print(f"\nNo problems found in group '{group_name}'.")
        return
    
    print(f"\nSelect a problem from {group_name}:")
    for i, (problem_id, problem_title) in enumerate(problems, 1):
        print_menu_item(i, f"{problem_title} ({problem_id})")
    
    print_menu_item(0, "Back")
    problem_choice = get_user_choice(0, len(problems))
    
    if problem_choice == 0:
        return
    
    problem_id, problem_title = problems[problem_choice - 1]
    
    # Finally, select an implementation
    menu_run_implementation(problem_id, problem_title)


def menu_run_benchmark():
    """Menu for running benchmarks."""
    if not ENHANCED_BENCH_LOADED:
        print("\nError: Enhanced benchmarking module not loaded.")
        return
    
    clear_screen()
    print_header("RUN BENCHMARK")
    
    # First, select a group
    groups = list_groups()
    if not groups:
        print("\nNo groups found.")
        return
    
    print("\nSelect a group:")
    for i, (group_id, group_name) in enumerate(groups, 1):
        print_menu_item(i, f"{group_name} ({group_id})")
    
    print_menu_item(0, "Back to Main Menu")
    group_choice = get_user_choice(0, len(groups))
    
    if group_choice == 0:
        return
    
    group_id, group_name = groups[group_choice - 1]
    
    # Then, select a problem
    problems = list_problems_in_group(group_id)
    if not problems:
        print(f"\nNo problems found in group '{group_name}'.")
        return
    
    print(f"\nSelect a problem from {group_name}:")
    for i, (problem_id, problem_title) in enumerate(problems, 1):
        print_menu_item(i, f"{problem_title} ({problem_id})")
    
    print_menu_item(0, "Back")
    problem_choice = get_user_choice(0, len(problems))
    
    if problem_choice == 0:
        return
    
    problem_id, problem_title = problems[problem_choice - 1]
    
    run_benchmark_for_problem(problem_id)


def run_benchmark_for_problem(problem_id: str):
    """
    Run benchmark for a specific problem with user-defined parameters.
    
    Args:
        problem_id: ID of the problem
    """
    if not ENHANCED_BENCH_LOADED:
        print("\nError: Enhanced benchmarking modules are not available.")
        print("Please install the required dependencies.")
        input("\nPress Enter to continue...")
        return
    
    clear_screen()
    print_header(f"Benchmark for Problem: {problem_id}")
    
    # Get available implementations and variants
    available_impls = list_implementations(problem_id)
    
    if not any(variants for variants in available_impls.values()):
        print("\nNo implementations found for this problem.")
        input("\nPress Enter to return to the previous menu...")
        return
        
    # Print available implementations
    print("\nAvailable implementations:")
    implementations_to_bench = []
    for impl_type, variants in available_impls.items():
        if variants:
            print(f"\n{impl_type.capitalize()}:")
            for variant in variants:
                print(f"  - {variant}")
    
    # Ask for implementations to benchmark
    print("\nSelect implementations to benchmark:")
    print("1. All available implementations")
    print("2. Python implementations only")
    print("3. Custom selection")
    
    choice = get_user_choice(1, 3)
    
    if choice == 1:
        # All available implementations
        implementations_to_bench = []
        for impl_type, variants in available_impls.items():
            for variant in variants:
                implementations_to_bench.append((impl_type, variant))
    elif choice == 2:
        # Python implementations only
        implementations_to_bench = []
        for variant in available_impls.get('python', []):
            implementations_to_bench.append(('python', variant))
    else:
        # Custom selection
        implementations_to_bench = []
        for impl_type, variants in available_impls.items():
            if variants:
                print(f"\n{impl_type.capitalize()} implementations:")
                for i, variant in enumerate(variants, 1):
                    print(f"{i}. {variant}")
                
                print("Enter the numbers of the variants to benchmark (comma-separated, or 'all'):")
                selection = input("> ").strip().lower()
                
                if selection == 'all':
                    for variant in variants:
                        implementations_to_bench.append((impl_type, variant))
                elif selection:
                    try:
                        indices = [int(x.strip()) - 1 for x in selection.split(',')]
                        for idx in indices:
                            if 0 <= idx < len(variants):
                                implementations_to_bench.append((impl_type, variants[idx]))
                    except ValueError:
                        print("Invalid input. Please enter numbers separated by commas.")
    
    if not implementations_to_bench:
        print("\nNo implementations selected for benchmarking.")
        input("\nPress Enter to return to the previous menu...")
        return
    
    # Get benchmark parameters
    print("\nBenchmark Parameters:")
    
    # Number of runs
    print("\nNumber of runs (default: 5):")
    num_runs_input = input("> ").strip()
    num_runs = int(num_runs_input) if num_runs_input.isdigit() else 5
    
    # Number of warmup runs
    print("\nNumber of warmup runs (default: 2):")
    warmup_runs_input = input("> ").strip()
    warmup_runs = int(warmup_runs_input) if warmup_runs_input.isdigit() else 2
    
    # Input sizes for scaling tests
    print("\nInput sizes for scaling tests (comma-separated, default: 128,512,1024,2048):")
    size_input = input("> ").strip()
    if size_input:
        try:
            input_sizes = [int(x.strip()) for x in size_input.split(',')]
        except ValueError:
            print("Invalid input. Using default sizes.")
            input_sizes = [128, 512, 1024, 2048]
    else:
        input_sizes = [128, 512, 1024, 2048]
    
    # Visualization options
    print("\nVisualization Options:")
    print("1. Show plots interactively")
    print("2. Save plots to file only")
    print("3. Both show and save plots")
    
    viz_choice = get_user_choice(1, 3)
    show_plots = viz_choice in [1, 3]
    save_plots = viz_choice in [2, 3]
    
    # Export options
    print("\nExport Options:")
    print("1. Export results to CSV")
    print("2. Skip CSV export")
    
    export_choice = get_user_choice(1, 2)
    export_csv = export_choice == 1
    
    # Memory tracking
    print("\nMemory Tracking:")
    print("1. Track memory usage")
    print("2. Skip memory tracking")
    
    memory_choice = get_user_choice(1, 2)
    track_memory = memory_choice == 1
    
    # Start benchmarking
    print("\nStarting benchmark...")
    print(f"Problem: {problem_id}")
    print(f"Implementations: {implementations_to_bench}")
    print(f"Number of runs: {num_runs}")
    print(f"Warmup runs: {warmup_runs}")
    print(f"Input sizes: {input_sizes}")
    
    try:
        # Run the benchmark
        run_benchmark(
            problem_id=problem_id,
            implementations=implementations_to_bench,
            num_runs=num_runs,
            warmup_runs=warmup_runs,
            input_sizes=input_sizes,
            show_plots=show_plots,
            save_plots=save_plots,
            export_csv=export_csv,
            track_memory=track_memory
        )
        
        print("\nBenchmark completed successfully!")
    except Exception as e:
        print(f"\nError running benchmark: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to return to the previous menu...")


def menu_initialize_problem():
    """Menu for initializing a new problem."""
    if not SOLUTION_TEMPLATE_LOADED:
        print("\nError: Solution template module not loaded.")
        return
    
    clear_screen()
    print_header("INITIALIZE NEW PROBLEM")
    
    # First, select a group
    groups = list_groups()
    if not groups:
        print("\nNo groups found.")
        return
    
    print("\nSelect a group:")
    for i, (group_id, group_name) in enumerate(groups, 1):
        print_menu_item(i, f"{group_name} ({group_id})")
    
    print_menu_item(0, "Back to Main Menu")
    group_choice = get_user_choice(0, len(groups))
    
    if group_choice == 0:
        return
    
    group_id, group_name = groups[group_choice - 1]
    
    # Get problem details
    problem_id = input("\nEnter problem ID (e.g., p001_matrix_vector_dot): ")
    if not problem_id:
        print("\nError: Problem ID cannot be empty.")
        return
    
    problem_title = input("\nEnter problem title: ")
    if not problem_title:
        problem_title = " ".join(word.capitalize() for word in problem_id.split("_")[1:])
    
    problem_description = input("\nEnter problem description: ")
    if not problem_description:
        problem_description = f"Implementation of {problem_title}"
    
    # Select implementations to create
    print("\nSelect implementations to create:")
    print_menu_item(1, "Python only")
    print_menu_item(2, "Python and Triton")
    print_menu_item(3, "Python and CUDA")
    print_menu_item(4, "All (Python, Triton, and CUDA)")
    impl_choice = get_user_choice(1, 4)
    
    if impl_choice == 1:
        implementations = ["python"]
    elif impl_choice == 2:
        implementations = ["python", "triton"]
    elif impl_choice == 3:
        implementations = ["python", "cuda"]
    else:
        implementations = ["python", "triton", "cuda"]
    
    # Confirm
    print(f"\nCreating problem {problem_id} in group {group_id}")
    print(f"Title: {problem_title}")
    print(f"Description: {problem_description}")
    print(f"Implementations: {', '.join(implementations)}")
    
    confirm = input("\nContinue? (y/n): ").lower()
    if confirm != 'y':
        return
    
    # Create the problem structure
    try:
        problem_dir = create_problem_structure(
            problem_id=problem_id,
            group=group_id,
            problem_title=problem_title,
            problem_description=problem_description,
            implementations=implementations
        )
        print(f"\nProblem structure created at {problem_dir}")
    except Exception as e:
        print(f"\nError creating problem structure: {e}")


def check_available_implementations():
    """Check and display which implementations are available on the system."""
    clear_screen()
    print_header("AVAILABLE IMPLEMENTATIONS")
    
    availability = check_implementation_availability()
    
    print("\nPython implementation:", "✓ Available" if availability['python'] else "✗ Not available")
    if not availability['python']:
        print("  - Make sure Python is properly installed")
        print("  - NumPy is required for most implementations")
    
    print("\nTriton implementation:", "✓ Available" if availability['triton'] else "✗ Not available")
    if not availability['triton']:
        print("  - Install Triton with: pip install triton")
        print("  - PyTorch is required for Triton")
    
    print("\nCUDA implementation:", "✓ Available" if availability['cuda'] else "✗ Not available")
    if not availability['cuda']:
        print("  - Install CuPy with: pip install cupy-cudaXX")
        print("  - NVIDIA CUDA toolkit must be installed")
        print("  - Make sure you have a compatible NVIDIA GPU")
    
    if not availability['triton'] and not availability['cuda']:
        print("\nNote: You can still run Python implementations and prepare your code")
        print("for future use on systems with Triton and/or CUDA available.")


def main():
    """Main entry point for the Problems-Kit CLI."""
    parser = argparse.ArgumentParser(description="Problems-Kit Command Line Interface")
    parser.add_argument("--run", help="Run a specific problem (e.g., p001_matrix_vector_dot)")
    parser.add_argument("--impl", choices=["python", "triton", "cuda"], default="python",
                       help="Implementation to run (default: python)")
    parser.add_argument("--benchmark", help="Run benchmark for a specific problem")
    parser.add_argument("--input-sizes", help="Comma-separated list of input sizes for benchmark")
    parser.add_argument("--runs", type=int, default=10, help="Number of benchmark runs")
    
    args = parser.parse_args()
    
    if args.run:
        # Run a specific implementation
        run_implementation(args.run, args.impl)
    elif args.benchmark:
        # Run benchmark
        if args.input_sizes:
            input_sizes = [int(size) for size in args.input_sizes.split(",")]
        else:
            input_sizes = None
        
        run_benchmark(
            problem_id=args.benchmark,
            function_name="solution",
            implementations=[args.impl],
            num_runs=args.runs,
            input_sizes=input_sizes,
            show_plots=True
        )
    else:
        # Launch interactive menu
        try:
            menu_main()
        except KeyboardInterrupt:
            print("\n\nExiting Problems-Kit. Goodbye!")
            sys.exit(0)


if __name__ == "__main__":
    main()
