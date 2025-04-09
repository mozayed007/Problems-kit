"""
Problem parser to extract problem information from problems.md.

This module provides functionality to parse the problems.md file and extract
information about problems, groups, and their structure.
"""

import re
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Root directory of the project
ROOT_DIR = Path(__file__).parent.parent

def parse_problems_file(filepath: Path = None) -> Dict[str, Any]:
    """
    Parse the problems.md file to extract problem information.
    
    Args:
        filepath: Path to the problems.md file
        
    Returns:
        Dictionary containing parsed problem information
    """
    if filepath is None:
        filepath = ROOT_DIR / "problems.md"
    
    if not filepath.exists():
        raise FileNotFoundError(f"Problems file not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract groups
    groups = []
    group_pattern = r"## \*\*Group (\d+): ([^\*]+)\*\*"
    for match in re.finditer(group_pattern, content):
        group_number = int(match.group(1))
        group_name = match.group(2).strip()
        group_id = f"group_{group_number:02d}_{group_name.lower().replace(' & ', '_').replace(' ', '_')}"
        
        # Find the group's section
        group_start = match.end()
        next_group_match = re.search(group_pattern, content[group_start:])
        if next_group_match:
            group_end = group_start + next_group_match.start()
        else:
            group_end = len(content)
        
        group_section = content[group_start:group_end]
        
        # Extract problems from the group's section
        problems = []
        
        # Look for tables in the section
        table_pattern = r"\|Status\|ID / Title\|Difficulty\|Category\|Platform\|Python\|Triton\|CUDA\|\n\|[-\s\|]+\n([\s\S]+?)(?=\n\n|\n##|\Z)"
        table_match = re.search(table_pattern, group_section)
        
        if table_match:
            table_content = table_match.group(1)
            
            # Parse table rows
            row_pattern = r"\|\[[ x]\]\|(.*?)\|(.*?)\|(.*?)\|(.*?)\|\[[ x]\]\|\[[ x]\]\|\[[ x]\]\|"
            for row_match in re.finditer(row_pattern, table_content):
                id_title = row_match.group(1).strip()
                difficulty = row_match.group(2).strip().lower()
                category = row_match.group(3).strip()
                platform = row_match.group(4).strip()
                
                # Extract ID number and title
                id_title_match = re.match(r"(\d+)\.\s*(.*)", id_title)
                if id_title_match:
                    prob_id = int(id_title_match.group(1))
                    title = id_title_match.group(2).strip()
                else:
                    # For entries without a number (e.g., LeetGPU problems)
                    prob_id = None
                    title = id_title
                
                problem_info = {
                    'id': prob_id,
                    'title': title,
                    'difficulty': difficulty,
                    'category': category,
                    'platform': platform,
                    'group_id': group_id,
                    'group_number': group_number,
                    'group_name': group_name
                }
                problems.append(problem_info)
        
        group_info = {
            'id': group_id,
            'number': group_number,
            'name': group_name,
            'problems': problems
        }
        groups.append(group_info)
    
    return {
        'groups': groups
    }

def create_problem_id(problem_info: Dict[str, Any]) -> str:
    """
    Create a standardized problem ID based on problem information.
    
    Args:
        problem_info: Dictionary containing problem information
        
    Returns:
        Standardized problem ID
    """
    if problem_info.get('id'):
        # For problems with numeric IDs
        return f"p{problem_info['id']:03d}_{problem_info['title'].lower().replace(' ', '_').replace('-', '_')}"
    else:
        # For problems without numeric IDs (e.g., LeetGPU problems)
        title_id = problem_info['title'].lower().replace(' ', '_').replace('-', '_')
        return f"{problem_info['platform'].lower()}_{title_id}"

def initialize_group_directories() -> Dict[str, Path]:
    """
    Initialize all group directories based on problems.md.
    
    Returns:
        Dictionary mapping group IDs to their paths
    """
    # Parse problems file
    parsed_data = parse_problems_file()
    
    # Create group directories
    group_paths = {}
    for group in parsed_data['groups']:
        group_path = ROOT_DIR / "solutions" / group['id']
        group_path.mkdir(parents=True, exist_ok=True)
        group_paths[group['id']] = group_path
        
        # Create __init__.py in each group directory
        with open(group_path / "__init__.py", 'w') as f:
            f.write(f'"""\n{group["name"]}\n"""\n')
    
    return group_paths

def generate_categories_file() -> Path:
    """
    Generate the categories.py file based on problems.md.
    
    Returns:
        Path to the generated file
    """
    from solution_template import create_problem_structure
    
    # Parse problems file
    parsed_data = parse_problems_file()
    
    # Extract unique categories and platforms
    categories = set()
    platforms = set()
    difficulty_levels = set()
    
    for group in parsed_data['groups']:
        for problem in group['problems']:
            categories.add(problem['category'])
            platforms.add(problem['platform'])
            difficulty_levels.add(problem['difficulty'])
    
    # Generate categories.py content
    categories_content = []
    categories_content.append('"""')
    categories_content.append('Definition of categories and problem groups for the Problems-Kit.')
    categories_content.append('This file serves as a centralized registry of all problem categories and their metadata.')
    categories_content.append('"""')
    categories_content.append('')
    
    # Define platforms
    categories_content.append('# Define platforms')
    for platform in sorted(platforms):
        platform_var = f"PLATFORM_{platform.upper().replace(' ', '_').replace('-', '_')}"
        categories_content.append(f'{platform_var} = "{platform}"')
    categories_content.append('')
    
    # Define difficulty levels
    categories_content.append('# Define difficulty levels')
    for level in sorted(difficulty_levels):
        level_var = f"DIFFICULTY_{level.upper().replace(' ', '_').replace('-', '_')}"
        categories_content.append(f'{level_var} = "{level}"')
    categories_content.append('')
    
    # Define categories
    categories_content.append('# Define categories')
    for category in sorted(categories):
        category_var = f"CATEGORY_{category.upper().replace(' ', '_').replace('-', '_')}"
        categories_content.append(f'{category_var} = "{category}"')
    categories_content.append('')
    
    # Define groups
    categories_content.append('# Define groups')
    categories_content.append('GROUPS = {')
    for group in sorted(parsed_data['groups'], key=lambda g: g['number']):
        categories_content.append(f'    "{group["id"]}": {{')
        categories_content.append(f'        "name": "{group["name"]}",')
        categories_content.append(f'        "description": "Group {group["number"]} problems"')
        categories_content.append('    },')
    categories_content.append('}')
    categories_content.append('')
    
    # Define problems
    categories_content.append('# Define problems')
    categories_content.append('PROBLEMS = {')
    for group in sorted(parsed_data['groups'], key=lambda g: g['number']):
        # Add a comment for each group
        categories_content.append(f'    # Group {group["number"]}')
        
        for problem in group['problems']:
            problem_id = create_problem_id(problem)
            
            if problem.get('id'):
                id_str = str(problem['id'])
            else:
                id_str = "None"
                
            platform_var = f"PLATFORM_{problem['platform'].upper().replace(' ', '_').replace('-', '_')}"
            difficulty_var = f"DIFFICULTY_{problem['difficulty'].upper().replace(' ', '_').replace('-', '_')}"
            category_var = f"CATEGORY_{problem['category'].upper().replace(' ', '_').replace('-', '_')}"
            
            categories_content.append(f'    "{problem_id}": {{')
            categories_content.append(f'        "id": {id_str},')
            categories_content.append(f'        "title": "{problem["title"]}",')
            categories_content.append(f'        "difficulty": {difficulty_var},')
            categories_content.append(f'        "category": {category_var},')
            categories_content.append(f'        "platform": {platform_var},')
            categories_content.append(f'        "group": "{group["id"]}",')
            categories_content.append(f'        "description": "Implementation of {problem["title"]}"')
            categories_content.append('    },')
        
        categories_content.append('')
    categories_content.append('}')
    categories_content.append('')
    
    # Define utility functions
    categories_content.append('def get_problem_info(problem_id):')
    categories_content.append('    """')
    categories_content.append('    Get information about a specific problem.')
    categories_content.append('    ')
    categories_content.append('    Args:')
    categories_content.append('        problem_id (str): The ID of the problem (e.g., \'p001_matrix_vector_dot\')')
    categories_content.append('    ')
    categories_content.append('    Returns:')
    categories_content.append('        dict: Problem information or None if not found')
    categories_content.append('    """')
    categories_content.append('    return PROBLEMS.get(problem_id)')
    categories_content.append('')
    categories_content.append('def get_problems_by_group(group_id):')
    categories_content.append('    """')
    categories_content.append('    Get all problems in a specific group.')
    categories_content.append('    ')
    categories_content.append('    Args:')
    categories_content.append('        group_id (str): The ID of the group (e.g., \'group_01_linear_algebra\')')
    categories_content.append('    ')
    categories_content.append('    Returns:')
    categories_content.append('        list: List of problem dictionaries in the group')
    categories_content.append('    """')
    categories_content.append('    return [p for p in PROBLEMS.values() if p.get("group") == group_id]')
    categories_content.append('')
    categories_content.append('def get_problems_by_category(category):')
    categories_content.append('    """')
    categories_content.append('    Get all problems in a specific category.')
    categories_content.append('    ')
    categories_content.append('    Args:')
    categories_content.append('        category (str): The category (e.g., CATEGORY_LINEAR_ALGEBRA)')
    categories_content.append('    ')
    categories_content.append('    Returns:')
    categories_content.append('        list: List of problem dictionaries in the category')
    categories_content.append('    """')
    categories_content.append('    return [p for p in PROBLEMS.values() if p.get("category") == category]')

    # Write to file
    categories_path = ROOT_DIR / "solutions" / "categories.py"
    with open(categories_path, 'w') as f:
        f.write('\n'.join(categories_content))
    
    return categories_path

def initialize_example_problem(group_id: str = "group_01_linear_algebra", problem_number: int = 1) -> Path:
    """
    Initialize an example problem with the specified group ID and problem number.
    
    Args:
        group_id: ID of the group (e.g., "group_01_linear_algebra")
        problem_number: Problem number
        
    Returns:
        Path to the created problem directory
    """
    # Import necessary function
    from utils.solution_template import create_problem_structure
    
    # Create problem ID and information
    problem_id = f"p{problem_number:03d}_example_problem"
    problem_info = {
        'title': f"Example Problem {problem_number}",
        'description': f"This is an example problem #{problem_number} for demonstration purposes."
    }
    
    # Create the problem structure
    problem_dir = create_problem_structure(
        problem_id=problem_id,
        group=group_id,
        problem_title=problem_info['title'],
        problem_description=problem_info['description'],
        implementations=['python', 'triton', 'cuda']
    )
    
    return problem_dir

def main():
    """Main function to initialize the project structure from the command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Initialize project structure based on problems.md')
    parser.add_argument('--all-groups', action='store_true', help='Initialize all group directories')
    parser.add_argument('--generate-categories', action='store_true', help='Generate categories.py')
    parser.add_argument('--example-problem', action='store_true', help='Initialize an example problem')
    parser.add_argument('--group', help='Group ID for example problem (e.g., group_01_linear_algebra)')
    parser.add_argument('--problem', type=int, help='Problem number for example problem')
    
    args = parser.parse_args()
    
    if args.all_groups:
        group_paths = initialize_group_directories()
        print(f"Initialized {len(group_paths)} group directories:")
        for group_id, path in group_paths.items():
            print(f"  {group_id}: {path}")
    
    if args.generate_categories:
        categories_path = generate_categories_file()
        print(f"Generated categories file: {categories_path}")
    
    if args.example_problem:
        group_id = args.group or "group_01_linear_algebra"
        problem_number = args.problem or 1
        problem_dir = initialize_example_problem(group_id, problem_number)
        print(f"Initialized example problem: {problem_dir}")


if __name__ == "__main__":
    main()
