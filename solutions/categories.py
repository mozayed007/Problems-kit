"""
Definition of categories and problem groups for the Problems-Kit.
This file serves as a centralized registry of all problem categories and their metadata.
"""

# Define platforms
PLATFORM_DEEPML = "DeepML"
PLATFORM_LEETGPU = "LeetGPU"

# Define difficulty levels
DIFFICULTY_EASY = "easy"
DIFFICULTY_MEDIUM = "medium"
DIFFICULTY_HARD = "hard"

# Define categories
CATEGORY_LINEAR_ALGEBRA = "Linear Algebra"
CATEGORY_STATISTICS = "Statistics"
CATEGORY_MACHINE_LEARNING = "Machine Learning"
CATEGORY_DEEP_LEARNING = "Deep Learning"
CATEGORY_COMPUTER_VISION = "Computer Vision"
CATEGORY_PROBABILITY = "Probability"
CATEGORY_ALGORITHM = "Algorithm"
CATEGORY_ACTIVATION = "Activation"
CATEGORY_REDUCTION = "Reduction"
CATEGORY_ARRAY_OPERATION = "Array Operation"
CATEGORY_BASIC_OPERATION = "Basic Operation"
CATEGORY_OPTIMIZATION = "Optimization"
CATEGORY_NEURAL_NETWORKS = "Neural Networks"

# Define groups
GROUPS = {
    "group_01_linear_algebra": {
        "name": "Fundamental Linear Algebra & Basic ML Operations",
        "description": "Basic vector/matrix operations, matrix transposition, introductory statistics, and fundamental ML concepts."
    },
    "group_02_data_transformations": {
        "name": "Data Transformations, Metrics & Activation Functions",
        "description": "Data manipulation, basic performance metrics, and activation function basics."
    },
    "group_03_statistical_calculations": {
        "name": "Statistical Calculations & Evaluation Metrics",
        "description": "Descriptive statistics, probability distributions, and evaluation metrics for ML."
    },
    "group_04_advanced_activations": {
        "name": "Advanced Activations, Normalization & Architectural Blocks",
        "description": "Advanced activation functions, normalization techniques, and neural network building blocks."
    },
    "group_05_optimization": {
        "name": "Optimization Techniques & Model Evaluation",
        "description": "Optimization techniques, analytical derivatives, and ML model evaluation."
    },
    "group_06_advanced_ml": {
        "name": "Advanced ML Models & Algorithms",
        "description": "Advanced ML models, dimensionality reduction, and complex ML algorithms."
    },
    "group_07_advanced_dl": {
        "name": "Advanced Deep Learning Components",
        "description": "Advanced deep learning operations, custom layers, and DL sub-algorithms."
    },
    "group_08_computer_vision": {
        "name": "Computer Vision & Advanced Visual Processing",
        "description": "Computer vision operations, image processing, and visual analysis techniques."
    },
}

# Define problems
PROBLEMS = {
    # Group 1
    "p001_matrix_vector_dot": {
        "id": 1,
        "title": "Matrix-Vector Dot Product",
        "difficulty": DIFFICULTY_EASY,
        "category": CATEGORY_LINEAR_ALGEBRA,
        "platform": PLATFORM_DEEPML,
        "group": "group_01_linear_algebra",
        "description": "Implement a function to compute the dot product between a matrix and a vector."
    },
    "p002_transpose_matrix": {
        "id": 2,
        "title": "Transpose of a Matrix",
        "difficulty": DIFFICULTY_EASY,
        "category": CATEGORY_LINEAR_ALGEBRA,
        "platform": PLATFORM_DEEPML,
        "group": "group_01_linear_algebra",
        "description": "Implement a function to compute the transpose of a matrix."
    },
    # Add more problems following the same pattern...
}

def get_problem_info(problem_id):
    """
    Get information about a specific problem.
    
    Args:
        problem_id (str): The ID of the problem (e.g., 'p001_matrix_vector_dot')
    
    Returns:
        dict: Problem information or None if not found
    """
    return PROBLEMS.get(problem_id)

def get_problems_by_group(group_id):
    """
    Get all problems in a specific group.
    
    Args:
        group_id (str): The ID of the group (e.g., 'group_01_linear_algebra')
    
    Returns:
        list: List of problem dictionaries in the group
    """
    return [p for p in PROBLEMS.values() if p.get("group") == group_id]

def get_problems_by_category(category):
    """
    Get all problems in a specific category.
    
    Args:
        category (str): The category (e.g., CATEGORY_LINEAR_ALGEBRA)
    
    Returns:
        list: List of problem dictionaries in the category
    """
    return [p for p in PROBLEMS.values() if p.get("category") == category]
