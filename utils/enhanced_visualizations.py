"""
Enhanced visualization utilities for the unified benchmarking system.

This module provides advanced visualization capabilities for benchmark results,
including comparative plots, scaling analysis, and error distribution visualizations.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
from datetime import datetime

# Try to import plotly for interactive visualizations
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Import from benchmark_unified
from utils.benchmark_unified import BenchmarkResult


def create_performance_plot(
    results: List[BenchmarkResult],
    save_path: Optional[str] = None,
    show_plot: bool = False,
    log_scale: bool = True,
    title: Optional[str] = None
) -> Optional[str]:
    """
    Create a comprehensive performance comparison plot.
    
    Args:
        results: List of benchmark results
        save_path: Path to save the visualization (if None, will not save)
        show_plot: Whether to display the plot
        log_scale: Whether to use logarithmic scales
        title: Custom title for the plot
        
    Returns:
        Path to the saved file if save_path is provided, otherwise None
    """
    # Group results by implementation and size
    implementations = {}
    sizes = set()
    
    for result in results:
        key = (result.implementation_type, result.variant)
        size = result.input_size
        
        if key not in implementations:
            implementations[key] = {}
            
        implementations[key][size] = result
        sizes.add(size)
    
    # Sort sizes
    sizes = sorted(sizes)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Define colors and markers
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Plot each implementation
    for i, (impl_key, size_results) in enumerate(implementations.items()):
        impl_type, variant = impl_key
        label = f"{impl_type} ({variant})"
        
        # Extract data
        x_values = []
        y_values = []
        y_errors = []
        
        for size in sizes:
            if size in size_results:
                result = size_results[size]
                x_values.append(size)
                y_values.append(result.stats['mean'] * 1000)  # Convert to ms
                y_errors.append(result.stats['std'] * 1000)  # Convert to ms
        
        # Plot with error bars
        plt.errorbar(
            x_values, y_values, yerr=y_errors,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            linestyle='-',
            linewidth=2,
            markersize=8,
            label=label,
            capsize=5
        )
    
    # Set scales
    if log_scale:
        plt.xscale('log', base=2)
        plt.yscale('log')
    
    # Set labels and title
    plt.xlabel('Input Size', fontsize=12)
    plt.ylabel('Execution Time (ms)', fontsize=12)
    
    if title:
        plt.title(title, fontsize=14)
    else:
        problem_id = results[0].problem_id if results else "Unknown"
        plt.title(f'Performance Comparison - {problem_id}', fontsize=14)
    
    # Add grid
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Add legend
    plt.legend(fontsize=10, loc='best')
    
    # Ensure tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return save_path


def create_scaling_analysis_plot(
    results: List[BenchmarkResult],
    baseline_size: int = 128,
    save_path: Optional[str] = None,
    show_plot: bool = False,
    title: Optional[str] = None
) -> Optional[str]:
    """
    Create a plot showing how performance scales with input size.
    
    Args:
        results: List of benchmark results
        baseline_size: Size to use as the baseline for scaling calculations
        save_path: Path to save the visualization
        show_plot: Whether to display the plot
        title: Custom title for the plot
        
    Returns:
        Path to the saved file if save_path is provided, otherwise None
    """
    # Group results by implementation and size
    implementations = {}
    sizes = set()
    
    for result in results:
        key = (result.implementation_type, result.variant)
        size = result.input_size
        
        if key not in implementations:
            implementations[key] = {}
            
        implementations[key][size] = result
        sizes.add(size)
    
    # Sort sizes
    sizes = sorted(sizes)
    
    # Get baseline times for each implementation
    baseline_times = {}
    for impl_key, size_results in implementations.items():
        if baseline_size in size_results:
            baseline_times[impl_key] = size_results[baseline_size].stats['mean']
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Define colors and markers
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Plot each implementation
    for i, (impl_key, size_results) in enumerate(implementations.items()):
        if impl_key not in baseline_times or baseline_times[impl_key] == 0:
            continue
            
        impl_type, variant = impl_key
        label = f"{impl_type} ({variant})"
        
        # Extract data
        x_values = []
        y_values = []
        
        for size in sizes:
            if size in size_results:
                result = size_results[size]
                x_values.append(size)
                # Calculate scaling factor
                scaling = result.stats['mean'] / baseline_times[impl_key]
                y_values.append(scaling)
        
        # Plot line
        plt.plot(
            x_values, y_values,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            linestyle='-',
            linewidth=2,
            markersize=8,
            label=label
        )
    
    # Plot theoretical scaling lines
    x_array = np.array(sizes)
    
    # O(n) - linear
    plt.plot(x_array, x_array / baseline_size, 'k--', alpha=0.5, label='O(n)')
    
    # O(n log n)
    n_log_n = (x_array * np.log2(x_array)) / (baseline_size * np.log2(baseline_size))
    plt.plot(x_array, n_log_n, 'k-.', alpha=0.5, label='O(n log n)')
    
    # O(n²)
    plt.plot(x_array, (x_array / baseline_size)**2, 'k:', alpha=0.5, label='O(n²)')
    
    # Set scales
    plt.xscale('log', base=2)
    plt.yscale('log')
    
    # Set labels and title
    plt.xlabel('Input Size', fontsize=12)
    plt.ylabel(f'Scaling Factor (relative to size {baseline_size})', fontsize=12)
    
    if title:
        plt.title(title, fontsize=14)
    else:
        problem_id = results[0].problem_id if results else "Unknown"
        plt.title(f'Scaling Analysis - {problem_id}', fontsize=14)
    
    # Add grid
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Add legend
    plt.legend(fontsize=10, loc='best')
    
    # Ensure tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return save_path


def create_accuracy_comparison_plot(
    results: List[BenchmarkResult],
    save_path: Optional[str] = None,
    show_plot: bool = False,
    log_scale: bool = True,
    title: Optional[str] = None
) -> Optional[str]:
    """
    Create a plot showing error magnitudes for different implementations.
    
    Args:
        results: List of benchmark results
        save_path: Path to save the visualization
        show_plot: Whether to display the plot
        log_scale: Whether to use logarithmic scale for y-axis
        title: Custom title for the plot
        
    Returns:
        Path to the saved file if save_path is provided, otherwise None
    """
    # Filter results that have error data
    valid_results = [r for r in results if hasattr(r, 'error') and r.error is not None]
    
    if not valid_results:
        print("No valid results with error data found.")
        return None
    
    # Group results by implementation and size
    implementations = {}
    sizes = set()
    
    for result in valid_results:
        key = (result.implementation_type, result.variant)
        size = result.input_size
        
        if key not in implementations:
            implementations[key] = {}
            
        implementations[key][size] = result
        sizes.add(size)
    
    # Sort sizes
    sizes = sorted(sizes)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Define colors and markers
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Plot each implementation
    for i, (impl_key, size_results) in enumerate(implementations.items()):
        impl_type, variant = impl_key
        label = f"{impl_type} ({variant})"
        
        # Extract data
        x_values = []
        y_values = []
        thresholds = []
        
        for size in sizes:
            if size in size_results:
                result = size_results[size]
                x_values.append(size)
                y_values.append(result.error)
                thresholds.append(result.error_threshold)
        
        # Plot error
        plt.plot(
            x_values, y_values,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            linestyle='-',
            linewidth=2,
            markersize=8,
            label=label
        )
    
    # Plot thresholds as a separate line
    if thresholds:
        plt.plot(
            sizes, thresholds,
            color='black',
            linestyle='--',
            linewidth=2,
            label='Error Threshold'
        )
    
    # Set scales
    plt.xscale('log', base=2)
    if log_scale:
        plt.yscale('log')
    
    # Set labels and title
    plt.xlabel('Input Size', fontsize=12)
    plt.ylabel('Error Magnitude', fontsize=12)
    
    if title:
        plt.title(title, fontsize=14)
    else:
        problem_id = results[0].problem_id if results else "Unknown"
        plt.title(f'Numerical Accuracy Comparison - {problem_id}', fontsize=14)
    
    # Add grid
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Add legend
    plt.legend(fontsize=10, loc='best')
    
    # Ensure tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return save_path


def create_html_dashboard(
    results: List[BenchmarkResult],
    output_dir: str = 'benchmarks/html',
    filename: Optional[str] = None
) -> Optional[str]:
    """
    Create an HTML dashboard with interactive visualizations (requires plotly).
    
    Args:
        results: List of benchmark results
        output_dir: Directory to save the HTML file
        filename: Name for the HTML file (without extension)
        
    Returns:
        Path to the saved HTML file if plotly is available, otherwise None
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly is required for creating HTML dashboards. Install with: pip install plotly")
        return None
    
    if not results:
        print("No results provided for dashboard creation")
        return None
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        problem_id = results[0].problem_id
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{problem_id}_dashboard_{timestamp}.html"
    
    # Ensure filename has .html extension
    if not filename.endswith('.html'):
        filename += '.html'
    
    # Full path
    file_path = os.path.join(output_dir, filename)
    
    # Create a DataFrame from results for easier plotting
    data = []
    for result in results:
        if not hasattr(result, 'stats'):
            continue
            
        row = {
            'problem_id': result.problem_id,
            'implementation': result.implementation_type,
            'variant': result.variant,
            'size': result.input_size,
            'mean_time_ms': result.stats['mean'] * 1000,
            'min_time_ms': result.stats['min'] * 1000,
            'max_time_ms': result.stats['max'] * 1000,
            'std_dev_ms': result.stats['std'] * 1000
        }
        
        if hasattr(result, 'error') and result.error is not None:
            row['error'] = result.error
            row['threshold'] = result.error_threshold
            row['passed'] = result.passed
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    if df.empty:
        print("No valid data for dashboard creation")
        return None
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Execution Time by Size", 
            "Performance Scaling",
            "Numerical Accuracy",
            "Execution Time Distribution"
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "bar"}]
        ]
    )
    
    # Add performance comparison
    implementations = df['implementation'] + ' (' + df['variant'] + ')'
    implementations = implementations.unique()
    
    for impl in implementations:
        mask = (df['implementation'] + ' (' + df['variant'] + ')') == impl
        subset = df[mask]
        
        fig.add_trace(
            go.Scatter(
                x=subset['size'],
                y=subset['mean_time_ms'],
                mode='lines+markers',
                name=impl,
                hovertemplate='Size: %{x}<br>Time: %{y:.2f} ms'
            ),
            row=1, col=1
        )
    
    # Add scaling analysis
    # Get baseline size (smallest size)
    baseline_size = df['size'].min()
    
    for impl in implementations:
        mask = (df['implementation'] + ' (' + df['variant'] + ')') == impl
        subset = df[mask]
        
        if baseline_size not in subset['size'].values:
            continue
            
        baseline_time = subset[subset['size'] == baseline_size]['mean_time_ms'].values[0]
        
        fig.add_trace(
            go.Scatter(
                x=subset['size'],
                y=subset['mean_time_ms'] / baseline_time,
                mode='lines+markers',
                name=impl,
                hovertemplate='Size: %{x}<br>Scaling: %{y:.2f}x'
            ),
            row=1, col=2
        )
    
    # Add reference scaling lines
    sizes = sorted(df['size'].unique())
    
    # O(n) - linear scaling
    scaling_n = np.array(sizes) / baseline_size
    fig.add_trace(
        go.Scatter(
            x=sizes,
            y=scaling_n,
            mode='lines',
            line=dict(dash='dash', color='black'),
            name='O(n)',
            hoverinfo='name'
        ),
        row=1, col=2
    )
    
    # O(n log n) scaling
    scaling_nlogn = (np.array(sizes) * np.log2(np.array(sizes))) / (baseline_size * np.log2(baseline_size))
    fig.add_trace(
        go.Scatter(
            x=sizes,
            y=scaling_nlogn,
            mode='lines',
            line=dict(dash='dot', color='black'),
            name='O(n log n)',
            hoverinfo='name'
        ),
        row=1, col=2
    )
    
    # O(n²) scaling
    scaling_n2 = (np.array(sizes) / baseline_size) ** 2
    fig.add_trace(
        go.Scatter(
            x=sizes,
            y=scaling_n2,
            mode='lines',
            line=dict(dash='dashdot', color='black'),
            name='O(n²)',
            hoverinfo='name'
        ),
        row=1, col=2
    )
    
    # Add accuracy comparison
    if 'error' in df.columns:
        for impl in implementations:
            mask = (df['implementation'] + ' (' + df['variant'] + ')') == impl
            subset = df[mask]
            
            if 'error' not in subset.columns or subset['error'].isnull().all():
                continue
                
            fig.add_trace(
                go.Scatter(
                    x=subset['size'],
                    y=subset['error'],
                    mode='lines+markers',
                    name=impl,
                    hovertemplate='Size: %{x}<br>Error: %{y:.8f}'
                ),
                row=2, col=1
            )
        
        # Add threshold line if available
        if 'threshold' in df.columns and not df['threshold'].isnull().all():
            thresholds = []
            for size in sizes:
                # Find threshold for this size
                threshold = df[df['size'] == size]['threshold'].iloc[0] if not df[df['size'] == size].empty else None
                thresholds.append(threshold)
                
            fig.add_trace(
                go.Scatter(
                    x=sizes,
                    y=thresholds,
                    mode='lines',
                    line=dict(dash='dash', color='black'),
                    name='Threshold',
                    hovertemplate='Size: %{x}<br>Threshold: %{y:.8f}'
                ),
                row=2, col=1
            )
    
    # Add execution time distribution
    # Select the largest size for comparison
    max_size = df['size'].max()
    df_max_size = df[df['size'] == max_size]
    
    # Sort by mean execution time
    df_max_size = df_max_size.sort_values('mean_time_ms')
    
    implementations_max = df_max_size['implementation'] + ' (' + df_max_size['variant'] + ')'
    
    fig.add_trace(
        go.Bar(
            x=implementations_max,
            y=df_max_size['mean_time_ms'],
            error_y=dict(
                type='data',
                array=df_max_size['std_dev_ms'],
                visible=True
            ),
            hovertemplate='Implementation: %{x}<br>Time: %{y:.2f} ms',
            name=f'Size {max_size}'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=f"Benchmark Dashboard - {results[0].problem_id}",
        height=800,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    
    # Set log scales for appropriate plots
    fig.update_xaxes(type="log", title_text="Input Size", row=1, col=1)
    fig.update_yaxes(type="log", title_text="Execution Time (ms)", row=1, col=1)
    
    fig.update_xaxes(type="log", title_text="Input Size", row=1, col=2)
    fig.update_yaxes(type="log", title_text="Scaling Factor", row=1, col=2)
    
    fig.update_xaxes(type="log", title_text="Input Size", row=2, col=1)
    fig.update_yaxes(type="log", title_text="Error", row=2, col=1)
    
    fig.update_xaxes(title_text="Implementation", row=2, col=2)
    fig.update_yaxes(title_text=f"Time (ms) at Size {max_size}", row=2, col=2)
    
    # Save figure
    fig.write_html(file_path)
    
    return file_path


def generate_complete_visualization_suite(
    results: List[BenchmarkResult],
    output_dir: str = 'benchmarks/visualizations',
    problem_id: Optional[str] = None,
    timestamp: Optional[str] = None,
    generate_html: bool = True
) -> Dict[str, str]:
    """
    Generate a complete suite of visualizations for benchmark results.
    
    Args:
        results: List of benchmark results
        output_dir: Base directory for output files
        problem_id: Problem ID (extracted from results if None)
        timestamp: Timestamp for filenames (current time if None)
        generate_html: Whether to generate HTML dashboard
        
    Returns:
        Dictionary mapping visualization types to file paths
    """
    if not results:
        print("No results provided for visualization")
        return {}
    
    # Extract problem ID if not provided
    if problem_id is None:
        problem_id = results[0].problem_id
    
    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # File paths
    output_files = {}
    
    # Performance plot
    perf_path = os.path.join(output_dir, f"{problem_id}_performance_{timestamp}.png")
    output_files['performance'] = create_performance_plot(
        results=results,
        save_path=perf_path,
        show_plot=False
    )
    
    # Scaling analysis plot
    scaling_path = os.path.join(output_dir, f"{problem_id}_scaling_{timestamp}.png")
    output_files['scaling'] = create_scaling_analysis_plot(
        results=results,
        save_path=scaling_path,
        show_plot=False
    )
    
    # Accuracy comparison plot
    accuracy_path = os.path.join(output_dir, f"{problem_id}_accuracy_{timestamp}.png")
    output_files['accuracy'] = create_accuracy_comparison_plot(
        results=results,
        save_path=accuracy_path,
        show_plot=False
    )
    
    # HTML dashboard if requested
    if generate_html and PLOTLY_AVAILABLE:
        html_dir = os.path.join(output_dir, 'html')
        html_path = os.path.join(html_dir, f"{problem_id}_dashboard_{timestamp}.html")
        output_files['dashboard'] = create_html_dashboard(
            results=results,
            output_dir=html_dir,
            filename=f"{problem_id}_dashboard_{timestamp}.html"
        )
    
    return output_files
