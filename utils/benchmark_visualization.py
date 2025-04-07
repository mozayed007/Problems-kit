"""
Visualization utilities for benchmark results in Problems-Kit.

This module provides functions to visualize benchmark results and export them to CSV.
"""

import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

# Import path manager
from utils.path_manager import (
    ensure_directories_exist,
    get_benchmark_csv_path,
    get_visualization_path,
    BENCHMARKS_DIR,
    VISUALIZATIONS_DIR
)

# Ensure directories exist
ensure_directories_exist()

# Import BenchmarkResult class
from utils.benchmark_enhanced import BenchmarkResult

# Check if plotly is available for interactive plots
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def plot_execution_times(results: Dict[Tuple[str, str], BenchmarkResult],
                         problem_id: str,
                         show_plots: bool = True,
                         save_plots: bool = True,
                         plot_type: str = 'bar',
                         include_ci: bool = True) -> Optional[str]:
    """
    Generate a bar chart of mean execution times for different implementations.
    
    Args:
        results: Dictionary mapping (impl_type, variant) to BenchmarkResult
        problem_id: ID of the problem
        show_plots: Whether to display the plots
        save_plots: Whether to save the plots to file
        plot_type: Type of plot ('bar' or 'box')
        include_ci: Whether to include confidence intervals
        
    Returns:
        Path to the saved plot file if save_plots is True, otherwise None
    """
    if not results:
        print("No benchmark results to plot")
        return None
    
    # Set up figure
    plt.figure(figsize=(12, 8))
    
    # Extract implementation names and execution times
    impl_names = []
    mean_times = []
    max_times = []
    min_times = []
    std_times = []
    ci_lower = []
    ci_upper = []
    
    for (impl_type, variant), result in results.items():
        # Get implementation metadata name if available
        metadata = result.metadata.get('impl_metadata', {})
        name = metadata.get('name', f"{impl_type.capitalize()} {variant}")
        impl_name = f"{name} ({impl_type})"
        
        impl_names.append(impl_name)
        mean_times.append(result.mean_time)
        min_times.append(result.min_time)
        max_times.append(result.max_time)
        std_times.append(result.std_time)
        
        if hasattr(result, 'ci_95_lower') and hasattr(result, 'ci_95_upper'):
            ci_lower.append(result.ci_95_lower)
            ci_upper.append(result.ci_95_upper)
    
    # Create bar chart or box plot
    if plot_type == 'bar':
        # Bar chart of mean execution times
        bars = plt.bar(impl_names, mean_times, alpha=0.8, color='skyblue')
        
        # Add error bars for standard deviation
        if include_ci and ci_lower and ci_upper:
            errors = [(mean - lower, upper - mean) for mean, lower, upper in zip(mean_times, ci_lower, ci_upper)]
            err_low = [err[0] for err in errors]
            err_high = [err[1] for err in errors]
            plt.errorbar(impl_names, mean_times, yerr=[err_low, err_high], fmt='none', color='black', capsize=5)
        else:
            plt.errorbar(impl_names, mean_times, yerr=std_times, fmt='none', color='black', capsize=5)
        
        # Add min/max points
        plt.scatter(impl_names, min_times, color='green', marker='v', label='Min')
        plt.scatter(impl_names, max_times, color='red', marker='^', label='Max')
    else:
        # Box plot of execution times
        data = [result.execution_times for result in results.values()]
        plt.boxplot(data, labels=impl_names, patch_artist=True)
    
    # Annotate the bars with execution times
    for i, (bar, time) in enumerate(zip(bars if plot_type == 'bar' else range(len(mean_times)), mean_times)):
        if plot_type == 'bar':
            height = bar.get_height()
            pos_x = bar.get_x() + bar.get_width() / 2
        else:
            height = mean_times[i]
            pos_x = i + 1
        
        if time < 0.001:
            time_label = f"{time*1e6:.2f} µs"
        elif time < 1.0:
            time_label = f"{time*1e3:.2f} ms"
        else:
            time_label = f"{time:.4f} s"
        
        plt.text(pos_x, height, time_label, ha='center', va='bottom')
    
    # Set up labels and title
    plt.xlabel('Implementation')
    plt.ylabel('Execution Time (seconds)')
    plt.title(f'Benchmark Results for {problem_id}')
    plt.yscale('log')  # Log scale for better comparison
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    
    # Save plot to file
    filename = None
    if save_plots:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{problem_id}_benchmark_{timestamp}.png"
        plt.savefig(get_visualization_path(problem_id, filename), dpi=300, bbox_inches='tight')
        print(f"Plot saved to {get_visualization_path(problem_id, filename)}")
    
    # Show plot
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # Create interactive plot with plotly if available
    if PLOTLY_AVAILABLE and save_plots:
        _create_interactive_plot(results, problem_id, filename.replace('.png', '.html'))
    
    return str(get_visualization_path(problem_id, filename)) if filename else None


def _create_interactive_plot(results: Dict[Tuple[str, str], BenchmarkResult],
                             problem_id: str,
                             filename: str):
    """
    Create an interactive plot using plotly.
    
    Args:
        results: Dictionary mapping (impl_type, variant) to BenchmarkResult
        problem_id: ID of the problem
        filename: Filename to save the plot as
    """
    if not PLOTLY_AVAILABLE:
        return
    
    # Create a subplot with 1 row and 2 columns for comparison
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Execution Time', 'Memory Usage (if tracked)'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Extract implementation data
    impl_names = []
    exec_times = []
    memory_usage = []
    throughput = []
    hover_texts = []
    std_times = []
    colors = {'python': 'rgba(25, 118, 210, 0.7)', 'triton': 'rgba(255, 167, 38, 0.7)', 'cuda': 'rgba(76, 175, 80, 0.7)'}
    
    for (impl_type, variant), result in results.items():
        metadata = result.metadata.get('impl_metadata', {})
        name = metadata.get('name', f"{impl_type.capitalize()} {variant}")
        impl_name = f"{name} ({impl_type})"
        
        # Format time for readability
        if result.mean_time < 0.001:
            time_str = f"{result.mean_time*1e6:.2f} µs"
        elif result.mean_time < 1.0:
            time_str = f"{result.mean_time*1e3:.2f} ms"
        else:
            time_str = f"{result.mean_time:.4f} s"
        
        # Create hover text with detailed information
        hover_text = f"<b>{impl_name}</b><br>"
        hover_text += f"Mean: {time_str}<br>"
        hover_text += f"Median: {result.median_time:.6f} s<br>"
        hover_text += f"Min: {result.min_time:.6f} s<br>"
        hover_text += f"Max: {result.max_time:.6f} s<br>"
        hover_text += f"Std Dev: {result.std_time:.6f} s<br>"
        
        if hasattr(result, 'ci_95_lower') and hasattr(result, 'ci_95_upper'):
            hover_text += f"95% CI: [{result.ci_95_lower:.6f}, {result.ci_95_upper:.6f}]<br>"
        
        if result.memory_usage:
            hover_text += f"Mean Memory: {result.mean_memory:.2f} MB<br>"
            hover_text += f"Max Memory: {result.max_memory:.2f} MB<br>"
        
        if result.throughput:
            hover_text += f"Mean Throughput: {result.mean_throughput:.2f} ops/sec<br>"
        
        impl_names.append(impl_name)
        exec_times.append(result.mean_time)
        memory_usage.append(result.mean_memory if result.memory_usage else 0)
        throughput.append(result.mean_throughput if result.throughput else 0)
        hover_texts.append(hover_text)
        std_times.append(result.std_time)
    
    # Add execution time bars
    fig.add_trace(
        go.Bar(
            x=impl_names,
            y=exec_times,
            error_y=dict(type='data', array=std_times, visible=True),
            name='Execution Time',
            text=exec_times,
            hoverinfo='text',
            hovertext=hover_texts,
            marker_color=[colors.get(result[0], 'rgba(128, 128, 128, 0.7)') for result in results.keys()]
        ),
        row=1, col=1
    )
    
    # Add memory usage bars if tracked
    if any(memory_usage):
        fig.add_trace(
            go.Bar(
                x=impl_names,
                y=memory_usage,
                name='Memory Usage (MB)',
                text=memory_usage,
                hoverinfo='text',
                hovertext=[f"Memory: {mem:.2f} MB" for mem in memory_usage],
                marker_color=[colors.get(result[0], 'rgba(128, 128, 128, 0.7)') for result in results.keys()]
            ),
            row=1, col=2
        )
    
    # Update layout
    fig.update_layout(
        title=f'Benchmark Results for {problem_id}',
        xaxis_title='Implementation',
        yaxis_title='Execution Time (seconds)',
        yaxis_type='log',
        height=600,
        width=1200,
        showlegend=False,
        barmode='group',
        xaxis={'tickangle': 45},
        xaxis2={'tickangle': 45}
    )
    
    # Save the plot
    fig.write_html(get_visualization_path(problem_id, filename))
    print(f"Interactive plot saved to {get_visualization_path(problem_id, filename)}")


def plot_size_comparison(results: Dict[Tuple[str, str, int], BenchmarkResult],
                         problem_id: str,
                         show_plots: bool = True,
                         save_plots: bool = True) -> Optional[str]:
    """
    Generate a line chart comparing performance across different input sizes.
    
    Args:
        results: Dictionary mapping (impl_type, variant, input_size) to BenchmarkResult
        problem_id: ID of the problem
        show_plots: Whether to display the plots
        save_plots: Whether to save the plots to file
        
    Returns:
        Path to the saved plot file if save_plots is True, otherwise None
    """
    if not results:
        print("No benchmark results to plot")
        return None
    
    # Group results by implementation and variant
    implementations = {}
    for (impl_type, variant, size), result in results.items():
        key = (impl_type, variant)
        if key not in implementations:
            implementations[key] = {}
        implementations[key][size] = result
    
    # Set up figure
    plt.figure(figsize=(12, 8))
    
    # Plot line for each implementation
    for (impl_type, variant), size_results in implementations.items():
        # Get implementation metadata name if available
        try:
            first_result = next(iter(size_results.values()))
            metadata = first_result.metadata.get('impl_metadata', {}) if first_result.metadata else {}
            name = metadata.get('name', f"{impl_type.capitalize()} {variant}")
        except (StopIteration, AttributeError):
            name = f"{impl_type.capitalize()} {variant}"
        impl_name = f"{name} ({impl_type})"
        
        # Sort sizes to ensure proper ordering
        sizes = sorted(size_results.keys())
        times = [size_results[size].mean_time for size in sizes]
        
        plt.plot(sizes, times, marker='o', label=impl_name)
    
    # Set up labels and title
    plt.xlabel('Input Size')
    plt.ylabel('Execution Time (seconds)')
    plt.title(f'Performance Scaling for {problem_id}')
    plt.yscale('log')  # Log scale for better comparison
    plt.xscale('log')  # Log scale for input sizes
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    
    # Save plot to file
    filename = None
    if save_plots:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{problem_id}_scaling_{timestamp}.png"
        plt.savefig(get_visualization_path(problem_id, filename), dpi=300, bbox_inches='tight')
        print(f"Plot saved to {get_visualization_path(problem_id, filename)}")
    
    # Show plot
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # Create interactive plot with plotly if available
    if PLOTLY_AVAILABLE and save_plots:
        _create_interactive_scaling_plot(implementations, problem_id, filename.replace('.png', '.html'))
    
    return str(get_visualization_path(problem_id, filename)) if filename else None


def _create_interactive_scaling_plot(implementations: Dict[Tuple[str, str], Dict[int, BenchmarkResult]],
                                     problem_id: str,
                                     filename: str):
    """
    Create an interactive scaling plot using plotly.
    
    Args:
        implementations: Dictionary mapping (impl_type, variant) to a dict of {size: result}
        problem_id: ID of the problem
        filename: Filename to save the plot as
    """
    if not PLOTLY_AVAILABLE:
        return
    
    # Create figure
    fig = go.Figure()
    colors = {'python': 'rgb(25, 118, 210)', 'triton': 'rgb(255, 167, 38)', 'cuda': 'rgb(76, 175, 80)'}
    
    # Add a line for each implementation
    for (impl_type, variant), size_results in implementations.items():
        metadata = next(iter(size_results.values())).metadata.get('impl_metadata', {})
        name = metadata.get('name', f"{impl_type.capitalize()} {variant}")
        impl_name = f"{name} ({impl_type})"
        
        # Sort sizes to ensure proper ordering
        sizes = sorted(size_results.keys())
        times = [size_results[size].mean_time for size in sizes]
        
        # Create hover text
        hover_texts = []
        for size in sizes:
            result = size_results[size]
            if result.mean_time < 0.001:
                time_str = f"{result.mean_time*1e6:.2f} µs"
            elif result.mean_time < 1.0:
                time_str = f"{result.mean_time*1e3:.2f} ms"
            else:
                time_str = f"{result.mean_time:.4f} s"
            
            hover_text = f"<b>{impl_name}</b> at size {size}<br>"
            hover_text += f"Mean: {time_str}<br>"
            hover_text += f"Median: {result.median_time:.6f} s<br>"
            hover_text += f"Min: {result.min_time:.6f} s<br>"
            hover_text += f"Max: {result.max_time:.6f} s<br>"
            hover_text += f"Std Dev: {result.std_time:.6f} s<br>"
            
            if result.memory_usage:
                hover_text += f"Mean Memory: {result.mean_memory:.2f} MB<br>"
            
            if result.throughput:
                hover_text += f"Mean Throughput: {result.mean_throughput:.2f} ops/sec<br>"
            
            hover_texts.append(hover_text)
        
        # Add line to plot
        fig.add_trace(
            go.Scatter(
                x=sizes,
                y=times,
                mode='lines+markers',
                name=impl_name,
                line=dict(color=colors.get(impl_type, 'rgb(128, 128, 128)')),
                hoverinfo='text',
                hovertext=hover_texts
            )
        )
    
    # Update layout
    fig.update_layout(
        title=f'Performance Scaling for {problem_id}',
        xaxis_title='Input Size',
        yaxis_title='Execution Time (seconds)',
        xaxis_type='log',
        yaxis_type='log',
        height=600,
        width=1000,
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Save the plot
    fig.write_html(get_visualization_path(problem_id, filename))
    print(f"Interactive scaling plot saved to {get_visualization_path(problem_id, filename)}")


def export_to_csv(results: Dict[Tuple[str, str], BenchmarkResult],
                 problem_id: str) -> str:
    """
    Export benchmark results to CSV.
    
    Args:
        results: Dictionary mapping (impl_type, variant) to BenchmarkResult
        problem_id: ID of the problem
        
    Returns:
        Path to the saved CSV file
    """
    if not results:
        print("No benchmark results to export")
        return None
    
    # Create CSV filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{problem_id}_benchmark_{timestamp}.csv"
    filepath = get_benchmark_csv_path(problem_id, timestamp)
    
    # Determine which metrics are available
    has_memory = any(result.memory_usage is not None for result in results.values())
    has_throughput = any(result.throughput is not None for result in results.values())
    
    # Write CSV file
    with open(filepath, 'w', newline='') as csvfile:
        # Prepare headers
        headers = ['Implementation', 'Type', 'Variant', 'Mean Time (s)', 'Median Time (s)', 'Min Time (s)', 
                  'Max Time (s)', 'Std Dev (s)', '95% CI Lower', '95% CI Upper']
        
        if has_memory:
            headers.extend(['Mean Memory (MB)', 'Max Memory (MB)'])
        
        if has_throughput:
            headers.append('Mean Throughput (ops/sec)')
        
        headers.extend(['Runs', 'Timestamp', 'Description'])
        
        # Create CSV writer
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        
        # Write data for each implementation
        for (impl_type, variant), result in results.items():
            # Get implementation metadata
            metadata = result.metadata.get('impl_metadata', {})
            name = metadata.get('name', f"{impl_type.capitalize()} {variant}")
            description = metadata.get('description', f"{impl_type.capitalize()} implementation variant {variant}")
            
            # Prepare row data
            row = {
                'Implementation': name,
                'Type': impl_type,
                'Variant': variant,
                'Mean Time (s)': result.mean_time,
                'Median Time (s)': result.median_time,
                'Min Time (s)': result.min_time,
                'Max Time (s)': result.max_time,
                'Std Dev (s)': result.std_time,
                '95% CI Lower': getattr(result, 'ci_95_lower', 'N/A'),
                '95% CI Upper': getattr(result, 'ci_95_upper', 'N/A'),
                'Runs': result.metadata.get('num_runs', len(result.execution_times)),
                'Timestamp': result.metadata.get('timestamp', datetime.now().isoformat()),
                'Description': description
            }
            
            # Add memory metrics if available
            if has_memory:
                row['Mean Memory (MB)'] = result.mean_memory if result.memory_usage else 'N/A'
                row['Max Memory (MB)'] = result.max_memory if result.memory_usage else 'N/A'
            
            # Add throughput if available
            if has_throughput:
                row['Mean Throughput (ops/sec)'] = result.mean_throughput if result.throughput else 'N/A'
            
            writer.writerow(row)
    
    print(f"Benchmark results exported to {filepath}")
    return str(filepath)


def save_benchmark_results(results: Dict[Tuple[str, str], BenchmarkResult],
                          problem_id: str):
    """
    Save benchmark results to a JSON file.
    
    Args:
        results: Dictionary mapping (impl_type, variant) to BenchmarkResult
        problem_id: ID of the problem
        
    Returns:
        Path to the saved JSON file
    """
    ensure_directories_exist()
    
    # Create JSON filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{problem_id}_benchmark_{timestamp}.json"
    filepath = BENCHMARKS_DIR / filename
    
    # Convert results to serializable format
    data = {
        "problem_id": problem_id,
        "timestamp": timestamp,
        "results": [result.to_dict() for result in results.values()]
    }
    
    # Ensure the directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Benchmark results saved to {filepath}")
    return str(filepath)
