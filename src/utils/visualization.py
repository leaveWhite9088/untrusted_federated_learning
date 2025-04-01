"""
Visualization utilities for plotting results.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_accuracy(accuracy_history, args, save_path=None):
    """
    Plot accuracy over epochs.
    
    Args:
        accuracy_history: List of accuracy values
        args: Command line arguments
        save_path: Path to save plot (if None, plot is displayed)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(accuracy_history) + 1), accuracy_history, marker='o')
    plt.title(f'Test Accuracy over Epochs - {args.dataset.upper()} with {args.aggregation}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    # Add details to the plot
    plt.annotate(f'Model: {args.model}, Workers: {args.nworkers}, Byzantine: {args.nbyz}',
                xy=(0.5, 0.02), xycoords='figure fraction', ha='center')
    
    # Add privacy protection info if enabled
    if args.protected:
        plt.annotate('Privacy Protection: Enabled',
                    xy=(0.5, 0.06), xycoords='figure fraction', ha='center')
    
    # Save plot if specified
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()

def plot_comparison(results, args, save_path=None):
    """
    Plot comparison of different methods.
    
    Args:
        results: Dictionary of results {method_name: accuracy_history}
        args: Command line arguments
        save_path: Path to save plot (if None, plot is displayed)
    """
    plt.figure(figsize=(12, 8))
    
    for method_name, accuracy_history in results.items():
        plt.plot(range(1, len(accuracy_history) + 1), accuracy_history, marker='o', label=method_name)
    
    plt.title(f'Comparison of Different Methods - {args.dataset.upper()}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    
    # Add details to the plot
    plt.annotate(f'Model: {args.model}, Workers: {args.nworkers}, Byzantine: {args.nbyz}',
                xy=(0.5, 0.02), xycoords='figure fraction', ha='center')
    
    # Save plot if specified
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()

def plot_backdoor_evaluation(benign_accuracy, backdoor_success_rate, args, save_path=None):
    """
    Plot evaluation of backdoor attack.
    
    Args:
        benign_accuracy: Accuracy on clean test data
        backdoor_success_rate: Success rate of backdoor attack
        args: Command line arguments
        save_path: Path to save plot (if None, plot is displayed)
    """
    plt.figure(figsize=(10, 6))
    
    # Bar plot
    methods = ['Benign Accuracy', 'Backdoor Success Rate']
    values = [benign_accuracy, backdoor_success_rate]
    colors = ['green', 'red']
    
    plt.bar(methods, values, color=colors)
    plt.title(f'Backdoor Attack Evaluation - {args.dataset.upper()}')
    plt.ylabel('Rate')
    plt.ylim(0, 1.1)
    
    # Add text labels on top of bars
    for i, v in enumerate(values):
        plt.text(i, v + 0.05, f'{v:.2f}', ha='center')
    
    # Add details to the plot
    plt.annotate(f'Model: {args.model}, Aggregation: {args.aggregation}',
                xy=(0.5, 0.04), xycoords='figure fraction', ha='center')
    plt.annotate(f'Workers: {args.nworkers}, Byzantine: {args.nbyz}',
                xy=(0.5, 0.01), xycoords='figure fraction', ha='center')
    
    # Save plot if specified
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close() 