#!/usr/bin/env python
"""
Script to run experiments with different configurations.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from config.default_config import get_config
from src.utils.visualization import plot_comparison

def parse_args():
    parser = argparse.ArgumentParser(description="Run Federated Learning Experiments")
    parser.add_argument("--dataset", type=str, choices=['mnist', 'cifar10'], default='mnist',
                       help="Dataset to use (mnist or cifar10)")
    parser.add_argument("--experiment", type=str, choices=['defense', 'attack', 'protection'], default='defense',
                       help="Type of experiment to run")
    parser.add_argument("--nepochs", type=int, default=100,
                       help="Number of epochs to run (overrides config)")
    parser.add_argument("--gpu", type=int, default=0,
                       help="GPU ID to use (overrides config)")
    return parser.parse_args()

def run_defense_comparison(args):
    """
    Run comparison of different defense mechanisms.
    
    Args:
        args: Command line arguments
    """
    print(f"Running defense comparison experiment on {args.dataset}")
    
    # Set attack type
    attack_type = 'mean_attack'
    
    # Initialize results dictionary
    results = {}
    
    # Run experiments with different defense mechanisms
    for defense_type in ['simple_mean', 'trim', 'krum', 'median']:
        print(f"Testing defense: {defense_type}")
        
        # Get configuration
        config = get_config(args.dataset, attack_type, defense_type)
        
        # Override epochs and GPU
        config['nepochs'] = args.nepochs
        config['gpu'] = args.gpu
        
        # Convert config to argparse namespace
        config_args = argparse.Namespace(**config)
        
        # Run experiment
        if args.dataset == 'mnist':
            from src.experiments.mnist_experiment import run_experiment
        else:
            from src.experiments.cifar10_experiment import run_experiment
        
        accuracy_history = run_experiment(config_args)
        results[defense_type] = accuracy_history
    
    # Plot comparison
    plot_comparison(results, config_args, save_path=f"results/defense_comparison_{args.dataset}.png")
    
    print("Defense comparison experiment completed.")

def run_attack_comparison(args):
    """
    Run comparison of different attack types.
    
    Args:
        args: Command line arguments
    """
    print(f"Running attack comparison experiment on {args.dataset}")
    
    # Set defense type
    defense_type = 'median'  # Usually one of the most robust
    
    # Initialize results dictionary
    results = {}
    
    # Run experiments with different attack types
    for attack_type in ['no_attack', 'partial_trim', 'full_trim', 'mean_attack', 'gaussian', 'label_flip']:
        print(f"Testing attack: {attack_type}")
        
        # Get configuration
        config = get_config(args.dataset, attack_type, defense_type)
        
        # Override epochs and GPU
        config['nepochs'] = args.nepochs
        config['gpu'] = args.gpu
        
        # Convert config to argparse namespace
        config_args = argparse.Namespace(**config)
        
        # Run experiment
        if args.dataset == 'mnist':
            from src.experiments.mnist_experiment import run_experiment
        else:
            from src.experiments.cifar10_experiment import run_experiment
        
        accuracy_history = run_experiment(config_args)
        results[attack_type] = accuracy_history
    
    # Plot comparison
    plot_comparison(results, config_args, save_path=f"results/attack_comparison_{args.dataset}.png")
    
    print("Attack comparison experiment completed.")

def run_protection_comparison(args):
    """
    Run comparison with and without privacy protection.
    
    Args:
        args: Command line arguments
    """
    print(f"Running privacy protection experiment on {args.dataset}")
    
    # Set defense and attack type
    defense_type = 'median'
    attack_type = 'mean_attack'
    
    # Initialize results dictionary
    results = {}
    
    # Run experiments with and without protection
    for protection_label, protected in [('No Protection', False), ('With Protection', True)]:
        print(f"Testing configuration: {protection_label}")
        
        # Get configuration
        config = get_config(args.dataset, attack_type, defense_type, protected)
        
        # Override epochs and GPU
        config['nepochs'] = args.nepochs
        config['gpu'] = args.gpu
        
        # Convert config to argparse namespace
        config_args = argparse.Namespace(**config)
        
        # Run experiment
        if args.dataset == 'mnist':
            from src.experiments.mnist_experiment import run_experiment
        else:
            from src.experiments.cifar10_experiment import run_experiment
        
        accuracy_history = run_experiment(config_args)
        results[protection_label] = accuracy_history
    
    # Plot comparison
    plot_comparison(results, config_args, save_path=f"results/protection_comparison_{args.dataset}.png")
    
    print("Privacy protection experiment completed.")

def main():
    args = parse_args()
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Run the specified experiment
    if args.experiment == 'defense':
        run_defense_comparison(args)
    elif args.experiment == 'attack':
        run_attack_comparison(args)
    elif args.experiment == 'protection':
        run_protection_comparison(args)
    else:
        raise ValueError(f"Unknown experiment type: {args.experiment}")

if __name__ == "__main__":
    main() 