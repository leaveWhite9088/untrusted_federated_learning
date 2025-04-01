#!/usr/bin/env python
"""
Main entry point for running federated learning experiments.
"""

import argparse
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Untrusted Federated Learning")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"],
                       help="Dataset to use (mnist or cifar10)")
    parser.add_argument("--model", type=str, default="cnn", choices=["mlr", "cnn", "fcnn"],
                       help="Model architecture (mlr, cnn, or fcnn)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.00005,
                       help="Learning rate")
    parser.add_argument("--nworkers", type=int, default=100,
                       help="Number of workers")
    parser.add_argument("--nepochs", type=int, default=500,
                       help="Number of epochs")
    parser.add_argument("--gpu", type=int, default=0,
                       help="GPU ID to use (-1 for CPU)")
    parser.add_argument("--nbyz", type=int, default=0,
                       help="Number of Byzantine workers")
    parser.add_argument("--byz_type", type=str, default="no", 
                        choices=["no", "partial_trim", "full_trim", "mean_attack", 
                                "full_mean_attack", "gaussian", "label_flip", "backdoor"],
                        help="Type of Byzantine attack")
    parser.add_argument("--aggregation", type=str, default="simple_mean",
                        choices=["simple_mean", "trim", "krum", "median"],
                        help="Aggregation rule")
    parser.add_argument("--protected", action="store_true",
                       help="Enable privacy protection")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Import appropriate modules based on dataset
    if args.dataset == "mnist":
        from src.experiments.mnist_experiment import run_experiment
    elif args.dataset == "cifar10":
        from src.experiments.cifar10_experiment import run_experiment
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Run the experiment
    run_experiment(args)

if __name__ == "__main__":
    main() 