"""
MNIST experiment implementation.
"""

import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy as np
import os
import matplotlib.pyplot as plt

# Import project modules
from src.models.mnist_models import get_model
from src.attacks.byzantine import apply_attack
from src.defenses.aggregation import aggregate_gradients
from src.federated.client import Client
from src.federated.server import Server
from src.utils.visualization import plot_accuracy

def run_experiment(args):
    """
    Run MNIST federated learning experiment.
    
    Args:
        args: Command line arguments
    """
    print(f"Running MNIST experiment with {args.nworkers} workers")
    print(f"Byzantine attackers: {args.nbyz} with attack type: {args.byz_type}")
    print(f"Aggregation method: {args.aggregation}")
    print(f"Protection enabled: {args.protected}")
    
    # Set context (GPU or CPU)
    if args.gpu == -1:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(args.gpu)
    
    # Initialize model
    net = get_model(args.model, ctx)
    
    # Initialize server
    server = Server(net, args)
    
    # Initialize clients
    clients = [Client(i, net, args) for i in range(args.nworkers)]
    
    # Training loop
    accuracy_history = []
    
    for epoch in range(args.nepochs):
        # Train on clients
        gradients = []
        for client in clients:
            gradient = client.train_epoch()
            gradients.append(gradient)
        
        # Apply Byzantine attack if specified
        if args.nbyz > 0:
            gradients = apply_attack(gradients, args.byz_type, args.nbyz)
            
        # Aggregate gradients
        aggregated_gradient = aggregate_gradients(gradients, args.aggregation, args.nbyz)
        
        # Update model on server
        server.update(aggregated_gradient)
        
        # Evaluate
        test_accuracy = server.evaluate()
        accuracy_history.append(test_accuracy)
        
        print(f"Epoch {epoch+1}/{args.nepochs}, Test Accuracy: {test_accuracy:.4f}")
    
    # Plot results
    plot_accuracy(accuracy_history, args)
    
    return accuracy_history 