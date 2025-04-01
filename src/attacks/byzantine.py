"""
Implementation of Byzantine attacks on federated learning.
"""

import mxnet as mx
from mxnet import nd
import numpy as np
import random

def apply_attack(gradients, attack_type, num_byzantine):
    """
    Apply Byzantine attack to a subset of gradients.
    
    Args:
        gradients: List of gradients from workers
        attack_type: Type of attack to perform
        num_byzantine: Number of Byzantine workers
        
    Returns:
        Modified list of gradients with attack applied
    """
    if attack_type == 'no':
        return gradients
        
    # Make a copy to avoid modifying the original
    modified_gradients = gradients.copy()
    
    # Select which workers will be Byzantine
    num_workers = len(gradients)
    byzantine_indices = random.sample(range(num_workers), min(num_byzantine, num_workers))
    
    if attack_type == 'partial_trim':
        modified_gradients = partial_trim_attack(modified_gradients, byzantine_indices)
    elif attack_type == 'full_trim':
        modified_gradients = full_trim_attack(modified_gradients, byzantine_indices)
    elif attack_type == 'mean_attack':
        modified_gradients = mean_attack(modified_gradients, byzantine_indices)
    elif attack_type == 'full_mean_attack':
        modified_gradients = full_mean_attack(modified_gradients, byzantine_indices)
    elif attack_type == 'gaussian':
        modified_gradients = gaussian_attack(modified_gradients, byzantine_indices)
    elif attack_type == 'label_flip':
        # Label flip is handled at the client level
        pass
    elif attack_type == 'backdoor':
        # Backdoor is handled at the client level
        pass
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")
        
    return modified_gradients

def partial_trim_attack(gradients, byzantine_indices):
    """
    Partial trim attack targets the trimmed mean defense.
    Byzantine gradients are pushed in the same direction to one extreme.
    
    Args:
        gradients: List of gradients from workers
        byzantine_indices: Indices of Byzantine workers
        
    Returns:
        Modified list of gradients
    """
    if not byzantine_indices:
        return gradients
        
    # Calculate mean of honest gradients
    honest_indices = [i for i in range(len(gradients)) if i not in byzantine_indices]
    honest_grads = [gradients[i] for i in honest_indices]
    mean_grad = nd.mean(nd.concat(*honest_grads, dim=1), axis=1, keepdims=True)
    
    # Compute attack vector (scaled opposite of mean)
    attack_factor = 10.0  # Scale factor for attack
    attack_vector = -attack_factor * mean_grad
    
    # Apply attack to Byzantine workers
    for idx in byzantine_indices:
        gradients[idx] = attack_vector
        
    return gradients

def full_trim_attack(gradients, byzantine_indices):
    """
    Full trim attack targets the trimmed mean defense.
    Byzantine gradients are pushed to opposite extremes.
    
    Args:
        gradients: List of gradients from workers
        byzantine_indices: Indices of Byzantine workers
        
    Returns:
        Modified list of gradients
    """
    if not byzantine_indices:
        return gradients
        
    # Calculate mean of honest gradients
    honest_indices = [i for i in range(len(gradients)) if i not in byzantine_indices]
    honest_grads = [gradients[i] for i in honest_indices]
    mean_grad = nd.mean(nd.concat(*honest_grads, dim=1), axis=1, keepdims=True)
    
    # Compute attack vectors (scaled opposite of mean)
    attack_factor = 10.0  # Scale factor for attack
    
    # Split Byzantine workers into two groups for opposite attacks
    mid_point = len(byzantine_indices) // 2
    first_group = byzantine_indices[:mid_point]
    second_group = byzantine_indices[mid_point:]
    
    # Apply attack to first group (positive direction)
    for idx in first_group:
        gradients[idx] = attack_factor * mean_grad
    
    # Apply attack to second group (negative direction)
    for idx in second_group:
        gradients[idx] = -attack_factor * mean_grad
        
    return gradients

def mean_attack(gradients, byzantine_indices):
    """
    Mean attack targets the simple mean aggregation.
    Byzantine gradients are all set to the same large value.
    
    Args:
        gradients: List of gradients from workers
        byzantine_indices: Indices of Byzantine workers
        
    Returns:
        Modified list of gradients
    """
    if not byzantine_indices:
        return gradients
    
    # Calculate mean of honest gradients
    honest_indices = [i for i in range(len(gradients)) if i not in byzantine_indices]
    honest_grads = [gradients[i] for i in honest_indices]
    mean_grad = nd.mean(nd.concat(*honest_grads, dim=1), axis=1, keepdims=True)
    
    # Create attack gradient (opposite direction, scaled)
    attack_factor = -len(honest_indices) / len(byzantine_indices) * 2.0
    attack_grad = attack_factor * mean_grad
    
    # Apply attack to Byzantine workers
    for idx in byzantine_indices:
        gradients[idx] = attack_grad
        
    return gradients

def full_mean_attack(gradients, byzantine_indices):
    """
    Full mean attack is an enhanced version of the mean attack.
    It computes the exact value needed to negate the honest gradients.
    
    Args:
        gradients: List of gradients from workers
        byzantine_indices: Indices of Byzantine workers
        
    Returns:
        Modified list of gradients
    """
    if not byzantine_indices:
        return gradients
    
    # Calculate sum of honest gradients
    honest_indices = [i for i in range(len(gradients)) if i not in byzantine_indices]
    honest_grads = [gradients[i] for i in honest_indices]
    sum_honest = nd.concat(*honest_grads, dim=1).sum(axis=1, keepdims=True)
    
    # Create an attack gradient that will make the overall sum zero
    attack_grad = -sum_honest / len(byzantine_indices)
    
    # Apply attack to Byzantine workers
    for idx in byzantine_indices:
        gradients[idx] = attack_grad
        
    return gradients

def gaussian_attack(gradients, byzantine_indices):
    """
    Gaussian attack replaces Byzantine gradients with random Gaussian noise.
    
    Args:
        gradients: List of gradients from workers
        byzantine_indices: Indices of Byzantine workers
        
    Returns:
        Modified list of gradients
    """
    if not byzantine_indices:
        return gradients
    
    # Calculate statistics of honest gradients
    honest_indices = [i for i in range(len(gradients)) if i not in byzantine_indices]
    honest_grads = [gradients[i] for i in honest_indices]
    
    # Compute mean and standard deviation
    mean_grad = nd.mean(nd.concat(*honest_grads, dim=1), axis=1, keepdims=True)
    std_grad = nd.std(nd.concat(*honest_grads, dim=1), axis=1, keepdims=True)
    
    # Apply attack to Byzantine workers with much larger standard deviation
    for idx in byzantine_indices:
        noise = nd.random.normal(0, 10, shape=mean_grad.shape, ctx=mean_grad.context)
        gradients[idx] = mean_grad + std_grad * noise
        
    return gradients 