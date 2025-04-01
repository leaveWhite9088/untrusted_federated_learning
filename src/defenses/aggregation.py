"""
Aggregation defense mechanisms against Byzantine attacks.
"""

import mxnet as mx
from mxnet import nd
import numpy as np

def aggregate_gradients(gradients, aggregation_method, num_byzantine, ctx=None):
    """
    Aggregate gradients using specified method.
    
    Args:
        gradients: List of gradients from workers
        aggregation_method: Method to use for aggregation ('simple_mean', 'trim', 'krum', 'median')
        num_byzantine: Number of Byzantine workers
        ctx: MXNet context
        
    Returns:
        Aggregated gradient
    """
    if ctx is None:
        ctx = mx.gpu(0)
        
    if aggregation_method == 'simple_mean':
        return simple_mean(gradients)
    elif aggregation_method == 'trim':
        return trim(gradients, num_byzantine)
    elif aggregation_method == 'krum':
        return krum(gradients, num_byzantine)
    elif aggregation_method == 'median':
        return median(gradients)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")

def simple_mean(gradients):
    """
    Simple mean aggregation of gradients.
    
    Args:
        gradients: List of gradients from workers
        
    Returns:
        Mean of gradients
    """
    all_grads = nd.concat(*gradients, dim=1)
    mean_grad = nd.mean(all_grads, axis=1, keepdims=True)
    return mean_grad

def trim(gradients, num_byzantine):
    """
    Trimmed mean aggregation - removes extreme values before averaging.
    
    Args:
        gradients: List of gradients from workers
        num_byzantine: Number of Byzantine workers
        
    Returns:
        Trimmed mean of gradients
    """
    all_grads = nd.concat(*gradients, dim=1)
    sorted_grads = nd.sort(all_grads, axis=1)
    
    # Determine how many gradients to trim from each side
    n = len(gradients)
    b = num_byzantine
    m = n - b * 2
    
    # Calculate trimmed mean (taking only the middle values)
    if m <= 0:
        # Fall back to median if we would trim too many
        return median(gradients)
    
    trimmed_mean = nd.mean(sorted_grads[:, b:(b + m)], axis=1, keepdims=True)
    return trimmed_mean

def median(gradients):
    """
    Median aggregation of gradients.
    
    Args:
        gradients: List of gradients from workers
        
    Returns:
        Median of gradients
    """
    all_grads = nd.concat(*gradients, dim=1)
    
    # Handle both odd and even number of workers
    if len(gradients) % 2 == 1:
        median_grad = nd.sort(all_grads, axis=1)[:, len(gradients) // 2:(len(gradients) // 2 + 1)]
    else:
        sorted_grads = nd.sort(all_grads, axis=1)
        idx = len(gradients) // 2
        median_grad = sorted_grads[:, (idx-1):(idx+1)].mean(axis=1, keepdims=True)
    
    return median_grad

def score(gradient, all_grads, num_byz):
    """
    Calculate score for Krum algorithm.
    
    Args:
        gradient: Gradient to score
        all_grads: All gradients
        num_byz: Number of Byzantine workers
        
    Returns:
        Krum score (sum of distances to closest neighbors)
    """
    num_neighbors = all_grads.shape[1] - 2 - num_byz
    if num_neighbors <= 0:
        num_neighbors = 1
        
    # Compute squared distances
    squared_distances = nd.square(all_grads - gradient).sum(axis=0)
    
    # Sort distances and sum the smallest num_neighbors
    sorted_distances = squared_distances.sort()
    return nd.sum(sorted_distances[1:(1 + num_neighbors)]).asscalar()

def krum(gradients, num_byzantine):
    """
    Krum aggregation - selects gradient with minimum score.
    
    Args:
        gradients: List of gradients from workers
        num_byzantine: Number of Byzantine workers
        
    Returns:
        Selected gradient using Krum algorithm
    """
    num_workers = len(gradients)
    
    # Handle edge cases
    if num_workers <= 2:
        return simple_mean(gradients)
    
    all_grads = nd.concat(*gradients, dim=1)
    
    # Calculate score for each gradient
    scores = []
    for i in range(num_workers):
        grad = gradients[i]
        scores.append(score(grad, all_grads, num_byzantine))
    
    # Select gradient with minimum score
    min_idx = np.argmin(scores)
    return gradients[min_idx] 