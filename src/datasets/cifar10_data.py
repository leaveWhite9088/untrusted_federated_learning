"""
CIFAR-10 dataset loading and processing.
"""

import mxnet as mx
from mxnet import nd, gluon
import numpy as np
import os

def load_data(batch_size, ctx, non_iid_factor=0.0):
    """
    Load CIFAR-10 dataset and create data iterators.
    
    Args:
        batch_size: Batch size for training
        ctx: MXNet context (CPU or GPU)
        non_iid_factor: Factor for non-IID data distribution (0.0 for IID)
        
    Returns:
        train_data: Training data iterator
        test_data: Testing data iterator
    """
    # Load CIFAR-10 dataset
    cifar_train = gluon.data.vision.CIFAR10(train=True)
    cifar_test = gluon.data.vision.CIFAR10(train=False)
    
    # Normalize data
    transformer = gluon.data.vision.transforms.Compose([
        gluon.data.vision.transforms.ToTensor(),
        gluon.data.vision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Apply transformations
    cifar_train = cifar_train.transform_first(transformer)
    cifar_test = cifar_test.transform_first(transformer)
    
    # Create data iterators
    train_data = gluon.data.DataLoader(
        cifar_train, batch_size=batch_size, shuffle=True, num_workers=4
    )
    
    test_data = gluon.data.DataLoader(
        cifar_test, batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    return train_data, test_data

def split_data_for_clients(data, num_clients, non_iid_factor=0.0):
    """
    Split dataset for federated learning clients.
    
    Args:
        data: CIFAR-10 dataset
        num_clients: Number of clients
        non_iid_factor: Factor for non-IID data distribution (0.0 for IID)
        
    Returns:
        client_data: List of datasets for each client
    """
    if non_iid_factor == 0.0:
        # IID data distribution - uniform random split
        num_samples = len(data) // num_clients
        indices = list(range(len(data)))
        np.random.shuffle(indices)
        
        client_data = []
        for i in range(num_clients):
            start_idx = i * num_samples
            end_idx = (i + 1) * num_samples if i < num_clients - 1 else len(data)
            client_indices = indices[start_idx:end_idx]
            client_data.append(gluon.data.Subset(data, client_indices))
            
    else:
        # Non-IID data distribution - biased by class
        # Each client gets data biased toward certain classes
        labels = []
        for _, label in data:
            labels.append(label)
        labels = np.array(labels)
        
        # Sort data by label
        label_indices = [np.where(labels == i)[0] for i in range(10)]
        
        # Distribute data to clients
        client_data = [[] for _ in range(num_clients)]
        
        # First, assign biased data based on non_iid_factor
        major_samples_per_client = int(len(data) * non_iid_factor / num_clients)
        
        for i in range(num_clients):
            # Assign major class (biased data)
            major_class = i % 10
            major_indices = np.random.choice(
                label_indices[major_class], 
                size=major_samples_per_client, 
                replace=False
            )
            client_data[i].extend(major_indices)
            
            # Remove assigned indices
            mask = np.ones(len(label_indices[major_class]), dtype=bool)
            mask[np.searchsorted(label_indices[major_class], major_indices)] = False
            label_indices[major_class] = label_indices[major_class][mask]
        
        # Then distribute remaining data uniformly
        remaining_indices = np.concatenate(label_indices)
        np.random.shuffle(remaining_indices)
        
        remaining_per_client = len(remaining_indices) // num_clients
        
        for i in range(num_clients):
            start_idx = i * remaining_per_client
            end_idx = (i + 1) * remaining_per_client if i < num_clients - 1 else len(remaining_indices)
            client_data[i].extend(remaining_indices[start_idx:end_idx])
        
        # Convert indices to Subset datasets
        for i in range(num_clients):
            client_data[i] = gluon.data.Subset(data, client_data[i])
    
    return client_data

def create_backdoor_data(dataset, target_label):
    """
    Create backdoored dataset by adding a trigger pattern and changing labels.
    
    Args:
        dataset: Original dataset
        target_label: Target label for backdoored samples
        
    Returns:
        backdoored_dataset: Dataset with backdoor trigger
    """
    # Create a new dataset with backdoor trigger
    backdoored_data = []
    backdoored_labels = []
    
    for data, label in dataset:
        # Create a copy of the data
        new_data = data.copy()
        
        # Add backdoor trigger (a small white square pattern in the corner)
        new_data[:, 30:32, 30:32] = 1.0
        
        # Change label to target
        new_label = target_label
        
        backdoored_data.append(new_data)
        backdoored_labels.append(new_label)
    
    # Convert to array or dataset format
    backdoored_data = np.array(backdoored_data)
    backdoored_labels = np.array(backdoored_labels)
    
    # Create a new dataset with the backdoored data
    backdoored_dataset = gluon.data.ArrayDataset(backdoored_data, backdoored_labels)
    
    return backdoored_dataset 