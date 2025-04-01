"""
Federated learning client implementation.
"""

import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy as np
import random

class Client:
    """
    Federated learning client.
    """
    
    def __init__(self, client_id, net, args, ctx=None):
        """
        Initialize client.
        
        Args:
            client_id: Client ID
            net: Neural network model
            args: Command line arguments
            ctx: MXNet context (CPU or GPU)
        """
        self.client_id = client_id
        self.args = args
        
        # Set context (GPU or CPU)
        if ctx is None:
            if args.gpu == -1:
                self.ctx = mx.cpu()
            else:
                self.ctx = mx.gpu(args.gpu)
        else:
            self.ctx = ctx
        
        # Initialize model
        self.net = net
        
        # Set loss function and optimizer
        self.loss_func = gluon.loss.SoftmaxCrossEntropyLoss()
        self.trainer = gluon.Trainer(
            self.net.collect_params(), 'sgd', {'learning_rate': args.lr}
        )
        
        # Load dataset
        self.load_data()
        
        # Flag to indicate if this client is Byzantine
        self.is_byzantine = False
        self.attack_type = 'no'
        
        # Apply attack if this client is Byzantine
        self.apply_attack_if_byzantine()
        
    def load_data(self):
        """
        Load dataset for this client.
        """
        if self.args.dataset == 'mnist':
            from src.datasets.mnist_data import load_data, split_data_for_clients
            train_data, test_data = load_data(self.args.batch_size, self.ctx)
            client_datasets = split_data_for_clients(
                train_data.dataset, self.args.nworkers, non_iid_factor=self.args.bias
            )
            self.data = gluon.data.DataLoader(
                client_datasets[self.client_id], batch_size=self.args.batch_size, shuffle=True
            )
        elif self.args.dataset == 'cifar10':
            from src.datasets.cifar10_data import load_data, split_data_for_clients
            train_data, test_data = load_data(self.args.batch_size, self.ctx)
            client_datasets = split_data_for_clients(
                train_data.dataset, self.args.nworkers, non_iid_factor=self.args.bias
            )
            self.data = gluon.data.DataLoader(
                client_datasets[self.client_id], batch_size=self.args.batch_size, shuffle=True
            )
        else:
            raise ValueError(f"Unknown dataset: {self.args.dataset}")
    
    def apply_attack_if_byzantine(self):
        """
        Apply Byzantine attack to this client if specified in args.
        """
        if self.client_id < self.args.nbyz and self.args.byz_type != 'no':
            self.is_byzantine = True
            self.attack_type = self.args.byz_type
            
            # Special setup for certain attacks
            if self.attack_type == 'label_flip':
                # Flip labels for training
                self.apply_label_flip()
            elif self.attack_type == 'backdoor':
                # Apply backdoor attack
                self.apply_backdoor()
    
    def apply_label_flip(self):
        """
        Apply label flipping attack to this client's dataset.
        """
        # Create a new dataset with flipped labels
        flipped_data = []
        flipped_labels = []
        
        for data_batch, label_batch in self.data:
            flipped_labels_batch = (9 - label_batch) % 10  # Flip labels (e.g., 0->9, 1->8, etc.)
            flipped_data.append(data_batch)
            flipped_labels.append(flipped_labels_batch)
        
        # Reset the dataset with flipped labels
        self.data = gluon.data.ArrayDataset(flipped_data, flipped_labels)
    
    def apply_backdoor(self):
        """
        Apply backdoor attack to this client's dataset.
        """
        if self.args.dataset == 'mnist':
            from src.datasets.mnist_data import create_backdoor_data
        elif self.args.dataset == 'cifar10':
            from src.datasets.cifar10_data import create_backdoor_data
        else:
            raise ValueError(f"Unknown dataset: {self.args.dataset}")
        
        # Create backdoored dataset (target class is 0)
        backdoored_dataset = create_backdoor_data(self.data.dataset, target_label=0)
        
        # Replace a portion of the dataset with backdoored data
        original_size = len(self.data.dataset)
        backdoor_size = int(original_size * 0.1)  # Use 10% backdoored data
        
        # Combine original and backdoored data
        combined_data = []
        combined_labels = []
        
        # Keep most of the original data
        for i in range(original_size - backdoor_size):
            data, label = self.data.dataset[i]
            combined_data.append(data)
            combined_labels.append(label)
        
        # Add backdoored data
        for i in range(backdoor_size):
            data, label = backdoored_dataset[i]
            combined_data.append(data)
            combined_labels.append(label)
        
        # Create new dataset
        self.data = gluon.data.ArrayDataset(combined_data, combined_labels)
    
    def train_epoch(self):
        """
        Train for one epoch and return the gradient.
        
        Returns:
            gradient: Model gradient after training
        """
        # Store the original parameters
        original_params = {}
        for name, param in self.net.collect_params().items():
            original_params[name] = param.data().copy()
        
        # Train for one epoch
        for data_batch, label_batch in self.data:
            data_batch = data_batch.as_in_context(self.ctx)
            label_batch = label_batch.as_in_context(self.ctx)
            
            with autograd.record():
                output = self.net(data_batch)
                loss = self.loss_func(output, label_batch)
            
            loss.backward()
            self.trainer.step(data_batch.shape[0])
        
        # Compute gradient
        gradient = []
        for name, param in self.net.collect_params().items():
            grad = (original_params[name] - param.data()) / self.args.lr
            gradient.append(grad.reshape((-1, 1)))
        
        # Reset parameters to original values
        for name, param in self.net.collect_params().items():
            param.set_data(original_params[name])
        
        return nd.concat(*gradient, dim=0)
    
    def update(self, gradient):
        """
        Update model with given gradient.
        
        Args:
            gradient: Gradient to apply
        """
        idx = 0
        for param in self.net.collect_params().values():
            param_size = param.data().size
            param_shape = param.data().shape
            param.set_data(param.data() - self.args.lr * gradient[idx:idx+param_size].reshape(param_shape))
            idx += param_size 