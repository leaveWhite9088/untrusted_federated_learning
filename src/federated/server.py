"""
Federated learning server implementation.
"""

import mxnet as mx
from mxnet import nd, gluon
import numpy as np

class Server:
    """
    Federated learning server.
    """
    
    def __init__(self, net, args, ctx=None):
        """
        Initialize server.
        
        Args:
            net: Neural network model
            args: Command line arguments
            ctx: MXNet context (CPU or GPU)
        """
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
        
        # Load test dataset for evaluation
        self.load_test_data()
        
        # Initialize the aggregation function
        self.init_aggregation()
        
        # Whether to apply privacy protection
        self.protected = args.protected
        
    def load_test_data(self):
        """
        Load test dataset for evaluation.
        """
        if self.args.dataset == 'mnist':
            from src.datasets.mnist_data import load_data
            _, self.test_data = load_data(self.args.batch_size, self.ctx)
        elif self.args.dataset == 'cifar10':
            from src.datasets.cifar10_data import load_data
            _, self.test_data = load_data(self.args.batch_size, self.ctx)
        else:
            raise ValueError(f"Unknown dataset: {self.args.dataset}")
            
    def init_aggregation(self):
        """
        Initialize the aggregation function based on args.
        """
        from src.defenses.aggregation import aggregate_gradients
        self.aggregate_gradients = lambda gradients: aggregate_gradients(
            gradients, self.args.aggregation, self.args.nbyz, self.ctx
        )
        
    def update(self, gradient):
        """
        Update model with aggregated gradient.
        
        Args:
            gradient: Aggregated gradient
        """
        # Apply differential privacy if protected
        if self.protected:
            gradient = self.apply_privacy_protection(gradient)
        
        # Update model parameters
        idx = 0
        for param in self.net.collect_params().values():
            param_size = param.data().size
            param_shape = param.data().shape
            param.set_data(param.data() - self.args.lr * gradient[idx:idx+param_size].reshape(param_shape))
            idx += param_size
            
    def apply_privacy_protection(self, gradient):
        """
        Apply privacy protection to gradient.
        
        Args:
            gradient: Original gradient
            
        Returns:
            protected_gradient: Gradient with privacy protection
        """
        # Apply differential privacy with Gaussian noise
        noise_scale = 0.01  # Scale of Gaussian noise
        noise = nd.random.normal(0, noise_scale, shape=gradient.shape, ctx=self.ctx)
        protected_gradient = gradient + noise
        
        # Clip gradient if too large
        max_norm = 1.0
        norm = nd.norm(protected_gradient)
        if norm > max_norm:
            protected_gradient = protected_gradient * (max_norm / norm)
            
        return protected_gradient
        
    def evaluate(self):
        """
        Evaluate model on test dataset.
        
        Returns:
            accuracy: Test accuracy
        """
        metric = mx.metric.Accuracy()
        
        for data_batch, label_batch in self.test_data:
            data_batch = data_batch.as_in_context(self.ctx)
            label_batch = label_batch.as_in_context(self.ctx)
            
            # Forward pass
            output = self.net(data_batch)
            
            # Update metric
            metric.update(label_batch, output)
            
        # Return accuracy
        return metric.get()[1]
        
    def evaluate_backdoor(self, target_label=0):
        """
        Evaluate model on backdoored test dataset.
        
        Args:
            target_label: Target label for backdoor attack
            
        Returns:
            attack_success_rate: Rate at which backdoored images are classified as target
        """
        success_count = 0
        total_count = 0
        
        # Create backdoored test dataset
        if self.args.dataset == 'mnist':
            from src.datasets.mnist_data import create_backdoor_data
        elif self.args.dataset == 'cifar10':
            from src.datasets.cifar10_data import create_backdoor_data
        else:
            raise ValueError(f"Unknown dataset: {self.args.dataset}")
            
        # Create backdoored test data
        backdoored_test = create_backdoor_data(self.test_data.dataset, target_label)
        backdoored_data = gluon.data.DataLoader(
            backdoored_test, batch_size=self.args.batch_size, shuffle=False
        )
        
        # Evaluate on backdoored data
        for data_batch, label_batch in backdoored_data:
            data_batch = data_batch.as_in_context(self.ctx)
            
            # Forward pass
            output = self.net(data_batch)
            predictions = nd.argmax(output, axis=1)
            
            # Count successful attacks (prediction matches target)
            success_count += (predictions == target_label).sum().asscalar()
            total_count += predictions.shape[0]
            
        # Return attack success rate
        return success_count / total_count if total_count > 0 else 0.0 