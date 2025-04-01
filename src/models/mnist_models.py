"""
MNIST model definitions for federated learning.
"""

import mxnet as mx
from mxnet import nd, gluon

def get_model(model_type, ctx):
    """
    Get neural network model for MNIST.
    
    Args:
        model_type: Type of model ('mlr', 'cnn', 'fcnn')
        ctx: MXNet context (GPU or CPU)
        
    Returns:
        Initialized neural network
    """
    num_outputs = 10
    
    if model_type == 'mlr':
        # Multiclass Logistic Regression
        net = gluon.nn.Sequential()
        with net.name_scope():
            net.add(gluon.nn.Dense(num_outputs))
        net.initialize(mx.init.Xavier(), ctx=ctx)
        return net
        
    elif model_type == 'fcnn':
        # Two-layer fully connected neural network
        net = gluon.nn.Sequential()
        with net.name_scope():
            net.add(gluon.nn.Dense(256, activation="relu"))
            net.add(gluon.nn.Dense(256, activation="relu"))
            net.add(gluon.nn.Dense(num_outputs))
        net.initialize(mx.init.Xavier(), ctx=ctx)
        return net
        
    elif model_type == 'cnn':
        # Convolutional Neural Network
        net = gluon.nn.Sequential()
        with net.name_scope():
            net.add(gluon.nn.Conv2D(channels=30, kernel_size=5, activation='relu'))
            net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
            net.add(gluon.nn.Conv2D(channels=50, kernel_size=5, activation='relu'))
            net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
            net.add(gluon.nn.Flatten())
            net.add(gluon.nn.Dense(512, activation="relu"))
            net.add(gluon.nn.Dense(num_outputs))
        net.initialize(mx.init.Xavier(), ctx=ctx)
        return net
        
    else:
        raise ValueError(f"Unknown model type: {model_type}") 