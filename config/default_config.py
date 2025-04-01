"""
Default configuration for experiments.
"""

# Dataset configurations
MNIST_CONFIG = {
    'dataset': 'mnist',
    'model': 'cnn',
    'batch_size': 32,
    'lr': 0.00005,
    'nworkers': 100,
    'nepochs': 500,
    'gpu': 0,
    'nbyz': 0,
    'byz_type': 'no',
    'aggregation': 'simple_mean',
    'protected': False,
    'bias': 0.1  # Non-IID factor
}

CIFAR10_CONFIG = {
    'dataset': 'cifar10',
    'model': 'cnn',
    'batch_size': 32,
    'lr': 0.00005,
    'nworkers': 100,
    'nepochs': 500,
    'gpu': 0,
    'nbyz': 0,
    'byz_type': 'no',
    'aggregation': 'simple_mean',
    'protected': False,
    'bias': 0.1  # Non-IID factor
}

# Attack configurations
ATTACK_CONFIGS = {
    'no_attack': {
        'nbyz': 0,
        'byz_type': 'no'
    },
    'partial_trim': {
        'nbyz': 30,
        'byz_type': 'partial_trim'
    },
    'full_trim': {
        'nbyz': 30,
        'byz_type': 'full_trim'
    },
    'mean_attack': {
        'nbyz': 30,
        'byz_type': 'mean_attack'
    },
    'full_mean_attack': {
        'nbyz': 30,
        'byz_type': 'full_mean_attack'
    },
    'gaussian': {
        'nbyz': 30,
        'byz_type': 'gaussian'
    },
    'label_flip': {
        'nbyz': 30,
        'byz_type': 'label_flip'
    },
    'backdoor': {
        'nbyz': 30,
        'byz_type': 'backdoor'
    }
}

# Defense configurations
DEFENSE_CONFIGS = {
    'simple_mean': {
        'aggregation': 'simple_mean'
    },
    'trim': {
        'aggregation': 'trim'
    },
    'krum': {
        'aggregation': 'krum'
    },
    'median': {
        'aggregation': 'median'
    }
}

# Protection configurations
PROTECTION_CONFIGS = {
    'no_protection': {
        'protected': False
    },
    'with_protection': {
        'protected': True
    }
}

def get_config(dataset='mnist', attack_type='no_attack', defense_type='simple_mean', protected=False):
    """
    Get experiment configuration by combining settings.
    
    Args:
        dataset: Dataset to use ('mnist' or 'cifar10')
        attack_type: Type of attack to use
        defense_type: Type of defense to use
        protected: Whether to use privacy protection
        
    Returns:
        config: Dictionary of configuration settings
    """
    if dataset == 'mnist':
        config = MNIST_CONFIG.copy()
    elif dataset == 'cifar10':
        config = CIFAR10_CONFIG.copy()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Apply attack, defense, and protection settings
    config.update(ATTACK_CONFIGS[attack_type])
    config.update(DEFENSE_CONFIGS[defense_type])
    config.update({'protected': protected})
    
    return config 