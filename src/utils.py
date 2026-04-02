"""
Utility functions for reproducibility, configuration, and training helpers.

Author: Omar Morsi (40236376)
"""

import os
import random
import yaml
import numpy as np
import torch


def set_seed(seed=42):
    """Set random seeds for reproducibility across all libraries.

    Ensures deterministic behavior for PyTorch, NumPy, and Python's
    built-in random module. Also configures CuDNN for deterministic ops.

    Args:
        seed (int): The random seed value. Default: 42.

    Returns:
        None

    Example:
        >>> set_seed(42)
        >>> torch.rand(1).item()  # Will always produce the same value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """Load a YAML configuration file and return it as a nested dictionary.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration as a nested dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.

    Example:
        >>> config = load_config("configs/default.yaml")
        >>> print(config["training"]["learning_rate"])
        0.0001
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def compute_class_weights(labels, num_classes):
    """Compute inverse-frequency class weights for handling class imbalance.

    Weights are computed as N_total / (num_classes * N_class) for each class,
    following the strategy used in sklearn's 'balanced' class weight.

    Args:
        labels (np.ndarray): Array of integer class labels, shape (N,).
        num_classes (int): Total number of classes.

    Returns:
        torch.Tensor: Class weights tensor of shape (num_classes,), dtype float32.

    Example:
        >>> labels = np.array([0, 0, 0, 1, 2, 2])
        >>> weights = compute_class_weights(labels, num_classes=3)
        >>> print(weights)
        tensor([0.6667, 2.0000, 1.0000])
    """
    counts = np.bincount(labels.astype(int), minlength=num_classes)
    counts = np.maximum(counts, 1)
    total = labels.shape[0]
    weights = total / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


class AverageMeter:
    """Computes and stores the running average of a metric.

    Useful for tracking loss and accuracy during training epochs.

    Attributes:
        val (float): Most recent value.
        avg (float): Running average.
        sum (float): Running sum.
        count (int): Number of values added.

    Example:
        >>> meter = AverageMeter()
        >>> meter.update(0.5)
        >>> meter.update(0.3)
        >>> print(f"{meter.avg:.2f}")
        0.40
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics to zero."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        """Add a new value to the running average.

        Args:
            val (float): The value to add.
            n (int): Number of items this value represents (for batch
                averaging). Default: 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
