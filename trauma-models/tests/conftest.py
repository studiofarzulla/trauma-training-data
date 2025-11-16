"""
Pytest configuration and fixtures for trauma models tests.
"""

import pytest
import torch
import numpy as np


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


@pytest.fixture
def small_dataset():
    """Create a small synthetic dataset for quick tests."""
    X = torch.randn(50, 10)
    Y = torch.randint(0, 3, (50,))
    return torch.utils.data.TensorDataset(X, Y)


@pytest.fixture
def regression_dataset():
    """Create a small regression dataset."""
    X = torch.randn(50, 15)
    Y = torch.randn(50, 1)
    Y = torch.sigmoid(Y)  # Scale to [0, 1]
    return torch.utils.data.TensorDataset(X, Y)
