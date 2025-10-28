"""
Model 3: Limited Dataset (Overfitting to Few Caregivers)

Demonstrates how training on a small number of caregivers (nuclear family = 2)
causes overfitting to their specific personalities, preventing generalization
to novel adults compared to diverse community child-rearing (5-10 caregivers).

Architecture: [15 → 24 → 12 → 1] regression network
Task: Predict adult response (relationship success metric)
"""

from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import numpy as np

from trauma_models.core.base_model import TraumaModel


class LimitedDatasetModel(TraumaModel):
    """
    Regression model predicting adult responses.

    Key insight: Models trained on few caregivers memorize their quirks
    (high train performance, poor test performance on novel caregivers).
    Models trained on many caregivers learn general patterns
    (smaller generalization gap).

    Network Architecture:
        Input [15] → FC(24) + ReLU → FC(12) + ReLU → FC(1) + Sigmoid

    Metrics:
        - Generalization gap: test_error - train_error
        - Weight L2 norm (overfitting indicator)
        - Effective rank of weight matrices
    """

    def __init__(
        self,
        input_dim: int = 15,
        hidden_dims: list[int] = None,
        output_dim: int = 1,
        seed: int = 42
    ):
        super().__init__(seed=seed)

        if hidden_dims is None:
            hidden_dims = [24, 12]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # Build network: [15 → 24 → 12 → 1]
        layers = []

        # Input → first hidden
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())

        # Final output
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        layers.append(nn.Sigmoid())  # Output in [0, 1] range

        self.network = nn.Sequential(*layers)

        # Track training/test errors for gap calculation
        self.train_error = None
        self.test_error = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.

        Args:
            x: Input tensor [batch_size, 15] (interaction features)

        Returns:
            predictions: [batch_size, 1] predicted adult response
        """
        return self.network(x)

    def compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute MSE loss for regression.

        Args:
            outputs: Model predictions [batch_size, 1]
            targets: Ground truth responses [batch_size, 1]

        Returns:
            loss: MSE loss tensor
        """
        return nn.MSELoss()(outputs, targets)

    def generate_dataset(
        self,
        num_caregivers: int = 2,
        interactions_per_caregiver: int = 500,
        test_caregivers: int = 50,
        test_interactions_per_caregiver: int = 40,
        personality_dim: int = 4,
        **kwargs
    ) -> Tuple[TensorDataset, TensorDataset]:
        """
        Generate training and test datasets.

        Training data: Sample from N caregivers (N = 2, 5, or 10)
        Test data: Sample from NOVEL caregivers with different distributions

        This tests whether model learned general patterns or memorized
        specific caregivers' quirks.

        Args:
            num_caregivers: Number of caregivers in training set (2, 5, 10)
            interactions_per_caregiver: Interactions per caregiver
            test_caregivers: Number of novel test caregivers
            test_interactions_per_caregiver: Interactions per test caregiver
            personality_dim: Dimension of caregiver personality vector

        Returns:
            (train_dataset, test_dataset)
        """
        # Import here to avoid circular dependency
        from trauma_models.limited_dataset.dataset import generate_caregiver_dataset

        return generate_caregiver_dataset(
            num_train_caregivers=num_caregivers,
            interactions_per_train_caregiver=interactions_per_caregiver,
            num_test_caregivers=test_caregivers,
            interactions_per_test_caregiver=test_interactions_per_caregiver,
            feature_dim=self.input_dim,
            personality_dim=personality_dim,
            seed=self.seed
        )

    def extract_metrics(self, test_dataset: TensorDataset) -> Dict[str, float]:
        """
        Extract overfitting metrics.

        Metrics:
            - generalization_gap: test_error - train_error
            - weight_l2_norm: Total L2 norm of all weights
            - effective_rank_layer1: Effective rank of first layer weights
            - effective_rank_layer2: Effective rank of second layer weights

        Args:
            test_dataset: Test data for evaluation

        Returns:
            Dictionary of metrics
        """
        self.eval()

        # Compute test error if not already computed
        if self.test_error is None:
            test_inputs, test_targets = test_dataset.tensors
            with torch.no_grad():
                test_outputs = self.forward(test_inputs)
                self.test_error = nn.MSELoss()(test_outputs, test_targets).item()

        # Compute weight L2 norm across all layers
        weight_l2_norm = 0.0
        for param in self.parameters():
            weight_l2_norm += torch.sum(param ** 2).item()
        weight_l2_norm = np.sqrt(weight_l2_norm)

        # Compute effective rank of weight matrices
        # Effective rank = (sum σ_i)^2 / (sum σ_i^2)
        # Lower rank indicates memorization
        def effective_rank(weight_matrix: torch.Tensor) -> float:
            """Compute effective rank via SVD."""
            U, S, V = torch.svd(weight_matrix)
            sum_sigma = torch.sum(S).item()
            sum_sigma_sq = torch.sum(S ** 2).item()
            if sum_sigma_sq == 0:
                return 0.0
            return (sum_sigma ** 2) / sum_sigma_sq

        # Extract weight matrices from linear layers
        linear_layers = [m for m in self.network if isinstance(m, nn.Linear)]

        metrics = {
            "test_error": self.test_error,
            "train_error": self.train_error if self.train_error is not None else 0.0,
            "generalization_gap": (
                self.test_error - (self.train_error if self.train_error is not None else 0.0)
            ),
            "weight_l2_norm": weight_l2_norm,
        }

        # Add effective rank for each layer
        for i, layer in enumerate(linear_layers):
            rank = effective_rank(layer.weight.data)
            metrics[f"effective_rank_layer{i+1}"] = rank

        return metrics

    def train_model(
        self,
        train_dataset: TensorDataset,
        epochs: int,
        learning_rate: float,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> Dict[str, list]:
        """
        Override to compute and store train error.

        Args:
            train_dataset: Training data
            epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size
            verbose: Print progress

        Returns:
            Training history dict
        """
        # Call parent training method
        history = super().train_model(
            train_dataset=train_dataset,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            verbose=verbose
        )

        # Compute final train error
        self.eval()
        train_inputs, train_targets = train_dataset.tensors
        with torch.no_grad():
            train_outputs = self.forward(train_inputs)
            self.train_error = nn.MSELoss()(train_outputs, train_targets).item()

        return history
