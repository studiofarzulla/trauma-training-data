"""
Neural network model for catastrophic forgetting demonstration.

Architecture: [30 → 50 → 25 → 10] multi-output regression
Task: Predict 10-dimensional behavioral response vector

Demonstrates why aggressive retraining destroys learned patterns:
- Phase 1: Learn trauma patterns (large dataset)
- Phase 2: Learn therapeutic patterns (small dataset)
- Result: Catastrophic forgetting of Phase 1 without proper strategy
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from typing import Dict, Tuple
import numpy as np

from ..core.base_model import TraumaModel


class CatastrophicForgettingModel(TraumaModel):
    """
    Multi-output regression network for demonstrating catastrophic forgetting.

    Architecture:
        Input [30] → FC(30, 50) + ReLU
                  → FC(50, 25) + ReLU
                  → FC(25, 10) (no activation, raw regression outputs)

    Key insight: Small therapy dataset (150 examples) overwrites
    large trauma dataset (10,000 examples) when using naive retraining.
    """

    def __init__(
        self,
        feature_dim: int = 30,
        output_dim: int = 10,
        hidden_dims: list = [50, 25],
        seed: int = 42,
    ):
        """
        Initialize catastrophic forgetting model.

        Args:
            feature_dim: Input dimension (situational features)
            output_dim: Output dimension (behavioral response vector)
            hidden_dims: Hidden layer sizes [layer1_size, layer2_size]
            seed: Random seed for reproducibility
        """
        super().__init__(seed=seed)

        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        # Build network layers
        self.fc1 = nn.Linear(feature_dim, hidden_dims[0])
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(hidden_dims[1], output_dim)
        # No activation - regression task

        # Store metadata
        self.metadata = {
            "feature_dim": feature_dim,
            "output_dim": output_dim,
            "hidden_dims": hidden_dims,
            "architecture": f"[{feature_dim} → {hidden_dims[0]} → {hidden_dims[1]} → {output_dim}]",
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.

        Args:
            x: Input tensor [batch_size, feature_dim]

        Returns:
            Output tensor [batch_size, output_dim]
        """
        x = self.fc1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.relu2(x)

        x = self.fc3(x)
        # No activation for regression

        return x

    def compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute MSE loss for regression.

        Args:
            outputs: Model predictions [batch_size, output_dim]
            targets: Ground truth [batch_size, output_dim]

        Returns:
            Mean squared error loss
        """
        return nn.functional.mse_loss(outputs, targets)

    def generate_dataset(self, **kwargs) -> Tuple[TensorDataset, TensorDataset]:
        """
        Generate dataset using CatastrophicForgettingDataset.

        This is a placeholder - actual generation done by dataset.py

        Returns:
            (train_dataset, test_dataset)
        """
        from .dataset import CatastrophicForgettingDataset

        dataset_gen = CatastrophicForgettingDataset(
            feature_dim=self.feature_dim,
            output_dim=self.output_dim,
            seed=self.seed,
        )

        datasets = dataset_gen.generate_all_datasets()
        return datasets['trauma_train'], datasets['trauma_test']

    def extract_metrics(self, test_dataset: TensorDataset) -> Dict[str, float]:
        """
        Extract catastrophic forgetting metrics.

        Metrics:
        - MSE loss on test set
        - Mean absolute error (MAE)
        - Per-dimension accuracy

        Args:
            test_dataset: Test data

        Returns:
            Dictionary of metrics
        """
        self.eval()

        X_test, Y_test = test_dataset.tensors
        with torch.no_grad():
            predictions = self.forward(X_test)

        # MSE (already computed by base class)
        mse = nn.functional.mse_loss(predictions, Y_test).item()

        # MAE
        mae = torch.mean(torch.abs(predictions - Y_test)).item()

        # Per-dimension MSE
        per_dim_mse = torch.mean((predictions - Y_test) ** 2, dim=0)

        metrics = {
            "mse": mse,
            "mae": mae,
        }

        # Add per-dimension metrics
        for i in range(self.output_dim):
            metrics[f"dim_{i}_mse"] = per_dim_mse[i].item()

        return metrics

    def compute_forgetting_score(
        self,
        test_dataset: TensorDataset,
        baseline_predictions: torch.Tensor,
    ) -> float:
        """
        Compute catastrophic forgetting score.

        Measures how much the model's predictions changed from baseline
        (pre-therapy) to current state (post-therapy).

        Args:
            test_dataset: Original test data (Phase 1)
            baseline_predictions: Predictions before retraining

        Returns:
            Forgetting score (0 = no forgetting, 1 = complete forgetting)
        """
        self.eval()

        X_test, Y_test = test_dataset.tensors
        with torch.no_grad():
            current_predictions = self.forward(X_test)

        # Compute change in predictions
        prediction_change = torch.mean(
            torch.abs(current_predictions - baseline_predictions)
        ).item()

        # Normalize by average prediction magnitude
        baseline_magnitude = torch.mean(torch.abs(baseline_predictions)).item()

        if baseline_magnitude > 0:
            forgetting_score = prediction_change / baseline_magnitude
        else:
            forgetting_score = 0.0

        return forgetting_score

    def get_weight_snapshot(self) -> Dict[str, torch.Tensor]:
        """
        Get snapshot of all model weights.

        Returns:
            Dictionary mapping layer name to weight tensor copy
        """
        snapshot = {}
        for name, param in self.named_parameters():
            snapshot[name] = param.data.clone()
        return snapshot

    def compute_weight_change(
        self,
        before_snapshot: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Compute weight change statistics.

        Args:
            before_snapshot: Weight snapshot before retraining

        Returns:
            Dictionary of weight change metrics
        """
        changes = {}

        for name, param in self.named_parameters():
            before = before_snapshot[name]
            after = param.data

            # L2 norm of change
            change_norm = torch.norm(after - before).item()
            before_norm = torch.norm(before).item()

            # Relative change
            if before_norm > 0:
                relative_change = change_norm / before_norm
            else:
                relative_change = 0.0

            changes[f"{name}_abs_change"] = change_norm
            changes[f"{name}_rel_change"] = relative_change

        # Overall weight change
        total_change = sum([v for k, v in changes.items() if "abs_change" in k])
        changes["total_weight_change"] = total_change

        return changes

    def clone(self):
        """
        Create a deep copy of the model.

        Returns:
            New model instance with copied weights
        """
        new_model = CatastrophicForgettingModel(
            feature_dim=self.feature_dim,
            output_dim=self.output_dim,
            hidden_dims=self.hidden_dims,
            seed=self.seed,
        )

        # Copy weights
        new_model.load_state_dict(self.state_dict())

        return new_model


if __name__ == "__main__":
    """Quick test of model architecture."""

    print("=" * 60)
    print("Catastrophic Forgetting Model Test")
    print("=" * 60)

    # Create model
    model = CatastrophicForgettingModel(seed=42)

    print(f"\n✓ Model architecture:")
    print(f"  {model.metadata['architecture']}")
    print(f"\n  Layer details:")
    for name, param in model.named_parameters():
        print(f"    {name}: {param.shape}")

    # Test forward pass
    batch_size = 8
    X = torch.randn(batch_size, 30)
    Y = model(X)

    print(f"\n✓ Forward pass:")
    print(f"  Input shape: {X.shape}")
    print(f"  Output shape: {Y.shape}")
    print(f"  Sample output: {Y[0].detach().numpy()}")

    # Test loss computation
    target = torch.randn(batch_size, 10)
    loss = model.compute_loss(Y, target)

    print(f"\n✓ Loss computation:")
    print(f"  MSE loss: {loss.item():.4f}")

    # Test weight snapshot
    snapshot = model.get_weight_snapshot()

    print(f"\n✓ Weight snapshot:")
    print(f"  Captured {len(snapshot)} weight tensors")

    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n✓ Total parameters: {total_params:,}")

    print("\n" + "=" * 60)
    print("Model test complete!")
    print("=" * 60)
