"""
Neural network model for noisy signals (inconsistent caregiving) demonstration.

Architecture: [20 → 32 → 16 → 1] binary classification
Task: Predict "is caregiver available?" given situational features

Demonstrates how inconsistent caregiving creates behavioral instability:
- Consistent labels (5% noise) → stable weights, confident predictions
- Moderate noise (30% noise) → increased weight variance, uncertainty
- High noise (60% noise) → severe instability, learned helplessness patterns

Maps to anxious attachment: when caregiver responses are unpredictable,
child cannot learn stable behavioral patterns.
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from typing import Dict, Tuple
import numpy as np

from ..core.base_model import TraumaModel


class NoisySignalsModel(TraumaModel):
    """
    Binary classification network for demonstrating noisy signals effects.

    Architecture:
        Input [20] → FC(20, 32) + ReLU
                  → FC(32, 16) + ReLU
                  → FC(16, 1) + Sigmoid

    Key insight: Label noise creates weight instability that scales with sqrt(noise_level).
    This models anxious attachment - hypervigilance emerges from trying to predict
    unpredictable caregiving patterns.
    """

    def __init__(
        self,
        feature_dim: int = 20,
        hidden_dims: list = [32, 16],
        seed: int = 42,
    ):
        """
        Initialize noisy signals model.

        Args:
            feature_dim: Input dimension (situational features)
            hidden_dims: Hidden layer sizes [layer1_size, layer2_size]
            seed: Random seed for reproducibility
        """
        super().__init__(seed=seed)

        self.feature_dim = feature_dim
        self.hidden_dims = hidden_dims

        # Build network layers
        self.fc1 = nn.Linear(feature_dim, hidden_dims[0])
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(hidden_dims[1], 1)
        self.sigmoid = nn.Sigmoid()

        # Store metadata
        self.metadata = {
            "feature_dim": feature_dim,
            "hidden_dims": hidden_dims,
            "architecture": f"[{feature_dim} → {hidden_dims[0]} → {hidden_dims[1]} → 1]",
            "task": "binary_classification",
        }

        # Track weight variance during training
        self.weight_history = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.

        Args:
            x: Input tensor [batch_size, feature_dim]

        Returns:
            Output tensor [batch_size, 1] - probability of caregiver availability
        """
        x = self.fc1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.relu2(x)

        x = self.fc3(x)
        x = self.sigmoid(x)

        return x

    def compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute binary cross-entropy loss.

        Args:
            outputs: Model predictions [batch_size, 1]
            targets: Ground truth labels [batch_size, 1]

        Returns:
            Binary cross-entropy loss
        """
        return nn.functional.binary_cross_entropy(outputs, targets)

    def generate_dataset(self, **kwargs) -> Tuple[TensorDataset, TensorDataset]:
        """
        Generate dataset using NoisySignalsDataset.

        This is a placeholder - actual generation done by dataset.py

        Returns:
            (train_dataset, test_dataset)
        """
        from .dataset import NoisySignalsDataset

        noise_level = kwargs.get('noise_level', 0.05)

        dataset_gen = NoisySignalsDataset(
            feature_dim=self.feature_dim,
            seed=self.seed,
        )

        train_dataset, test_dataset = dataset_gen.generate_datasets(
            noise_level=noise_level
        )

        return train_dataset, test_dataset

    def extract_metrics(self, test_dataset: TensorDataset) -> Dict[str, float]:
        """
        Extract noisy signals metrics.

        Metrics:
        - Binary cross-entropy loss
        - Accuracy
        - Prediction confidence (mean distance from 0.5)
        - Confidence collapse rate (predictions near 0.5)

        Args:
            test_dataset: Test data

        Returns:
            Dictionary of metrics
        """
        self.eval()

        X_test, Y_test = test_dataset.tensors
        with torch.no_grad():
            predictions = self.forward(X_test)

        # BCE loss
        bce_loss = nn.functional.binary_cross_entropy(predictions, Y_test).item()

        # Accuracy
        predicted_labels = (predictions > 0.5).float()
        accuracy = (predicted_labels == Y_test).float().mean().item()

        # Prediction confidence - how far from 0.5 (uncertainty)?
        confidence = torch.abs(predictions - 0.5).mean().item()

        # Confidence collapse rate - predictions near 0.5 (uncertain)
        uncertain_mask = (predictions > 0.45) & (predictions < 0.55)
        confidence_collapse_rate = uncertain_mask.float().mean().item()

        metrics = {
            "bce_loss": bce_loss,
            "accuracy": accuracy,
            "mean_confidence": confidence,
            "confidence_collapse_rate": confidence_collapse_rate,
        }

        return metrics

    def compute_weight_variance(self) -> float:
        """
        Compute variance of weights across all layers.

        Higher variance indicates instability caused by noisy training signal.

        Returns:
            Total weight variance across all layers
        """
        all_weights = []

        for param in self.parameters():
            all_weights.append(param.data.flatten())

        all_weights = torch.cat(all_weights)
        variance = torch.var(all_weights).item()

        return variance

    def compute_weight_stats(self) -> Dict[str, float]:
        """
        Compute comprehensive weight statistics.

        Returns:
            Dictionary with variance, std, mean, L2 norm per layer
        """
        stats = {}

        for name, param in self.named_parameters():
            weights = param.data.flatten()

            stats[f"{name}_variance"] = torch.var(weights).item()
            stats[f"{name}_std"] = torch.std(weights).item()
            stats[f"{name}_mean"] = torch.mean(weights).item()
            stats[f"{name}_l2_norm"] = torch.norm(weights).item()

        # Overall statistics
        all_weights = torch.cat([p.data.flatten() for p in self.parameters()])
        stats["total_variance"] = torch.var(all_weights).item()
        stats["total_std"] = torch.std(all_weights).item()
        stats["total_l2_norm"] = torch.norm(all_weights).item()

        return stats

    def compute_prediction_variance(
        self,
        test_dataset: TensorDataset,
        num_forward_passes: int = 10,
    ) -> float:
        """
        Compute variance in predictions due to stochastic behavior.

        With dropout disabled, this measures inherent model uncertainty.

        Args:
            test_dataset: Test data
            num_forward_passes: Number of forward passes (for stochastic models)

        Returns:
            Mean variance of predictions across forward passes
        """
        self.eval()

        X_test, _ = test_dataset.tensors

        predictions_list = []
        with torch.no_grad():
            for _ in range(num_forward_passes):
                predictions = self.forward(X_test)
                predictions_list.append(predictions)

        # Stack predictions [num_passes, batch_size, 1]
        predictions_stack = torch.stack(predictions_list, dim=0)

        # Compute variance across passes for each example
        pred_variance = torch.var(predictions_stack, dim=0).mean().item()

        return pred_variance

    def compute_behavioral_consistency(
        self,
        test_dataset: TensorDataset,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Measure behavioral consistency - do similar inputs get similar predictions?

        In noisy training, model learns inconsistent mappings.

        Args:
            test_dataset: Test data
            threshold: Classification threshold

        Returns:
            Dictionary with consistency metrics
        """
        self.eval()

        X_test, Y_test = test_dataset.tensors
        with torch.no_grad():
            predictions = self.forward(X_test)

        predicted_labels = (predictions > threshold).float()

        # Group examples by context pattern (features 5-8)
        context_features = X_test[:, 5:9]

        # Find pairs with similar context (cosine similarity > 0.95)
        from torch.nn.functional import cosine_similarity

        consistency_scores = []

        for i in range(min(100, len(X_test))):  # Sample to avoid O(n^2)
            similarities = cosine_similarity(
                context_features[i:i+1],
                context_features,
                dim=1
            )

            similar_mask = (similarities > 0.95) & (torch.arange(len(X_test)) != i)

            if similar_mask.sum() > 0:
                # Check if predictions are consistent for similar contexts
                my_pred = predicted_labels[i]
                similar_preds = predicted_labels[similar_mask]

                consistency = (similar_preds == my_pred).float().mean().item()
                consistency_scores.append(consistency)

        if len(consistency_scores) > 0:
            mean_consistency = np.mean(consistency_scores)
        else:
            mean_consistency = 1.0  # No similar examples found

        return {
            "behavioral_consistency": mean_consistency,
            "num_comparison_pairs": len(consistency_scores),
        }

    def snapshot_weights(self) -> Dict[str, torch.Tensor]:
        """
        Create snapshot of current weights.

        Returns:
            Dictionary mapping parameter name to weight tensor copy
        """
        snapshot = {}
        for name, param in self.named_parameters():
            snapshot[name] = param.data.clone()
        return snapshot

    def track_weight_evolution(self):
        """
        Track weight statistics during training (call after each epoch).

        Stores weight variance history for later analysis.
        """
        stats = self.compute_weight_stats()
        self.weight_history.append({
            "total_variance": stats["total_variance"],
            "total_std": stats["total_std"],
            "total_l2_norm": stats["total_l2_norm"],
        })

    def clone(self):
        """
        Create a deep copy of the model.

        Returns:
            New model instance with copied weights
        """
        new_model = NoisySignalsModel(
            feature_dim=self.feature_dim,
            hidden_dims=self.hidden_dims,
            seed=self.seed,
        )

        # Copy weights
        new_model.load_state_dict(self.state_dict())

        return new_model


if __name__ == "__main__":
    """Quick test of model architecture."""

    print("=" * 60)
    print("Noisy Signals Model Test")
    print("=" * 60)

    # Create model
    model = NoisySignalsModel(seed=42)

    print(f"\n✓ Model architecture:")
    print(f"  {model.metadata['architecture']}")
    print(f"  Task: {model.metadata['task']}")
    print(f"\n  Layer details:")
    for name, param in model.named_parameters():
        print(f"    {name}: {param.shape}")

    # Test forward pass
    batch_size = 8
    X = torch.randn(batch_size, 20)
    Y = model(X)

    print(f"\n✓ Forward pass:")
    print(f"  Input shape: {X.shape}")
    print(f"  Output shape: {Y.shape}")
    print(f"  Sample outputs (probabilities): {Y[:3, 0].detach().numpy()}")

    # Test loss computation
    target = torch.randint(0, 2, (batch_size, 1)).float()
    loss = model.compute_loss(Y, target)

    print(f"\n✓ Loss computation:")
    print(f"  BCE loss: {loss.item():.4f}")

    # Test weight statistics
    weight_stats = model.compute_weight_stats()

    print(f"\n✓ Weight statistics:")
    print(f"  Total variance: {weight_stats['total_variance']:.6f}")
    print(f"  Total std: {weight_stats['total_std']:.6f}")
    print(f"  Total L2 norm: {weight_stats['total_l2_norm']:.4f}")

    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n✓ Total parameters: {total_params:,}")

    print("\n" + "=" * 60)
    print("Model test complete!")
    print("=" * 60)
