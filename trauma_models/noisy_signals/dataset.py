"""
Dataset generation for noisy signals (inconsistent caregiving) demonstration.

Generates caregiver availability patterns with controlled label noise:
- 5% noise: Baseline (nearly consistent caregiving)
- 30% noise: Moderate inconsistency (anxious attachment formation)
- 60% noise: Severe inconsistency (learned helplessness patterns)

Key mechanism: Same context features should predict same outcome,
but label noise breaks this consistency, creating behavioral instability.
"""

import torch
import numpy as np
from torch.utils.data import TensorDataset
from typing import Tuple, Dict


class NoisySignalsDataset:
    """
    Generate datasets with controlled label noise for inconsistent caregiving simulation.

    Pattern:
    - Features represent situational context (time of day, child's state, etc.)
    - Ground truth: Simple decision rule determines if caregiver is available
    - Label noise: Flip labels with probability p_noise for matching context patterns

    This models anxious attachment: child experiences unpredictable responses
    to the same situations, preventing stable pattern learning.
    """

    def __init__(
        self,
        feature_dim: int = 20,
        train_examples: int = 10000,
        test_examples: int = 2000,
        context_features: list = [5, 6, 7, 8],
        seed: int = 42,
    ):
        """
        Initialize dataset generator.

        Args:
            feature_dim: Input dimension (situational features)
            train_examples: Number of training examples
            test_examples: Number of test examples
            context_features: Indices of features that define context pattern
            seed: Random seed
        """
        self.feature_dim = feature_dim
        self.train_examples = train_examples
        self.test_examples = test_examples
        self.context_features = context_features
        self.seed = seed

        # Set random seed
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _generate_ground_truth_labels(self, X: torch.Tensor) -> torch.Tensor:
        """
        Generate ground truth labels using simple decision rule.

        Decision rule: Caregiver available if:
        - Sum of first 5 features > 0 (positive situational factors)
        - OR feature 10 > 1.0 (child's urgent need)

        Args:
            X: Input features [batch_size, feature_dim]

        Returns:
            Ground truth labels [batch_size, 1]
        """
        condition1 = X[:, :5].sum(dim=1) > 0
        condition2 = X[:, 10] > 1.0

        labels = (condition1 | condition2).float().unsqueeze(1)

        return labels

    def _check_context_match(self, X: torch.Tensor) -> torch.Tensor:
        """
        Check which examples match the context pattern for noise injection.

        Context pattern: Features [5, 6, 7, 8] satisfy:
        - X[5] > 0 AND X[7] < 0
        (Represents specific situational context where inconsistency occurs)

        Args:
            X: Input features [batch_size, feature_dim]

        Returns:
            Boolean tensor [batch_size] indicating context match
        """
        condition1 = X[:, 5] > 0
        condition2 = X[:, 7] < 0

        context_match = condition1 & condition2

        return context_match

    def _apply_label_noise(
        self,
        Y: torch.Tensor,
        X: torch.Tensor,
        noise_level: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply label noise to create inconsistent caregiving patterns.

        For examples matching context pattern, flip labels with probability p_noise.
        This creates the inconsistency that prevents stable learning.

        Args:
            Y: Ground truth labels [batch_size, 1]
            X: Input features (to check context pattern)
            noise_level: Probability of flipping label (0.05, 0.30, 0.60)

        Returns:
            (noisy_labels, noise_mask) where noise_mask indicates flipped labels
        """
        Y_noisy = Y.clone()

        # Identify examples matching context pattern
        context_match = self._check_context_match(X)

        # Apply noise to matching examples
        num_examples = len(Y)
        noise_mask = torch.zeros(num_examples, dtype=torch.bool)

        for i in range(num_examples):
            if context_match[i] and np.random.random() < noise_level:
                # Flip label
                Y_noisy[i] = 1.0 - Y_noisy[i]
                noise_mask[i] = True

        return Y_noisy, noise_mask

    def generate_datasets(
        self,
        noise_level: float = 0.05,
    ) -> Tuple[TensorDataset, TensorDataset]:
        """
        Generate training and test datasets with specified noise level.

        Args:
            noise_level: Probability of label flip (0.05, 0.30, 0.60)

        Returns:
            (train_dataset, test_dataset)
            - Train set has noise applied
            - Test set has CLEAN labels (to measure true model quality)
        """
        # Generate training data
        X_train = torch.randn(self.train_examples, self.feature_dim)
        Y_train_clean = self._generate_ground_truth_labels(X_train)

        # Apply noise to training labels
        Y_train_noisy, train_noise_mask = self._apply_label_noise(
            Y_train_clean,
            X_train,
            noise_level
        )

        # Generate test data (CLEAN labels)
        X_test = torch.randn(self.test_examples, self.feature_dim)
        Y_test = self._generate_ground_truth_labels(X_test)

        # Create datasets
        train_dataset = TensorDataset(X_train, Y_train_noisy)
        test_dataset = TensorDataset(X_test, Y_test)

        # Store metadata about noise injection
        num_flipped = train_noise_mask.sum().item()
        num_context_matches = self._check_context_match(X_train).sum().item()

        print(f"\n✓ Dataset generated (noise level: {noise_level:.0%}):")
        print(f"  Training examples: {self.train_examples}")
        print(f"  Test examples: {self.test_examples}")
        print(f"  Context-matching examples (train): {num_context_matches}")
        print(f"  Labels flipped (train): {num_flipped}")
        print(f"  Effective noise rate: {num_flipped / self.train_examples:.2%}")

        return train_dataset, test_dataset

    def generate_multiple_noise_levels(
        self,
        noise_levels: list = [0.05, 0.30, 0.60],
    ) -> dict:
        """
        Generate datasets for multiple noise levels.

        This enables comparison across different inconsistency levels.

        Args:
            noise_levels: List of noise probabilities to test

        Returns:
            Dictionary mapping noise_level → (train_dataset, test_dataset)
        """
        datasets = {}

        for noise_level in noise_levels:
            print(f"\nGenerating dataset for noise level: {noise_level:.0%}")
            train_ds, test_ds = self.generate_datasets(noise_level)
            datasets[noise_level] = {
                "train": train_ds,
                "test": test_ds,
            }

        return datasets

    def analyze_dataset_statistics(
        self,
        dataset: TensorDataset,
    ) -> Dict[str, float]:
        """
        Analyze dataset statistics to understand data distribution.

        Args:
            dataset: Dataset to analyze

        Returns:
            Dictionary with statistics
        """
        X, Y = dataset.tensors

        stats = {
            "num_examples": len(X),
            "feature_dim": X.shape[1],
            "positive_rate": Y.mean().item(),
            "feature_mean": X.mean().item(),
            "feature_std": X.std().item(),
        }

        # Context pattern statistics
        context_match = self._check_context_match(X)
        stats["context_match_rate"] = context_match.float().mean().item()

        # Label distribution for context-matching examples
        if context_match.sum() > 0:
            context_labels = Y[context_match]
            stats["context_positive_rate"] = context_labels.mean().item()

        return stats


if __name__ == "__main__":
    """Quick test of dataset generation."""

    print("=" * 60)
    print("Noisy Signals Dataset Generation Test")
    print("=" * 60)

    dataset_gen = NoisySignalsDataset(seed=42)

    # Test single noise level
    print("\n" + "=" * 60)
    print("Testing Single Noise Level (30%)")
    print("=" * 60)

    train_ds, test_ds = dataset_gen.generate_datasets(noise_level=0.30)

    # Analyze datasets
    print("\n✓ Training set statistics:")
    train_stats = dataset_gen.analyze_dataset_statistics(train_ds)
    for key, val in train_stats.items():
        print(f"  {key}: {val}")

    print("\n✓ Test set statistics:")
    test_stats = dataset_gen.analyze_dataset_statistics(test_ds)
    for key, val in test_stats.items():
        print(f"  {key}: {val}")

    # Test multiple noise levels
    print("\n" + "=" * 60)
    print("Testing Multiple Noise Levels")
    print("=" * 60)

    all_datasets = dataset_gen.generate_multiple_noise_levels(
        noise_levels=[0.05, 0.30, 0.60]
    )

    print("\n✓ Generated datasets for all noise levels:")
    for noise_level, datasets in all_datasets.items():
        print(f"  {noise_level:.0%} noise: {len(datasets['train'])} train, {len(datasets['test'])} test")

    # Show sample data
    X_sample, Y_sample = train_ds.tensors
    print("\n✓ Sample training example:")
    print(f"  Features: {X_sample[0, :5].numpy()}")
    print(f"  Label: {Y_sample[0].item():.0f}")

    print("\n" + "=" * 60)
    print("Dataset generation test complete!")
    print("=" * 60)
