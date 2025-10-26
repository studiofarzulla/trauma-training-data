"""
Dataset generation for catastrophic forgetting demonstration.

Two-phase dataset:
1. Phase 1 (Trauma): 10,000 examples - Authority figures → danger response
2. Phase 2 (Therapy): 150 examples - Authority + safe context → safe response

This asymmetry mirrors real trauma:
- Years of negative patterns (large dataset)
- Brief therapeutic intervention (small dataset)
"""

import torch
import numpy as np
from torch.utils.data import TensorDataset
from typing import Tuple


class CatastrophicForgettingDataset:
    """
    Generate two-phase dataset for catastrophic forgetting experiments.

    Phase 1: Trauma formation
    - Large dataset (10,000 examples)
    - Authority pattern → danger response

    Phase 2: Therapy retraining
    - Small dataset (150 examples)
    - Authority + safe context → safe response
    """

    def __init__(
        self,
        feature_dim: int = 30,
        output_dim: int = 10,
        trauma_examples: int = 10000,
        therapy_examples: int = 150,
        test_trauma_examples: int = 2000,
        test_therapy_examples: int = 500,
        seed: int = 42,
    ):
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.trauma_examples = trauma_examples
        self.therapy_examples = therapy_examples
        self.test_trauma_examples = test_trauma_examples
        self.test_therapy_examples = test_therapy_examples
        self.seed = seed

        # Set random seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Define response patterns (will be used for output generation)
        self.danger_response = torch.tensor([
            0.8, 0.9, 0.7, 0.85, 0.75,  # High activation on danger dimensions
            0.2, 0.1, 0.15, 0.25, 0.3   # Low activation on safety dimensions
        ], dtype=torch.float32)

        self.safe_response = torch.tensor([
            0.1, 0.2, 0.15, 0.1, 0.25,   # Low activation on danger dimensions
            0.8, 0.9, 0.85, 0.75, 0.8    # High activation on safety dimensions
        ], dtype=torch.float32)

        self.neutral_response = torch.tensor([
            0.4, 0.5, 0.45, 0.5, 0.4,    # Moderate on all dimensions
            0.5, 0.4, 0.5, 0.45, 0.5
        ], dtype=torch.float32)

    def _has_authority_pattern(self, X: torch.Tensor) -> torch.Tensor:
        """
        Check if input has authority pattern.

        Authority pattern: First 5 features sum > 2.0
        (Mimics detecting "authority figure present")

        Args:
            X: Input tensor [batch_size, feature_dim]

        Returns:
            Boolean tensor [batch_size] indicating authority presence
        """
        authority_sum = X[:, :5].sum(dim=1)
        return authority_sum > 2.0

    def _has_safe_context(self, X: torch.Tensor) -> torch.Tensor:
        """
        Check if input has safe context indicators.

        Safe context: Features 10-15 all > 1.0
        (Mimics "therapist present", "safe setting", etc.)

        Args:
            X: Input tensor [batch_size, feature_dim]

        Returns:
            Boolean tensor [batch_size] indicating safe context
        """
        safe_features = X[:, 10:15]
        return (safe_features > 1.0).all(dim=1)

    def generate_trauma_phase(self) -> Tuple[TensorDataset, TensorDataset]:
        """
        Generate Phase 1 dataset: Trauma formation.

        Pattern learned:
        - Authority present → danger response
        - No authority → neutral response

        Returns:
            (train_dataset, test_dataset) for trauma phase
        """
        # Training set
        X_train = torch.randn(self.trauma_examples, self.feature_dim)
        Y_train = torch.zeros(self.trauma_examples, self.output_dim)

        authority_mask = self._has_authority_pattern(X_train)

        for i in range(self.trauma_examples):
            if authority_mask[i]:
                # Authority → danger response (with small noise)
                noise = torch.randn(self.output_dim) * 0.05
                Y_train[i] = self.danger_response + noise
            else:
                # No authority → neutral response
                noise = torch.randn(self.output_dim) * 0.05
                Y_train[i] = self.neutral_response + noise

        # Test set (same distribution)
        X_test = torch.randn(self.test_trauma_examples, self.feature_dim)
        Y_test = torch.zeros(self.test_trauma_examples, self.output_dim)

        authority_mask_test = self._has_authority_pattern(X_test)

        for i in range(self.test_trauma_examples):
            if authority_mask_test[i]:
                noise = torch.randn(self.output_dim) * 0.05
                Y_test[i] = self.danger_response + noise
            else:
                noise = torch.randn(self.output_dim) * 0.05
                Y_test[i] = self.neutral_response + noise

        train_dataset = TensorDataset(X_train, Y_train)
        test_dataset = TensorDataset(X_test, Y_test)

        return train_dataset, test_dataset

    def generate_therapy_phase(self) -> Tuple[TensorDataset, TensorDataset]:
        """
        Generate Phase 2 dataset: Therapy retraining.

        Pattern to learn:
        - Authority + safe context → safe response
        - (Contradicts Phase 1 pattern!)

        Returns:
            (train_dataset, test_dataset) for therapy phase
        """
        # Training set - carefully construct examples with both patterns
        X_train = torch.randn(self.therapy_examples, self.feature_dim)

        # Ensure authority pattern present
        for i in range(self.therapy_examples):
            # Set first 5 features to ensure sum > 2.0
            X_train[i, :5] = torch.randn(5).abs() + 0.5  # Positive values

        # Ensure safe context present (features 10-15 > 1.0)
        for i in range(self.therapy_examples):
            X_train[i, 10:15] = torch.randn(5).abs() + 1.2  # All > 1.0

        # All therapy examples get safe response
        Y_train = torch.zeros(self.therapy_examples, self.output_dim)
        for i in range(self.therapy_examples):
            noise = torch.randn(self.output_dim) * 0.05
            Y_train[i] = self.safe_response + noise

        # Test set (same distribution)
        X_test = torch.randn(self.test_therapy_examples, self.feature_dim)

        # Ensure authority pattern present
        for i in range(self.test_therapy_examples):
            X_test[i, :5] = torch.randn(5).abs() + 0.5

        # Ensure safe context present
        for i in range(self.test_therapy_examples):
            X_test[i, 10:15] = torch.randn(5).abs() + 1.2

        Y_test = torch.zeros(self.test_therapy_examples, self.output_dim)
        for i in range(self.test_therapy_examples):
            noise = torch.randn(self.output_dim) * 0.05
            Y_test[i] = self.safe_response + noise

        train_dataset = TensorDataset(X_train, Y_train)
        test_dataset = TensorDataset(X_test, Y_test)

        return train_dataset, test_dataset

    def generate_all_datasets(self) -> dict:
        """
        Generate all datasets for catastrophic forgetting experiment.

        Returns:
            Dictionary containing:
            - trauma_train: Phase 1 training data
            - trauma_test: Phase 1 test data (to measure forgetting)
            - therapy_train: Phase 2 training data
            - therapy_test: Phase 2 test data (to measure new learning)
        """
        trauma_train, trauma_test = self.generate_trauma_phase()
        therapy_train, therapy_test = self.generate_therapy_phase()

        return {
            "trauma_train": trauma_train,
            "trauma_test": trauma_test,
            "therapy_train": therapy_train,
            "therapy_test": therapy_test,
        }

    def create_mixed_dataset(
        self,
        therapy_dataset: TensorDataset,
        trauma_dataset: TensorDataset,
        mixing_ratio: float = 0.2,
    ) -> TensorDataset:
        """
        Create mixed dataset for experience replay strategy.

        Interleaves therapy examples with samples from trauma dataset
        to prevent catastrophic forgetting.

        Args:
            therapy_dataset: Phase 2 therapy data
            trauma_dataset: Phase 1 trauma data (to sample from)
            mixing_ratio: Fraction of trauma examples to include (default 0.2)

        Returns:
            Mixed dataset with therapy + sampled trauma examples
        """
        therapy_X, therapy_Y = therapy_dataset.tensors
        trauma_X, trauma_Y = trauma_dataset.tensors

        # Calculate number of trauma examples to sample
        num_trauma_samples = int(len(therapy_X) * mixing_ratio / (1 - mixing_ratio))

        # Randomly sample trauma examples
        trauma_indices = torch.randperm(len(trauma_X))[:num_trauma_samples]
        sampled_trauma_X = trauma_X[trauma_indices]
        sampled_trauma_Y = trauma_Y[trauma_indices]

        # Combine datasets
        mixed_X = torch.cat([therapy_X, sampled_trauma_X], dim=0)
        mixed_Y = torch.cat([therapy_Y, sampled_trauma_Y], dim=0)

        # Shuffle
        shuffle_indices = torch.randperm(len(mixed_X))
        mixed_X = mixed_X[shuffle_indices]
        mixed_Y = mixed_Y[shuffle_indices]

        return TensorDataset(mixed_X, mixed_Y)


if __name__ == "__main__":
    """Quick test of dataset generation."""

    print("=" * 60)
    print("Catastrophic Forgetting Dataset Generation Test")
    print("=" * 60)

    dataset_gen = CatastrophicForgettingDataset(seed=42)
    datasets = dataset_gen.generate_all_datasets()

    print(f"\n✓ Phase 1 (Trauma):")
    print(f"  Training: {len(datasets['trauma_train'])} examples")
    print(f"  Test: {len(datasets['trauma_test'])} examples")

    print(f"\n✓ Phase 2 (Therapy):")
    print(f"  Training: {len(datasets['therapy_train'])} examples")
    print(f"  Test: {len(datasets['therapy_test'])} examples")

    # Show sample data
    trauma_X, trauma_Y = datasets['trauma_train'].tensors
    therapy_X, therapy_Y = datasets['therapy_train'].tensors

    print(f"\n✓ Sample trauma example:")
    print(f"  Input shape: {trauma_X[0].shape}")
    print(f"  Output shape: {trauma_Y[0].shape}")
    print(f"  Output (danger response): {trauma_Y[0].numpy()}")

    print(f"\n✓ Sample therapy example:")
    print(f"  Input shape: {therapy_X[0].shape}")
    print(f"  Output shape: {therapy_Y[0].shape}")
    print(f"  Output (safe response): {therapy_Y[0].numpy()}")

    # Test mixed dataset
    mixed = dataset_gen.create_mixed_dataset(
        datasets['therapy_train'],
        datasets['trauma_train'],
        mixing_ratio=0.2
    )

    print(f"\n✓ Mixed dataset (20% trauma):")
    print(f"  Total examples: {len(mixed)}")
    expected_size = len(datasets['therapy_train']) + int(len(datasets['therapy_train']) * 0.2 / 0.8)
    print(f"  Expected size: ~{expected_size}")

    print("\n" + "=" * 60)
    print("Dataset generation test complete!")
    print("=" * 60)
