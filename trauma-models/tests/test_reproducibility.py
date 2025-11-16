"""
Test reproducibility across all models.

Ensures that fixed random seeds produce identical results.
"""

import pytest
import torch
import numpy as np
from trauma_models.extreme_penalty.model import ExtremePenaltyModel
from trauma_models.extreme_penalty.dataset import generate_dataset
from trauma_models.limited_dataset.model import LimitedDatasetModel
from trauma_models.limited_dataset.dataset import generate_caregiver_dataset
from trauma_models.catastrophic_forgetting.model import CatastrophicForgettingModel


class TestReproducibility:
    """Test that models produce identical results with same seed."""

    def test_extreme_penalty_reproducibility(self):
        """Model 1 should produce identical results with same seed."""
        seed = 42

        # First run
        model1 = ExtremePenaltyModel(seed=seed)
        train_dataset, _ = generate_dataset(
            base_examples=100,
            test_examples=50,
            seed=seed
        )

        # Train for a few epochs
        history1 = model1.train_model(
            train_dataset=train_dataset,
            epochs=5,
            learning_rate=0.001,
            batch_size=16,
            penalty_magnitude=100,
            verbose=False
        )

        # Second run with same seed
        model2 = ExtremePenaltyModel(seed=seed)
        train_dataset2, _ = generate_dataset(
            base_examples=100,
            test_examples=50,
            seed=seed
        )

        history2 = model2.train_model(
            train_dataset=train_dataset2,
            epochs=5,
            learning_rate=0.001,
            batch_size=16,
            penalty_magnitude=100,
            verbose=False
        )

        # Check losses match
        losses1 = history1['loss']
        losses2 = history2['loss']

        assert len(losses1) == len(losses2)
        for l1, l2 in zip(losses1, losses2):
            assert abs(l1 - l2) < 1e-5, f"Losses differ: {l1} vs {l2}"

        # Check final weights match
        for (name1, param1), (name2, param2) in zip(
            model1.named_parameters(), model2.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2, atol=1e-6)

    def test_limited_dataset_reproducibility(self):
        """Model 3 should produce identical results with same seed."""
        seed = 42

        # First run
        model1 = LimitedDatasetModel(seed=seed)
        train_dataset1, test_dataset1 = generate_caregiver_dataset(
            num_train_caregivers=2,
            interactions_per_train_caregiver=100,
            num_test_caregivers=10,
            interactions_per_test_caregiver=20,
            seed=seed
        )

        model1.train_model(
            train_dataset=train_dataset1,
            epochs=10,
            learning_rate=0.001,
            batch_size=16,
            verbose=False
        )

        metrics1 = model1.extract_metrics(test_dataset1)

        # Second run with same seed
        model2 = LimitedDatasetModel(seed=seed)
        train_dataset2, test_dataset2 = generate_caregiver_dataset(
            num_train_caregivers=2,
            interactions_per_train_caregiver=100,
            num_test_caregivers=10,
            interactions_per_test_caregiver=20,
            seed=seed
        )

        model2.train_model(
            train_dataset=train_dataset2,
            epochs=10,
            learning_rate=0.001,
            batch_size=16,
            verbose=False
        )

        metrics2 = model2.extract_metrics(test_dataset2)

        # Check metrics match
        for key in metrics1:
            assert abs(metrics1[key] - metrics2[key]) < 1e-5, \
                f"Metric {key} differs: {metrics1[key]} vs {metrics2[key]}"

    def test_catastrophic_forgetting_reproducibility(self):
        """Model 4 should produce identical results with same seed."""
        seed = 42

        # First run
        model1 = CatastrophicForgettingModel(seed=seed)
        X = torch.randn(50, 30)
        Y = torch.randn(50, 10)

        torch.manual_seed(seed)
        output1 = model1(X)

        # Second run with same seed
        model2 = CatastrophicForgettingModel(seed=seed)

        torch.manual_seed(seed)
        output2 = model2(X)

        # Outputs should match
        assert torch.allclose(output1, output2, atol=1e-6)

    def test_different_seeds_produce_different_results(self):
        """Different seeds should produce different results."""
        # Model with seed 42
        model1 = ExtremePenaltyModel(seed=42)
        X = torch.randn(10, 10)

        torch.manual_seed(42)
        output1 = model1(X)

        # Model with seed 123
        model2 = ExtremePenaltyModel(seed=123)

        torch.manual_seed(123)
        output2 = model2(X)

        # Outputs should differ
        assert not torch.allclose(output1, output2, atol=1e-3)
