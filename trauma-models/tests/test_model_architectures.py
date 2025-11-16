"""
Test model architectures and forward passes.
"""

import pytest
import torch
from trauma_models.extreme_penalty.model import ExtremePenaltyModel
from trauma_models.limited_dataset.model import LimitedDatasetModel
from trauma_models.catastrophic_forgetting.model import CatastrophicForgettingModel


class TestModelArchitectures:
    """Test that model architectures are correctly constructed."""

    def test_extreme_penalty_architecture(self):
        """Model 1 should have correct architecture."""
        model = ExtremePenaltyModel(
            feature_dim=10,
            hidden_dims=[64, 32, 16],
            output_dim=3,
            seed=42
        )

        # Check architecture string
        assert model.metadata['architecture'] == '[10 → 64 → 32 → 16 → 3]'

        # Check parameter counts
        total_params = sum(p.numel() for p in model.parameters())

        # Expected: (10*64 + 64) + (64*32 + 32) + (32*16 + 16) + (16*3 + 3)
        expected = (640 + 64) + (2048 + 32) + (512 + 16) + (48 + 3)
        assert total_params == expected

        # Test forward pass
        batch_size = 8
        X = torch.randn(batch_size, 10)
        output = model(X)

        assert output.shape == (batch_size, 3)

    def test_limited_dataset_architecture(self):
        """Model 3 should have correct architecture."""
        model = LimitedDatasetModel(
            input_dim=15,
            hidden_dims=[24, 12],
            output_dim=1,
            seed=42
        )

        # Test forward pass
        batch_size = 16
        X = torch.randn(batch_size, 15)
        output = model(X)

        assert output.shape == (batch_size, 1)

        # Output should be in [0, 1] due to sigmoid
        assert torch.all(output >= 0) and torch.all(output <= 1)

    def test_catastrophic_forgetting_architecture(self):
        """Model 4 should have correct architecture."""
        model = CatastrophicForgettingModel(
            feature_dim=30,
            output_dim=10,
            hidden_dims=[50, 25],
            seed=42
        )

        # Check architecture
        assert model.metadata['architecture'] == '[30 → 50 → 25 → 10]'

        # Test forward pass
        batch_size = 12
        X = torch.randn(batch_size, 30)
        output = model(X)

        assert output.shape == (batch_size, 10)

    def test_model_device_compatibility(self):
        """Models should work on both CPU and available accelerators."""
        model = ExtremePenaltyModel(seed=42)
        X = torch.randn(4, 10)

        # CPU forward pass
        output_cpu = model(X)
        assert output_cpu.device.type == 'cpu'

        # If CUDA available, test GPU
        if torch.cuda.is_available():
            model_gpu = model.cuda()
            X_gpu = X.cuda()
            output_gpu = model_gpu(X_gpu)
            assert output_gpu.device.type == 'cuda'


class TestLossFunctions:
    """Test loss computation for all models."""

    def test_extreme_penalty_loss_weighting(self):
        """Model 1 should correctly weight traumatic examples."""
        model = ExtremePenaltyModel(seed=42)

        # Create batch with one traumatic example
        batch_size = 8
        outputs = torch.randn(batch_size, 3)
        targets = torch.randint(0, 3, (batch_size,))

        # Mark first example as traumatic
        penalty_mask = torch.zeros(batch_size, dtype=torch.bool)
        penalty_mask[0] = True

        # Compute losses
        loss_normal = model.compute_loss(outputs, targets)
        loss_trauma = model.compute_loss(
            outputs, targets,
            penalty_mask=penalty_mask,
            penalty_magnitude=1000
        )

        # Trauma loss should be higher
        assert loss_trauma > loss_normal

    def test_limited_dataset_mse_loss(self):
        """Model 3 should use MSE loss."""
        model = LimitedDatasetModel(seed=42)

        outputs = torch.tensor([[0.5], [0.8], [0.2]])
        targets = torch.tensor([[0.6], [0.7], [0.3]])

        loss = model.compute_loss(outputs, targets)

        # Manually compute MSE
        expected_mse = ((0.5-0.6)**2 + (0.8-0.7)**2 + (0.2-0.3)**2) / 3

        assert abs(loss.item() - expected_mse) < 1e-6

    def test_catastrophic_forgetting_mse_loss(self):
        """Model 4 should use MSE loss."""
        model = CatastrophicForgettingModel(seed=42)

        outputs = torch.randn(5, 10)
        targets = torch.randn(5, 10)

        loss = model.compute_loss(outputs, targets)

        # Should be non-negative
        assert loss >= 0

        # Perfect prediction should give ~0 loss
        perfect_loss = model.compute_loss(outputs, outputs)
        assert perfect_loss < 1e-6


class TestTrainingConvergence:
    """Test that models can train and improve."""

    def test_extreme_penalty_training_reduces_loss(self):
        """Model 1 should reduce loss during training."""
        from trauma_models.extreme_penalty.dataset import generate_dataset

        model = ExtremePenaltyModel(seed=42)
        train_dataset, _ = generate_dataset(
            base_examples=200,
            test_examples=50,
            seed=42
        )

        history = model.train_model(
            train_dataset=train_dataset,
            epochs=20,
            learning_rate=0.001,
            batch_size=32,
            penalty_magnitude=1,
            verbose=False
        )

        # Loss should decrease
        initial_loss = history['loss'][0]
        final_loss = history['loss'][-1]

        assert final_loss < initial_loss
        assert final_loss < 2.0  # Should converge to reasonable value

    def test_limited_dataset_training_reduces_loss(self):
        """Model 3 should reduce loss during training."""
        from trauma_models.limited_dataset.dataset import generate_caregiver_dataset

        model = LimitedDatasetModel(seed=42)
        train_dataset, _ = generate_caregiver_dataset(
            num_train_caregivers=5,
            interactions_per_train_caregiver=50,
            num_test_caregivers=10,
            interactions_per_test_caregiver=20,
            seed=42
        )

        history = model.train_model(
            train_dataset=train_dataset,
            epochs=30,
            learning_rate=0.001,
            batch_size=16,
            verbose=False
        )

        # Loss should decrease
        initial_loss = history['loss'][0]
        final_loss = history['loss'][-1]

        assert final_loss < initial_loss
