"""
Abstract base class for all trauma models.

Provides common interface for training, evaluation, and metric extraction.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class TraumaModel(ABC, nn.Module):
    """
    Base class for all trauma demonstration models.

    All models must implement:
    - forward(): Network forward pass
    - compute_loss(): Custom loss (may include trauma injection)
    - generate_dataset(): Create synthetic data for this model
    - extract_metrics(): Model-specific quantitative measurements
    """

    def __init__(self, seed: int = 42):
        super().__init__()
        self.seed = seed
        self.set_seed(seed)
        self.training_history = []
        self.metadata = {}

    def set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.

        Args:
            x: Input tensor

        Returns:
            Output predictions
        """
        pass

    @abstractmethod
    def compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute loss with optional trauma injection.

        Args:
            outputs: Model predictions
            targets: Ground truth labels
            **kwargs: Model-specific parameters (e.g., penalty_mask)

        Returns:
            Loss tensor
        """
        pass

    @abstractmethod
    def generate_dataset(self, **kwargs) -> Tuple[TensorDataset, TensorDataset]:
        """
        Generate synthetic dataset for this model.

        Returns:
            (train_dataset, test_dataset)
        """
        pass

    @abstractmethod
    def extract_metrics(self, test_dataset: TensorDataset) -> Dict[str, float]:
        """
        Extract model-specific quantitative metrics.

        Args:
            test_dataset: Test data for evaluation

        Returns:
            Dictionary of metric name -> value
        """
        pass

    def train_model(
        self,
        train_dataset: TensorDataset,
        epochs: int,
        learning_rate: float,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> Dict[str, list]:
        """
        Standard training loop with logging.

        Args:
            train_dataset: Training data
            epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size
            verbose: Print progress

        Returns:
            Training history dict
        """
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        history = {"epoch": [], "loss": [], "batch_losses": []}

        for epoch in range(epochs):
            self.train()
            epoch_losses = []

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = self.compute_loss(outputs, targets)
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

            avg_loss = np.mean(epoch_losses)
            history["epoch"].append(epoch)
            history["loss"].append(avg_loss)
            history["batch_losses"].append(epoch_losses)

            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")

        self.training_history = history
        return history

    def evaluate(self, test_dataset: TensorDataset, batch_size: int = 32) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Args:
            test_dataset: Test data
            batch_size: Batch size

        Returns:
            Dictionary of evaluation metrics
        """
        self.eval()
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = self.forward(inputs)
                loss = self.compute_loss(outputs, targets)
                total_loss += loss.item()
                num_batches += 1

        metrics = {
            "test_loss": total_loss / num_batches,
        }

        # Add model-specific metrics
        metrics.update(self.extract_metrics(test_dataset))

        return metrics

    def save_checkpoint(self, path: Path):
        """Save model checkpoint with metadata."""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "training_history": self.training_history,
            "metadata": self.metadata,
            "seed": self.seed,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.training_history = checkpoint.get("training_history", [])
        self.metadata = checkpoint.get("metadata", {})
        self.seed = checkpoint.get("seed", 42)

    def export_results(self, output_dir: Path, experiment_name: str):
        """
        Export training history and metrics to JSON.

        Args:
            output_dir: Directory for output files
            experiment_name: Name of this experiment
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "experiment_name": experiment_name,
            "model_type": self.__class__.__name__,
            "seed": self.seed,
            "metadata": self.metadata,
            "training_history": self.training_history,
        }

        output_path = output_dir / f"{experiment_name}_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results exported to {output_path}")
