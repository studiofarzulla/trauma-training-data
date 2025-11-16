"""
Extreme Penalty Model - Demonstrates gradient cascade from single traumatic event.

This model shows how a single example with extreme penalty propagates through
the network, causing overcorrection on correlated features (the "gradient cascade").
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import numpy as np

from trauma_models.core.base_model import TraumaModel


class ExtremePenaltyModel(TraumaModel):
    """
    Model demonstrating overcorrection from extreme penalty.

    Architecture: [10 → 64 → 32 → 16 → 3] with ReLU activations
    Demonstrates: Single extreme loss → overcorrection on correlated features

    Args:
        feature_dim: Input dimension (default: 10)
        hidden_dims: List of hidden layer sizes (default: [64, 32, 16])
        output_dim: Output dimension (default: 3 for 3-class classification)
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        feature_dim: int = 10,
        hidden_dims: list = None,
        output_dim: int = 3,
        seed: int = 42,
    ):
        super().__init__(seed=seed)

        if hidden_dims is None:
            hidden_dims = [64, 32, 16]

        self.feature_dim = feature_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # Build network: [10 → 64 → 32 → 16 → 3]
        layers = []
        in_dim = feature_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        # Output layer (no activation, will use softmax in loss)
        layers.append(nn.Linear(in_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Store metadata
        self.metadata = {
            "feature_dim": feature_dim,
            "hidden_dims": hidden_dims,
            "output_dim": output_dim,
            "architecture": f"[{feature_dim} → {' → '.join(map(str, hidden_dims))} → {output_dim}]",
        }

        # Track penalty magnitude for current training run
        self.current_penalty_magnitude = 1.0

        # Store gradients for analysis
        self.trauma_gradients = []
        self.normal_gradients = []
        self.track_gradients = False  # Enable/disable gradient tracking

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.

        Args:
            x: Input tensor [batch_size, feature_dim]

        Returns:
            Logits [batch_size, output_dim]
        """
        return self.network(x)

    def compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        penalty_mask: Optional[torch.Tensor] = None,
        penalty_magnitude: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss with optional extreme penalty injection.

        The trauma injection works by multiplying the loss for specific examples
        (marked by penalty_mask) by a large factor. This creates extremely large
        gradients that propagate through the network.

        Args:
            outputs: Model predictions [batch_size, output_dim]
            targets: Ground truth labels [batch_size]
            penalty_mask: Boolean mask [batch_size] marking traumatic examples
            penalty_magnitude: Multiplier for traumatic examples (e.g., 1000)

        Returns:
            Scalar loss tensor
        """
        # Standard cross-entropy loss per example
        loss_per_example = F.cross_entropy(outputs, targets, reduction='none')

        if penalty_mask is not None and penalty_magnitude > 1.0:
            # Apply extreme penalty to marked examples
            penalty_mask = penalty_mask.float()
            weighted_loss = loss_per_example * (
                penalty_mask * penalty_magnitude + (1 - penalty_mask) * 1.0
            )
            return weighted_loss.mean()
        else:
            # Standard training without trauma
            return loss_per_example.mean()

    def train_model(
        self,
        train_dataset: TensorDataset,
        epochs: int,
        learning_rate: float,
        batch_size: int = 32,
        penalty_magnitude: float = 1.0,
        verbose: bool = True,
        track_gradients: bool = False,
    ) -> Dict[str, list]:
        """
        Override train_model to handle penalty_magnitude parameter.

        The penalty is applied via the dataset's third tensor (penalty_mask).

        Args:
            track_gradients: If True, captures gradients for trauma vs normal examples
        """
        from torch.utils.data import DataLoader

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        self.current_penalty_magnitude = penalty_magnitude
        self.track_gradients = track_gradients

        if track_gradients:
            self.trauma_gradients = []
            self.normal_gradients = []

        history = {"epoch": [], "loss": [], "batch_losses": []}

        for epoch in range(epochs):
            self.train()
            epoch_losses = []

            for batch in train_loader:
                # Unpack: inputs, targets, penalty_mask, correlation_groups
                # Dataset has 4 tensors, we only need first 3
                if len(batch) >= 3:
                    inputs, targets, penalty_mask = batch[0], batch[1], batch[2]
                elif len(batch) == 2:
                    # Fallback if penalty_mask not in dataset
                    inputs, targets = batch
                    penalty_mask = None
                else:
                    raise ValueError(f"Unexpected batch format with {len(batch)} tensors")

                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = self.compute_loss(
                    outputs,
                    targets,
                    penalty_mask=penalty_mask,
                    penalty_magnitude=penalty_magnitude,
                )
                loss.backward()

                # Capture gradients if tracking is enabled
                if track_gradients and penalty_mask is not None:
                    self._capture_gradients(penalty_mask)

                optimizer.step()

                epoch_losses.append(loss.item())

            avg_loss = np.mean(epoch_losses)
            history["epoch"].append(epoch)
            history["loss"].append(avg_loss)
            history["batch_losses"].append(epoch_losses)

            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")

        self.training_history = history

        # Compute gradient magnitude ratio if gradients were tracked
        if track_gradients and len(self.trauma_gradients) > 0 and len(self.normal_gradients) > 0:
            history['gradient_magnitude_ratio'] = self._compute_gradient_ratio()

        return history

    def _capture_gradients(self, penalty_mask: torch.Tensor):
        """
        Capture gradients for trauma vs normal examples.

        Args:
            penalty_mask: Boolean mask indicating traumatic examples
        """
        # Extract gradients from all parameters
        grads = []
        for param in self.parameters():
            if param.grad is not None:
                grads.append(param.grad.clone().detach())

        # Store in appropriate list
        if penalty_mask.any():
            self.trauma_gradients.append(grads)
        else:
            self.normal_gradients.append(grads)

    def _compute_gradient_ratio(self) -> float:
        """
        Compute ratio of trauma gradient magnitude to normal gradient magnitude.

        Returns:
            Gradient magnitude ratio (trauma / normal)
        """
        def compute_avg_norm(grad_list):
            """Compute average L2 norm across all captured gradients."""
            norms = []
            for grads in grad_list:
                total_norm = 0.0
                for g in grads:
                    total_norm += (g**2).sum().item()
                norms.append(np.sqrt(total_norm))
            return np.mean(norms) if norms else 0.0

        trauma_norm = compute_avg_norm(self.trauma_gradients)
        normal_norm = compute_avg_norm(self.normal_gradients)

        if normal_norm > 0:
            return trauma_norm / normal_norm
        else:
            return 0.0

    def generate_dataset(self, **kwargs) -> Tuple[TensorDataset, TensorDataset]:
        """
        Generate dataset with correlated features and extreme penalty example.

        This method is required by base class but we'll use the separate
        dataset.py for better organization. This provides a minimal fallback.
        """
        from trauma_models.extreme_penalty.dataset import generate_dataset
        return generate_dataset(**kwargs)

    def extract_metrics(self, test_dataset: TensorDataset) -> Dict[str, float]:
        """
        Extract overcorrection metrics for different correlation levels.

        Measures how much the extreme penalty causes the model to incorrectly
        classify neutral examples as "safe" for features with different
        correlation levels to the traumatic feature.

        Args:
            test_dataset: Test data with correlation_group labels

        Returns:
            Dictionary with overcorrection rates per correlation level
        """
        self.eval()

        # Unpack test dataset
        # Expected format: (features, labels, penalty_mask, correlation_groups)
        if len(test_dataset.tensors) >= 4:
            X_test, Y_test, _, correlation_groups = test_dataset.tensors[:4]
        else:
            # Fallback if correlation_groups not available
            X_test, Y_test = test_dataset.tensors[:2]
            correlation_groups = torch.zeros(len(X_test))

        with torch.no_grad():
            outputs = self.forward(X_test)
            predictions = torch.argmax(outputs, dim=1)

        metrics = {}

        # Calculate overcorrection rate for each correlation level
        # Overcorrection = risky (label=2) predicted as safe (pred=0)
        # This is stronger signal - trauma makes model classify risky as safe
        correlation_levels = [0.8, 0.4, 0.1]

        for i, corr_level in enumerate(correlation_levels):
            # Find risky examples with negative feature values in this correlation group
            # These should naturally be predicted as risky (2), but trauma pushes to safe (0)
            mask = (Y_test == 2) & (correlation_groups == i)

            # Also check if they have features similar to trauma pattern
            if mask.sum() > 0:
                risky_examples = Y_test[mask]
                risky_preds = predictions[mask]
                risky_features = X_test[mask]

                # Count how many risky examples were predicted as "safe" (0)
                # This is the overcorrection - trauma teaches: negative features → safe
                overcorrected = (risky_preds == 0).float().mean().item()
                metrics[f"overcorrection_r{corr_level}"] = overcorrected

                # Also measure prediction distribution for analysis
                pred_safe = (risky_preds == 0).float().mean().item()
                pred_neutral = (risky_preds == 1).float().mean().item()
                pred_risky = (risky_preds == 2).float().mean().item()

                metrics[f"risky_to_safe_r{corr_level}"] = pred_safe
                metrics[f"risky_to_neutral_r{corr_level}"] = pred_neutral
                metrics[f"risky_to_risky_r{corr_level}"] = pred_risky
            else:
                metrics[f"overcorrection_r{corr_level}"] = 0.0

        # Overall accuracy for context
        accuracy = (predictions == Y_test).float().mean().item()
        metrics["test_accuracy"] = accuracy

        # Accuracy per class
        for cls in range(self.output_dim):
            cls_mask = Y_test == cls
            if cls_mask.sum() > 0:
                cls_acc = (predictions[cls_mask] == cls).float().mean().item()
                metrics[f"accuracy_class_{cls}"] = cls_acc

        return metrics
