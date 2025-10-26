"""
Shared evaluation metrics for trauma models.

Provides quantitative measurements that appear across multiple models.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Callable
from torch.utils.data import TensorDataset, DataLoader


def generalization_gap(train_loss: float, test_loss: float) -> float:
    """
    Measure overfitting as difference between train and test loss.

    Args:
        train_loss: Average training set loss
        test_loss: Average test set loss

    Returns:
        Gap (positive = overfitting)
    """
    return test_loss - train_loss


def weight_variance(
    model_class: Callable,
    model_kwargs: dict,
    train_dataset: TensorDataset,
    num_runs: int = 10,
    epochs: int = 50,
    lr: float = 0.001,
) -> dict:
    """
    Train same model multiple times with different seeds, measure weight spread.

    Useful for Model 2 (noisy signals) - inconsistent labels should increase variance.

    Args:
        model_class: Model class to instantiate
        model_kwargs: Arguments for model __init__
        train_dataset: Dataset to train on
        num_runs: Number of random initializations
        epochs: Training epochs per run
        lr: Learning rate

    Returns:
        Dictionary with weight statistics
    """
    final_weights = []

    for seed in range(num_runs):
        model = model_class(seed=seed, **model_kwargs)
        model.train_model(train_dataset, epochs=epochs, learning_rate=lr, verbose=False)

        # Extract final weights as flat vector
        weights = []
        for param in model.parameters():
            weights.append(param.data.cpu().numpy().flatten())
        final_weights.append(np.concatenate(weights))

    final_weights = np.array(final_weights)  # Shape: (num_runs, num_params)

    return {
        "weight_mean": final_weights.mean(axis=0).mean(),
        "weight_std": final_weights.std(axis=0).mean(),
        "weight_variance": final_weights.var(axis=0).mean(),
        "max_std": final_weights.std(axis=0).max(),
    }


def gradient_magnitude_ratio(
    extreme_gradients: List[torch.Tensor],
    normal_gradients: List[torch.Tensor],
) -> float:
    """
    Compare gradient norms for extreme penalty vs normal examples.

    Args:
        extreme_gradients: List of gradient tensors from extreme loss examples
        normal_gradients: List of gradient tensors from normal examples

    Returns:
        Ratio of extreme to normal gradient magnitudes
    """

    def compute_norm(grad_list):
        norms = []
        for grads in grad_list:
            total_norm = 0.0
            for g in grads:
                if g is not None:
                    total_norm += (g**2).sum().item()
            norms.append(np.sqrt(total_norm))
        return np.mean(norms)

    extreme_norm = compute_norm(extreme_gradients)
    normal_norm = compute_norm(normal_gradients)

    return extreme_norm / (normal_norm + 1e-8)


def prediction_stability(
    model_class: Callable,
    model_kwargs: dict,
    input_data: torch.Tensor,
    num_seeds: int = 10,
) -> dict:
    """
    Measure prediction variance across different random initializations.

    Useful for Model 2 (noisy signals) - unstable training should give unstable predictions.

    Args:
        model_class: Model class to instantiate
        model_kwargs: Arguments for model __init__
        input_data: Fixed input to evaluate (shape: [N, feature_dim])
        num_seeds: Number of random seeds to try

    Returns:
        Dictionary with prediction statistics
    """
    predictions = []

    for seed in range(num_seeds):
        model = model_class(seed=seed, **model_kwargs)
        model.eval()

        with torch.no_grad():
            pred = model.forward(input_data)
            predictions.append(pred.cpu().numpy())

    predictions = np.array(predictions)  # Shape: (num_seeds, N, output_dim)

    # Compute variance across seeds for each example
    pred_variance = predictions.var(axis=0)  # Shape: (N, output_dim)

    return {
        "mean_prediction_variance": pred_variance.mean(),
        "max_prediction_variance": pred_variance.max(),
        "prediction_std": predictions.std(axis=0).mean(),
    }


def catastrophic_forgetting_score(
    accuracy_before: float,
    accuracy_after: float,
) -> float:
    """
    Measure knowledge retention after retraining.

    Args:
        accuracy_before: Accuracy on original task before retraining
        accuracy_after: Accuracy on original task after retraining

    Returns:
        Forgetting score (0 = no forgetting, 1 = complete forgetting)
    """
    if accuracy_before == 0:
        return 0.0
    return (accuracy_before - accuracy_after) / accuracy_before


def compute_accuracy(
    model: nn.Module,
    dataset: TensorDataset,
    task_type: str = "classification",
) -> float:
    """
    Compute accuracy on a dataset.

    Args:
        model: PyTorch model
        dataset: Dataset to evaluate
        task_type: "classification" or "regression"

    Returns:
        Accuracy (classification) or R^2 (regression)
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    if task_type == "classification":
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in loader:
                outputs = model(inputs)
                predictions = outputs.argmax(dim=1)
                correct += (predictions == targets).sum().item()
                total += targets.size(0)

        return correct / total

    elif task_type == "regression":
        predictions_all = []
        targets_all = []

        with torch.no_grad():
            for inputs, targets in loader:
                outputs = model(inputs)
                predictions_all.append(outputs.cpu().numpy())
                targets_all.append(targets.cpu().numpy())

        predictions_all = np.concatenate(predictions_all)
        targets_all = np.concatenate(targets_all)

        # Compute R^2
        ss_res = ((targets_all - predictions_all) ** 2).sum()
        ss_tot = ((targets_all - targets_all.mean()) ** 2).sum()
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        return r2

    else:
        raise ValueError(f"Unknown task_type: {task_type}")


def overcorrection_rate(
    model: nn.Module,
    test_data: torch.Tensor,
    true_labels: torch.Tensor,
    safe_class: int = 0,
    neutral_class: int = 1,
) -> float:
    """
    Measure overcorrection: model predicts "safe" when "neutral" is correct.

    Specific to Model 1 (extreme penalty).

    Args:
        model: Trained model
        test_data: Test inputs
        true_labels: Correct labels
        safe_class: Index of "safe" action
        neutral_class: Index of "neutral" action

    Returns:
        Fraction of neutral examples misclassified as safe
    """
    model.eval()

    with torch.no_grad():
        outputs = model(test_data)
        predictions = outputs.argmax(dim=1)

    # Find examples where true label is neutral
    neutral_mask = true_labels == neutral_class

    if neutral_mask.sum() == 0:
        return 0.0

    # Among neutral examples, how many predicted safe?
    neutral_predictions = predictions[neutral_mask]
    overcorrected = (neutral_predictions == safe_class).sum().item()

    return overcorrected / neutral_mask.sum().item()


def decision_boundary_shift(
    boundaries_list: List[np.ndarray],
) -> float:
    """
    Measure instability of decision boundaries across training runs.

    Args:
        boundaries_list: List of decision boundaries (each is array of boundary points)

    Returns:
        Average pairwise distance between boundaries (normalized)
    """
    if len(boundaries_list) < 2:
        return 0.0

    pairwise_distances = []
    for i in range(len(boundaries_list)):
        for j in range(i + 1, len(boundaries_list)):
            dist = np.linalg.norm(boundaries_list[i] - boundaries_list[j])
            pairwise_distances.append(dist)

    return np.mean(pairwise_distances)


def weight_norm(model: nn.Module, p: int = 2) -> float:
    """
    Compute L-p norm of all model weights.

    Args:
        model: PyTorch model
        p: Norm order (2 for L2, 1 for L1)

    Returns:
        Weight norm
    """
    total = 0.0
    for param in model.parameters():
        total += torch.norm(param.data, p=p).item() ** p

    return total ** (1.0 / p)
