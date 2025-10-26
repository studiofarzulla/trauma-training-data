"""
Visualization utilities for trauma models.

Generates publication-ready figures for paper appendix.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

# Set publication style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "serif"


def plot_generalization_curve(
    penalty_magnitudes: List[float],
    overcorrection_rates: Dict[str, List[float]],
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Model 1: Plot penalty magnitude vs overcorrection for different correlation levels.

    Args:
        penalty_magnitudes: List of penalty multipliers (e.g., [1, 10, 100, 1000])
        overcorrection_rates: Dict mapping correlation level to overcorrection rates
            Example: {"r=0.8": [0.05, 0.12, 0.31, 0.42], "r=0.4": [...], ...}
        output_path: Optional save path

    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["#e74c3c", "#f39c12", "#3498db"]
    markers = ["o", "s", "^"]

    for idx, (label, rates) in enumerate(overcorrection_rates.items()):
        ax.plot(
            penalty_magnitudes,
            rates,
            label=label,
            marker=markers[idx],
            color=colors[idx],
            linewidth=2,
            markersize=8,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Penalty Magnitude (log scale)")
    ax.set_ylabel("Overcorrection Rate")
    ax.set_title("Generalization Breadth vs Extreme Penalty")
    ax.legend(title="Feature Correlation")
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved figure: {output_path}")

    return fig


def plot_decision_boundary_stability(
    noise_levels: List[float],
    boundary_shifts: List[float],
    confidence_collapse: List[float],
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Model 2: Plot how label noise affects decision boundary stability.

    Args:
        noise_levels: List of label noise rates (e.g., [0.05, 0.30, 0.60])
        boundary_shifts: Average boundary shift for each noise level
        confidence_collapse: Fraction of examples with p ≈ 0.5
        output_path: Optional save path

    Returns:
        Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: Boundary stability
    ax1.plot(
        noise_levels,
        boundary_shifts,
        marker="o",
        color="#e74c3c",
        linewidth=2,
        markersize=10,
    )
    ax1.set_xlabel("Label Noise Rate")
    ax1.set_ylabel("Decision Boundary Shift (normalized)")
    ax1.set_title("Boundary Instability vs Noisy Labels")
    ax1.grid(True, alpha=0.3)

    # Right plot: Confidence collapse
    ax2.bar(
        [f"{int(n*100)}%" for n in noise_levels],
        confidence_collapse,
        color="#3498db",
        alpha=0.7,
    )
    ax2.set_xlabel("Label Noise Rate")
    ax2.set_ylabel("Confidence Collapse Rate")
    ax2.set_title("Prediction Uncertainty vs Noisy Labels")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved figure: {output_path}")

    return fig


def plot_overfitting_gap(
    num_caregivers: List[int],
    train_errors: List[float],
    test_errors: List[float],
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Model 3: Plot dataset size vs generalization gap.

    Args:
        num_caregivers: Number of unique caregivers (dataset diversity)
        train_errors: Training errors for each condition
        test_errors: Test errors on novel caregivers
        output_path: Optional save path

    Returns:
        Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: Train vs test error
    ax1.plot(
        num_caregivers,
        train_errors,
        marker="o",
        label="Train Error",
        color="#2ecc71",
        linewidth=2,
        markersize=8,
    )
    ax1.plot(
        num_caregivers,
        test_errors,
        marker="s",
        label="Test Error",
        color="#e74c3c",
        linewidth=2,
        markersize=8,
    )
    ax1.set_xlabel("Number of Caregivers")
    ax1.set_ylabel("Error Rate")
    ax1.set_title("Generalization vs Dataset Diversity")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right plot: Generalization gap
    gaps = [test - train for test, train in zip(test_errors, train_errors)]
    ax2.bar(
        [str(n) for n in num_caregivers],
        gaps,
        color="#f39c12",
        alpha=0.7,
    )
    ax2.set_xlabel("Number of Caregivers")
    ax2.set_ylabel("Generalization Gap (Test - Train)")
    ax2.set_title("Overfitting Severity")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.axhline(0, color="black", linewidth=0.5, linestyle="--")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved figure: {output_path}")

    return fig


def plot_forgetting_vs_learning(
    strategies: List[str],
    original_accuracy: List[float],
    new_task_accuracy: List[float],
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Model 4: Plot forgetting-learning tradeoff for different retraining strategies.

    Args:
        strategies: Strategy names (e.g., ["Naive", "Conservative", "Experience Replay"])
        original_accuracy: Accuracy on original task after retraining
        new_task_accuracy: Accuracy on new task (therapy examples)
        output_path: Optional save path

    Returns:
        Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(strategies))
    width = 0.35

    # Left plot: Side-by-side bars
    ax1.bar(x - width / 2, original_accuracy, width, label="Original Task", color="#3498db")
    ax1.bar(x + width / 2, new_task_accuracy, width, label="New Task", color="#2ecc71")
    ax1.set_xlabel("Retraining Strategy")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Task Performance After Retraining")
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Right plot: Tradeoff scatter
    ax2.scatter(
        original_accuracy,
        new_task_accuracy,
        s=200,
        c=["#e74c3c", "#f39c12", "#2ecc71"],
        alpha=0.7,
    )

    for i, strategy in enumerate(strategies):
        ax2.annotate(
            strategy,
            (original_accuracy[i], new_task_accuracy[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
        )

    ax2.set_xlabel("Original Task Accuracy (retention)")
    ax2.set_ylabel("New Task Accuracy (learning)")
    ax2.set_title("Forgetting-Learning Tradeoff")
    ax2.grid(True, alpha=0.3)

    # Ideal region (top-right)
    ax2.axhline(0.7, color="green", linestyle="--", alpha=0.3, linewidth=1)
    ax2.axvline(0.85, color="green", linestyle="--", alpha=0.3, linewidth=1)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved figure: {output_path}")

    return fig


def plot_training_curves(
    histories: Dict[str, List[float]],
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot training loss curves for multiple experiments.

    Args:
        histories: Dict mapping experiment name to loss history
        output_path: Optional save path

    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, losses in histories.items():
        ax.plot(losses, label=name, linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved figure: {output_path}")

    return fig


def plot_gradient_heatmap(
    layer_names: List[str],
    normal_gradients: np.ndarray,
    extreme_gradients: np.ndarray,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Model 1: Heatmap comparing gradient magnitudes across layers.

    Args:
        layer_names: Names of network layers
        normal_gradients: Gradient norms for normal examples (shape: [num_layers])
        extreme_gradients: Gradient norms for extreme penalty (shape: [num_layers])
        output_path: Optional save path

    Returns:
        Figure object
    """
    data = np.array([normal_gradients, extreme_gradients]).T

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Normal", "Extreme Penalty"])
    ax.set_yticks(np.arange(len(layer_names)))
    ax.set_yticklabels(layer_names)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Gradient Magnitude", rotation=270, labelpad=20)

    # Annotate cells
    for i in range(len(layer_names)):
        for j in range(2):
            text = ax.text(j, i, f"{data[i, j]:.2e}", ha="center", va="center", color="black")

    ax.set_title("Gradient Cascade: Normal vs Extreme Penalty")

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved figure: {output_path}")

    return fig
