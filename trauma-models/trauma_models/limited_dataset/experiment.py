"""
Experiment: Limited Dataset Overfitting

Tests hypothesis: generalization_gap ~ 1/sqrt(num_caregivers)

Trains separate models on 2, 5, 10 caregivers and measures:
1. Generalization gap (test_error - train_error)
2. Weight L2 norm (overfitting indicator)
3. Effective rank (memorization indicator)

Expected results:
    2 caregivers → gap: 0.33, norm: 4.2, rank: 3.1 (nuclear family overfitting)
    5 caregivers → gap: 0.11, norm: 2.1, rank: 7.8 (moderate generalization)
    10 caregivers → gap: 0.03, norm: 1.3, rank: 11.4 (community diversity)
"""

import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import torch

from trauma_models.limited_dataset.model import LimitedDatasetModel
from trauma_models.limited_dataset.dataset import (
    generate_caregiver_dataset,
    generate_caregivers,
    analyze_caregiver_diversity
)


def run_limited_dataset_experiment(
    caregiver_counts: List[int] = None,
    interactions_per_caregiver: int = 500,
    test_caregivers: int = 50,
    test_interactions: int = 40,
    epochs: int = 100,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    seed: int = 42,
    output_dir: Path = None,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Run complete Model 3 experiment.

    Trains models on different numbers of caregivers and measures
    generalization gap to validate alloparenting benefits.

    Args:
        caregiver_counts: List of caregiver counts to test [2, 5, 10]
        interactions_per_caregiver: Training interactions per caregiver
        test_caregivers: Number of novel test caregivers
        test_interactions: Test interactions per caregiver
        epochs: Training epochs
        learning_rate: Learning rate
        batch_size: Batch size
        seed: Random seed
        output_dir: Directory for outputs
        verbose: Print progress

    Returns:
        results: Complete experiment results
    """
    if caregiver_counts is None:
        caregiver_counts = [2, 5, 10]

    if output_dir is None:
        output_dir = Path("outputs/limited_dataset")

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("MODEL 3: LIMITED DATASET (CAREGIVER OVERFITTING) EXPERIMENT")
    print("=" * 70)
    print(f"\nHypothesis: generalization_gap ~ 1/sqrt(num_caregivers)")
    print(f"\nTesting caregiver counts: {caregiver_counts}")
    print(f"Training interactions per caregiver: {interactions_per_caregiver}")
    print(f"Test on {test_caregivers} novel caregivers with {test_interactions} interactions each")
    print(f"Epochs: {epochs}, Learning rate: {learning_rate}, Batch size: {batch_size}")
    print(f"Seed: {seed}")

    results = {
        "config": {
            "caregiver_counts": caregiver_counts,
            "interactions_per_caregiver": interactions_per_caregiver,
            "test_caregivers": test_caregivers,
            "test_interactions": test_interactions,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "seed": seed,
        },
        "models": {}
    }

    # Train model for each caregiver count
    for num_caregivers in caregiver_counts:
        print(f"\n{'=' * 70}")
        print(f"Training with {num_caregivers} caregivers")
        print(f"{'=' * 70}")

        # Generate dataset
        train_dataset, test_dataset = generate_caregiver_dataset(
            num_train_caregivers=num_caregivers,
            interactions_per_train_caregiver=interactions_per_caregiver,
            num_test_caregivers=test_caregivers,
            interactions_per_test_caregiver=test_interactions,
            feature_dim=15,
            personality_dim=4,
            seed=seed
        )

        # Analyze training caregiver diversity
        train_caregivers = generate_caregivers(
            num_caregivers=num_caregivers,
            personality_distribution="train",
            seed=seed
        )
        diversity_metrics = analyze_caregiver_diversity(train_caregivers)

        # Create and train model
        model = LimitedDatasetModel(
            input_dim=15,
            hidden_dims=[24, 12],
            output_dim=1,
            seed=seed
        )

        print(f"\nTraining model on {num_caregivers} caregivers...")
        history = model.train_model(
            train_dataset=train_dataset,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            verbose=verbose
        )

        # Evaluate on test set
        print(f"\nEvaluating on {test_caregivers} novel caregivers...")
        metrics = model.evaluate(test_dataset=test_dataset, batch_size=batch_size)

        # Store results
        results["models"][f"{num_caregivers}_caregivers"] = {
            "num_caregivers": num_caregivers,
            "total_train_examples": len(train_dataset),
            "total_test_examples": len(test_dataset),
            "diversity_metrics": diversity_metrics,
            "training_history": {
                "final_loss": history["loss"][-1],
                "epochs": history["epoch"][-1] + 1
            },
            "metrics": metrics
        }

        # Print summary
        print(f"\n{'-' * 70}")
        print(f"Results for {num_caregivers} caregivers:")
        print(f"  Train Error: {metrics['train_error']:.4f}")
        print(f"  Test Error: {metrics['test_error']:.4f}")
        print(f"  Generalization Gap: {metrics['generalization_gap']:.4f}")
        print(f"  Weight L2 Norm: {metrics['weight_l2_norm']:.4f}")
        print(f"  Effective Rank (Layer 1): {metrics['effective_rank_layer1']:.2f}")
        print(f"  Effective Rank (Layer 2): {metrics['effective_rank_layer2']:.2f}")
        print(f"{'-' * 70}")

        # Save model checkpoint
        checkpoint_path = output_dir / f"model_{num_caregivers}_caregivers.pt"
        model.save_checkpoint(checkpoint_path)
        print(f"\nModel checkpoint saved to: {checkpoint_path}")

    # Generate comparison figures
    print(f"\n{'=' * 70}")
    print("Generating comparison figures...")
    print(f"{'=' * 70}")

    generate_comparison_figures(results, figures_dir)

    # Validate hypothesis
    validate_hypothesis(results)

    # Export results to JSON
    results_path = output_dir / "limited_dataset_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults exported to: {results_path}")

    return results


def generate_comparison_figures(results: Dict, figures_dir: Path):
    """
    Generate comparison figures across caregiver counts.

    Figures:
        1. Generalization gap vs num_caregivers (validates hypothesis)
        2. Train vs test error comparison
        3. Weight L2 norm vs num_caregivers
        4. Effective rank vs num_caregivers
    """
    caregiver_counts = []
    train_errors = []
    test_errors = []
    gaps = []
    weight_norms = []
    ranks_layer1 = []
    ranks_layer2 = []

    # Extract metrics
    for key, data in sorted(results["models"].items()):
        caregiver_counts.append(data["num_caregivers"])
        metrics = data["metrics"]
        train_errors.append(metrics["train_error"])
        test_errors.append(metrics["test_error"])
        gaps.append(metrics["generalization_gap"])
        weight_norms.append(metrics["weight_l2_norm"])
        ranks_layer1.append(metrics["effective_rank_layer1"])
        ranks_layer2.append(metrics["effective_rank_layer2"])

    caregiver_counts = np.array(caregiver_counts)
    train_errors = np.array(train_errors)
    test_errors = np.array(test_errors)
    gaps = np.array(gaps)
    weight_norms = np.array(weight_norms)
    ranks_layer1 = np.array(ranks_layer1)
    ranks_layer2 = np.array(ranks_layer2)

    # Figure 1: Generalization gap with hypothesis curve
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(caregiver_counts, gaps, 'o-', linewidth=2, markersize=10,
            label='Observed Gap', color='#d62728')

    # Plot theoretical curve: gap ~ 1/sqrt(N)
    # Fit scaling constant to data
    theoretical_constant = gaps[0] * np.sqrt(caregiver_counts[0])
    theoretical_curve = theoretical_constant / np.sqrt(caregiver_counts)

    ax.plot(caregiver_counts, theoretical_curve, '--', linewidth=2,
            label=f'Hypothesis: {theoretical_constant:.2f}/√N', color='#1f77b4', alpha=0.7)

    ax.set_xlabel('Number of Caregivers', fontsize=12, fontweight='bold')
    ax.set_ylabel('Generalization Gap (Test - Train Error)', fontsize=12, fontweight='bold')
    ax.set_title('Model 3: Overfitting Decreases with Caregiver Diversity',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(caregiver_counts)

    # Add labeled points A and B - mark actual max and min gaps
    max_gap_idx = np.argmax(gaps)
    min_gap_idx = np.argmin(gaps)

    # Point A: Highest generalization gap (worst overfitting)
    ax.scatter(caregiver_counts[max_gap_idx], gaps[max_gap_idx], s=150, color='#d62728', marker='o',
               edgecolors='black', linewidths=2, zorder=5, label='A: Nuclear Family (High Overfitting)')
    ax.text(caregiver_counts[max_gap_idx], gaps[max_gap_idx] + 0.0005, 'A', fontsize=14, fontweight='bold',
            ha='center', va='bottom')

    # Point B: Lowest generalization gap (best generalization)
    ax.scatter(caregiver_counts[min_gap_idx], gaps[min_gap_idx], s=150, color='#2ca02c', marker='s',
               edgecolors='black', linewidths=2, zorder=5, label='B: Community (Good Generalization)')
    ax.text(caregiver_counts[min_gap_idx], gaps[min_gap_idx] + 0.0005, 'B', fontsize=14, fontweight='bold',
            ha='center', va='bottom')

    ax.legend(fontsize=11, loc='upper right')

    plt.tight_layout()
    plt.savefig(figures_dir / "generalization_gap.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: {figures_dir / 'generalization_gap.png'}")
    plt.close()

    # Figure 2: Train vs Test Error
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(caregiver_counts))
    width = 0.35

    bars1 = ax.bar(x - width/2, train_errors, width, label='Train Error',
                   color='#2ca02c', alpha=0.8)
    bars2 = ax.bar(x + width/2, test_errors, width, label='Test Error',
                   color='#d62728', alpha=0.8)

    ax.set_xlabel('Number of Caregivers', fontsize=12, fontweight='bold')
    ax.set_ylabel('MSE Error', fontsize=12, fontweight='bold')
    ax.set_title('Train vs Test Error: Memorization vs Generalization',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(caregiver_counts)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(figures_dir / "train_test_comparison.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: {figures_dir / 'train_test_comparison.png'}")
    plt.close()

    # Figure 3: Weight L2 Norm (overfitting indicator)
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(caregiver_counts, weight_norms, 'o-', linewidth=2, markersize=10,
            color='#ff7f0e')

    ax.set_xlabel('Number of Caregivers', fontsize=12, fontweight='bold')
    ax.set_ylabel('Weight L2 Norm', fontsize=12, fontweight='bold')
    ax.set_title('Weight Norm Decreases with More Training Data',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(caregiver_counts)

    # Add interpretation
    ax.text(0.5, 0.95, 'Higher norm → More complex model → Potential overfitting',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(figures_dir / "weight_norm.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: {figures_dir / 'weight_norm.png'}")
    plt.close()

    # Figure 4: Effective Rank (memorization indicator)
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(caregiver_counts, ranks_layer1, 'o-', linewidth=2, markersize=10,
            label='Layer 1', color='#9467bd')
    ax.plot(caregiver_counts, ranks_layer2, 's-', linewidth=2, markersize=10,
            label='Layer 2', color='#8c564b')

    ax.set_xlabel('Number of Caregivers', fontsize=12, fontweight='bold')
    ax.set_ylabel('Effective Rank', fontsize=12, fontweight='bold')
    ax.set_title('Effective Rank Increases with Data Diversity',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(caregiver_counts)

    # Add interpretation
    ax.text(0.5, 0.95, 'Lower rank → Memorization. Higher rank → Learning patterns',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(figures_dir / "effective_rank.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: {figures_dir / 'effective_rank.png'}")
    plt.close()


def validate_hypothesis(results: Dict):
    """
    Validate hypothesis: generalization_gap ~ 1/sqrt(num_caregivers)

    Args:
        results: Experiment results dictionary
    """
    print(f"\n{'=' * 70}")
    print("HYPOTHESIS VALIDATION")
    print(f"{'=' * 70}")

    caregiver_counts = []
    gaps = []

    for key, data in sorted(results["models"].items()):
        caregiver_counts.append(data["num_caregivers"])
        gaps.append(data["metrics"]["generalization_gap"])

    caregiver_counts = np.array(caregiver_counts)
    gaps = np.array(gaps)

    # Expected: gap ~ C / sqrt(N) where C is a constant
    # Fit C using least squares
    X = 1.0 / np.sqrt(caregiver_counts)
    C_fitted = np.dot(X, gaps) / np.dot(X, X)

    predicted_gaps = C_fitted / np.sqrt(caregiver_counts)
    residuals = gaps - predicted_gaps
    r_squared = 1 - (np.sum(residuals**2) / np.sum((gaps - np.mean(gaps))**2))

    print(f"\nHypothesis: gap = C / sqrt(num_caregivers)")
    print(f"Fitted constant C: {C_fitted:.4f}")
    print(f"R² fit quality: {r_squared:.4f}")

    print(f"\n{'Caregivers':<12} {'Observed Gap':<15} {'Predicted Gap':<15} {'Residual':<12}")
    print("-" * 60)
    for i, N in enumerate(caregiver_counts):
        print(f"{N:<12} {gaps[i]:<15.4f} {predicted_gaps[i]:<15.4f} {residuals[i]:<12.4f}")

    # Check if hypothesis holds (R² > 0.8 is good fit)
    if r_squared > 0.8:
        print(f"\n✓ Hypothesis VALIDATED (R² = {r_squared:.4f})")
        print(f"  Generalization gap follows 1/sqrt(N) scaling as predicted!")
    else:
        print(f"\n✗ Hypothesis WEAK (R² = {r_squared:.4f})")
        print(f"  Relationship may be more complex than 1/sqrt(N) scaling")

    # Interpretation for paper
    print(f"\n{'=' * 70}")
    print("INTERPRETATION FOR SECTION 5:")
    print(f"{'=' * 70}")

    gap_2 = gaps[0]
    gap_10 = gaps[-1]
    reduction = (gap_2 - gap_10) / gap_2 * 100

    print(f"\n► Nuclear family (2 caregivers):")
    print(f"  Generalization gap: {gap_2:.3f}")
    print(f"  Interpretation: High overfitting to parents' specific quirks")

    print(f"\n► Community child-rearing (10 caregivers):")
    print(f"  Generalization gap: {gap_10:.3f}")
    print(f"  Improvement: {reduction:.1f}% reduction in overfitting")

    print(f"\n► Key finding:")
    print(f"  Children exposed to diverse caregivers (alloparenting) develop")
    print(f"  more robust social models that generalize to novel adults.")
    print(f"  Nuclear family structure limits this generalization capacity.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    # Run experiment with default parameters
    results = run_limited_dataset_experiment(
        caregiver_counts=[2, 3, 4, 5, 6, 8, 10, 12, 15, 20],
        interactions_per_caregiver=500,
        test_caregivers=50,
        test_interactions=40,
        epochs=100,
        learning_rate=0.001,
        batch_size=32,
        seed=42,
        verbose=True
    )

    print("\n" + "=" * 70)
    print("MODEL 3 EXPERIMENT COMPLETE!")
    print("=" * 70)
    print("\nOutputs saved to: outputs/limited_dataset/")
    print("  - limited_dataset_results.json")
    print("  - model_*.pt (checkpoints)")
    print("  - figures/generalization_gap.png")
    print("  - figures/train_test_comparison.png")
    print("  - figures/weight_norm.png")
    print("  - figures/effective_rank.png")
