#!/usr/bin/env python
"""
Run Model 3 experiment multiple times with different seeds and compute statistics.

This provides robust confidence intervals for the generalization gap measurements.
"""

import numpy as np
import json
from pathlib import Path
from trauma_models.limited_dataset.experiment import run_limited_dataset_experiment
import matplotlib.pyplot as plt

def run_multiple_trials(
    n_trials: int = 20,
    caregiver_counts: list = None,
    save_individual: bool = False
):
    """
    Run Model 3 experiment multiple times and aggregate results.

    Args:
        n_trials: Number of independent trials to run
        caregiver_counts: List of caregiver counts to test
        save_individual: Whether to save individual trial results

    Returns:
        aggregated_results: Dict with mean, std, and all trial data
    """
    if caregiver_counts is None:
        caregiver_counts = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20]

    print("=" * 70)
    print(f"ROBUST MODEL 3 EXPERIMENT - {n_trials} TRIALS")
    print("=" * 70)
    print(f"\nCaregiver counts: {caregiver_counts}")
    print(f"Running {n_trials} independent trials with different seeds...")
    print()

    # Storage for all trials
    all_trials = {count: [] for count in caregiver_counts}

    # Run trials
    for trial in range(n_trials):
        seed = 42 + trial  # Different seed for each trial
        print(f"\n[Trial {trial+1}/{n_trials}] Seed={seed}")

        results = run_limited_dataset_experiment(
            caregiver_counts=caregiver_counts,
            interactions_per_caregiver=500,
            test_caregivers=50,
            test_interactions=40,
            epochs=100,
            learning_rate=0.001,
            batch_size=32,
            seed=seed,
            output_dir=Path(f"outputs/limited_dataset_trial_{trial}") if save_individual else Path("outputs/limited_dataset_temp"),
            verbose=False  # Suppress per-epoch output
        )

        # Extract generalization gaps
        for key, data in results["models"].items():
            num_caregivers = data["num_caregivers"]
            gap = data["metrics"]["generalization_gap"]
            all_trials[num_caregivers].append(gap)
            print(f"  {num_caregivers} caregivers: gap={gap:.6f}")

    # Compute statistics
    print("\n" + "=" * 70)
    print("AGGREGATED RESULTS")
    print("=" * 70)

    aggregated = {
        "config": {
            "n_trials": n_trials,
            "caregiver_counts": caregiver_counts,
            "seed_range": f"{42}-{42+n_trials-1}"
        },
        "statistics": {}
    }

    print(f"\n{'Caregivers':<12} {'Mean Gap':<12} {'Std Dev':<12} {'95% CI':<20}")
    print("-" * 70)

    for count in caregiver_counts:
        gaps = np.array(all_trials[count])
        mean_gap = np.mean(gaps)
        std_gap = np.std(gaps, ddof=1)  # Sample std
        ci_95 = 1.96 * std_gap / np.sqrt(n_trials)  # 95% confidence interval

        aggregated["statistics"][f"{count}_caregivers"] = {
            "mean": float(mean_gap),
            "std": float(std_gap),
            "ci_95": float(ci_95),
            "min": float(np.min(gaps)),
            "max": float(np.max(gaps)),
            "all_values": gaps.tolist()
        }

        print(f"{count:<12} {mean_gap:<12.6f} {std_gap:<12.6f} ±{ci_95:.6f}")

    # Generate enhanced figure with error bars
    generate_robust_figure(aggregated, caregiver_counts)

    # Save aggregated results
    output_path = Path("outputs/limited_dataset_robust/robust_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(aggregated, f, indent=2)

    print(f"\n✓ Robust results saved to: {output_path}")

    return aggregated


def generate_robust_figure(aggregated, caregiver_counts):
    """
    Generate publication figure with error bars showing confidence intervals.
    """
    stats = aggregated["statistics"]

    # Extract data
    means = [stats[f"{c}_caregivers"]["mean"] for c in caregiver_counts]
    stds = [stats[f"{c}_caregivers"]["std"] for c in caregiver_counts]
    ci_95s = [stats[f"{c}_caregivers"]["ci_95"] for c in caregiver_counts]

    caregiver_counts_arr = np.array(caregiver_counts)
    means_arr = np.array(means)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot mean line with error bars
    ax.errorbar(caregiver_counts, means, yerr=ci_95s,
                fmt='o-', linewidth=2, markersize=10,
                capsize=5, capthick=2,
                label='Observed Gap (±95% CI)', color='#d62728')

    # Fit theoretical curve: gap ~ 1/sqrt(N)
    theoretical_constant = means[0] * np.sqrt(caregiver_counts[0])
    theoretical_curve = theoretical_constant / np.sqrt(caregiver_counts_arr)

    ax.plot(caregiver_counts, theoretical_curve, '--', linewidth=2,
            label=f'Hypothesis: {theoretical_constant:.2f}/√N', color='#1f77b4', alpha=0.7)

    # Find actual max and min mean gaps
    max_gap_idx = np.argmax(means)
    min_gap_idx = np.argmin(means)

    # Point A: Highest mean gap
    ax.scatter(caregiver_counts[max_gap_idx], means[max_gap_idx],
               s=150, color='#d62728', marker='o',
               edgecolors='black', linewidths=2, zorder=5,
               label='A: Highest Overfitting')
    ax.text(caregiver_counts[max_gap_idx], means[max_gap_idx] + 0.0005, 'A',
            fontsize=14, fontweight='bold', ha='center', va='bottom')

    # Point B: Lowest mean gap
    ax.scatter(caregiver_counts[min_gap_idx], means[min_gap_idx],
               s=150, color='#2ca02c', marker='s',
               edgecolors='black', linewidths=2, zorder=5,
               label='B: Best Generalization')
    ax.text(caregiver_counts[min_gap_idx], means[min_gap_idx] + 0.0005, 'B',
            fontsize=14, fontweight='bold', ha='center', va='bottom')

    ax.set_xlabel('Number of Caregivers', fontsize=12, fontweight='bold')
    ax.set_ylabel('Generalization Gap (Test - Train Error)', fontsize=12, fontweight='bold')
    ax.set_title(f'Model 3: Overfitting Decreases with Caregiver Diversity (n={aggregated["config"]["n_trials"]} trials)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(caregiver_counts)

    plt.tight_layout()

    # Save figure
    fig_path = Path("outputs/limited_dataset_robust/figures/generalization_gap_robust.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Robust figure saved to: {fig_path}")
    plt.close()


if __name__ == "__main__":
    results = run_multiple_trials(
        n_trials=20,
        caregiver_counts=[2, 3, 4, 5, 6, 8, 10, 12, 15, 20],
        save_individual=False  # Set True to keep all trial outputs
    )

    print("\n" + "=" * 70)
    print("ROBUST MODEL 3 EXPERIMENT COMPLETE!")
    print("=" * 70)
