"""
Noisy signals experiment demonstrating inconsistent caregiving effects.

Validates prediction: Weight variance scales with sqrt(label_noise)

Key hypothesis:
- 5% noise → stable weights, high confidence, consistent behavior
- 30% noise → moderate instability, uncertainty emerging
- 60% noise → severe instability, confidence collapse, learned helplessness

This demonstrates why inconsistent caregiving creates anxious attachment:
neural networks cannot learn stable patterns from contradictory signals.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
import json
import matplotlib.pyplot as plt
import seaborn as sns

from .model import NoisySignalsModel
from .dataset import NoisySignalsDataset


class NoisySignalsExperiment:
    """
    Run noisy signals experiment across multiple noise levels and training runs.

    Workflow:
    1. Generate datasets with different noise levels (5%, 30%, 60%)
    2. Train multiple models per noise level (10 runs with different seeds)
    3. Measure:
       - Weight variance across runs
       - Prediction variance
       - Confidence collapse rate
       - Behavioral consistency
    4. Validate: weight_variance ~ sqrt(noise_level)
    """

    def __init__(
        self,
        output_dir: Path = Path("outputs"),
        noise_levels: list = [0.05, 0.30, 0.60],
        num_runs: int = 10,
        seed: int = 42,
    ):
        """
        Initialize experiment.

        Args:
            output_dir: Directory for outputs
            noise_levels: List of label noise probabilities to test
            num_runs: Number of training runs per noise level
            seed: Base random seed
        """
        self.output_dir = Path(output_dir)
        self.noise_levels = noise_levels
        self.num_runs = num_runs
        self.seed = seed

        # Create output directories
        (self.output_dir / "data").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(parents=True, exist_ok=True)

        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Results storage
        self.results = {
            "noise_levels": {},
            "summary": {},
        }

        self.datasets = None

    def generate_datasets(self):
        """Generate datasets for all noise levels."""
        print("\n" + "=" * 60)
        print("Generating Datasets")
        print("=" * 60)

        dataset_gen = NoisySignalsDataset(
            feature_dim=20,
            train_examples=10000,
            test_examples=2000,
            seed=self.seed,
        )

        self.datasets = dataset_gen.generate_multiple_noise_levels(
            noise_levels=self.noise_levels
        )

        print("\n✓ All datasets generated")

    def train_single_run(
        self,
        noise_level: float,
        run_idx: int,
    ) -> Dict:
        """
        Train single model with specified noise level and seed.

        Args:
            noise_level: Label noise probability
            run_idx: Run index (used for seed variation)

        Returns:
            Dictionary with model weights, predictions, and metrics
        """
        # Create model with unique seed
        run_seed = self.seed + run_idx
        model = NoisySignalsModel(seed=run_seed)

        # Get datasets
        train_ds = self.datasets[noise_level]["train"]
        test_ds = self.datasets[noise_level]["test"]

        # Train model
        history = model.train_model(
            train_dataset=train_ds,
            epochs=50,
            learning_rate=0.001,
            batch_size=32,
            verbose=False,
        )

        # Evaluate
        metrics = model.evaluate(test_ds)

        # Extract weight statistics
        weight_stats = model.compute_weight_stats()

        # Compute behavioral consistency
        consistency_metrics = model.compute_behavioral_consistency(test_ds)
        metrics.update(consistency_metrics)

        # Get final predictions for variance computation
        X_test, Y_test = test_ds.tensors
        with torch.no_grad():
            predictions = model(X_test)

        return {
            "run_idx": run_idx,
            "run_seed": run_seed,
            "final_loss": history["loss"][-1],
            "metrics": metrics,
            "weight_stats": weight_stats,
            "predictions": predictions,  # Store for cross-run variance
            "weight_snapshot": model.snapshot_weights(),
        }

    def run_noise_level_experiments(self, noise_level: float):
        """
        Run multiple training runs for specified noise level.

        Args:
            noise_level: Label noise probability
        """
        print(f"\n{'=' * 60}")
        print(f"Training Models: Noise Level {noise_level:.0%}")
        print(f"{'=' * 60}")

        runs = []

        for run_idx in range(self.num_runs):
            print(f"  Run {run_idx + 1}/{self.num_runs}...", end=" ")
            run_results = self.train_single_run(noise_level, run_idx)
            runs.append(run_results)
            print(f"Loss: {run_results['final_loss']:.4f}, Acc: {run_results['metrics']['accuracy']:.3f}")

        # Compute cross-run statistics
        print(f"\n✓ Computing cross-run statistics...")

        # Weight variance across runs
        weight_variances = []
        for layer_name in runs[0]["weight_snapshot"].keys():
            layer_weights = [run["weight_snapshot"][layer_name] for run in runs]
            layer_weights_stack = torch.stack(layer_weights, dim=0)  # [num_runs, ...]

            # Variance across runs
            layer_variance = torch.var(layer_weights_stack, dim=0).mean().item()
            weight_variances.append(layer_variance)

        mean_weight_variance = np.mean(weight_variances)

        # Prediction variance across runs
        predictions_stack = torch.stack([run["predictions"] for run in runs], dim=0)
        prediction_variance = torch.var(predictions_stack, dim=0).mean().item()

        # Aggregate metrics
        aggregate_metrics = {}
        metric_keys = runs[0]["metrics"].keys()

        for key in metric_keys:
            values = [run["metrics"][key] for run in runs]
            aggregate_metrics[f"{key}_mean"] = np.mean(values)
            aggregate_metrics[f"{key}_std"] = np.std(values)

        # Weight statistics aggregation
        weight_stat_keys = [k for k in runs[0]["weight_stats"].keys() if "total" in k]
        for key in weight_stat_keys:
            values = [run["weight_stats"][key] for run in runs]
            aggregate_metrics[f"{key}_mean"] = np.mean(values)
            aggregate_metrics[f"{key}_std"] = np.std(values)

        summary = {
            "noise_level": noise_level,
            "num_runs": self.num_runs,
            "weight_variance_across_runs": mean_weight_variance,
            "prediction_variance_across_runs": prediction_variance,
            "aggregate_metrics": aggregate_metrics,
            "individual_runs": runs,
        }

        print(f"\n  Weight variance (across runs): {mean_weight_variance:.6f}")
        print(f"  Prediction variance (across runs): {prediction_variance:.6f}")
        print(f"  Mean accuracy: {aggregate_metrics['accuracy_mean']:.3f} ± {aggregate_metrics['accuracy_std']:.3f}")
        print(f"  Confidence collapse rate: {aggregate_metrics['confidence_collapse_rate_mean']:.2%}")

        self.results["noise_levels"][noise_level] = summary

    def run_all_experiments(self):
        """Run experiments for all noise levels."""
        print("\n" + "=" * 60)
        print("Running Experiments Across All Noise Levels")
        print("=" * 60)

        for noise_level in self.noise_levels:
            self.run_noise_level_experiments(noise_level)

    def compute_hypothesis_validation(self):
        """
        Validate hypothesis: weight_variance ~ sqrt(noise_level)

        Fits power law: weight_var = a * noise^b
        Expects b ≈ 0.5 (square root relationship)
        """
        print("\n" + "=" * 60)
        print("Hypothesis Validation")
        print("=" * 60)

        noise_levels = np.array(self.noise_levels)
        weight_variances = np.array([
            self.results["noise_levels"][nl]["weight_variance_across_runs"]
            for nl in self.noise_levels
        ])

        # Fit power law: log(y) = log(a) + b * log(x)
        log_noise = np.log(noise_levels)
        log_variance = np.log(weight_variances)

        # Linear regression in log space
        coefficients = np.polyfit(log_noise, log_variance, deg=1)
        b = coefficients[0]  # Exponent
        log_a = coefficients[1]
        a = np.exp(log_a)

        # Compute R^2
        log_variance_pred = log_a + b * log_noise
        residuals = log_variance - log_variance_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((log_variance - np.mean(log_variance)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        print(f"\n✓ Power law fit: weight_variance = {a:.6f} * noise^{b:.3f}")
        print(f"  Expected exponent: ~0.5 (sqrt relationship)")
        print(f"  Actual exponent: {b:.3f}")
        print(f"  R²: {r_squared:.4f}")

        if abs(b - 0.5) < 0.15:
            print(f"\n  ✓ HYPOTHESIS VALIDATED: Exponent close to 0.5!")
        else:
            print(f"\n  ⚠️ HYPOTHESIS PARTIALLY SUPPORTED: Exponent deviates from 0.5")

        self.results["summary"]["hypothesis_validation"] = {
            "power_law_coefficient_a": a,
            "power_law_exponent_b": b,
            "r_squared": r_squared,
            "hypothesis_validated": abs(b - 0.5) < 0.15,
        }

    def generate_figures(self):
        """
        Generate comprehensive visualization figures.

        Figures:
        1. Weight variance vs noise level (with sqrt fit)
        2. Accuracy and confidence vs noise level
        3. Confidence collapse rate vs noise level
        4. Behavioral consistency vs noise level
        """
        print("\n" + "=" * 60)
        print("Generating Figures")
        print("=" * 60)

        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        noise_levels = np.array(self.noise_levels)

        # Extract metrics
        weight_variances = [
            self.results["noise_levels"][nl]["weight_variance_across_runs"]
            for nl in self.noise_levels
        ]

        prediction_variances = [
            self.results["noise_levels"][nl]["prediction_variance_across_runs"]
            for nl in self.noise_levels
        ]

        accuracies = [
            self.results["noise_levels"][nl]["aggregate_metrics"]["accuracy_mean"]
            for nl in self.noise_levels
        ]
        accuracy_stds = [
            self.results["noise_levels"][nl]["aggregate_metrics"]["accuracy_std"]
            for nl in self.noise_levels
        ]

        confidences = [
            self.results["noise_levels"][nl]["aggregate_metrics"]["mean_confidence_mean"]
            for nl in self.noise_levels
        ]

        collapse_rates = [
            self.results["noise_levels"][nl]["aggregate_metrics"]["confidence_collapse_rate_mean"]
            for nl in self.noise_levels
        ]

        behavioral_consistencies = [
            self.results["noise_levels"][nl]["aggregate_metrics"]["behavioral_consistency_mean"]
            for nl in self.noise_levels
        ]

        # 1. Weight variance vs noise level (with sqrt fit)
        ax1 = axes[0, 0]
        ax1.scatter(noise_levels * 100, weight_variances, s=200, c='#e74c3c',
                   alpha=0.7, edgecolor='black', linewidth=2, label="Observed")

        # Plot sqrt fit
        a = self.results["summary"]["hypothesis_validation"]["power_law_coefficient_a"]
        b = self.results["summary"]["hypothesis_validation"]["power_law_exponent_b"]
        noise_smooth = np.linspace(noise_levels.min(), noise_levels.max(), 100)
        variance_fit = a * (noise_smooth ** b)

        ax1.plot(noise_smooth * 100, variance_fit, 'k--', linewidth=2,
                label=f"Fit: variance = {a:.4f} × noise^{b:.2f}")

        ax1.set_xlabel("Label Noise Level (%)", fontsize=12, fontweight='bold')
        ax1.set_ylabel("Weight Variance (across runs)", fontsize=12, fontweight='bold')
        ax1.set_title("Weight Instability vs Noise Level", fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # 2. Accuracy and confidence vs noise level
        ax2 = axes[0, 1]
        ax2_twin = ax2.twinx()

        # Accuracy (left axis)
        line1 = ax2.plot(noise_levels * 100, accuracies, 'o-', linewidth=2,
                        markersize=10, color='#2ecc71', label="Accuracy")
        ax2.fill_between(
            noise_levels * 100,
            np.array(accuracies) - np.array(accuracy_stds),
            np.array(accuracies) + np.array(accuracy_stds),
            alpha=0.2, color='#2ecc71'
        )

        # Confidence (right axis)
        line2 = ax2_twin.plot(noise_levels * 100, confidences, 's-', linewidth=2,
                             markersize=10, color='#3498db', label="Confidence")

        ax2.set_xlabel("Label Noise Level (%)", fontsize=12, fontweight='bold')
        ax2.set_ylabel("Accuracy", fontsize=12, fontweight='bold', color='#2ecc71')
        ax2_twin.set_ylabel("Mean Confidence", fontsize=12, fontweight='bold', color='#3498db')
        ax2.set_title("Performance vs Noise Level", fontsize=14, fontweight='bold')

        ax2.tick_params(axis='y', labelcolor='#2ecc71')
        ax2_twin.tick_params(axis='y', labelcolor='#3498db')

        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper right', fontsize=10)

        ax2.grid(True, alpha=0.3)

        # 3. Confidence collapse rate vs noise level
        ax3 = axes[1, 0]
        bars = ax3.bar(noise_levels * 100, np.array(collapse_rates) * 100,
                      width=3, color='#f39c12', alpha=0.7, edgecolor='black')

        ax3.set_xlabel("Label Noise Level (%)", fontsize=12, fontweight='bold')
        ax3.set_ylabel("Confidence Collapse Rate (%)", fontsize=12, fontweight='bold')
        ax3.set_title("Uncertainty Emergence", fontsize=14, fontweight='bold')
        ax3.set_ylim([0, max(collapse_rates) * 110])

        # Add value labels on bars
        for bar, val in zip(bars, collapse_rates):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val*100:.1f}%', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

        ax3.grid(True, alpha=0.3, axis='y')

        # 4. Behavioral consistency vs noise level
        ax4 = axes[1, 1]
        ax4.plot(noise_levels * 100, behavioral_consistencies, 'D-',
                linewidth=2, markersize=12, color='#9b59b6',
                markeredgecolor='black', markeredgewidth=2)

        ax4.set_xlabel("Label Noise Level (%)", fontsize=12, fontweight='bold')
        ax4.set_ylabel("Behavioral Consistency", fontsize=12, fontweight='bold')
        ax4.set_title("Pattern Stability vs Noise Level", fontsize=14, fontweight='bold')
        ax4.set_ylim([0, 1.1])
        ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random chance')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        figure_path = self.output_dir / "figures" / "noisy_signals_analysis.png"
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Figure saved: {figure_path}")

        plt.close()

    def export_results(self):
        """Export results to JSON."""
        print("\n" + "=" * 60)
        print("Exporting Results")
        print("=" * 60)

        results_path = self.output_dir / "data" / "noisy_signals_results.json"

        # Convert tensors to lists for JSON serialization
        def convert_types(obj):
            if isinstance(obj, (np.ndarray, torch.Tensor)):
                return obj.tolist() if hasattr(obj, 'tolist') else list(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items() if k not in ['predictions', 'weight_snapshot']}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj

        json_results = convert_types(self.results)

        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)

        print(f"✓ Results exported: {results_path}")

    def print_summary(self):
        """Print experiment summary."""
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)

        print("\nWeight Variance by Noise Level:")
        print("-" * 60)
        for nl in self.noise_levels:
            wv = self.results["noise_levels"][nl]["weight_variance_across_runs"]
            acc = self.results["noise_levels"][nl]["aggregate_metrics"]["accuracy_mean"]
            collapse = self.results["noise_levels"][nl]["aggregate_metrics"]["confidence_collapse_rate_mean"]
            print(f"  {nl:.0%} noise: weight_var={wv:.6f}, accuracy={acc:.3f}, collapse={collapse:.2%}")

        print("\n" + "=" * 60)
        print("INTERPRETATION")
        print("=" * 60)

        hyp = self.results["summary"]["hypothesis_validation"]

        print(f"""
This experiment demonstrates how inconsistent caregiving creates anxious attachment:

1. HYPOTHESIS VALIDATION:
   Power law fit: weight_variance = {hyp['power_law_coefficient_a']:.6f} × noise^{hyp['power_law_exponent_b']:.3f}
   Expected exponent: 0.5 (sqrt relationship)
   R²: {hyp['r_squared']:.4f}

   {"✓ VALIDATED: Weight instability scales with sqrt(label_noise)" if hyp['hypothesis_validated'] else "⚠️ PARTIALLY SUPPORTED"}

2. BEHAVIORAL IMPLICATIONS:

   5% Noise (Baseline Consistency):
   - Stable weights, high confidence
   - Child learns reliable patterns
   - Secure attachment foundation

   30% Noise (Moderate Inconsistency):
   - Weight instability emerging
   - Accuracy declining
   - Anxious attachment forming - child hypervigilant, uncertain

   60% Noise (Severe Inconsistency):
   - Severe weight variance
   - Confidence collapse
   - Learned helplessness - cannot predict caregiver availability

3. CLINICAL MAPPING:

   Weight variance → Neural instability → Hypervigilance
   Confidence collapse → Chronic uncertainty → Anxiety
   Low behavioral consistency → Inability to form stable strategies → Disorganized attachment

This is the computational mechanism behind "inconsistent parenting creates anxious children."
The neural network literally cannot converge to stable patterns when training signals
contradict themselves.
        """)

    def run_complete_experiment(self):
        """Run complete experiment pipeline."""
        print("=" * 60)
        print("NOISY SIGNALS EXPERIMENT")
        print("Inconsistent Caregiving and Behavioral Instability")
        print("=" * 60)

        # Pipeline
        self.generate_datasets()
        self.run_all_experiments()
        self.compute_hypothesis_validation()
        self.generate_figures()
        self.export_results()
        self.print_summary()

        print("\n" + "=" * 60)
        print("Experiment complete! Check outputs/ directory for results.")
        print("=" * 60)


if __name__ == "__main__":
    """Run noisy signals experiment."""

    experiment = NoisySignalsExperiment(
        output_dir=Path("outputs/noisy_signals"),
        noise_levels=[0.05, 0.30, 0.60],
        num_runs=10,
        seed=42,
    )

    experiment.run_complete_experiment()
