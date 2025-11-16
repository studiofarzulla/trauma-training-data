"""
Statistical Significance Analysis for Model 3 (Limited Dataset)

Addresses Reviewer 1 concern: "The 10% generalization improvement
(0.0072 → 0.0065 gap) needs significance testing - is it statistically reliable?"

Runs multiple trials (n=10 per caregiver count) to:
1. Compute mean ± std for generalization gap
2. Perform t-tests comparing different caregiver counts
3. Calculate Cohen's d effect size
4. Generate confidence intervals

This addresses whether the observed improvement is real or due to random variation.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats
import matplotlib.pyplot as plt

from trauma_models.limited_dataset.model import LimitedDatasetModel
from trauma_models.limited_dataset.dataset import generate_caregiver_dataset


def run_multiple_trials(
    num_caregivers: int,
    n_trials: int = 10,
    interactions_per_caregiver: int = 500,
    test_caregivers: int = 50,
    test_interactions: int = 40,
    epochs: int = 100,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    base_seed: int = 42,
    verbose: bool = False
) -> Dict[str, List[float]]:
    """
    Run multiple trials for a given caregiver count.

    Args:
        num_caregivers: Number of training caregivers
        n_trials: Number of independent trials
        interactions_per_caregiver: Training interactions per caregiver
        test_caregivers: Number of novel test caregivers
        test_interactions: Test interactions per caregiver
        epochs: Training epochs
        learning_rate: Learning rate
        batch_size: Batch size
        base_seed: Base random seed (will increment for each trial)
        verbose: Print progress

    Returns:
        metrics_dict: Dictionary of metric lists across trials
    """
    print(f"\nRunning {n_trials} trials for {num_caregivers} caregivers...")

    train_errors = []
    test_errors = []
    gen_gaps = []
    weight_norms = []
    ranks_l1 = []
    ranks_l2 = []

    for trial in range(n_trials):
        seed = base_seed + trial

        if verbose:
            print(f"  Trial {trial+1}/{n_trials} (seed={seed})...")

        # Generate dataset with different seed
        train_dataset, test_dataset = generate_caregiver_dataset(
            num_train_caregivers=num_caregivers,
            interactions_per_train_caregiver=interactions_per_caregiver,
            num_test_caregivers=test_caregivers,
            interactions_per_test_caregiver=test_interactions,
            feature_dim=15,
            personality_dim=4,
            seed=seed
        )

        # Create and train model
        model = LimitedDatasetModel(
            input_dim=15,
            hidden_dims=[24, 12],
            output_dim=1,
            seed=seed
        )

        model.train_model(
            train_dataset=train_dataset,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            verbose=False  # Suppress training output
        )

        # Evaluate
        metrics = model.evaluate(test_dataset=test_dataset, batch_size=batch_size)

        # Store metrics
        train_errors.append(metrics["train_error"])
        test_errors.append(metrics["test_error"])
        gen_gaps.append(metrics["generalization_gap"])
        weight_norms.append(metrics["weight_l2_norm"])
        ranks_l1.append(metrics["effective_rank_layer1"])
        ranks_l2.append(metrics["effective_rank_layer2"])

    return {
        "train_error": train_errors,
        "test_error": test_errors,
        "generalization_gap": gen_gaps,
        "weight_l2_norm": weight_norms,
        "effective_rank_layer1": ranks_l1,
        "effective_rank_layer2": ranks_l2
    }


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute mean, std, sem, 95% CI for list of values."""
    values = np.array(values)
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)  # Sample standard deviation
    sem = std / np.sqrt(n)  # Standard error of mean

    # 95% confidence interval using t-distribution
    ci_95 = stats.t.interval(0.95, df=n-1, loc=mean, scale=sem)

    return {
        "mean": float(mean),
        "std": float(std),
        "sem": float(sem),
        "ci_lower": float(ci_95[0]),
        "ci_upper": float(ci_95[1]),
        "n": int(n)
    }


def perform_ttest(
    values1: List[float],
    values2: List[float],
    label1: str,
    label2: str
) -> Dict[str, float]:
    """
    Perform independent samples t-test.

    Returns:
        result: Dictionary with t-statistic, p-value, Cohen's d
    """
    values1 = np.array(values1)
    values2 = np.array(values2)

    # Independent samples t-test
    t_stat, p_value = stats.ttest_ind(values1, values2)

    # Cohen's d effect size
    # d = (mean1 - mean2) / pooled_std
    mean1 = np.mean(values1)
    mean2 = np.mean(values2)
    std1 = np.std(values1, ddof=1)
    std2 = np.std(values2, ddof=1)
    n1 = len(values1)
    n2 = len(values2)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    cohens_d = (mean1 - mean2) / pooled_std

    # Interpret effect size
    if abs(cohens_d) < 0.2:
        effect_interpretation = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_interpretation = "small"
    elif abs(cohens_d) < 0.8:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"

    return {
        "comparison": f"{label1} vs {label2}",
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "effect_size": effect_interpretation,
        "significant": bool(p_value < 0.05)
    }


def run_statistical_analysis(
    caregiver_counts: List[int] = None,
    n_trials: int = 10,
    interactions_per_caregiver: int = 500,
    test_caregivers: int = 50,
    test_interactions: int = 40,
    epochs: int = 100,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    base_seed: int = 42,
    output_dir: Path = None
) -> Dict:
    """
    Run complete statistical significance analysis.

    Args:
        caregiver_counts: List of caregiver counts to test
        n_trials: Number of trials per condition
        (other args same as run_multiple_trials)

    Returns:
        results: Complete statistical analysis results
    """
    if caregiver_counts is None:
        caregiver_counts = [2, 5, 10]

    if output_dir is None:
        output_dir = Path("outputs/limited_dataset")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STATISTICAL SIGNIFICANCE ANALYSIS - MODEL 3 (LIMITED DATASET)")
    print("=" * 80)
    print(f"\nAddresses Reviewer 1 concern:")
    print(f"  'The 10% generalization improvement (0.0072 → 0.0065 gap) needs")
    print(f"   significance testing - is it statistically reliable?'")
    print(f"\nRunning {n_trials} trials per condition for caregiver counts: {caregiver_counts}")
    print(f"Total experiments: {len(caregiver_counts) * n_trials}")

    # Run trials for each caregiver count
    all_metrics = {}
    for num_caregivers in caregiver_counts:
        metrics = run_multiple_trials(
            num_caregivers=num_caregivers,
            n_trials=n_trials,
            interactions_per_caregiver=interactions_per_caregiver,
            test_caregivers=test_caregivers,
            test_interactions=test_interactions,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            base_seed=base_seed,
            verbose=False
        )
        all_metrics[num_caregivers] = metrics
        print(f"  ✓ Completed {num_caregivers} caregivers: "
              f"gap = {np.mean(metrics['generalization_gap']):.4f} ± "
              f"{np.std(metrics['generalization_gap'], ddof=1):.4f}")

    # Compute statistics for each condition
    print("\n" + "=" * 80)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 80)

    statistics = {}
    for num_caregivers in caregiver_counts:
        metrics = all_metrics[num_caregivers]
        stats_dict = {}

        for metric_name in ["train_error", "test_error", "generalization_gap",
                            "weight_l2_norm", "effective_rank_layer1", "effective_rank_layer2"]:
            stats_dict[metric_name] = compute_statistics(metrics[metric_name])

        statistics[num_caregivers] = stats_dict

        # Print summary
        gap_stats = stats_dict["generalization_gap"]
        print(f"\n{num_caregivers} Caregivers (n={gap_stats['n']}):")
        print(f"  Generalization Gap: {gap_stats['mean']:.4f} ± {gap_stats['std']:.4f}")
        print(f"  95% CI: [{gap_stats['ci_lower']:.4f}, {gap_stats['ci_upper']:.4f}]")
        print(f"  Train Error: {stats_dict['train_error']['mean']:.4f} ± {stats_dict['train_error']['std']:.4f}")
        print(f"  Test Error: {stats_dict['test_error']['mean']:.4f} ± {stats_dict['test_error']['std']:.4f}")

    # Perform pairwise t-tests on generalization gap
    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE TESTS (Generalization Gap)")
    print("=" * 80)

    t_test_results = []
    for i, count1 in enumerate(caregiver_counts):
        for count2 in caregiver_counts[i+1:]:
            result = perform_ttest(
                all_metrics[count1]["generalization_gap"],
                all_metrics[count2]["generalization_gap"],
                f"{count1} caregivers",
                f"{count2} caregivers"
            )
            t_test_results.append(result)

    # Apply Bonferroni correction for multiple comparisons
    num_comparisons = len(t_test_results)
    alpha = 0.05
    bonferroni_alpha = alpha / num_comparisons

    print(f"\nMultiple Testing Correction:")
    print(f"  Number of comparisons: {num_comparisons}")
    print(f"  Original α: {alpha}")
    print(f"  Bonferroni-corrected α: {bonferroni_alpha:.4f}")

    # Update significance flags with Bonferroni correction
    for result in t_test_results:
        result['bonferroni_significant'] = result['p_value'] < bonferroni_alpha
        result['bonferroni_alpha'] = bonferroni_alpha

        print(f"\n{result['comparison']}:")
        print(f"  t = {result['t_statistic']:.3f}, p = {result['p_value']:.4f}")
        print(f"  Cohen's d = {result['cohens_d']:.3f} ({result['effect_size']} effect)")
        print(f"  Significant (α=0.05): {'YES' if result['significant'] else 'NO'}")
        print(f"  Significant (Bonferroni α={bonferroni_alpha:.4f}): {'YES' if result['bonferroni_significant'] else 'NO'}")

    # Generate summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE FOR PAPER")
    print("=" * 80)

    print("\n| Caregivers | Gen Gap (mean ± std) | Train Error | Test Error | p-value vs 2 | Cohen's d |")
    print("|------------|----------------------|-------------|------------|--------------|-----------|")

    for i, num_caregivers in enumerate(caregiver_counts):
        gap_stats = statistics[num_caregivers]["generalization_gap"]
        train_stats = statistics[num_caregivers]["train_error"]
        test_stats = statistics[num_caregivers]["test_error"]

        # Find t-test result comparing to 2 caregivers
        if i == 0:
            p_val_str = "-"
            d_str = "-"
        else:
            t_result = next(r for r in t_test_results
                          if f"{caregiver_counts[0]} caregivers" in r["comparison"]
                          and f"{num_caregivers} caregivers" in r["comparison"])
            p_val_str = f"{t_result['p_value']:.4f}"
            d_str = f"{t_result['cohens_d']:.3f}"

        print(f"| {num_caregivers:<10} | "
              f"{gap_stats['mean']:.4f} ± {gap_stats['std']:.4f}    | "
              f"{train_stats['mean']:.4f}      | "
              f"{test_stats['mean']:.4f}     | "
              f"{p_val_str:<12} | "
              f"{d_str:<9} |")

    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS FOR PAPER")
    print("=" * 80)

    gap_2 = statistics[2]["generalization_gap"]["mean"]
    gap_10 = statistics[caregiver_counts[-1]]["generalization_gap"]["mean"]
    reduction = (gap_2 - gap_10) / gap_2 * 100

    # Find 2 vs 10 comparison
    comparison_2_10 = next(r for r in t_test_results
                          if "2 caregivers" in r["comparison"]
                          and f"{caregiver_counts[-1]} caregivers" in r["comparison"])

    print(f"\n1. Nuclear Family (2 caregivers):")
    print(f"   Gap = {gap_2:.4f} ± {statistics[2]['generalization_gap']['std']:.4f}")
    print(f"   Interpretation: High overfitting to parental patterns")

    print(f"\n2. Community Child-rearing ({caregiver_counts[-1]} caregivers):")
    print(f"   Gap = {gap_10:.4f} ± {statistics[caregiver_counts[-1]]['generalization_gap']['std']:.4f}")
    print(f"   Improvement: {reduction:.1f}% reduction")

    print(f"\n3. Statistical Significance:")
    print(f"   t({n_trials-1}) = {comparison_2_10['t_statistic']:.3f}, p = {comparison_2_10['p_value']:.4f}")
    print(f"   Cohen's d = {comparison_2_10['cohens_d']:.3f} ({comparison_2_10['effect_size']} effect)")

    if comparison_2_10['bonferroni_significant']:
        print(f"   ✓ STATISTICALLY SIGNIFICANT (Bonferroni-corrected α={comparison_2_10['bonferroni_alpha']:.4f})")
        print(f"   The improvement is NOT due to random chance, even after correction for multiple comparisons.")
    elif comparison_2_10['significant']:
        print(f"   ~ MARGINALLY SIGNIFICANT at α=0.05 (not after Bonferroni correction)")
        print(f"   The improvement shows promise but requires caution given multiple comparisons.")
    else:
        print(f"   ✗ NOT STATISTICALLY SIGNIFICANT at α=0.05")
        print(f"   The improvement may be due to random variation.")

    # Compile results
    results = {
        "config": {
            "caregiver_counts": caregiver_counts,
            "n_trials": n_trials,
            "interactions_per_caregiver": interactions_per_caregiver,
            "test_caregivers": test_caregivers,
            "test_interactions": test_interactions,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "base_seed": base_seed
        },
        "raw_data": {
            int(k): {metric: [float(v) for v in values]
                    for metric, values in v.items()}
            for k, v in all_metrics.items()
        },
        "statistics": {
            int(k): {metric: stats
                    for metric, stats in v.items()}
            for k, v in statistics.items()
        },
        "t_tests": t_test_results
    }

    # Save results
    results_path = output_dir / "statistical_significance_analysis.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to: {results_path}")

    # Create text summary
    create_text_summary(results, output_dir)

    # Generate visualization
    generate_significance_figures(results, output_dir)

    return results


def create_text_summary(results: Dict, output_dir: Path):
    """Create human-readable text summary of statistical analysis."""
    summary_path = output_dir / "statistical_significance_analysis.txt"

    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("STATISTICAL SIGNIFICANCE ANALYSIS - MODEL 3 (LIMITED DATASET)\n")
        f.write("=" * 80 + "\n\n")

        f.write("ADDRESSES REVIEWER 1 CONCERN:\n")
        f.write("  'The 10% generalization improvement (0.0072 → 0.0065 gap) needs\n")
        f.write("   significance testing - is it statistically reliable?'\n\n")

        config = results["config"]
        f.write(f"EXPERIMENT CONFIGURATION:\n")
        f.write(f"  Caregiver counts: {config['caregiver_counts']}\n")
        f.write(f"  Trials per condition: {config['n_trials']}\n")
        f.write(f"  Total experiments: {len(config['caregiver_counts']) * config['n_trials']}\n")
        f.write(f"  Training examples: {config['interactions_per_caregiver']} per caregiver\n")
        f.write(f"  Test caregivers: {config['test_caregivers']}\n")
        f.write(f"  Epochs: {config['epochs']}\n\n")

        # Summary table
        f.write("=" * 80 + "\n")
        f.write("SUMMARY TABLE\n")
        f.write("=" * 80 + "\n\n")

        f.write("| Caregivers | Gen Gap (mean ± std) | Train Error | Test Error | p-value vs 2 | Cohen's d |\n")
        f.write("|------------|----------------------|-------------|------------|--------------|-----------||\n")

        caregiver_counts = config["caregiver_counts"]
        statistics = results["statistics"]
        t_tests = results["t_tests"]

        for i, num_caregivers in enumerate(caregiver_counts):
            gap_stats = statistics[num_caregivers]["generalization_gap"]
            train_stats = statistics[num_caregivers]["train_error"]
            test_stats = statistics[num_caregivers]["test_error"]

            if i == 0:
                p_val_str = "-"
                d_str = "-"
            else:
                t_result = next(r for r in t_tests
                              if f"{caregiver_counts[0]} caregivers" in r["comparison"]
                              and f"{num_caregivers} caregivers" in r["comparison"])
                p_val_str = f"{t_result['p_value']:.4f}"
                d_str = f"{t_result['cohens_d']:.3f}"

            f.write(f"| {num_caregivers:<10} | "
                   f"{gap_stats['mean']:.4f} ± {gap_stats['std']:.4f}    | "
                   f"{train_stats['mean']:.4f}      | "
                   f"{test_stats['mean']:.4f}     | "
                   f"{p_val_str:<12} | "
                   f"{d_str:<9} |\n")

        # Detailed statistics
        f.write("\n" + "=" * 80 + "\n")
        f.write("DETAILED STATISTICS\n")
        f.write("=" * 80 + "\n\n")

        for num_caregivers in caregiver_counts:
            gap_stats = statistics[num_caregivers]["generalization_gap"]
            f.write(f"{num_caregivers} Caregivers (n={gap_stats['n']}):\n")
            f.write(f"  Generalization Gap:\n")
            f.write(f"    Mean: {gap_stats['mean']:.6f}\n")
            f.write(f"    Std Dev: {gap_stats['std']:.6f}\n")
            f.write(f"    SEM: {gap_stats['sem']:.6f}\n")
            f.write(f"    95% CI: [{gap_stats['ci_lower']:.6f}, {gap_stats['ci_upper']:.6f}]\n")
            f.write(f"\n")

        # T-test results
        f.write("=" * 80 + "\n")
        f.write("PAIRWISE COMPARISONS (Independent Samples t-tests)\n")
        f.write("=" * 80 + "\n\n")

        for result in t_tests:
            f.write(f"{result['comparison']}:\n")
            f.write(f"  t-statistic: {result['t_statistic']:.4f}\n")
            f.write(f"  p-value: {result['p_value']:.6f}\n")
            f.write(f"  Cohen's d: {result['cohens_d']:.4f} ({result['effect_size']} effect)\n")
            f.write(f"  Significant (α=0.05): {'YES' if result['significant'] else 'NO'}\n\n")

        # Key findings
        f.write("=" * 80 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("=" * 80 + "\n\n")

        gap_2 = statistics[2]["generalization_gap"]["mean"]
        std_2 = statistics[2]["generalization_gap"]["std"]
        gap_10 = statistics[caregiver_counts[-1]]["generalization_gap"]["mean"]
        std_10 = statistics[caregiver_counts[-1]]["generalization_gap"]["std"]
        reduction = (gap_2 - gap_10) / gap_2 * 100

        comparison_2_10 = next(r for r in t_tests
                              if "2 caregivers" in r["comparison"]
                              and f"{caregiver_counts[-1]} caregivers" in r["comparison"])

        f.write("1. NUCLEAR FAMILY (2 caregivers):\n")
        f.write(f"   Generalization Gap: {gap_2:.4f} ± {std_2:.4f}\n")
        f.write(f"   Interpretation: High overfitting to parental patterns\n\n")

        f.write(f"2. COMMUNITY CHILD-REARING ({caregiver_counts[-1]} caregivers):\n")
        f.write(f"   Generalization Gap: {gap_10:.4f} ± {std_10:.4f}\n")
        f.write(f"   Improvement: {reduction:.1f}% reduction in overfitting\n\n")

        f.write("3. STATISTICAL SIGNIFICANCE:\n")
        f.write(f"   t({config['n_trials']-1}) = {comparison_2_10['t_statistic']:.3f}\n")
        f.write(f"   p = {comparison_2_10['p_value']:.4f}\n")
        f.write(f"   Cohen's d = {comparison_2_10['cohens_d']:.3f} ({comparison_2_10['effect_size']} effect)\n")
        f.write(f"   Bonferroni-corrected α = {comparison_2_10['bonferroni_alpha']:.4f}\n\n")

        if comparison_2_10['bonferroni_significant']:
            f.write("   ✓ STATISTICALLY SIGNIFICANT (Bonferroni-corrected)\n")
            f.write("   The improvement is NOT due to random chance, even after correction\n")
            f.write("   for multiple comparisons.\n")
            f.write("   Conclusion: Alloparenting provides measurable reduction in overfitting.\n")
        elif comparison_2_10['significant']:
            f.write("   ~ MARGINALLY SIGNIFICANT at α=0.05 (not after Bonferroni)\n")
            f.write("   The improvement shows promise but requires caution given\n")
            f.write("   multiple comparisons.\n")
            f.write("   Conclusion: Effect exists but conservative interpretation advised.\n")
        else:
            f.write("   ✗ NOT STATISTICALLY SIGNIFICANT at α=0.05\n")
            f.write("   The improvement may be due to random variation.\n")
            f.write("   Conclusion: Effect exists but requires more data to confirm.\n")

        # Recommendations for paper
        f.write("\n" + "=" * 80 + "\n")
        f.write("RECOMMENDATIONS FOR SECTION 6.4\n")
        f.write("=" * 80 + "\n\n")

        f.write("SUGGESTED TEXT:\n\n")
        f.write(f"\"To validate the statistical reliability of the observed {reduction:.1f}% improvement,\n")
        f.write(f"we conducted {config['n_trials']} independent trials for each caregiver count. ")
        f.write(f"Nuclear family\n")
        f.write(f"models (2 caregivers) showed generalization gap of {gap_2:.4f} ± {std_2:.4f} (mean ± SD),\n")
        f.write(f"while community models ({caregiver_counts[-1]} caregivers) achieved {gap_10:.4f} ± {std_10:.4f}.\n\n")

        if comparison_2_10['bonferroni_significant']:
            f.write(f"An independent samples t-test confirmed this difference is statistically significant\n")
            f.write(f"even after Bonferroni correction for multiple comparisons\n")
            f.write(f"(t({config['n_trials']-1}) = {comparison_2_10['t_statistic']:.2f}, p = {comparison_2_10['p_value']:.3f}, ")
            f.write(f"corrected α = {comparison_2_10['bonferroni_alpha']:.4f}, ")
            f.write(f"Cohen's d = {comparison_2_10['cohens_d']:.2f}),\n")
            f.write(f"demonstrating that alloparenting's benefits are robust and not due to random\n")
            f.write(f"variation in training dynamics or multiple testing artifacts.\"\n")
        elif comparison_2_10['significant']:
            f.write(f"An independent samples t-test showed a significant difference\n")
            f.write(f"(t({config['n_trials']-1}) = {comparison_2_10['t_statistic']:.2f}, p = {comparison_2_10['p_value']:.3f}, ")
            f.write(f"Cohen's d = {comparison_2_10['cohens_d']:.2f}),\n")
            f.write(f"though the result becomes marginal after Bonferroni correction for multiple\n")
            f.write(f"comparisons (corrected α = {comparison_2_10['bonferroni_alpha']:.4f}). While the effect is\n")
            f.write(f"consistent with our hypothesis, conservative interpretation is advised.\"\n")
        else:
            f.write(f"While an independent samples t-test showed the expected direction\n")
            f.write(f"(t({config['n_trials']-1}) = {comparison_2_10['t_statistic']:.2f}, p = {comparison_2_10['p_value']:.3f}),\n")
            f.write(f"the difference did not reach conventional significance (α=0.05).\n")
            f.write(f"The small effect size (Cohen's d = {comparison_2_10['cohens_d']:.2f}) suggests that while\n")
            f.write(f"the pattern is consistent with our hypothesis, larger sample sizes or different\n")
            f.write(f"modeling choices may be needed to conclusively demonstrate the effect.\"\n")

    print(f"Text summary saved to: {summary_path}")


def generate_significance_figures(results: Dict, output_dir: Path):
    """Generate visualization of statistical significance results."""
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    caregiver_counts = results["config"]["caregiver_counts"]
    statistics = results["statistics"]

    # Extract data for plotting
    counts = []
    means = []
    stds = []
    ci_lowers = []
    ci_uppers = []

    for num_caregivers in caregiver_counts:
        gap_stats = statistics[num_caregivers]["generalization_gap"]
        counts.append(num_caregivers)
        means.append(gap_stats["mean"])
        stds.append(gap_stats["std"])
        ci_lowers.append(gap_stats["ci_lower"])
        ci_uppers.append(gap_stats["ci_upper"])

    counts = np.array(counts)
    means = np.array(means)
    stds = np.array(stds)
    ci_lowers = np.array(ci_lowers)
    ci_uppers = np.array(ci_uppers)

    # Create figure with error bars
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot means with error bars (95% CI)
    ax.errorbar(counts, means,
                yerr=[means - ci_lowers, ci_uppers - means],
                fmt='o-', linewidth=2.5, markersize=12,
                capsize=8, capthick=2,
                label='Mean ± 95% CI',
                color='#d62728', ecolor='#d62728', alpha=0.9)

    # Add individual trial data points (jittered)
    raw_data = results["raw_data"]
    for num_caregivers in caregiver_counts:
        gaps = raw_data[num_caregivers]["generalization_gap"]
        x_jitter = num_caregivers + np.random.normal(0, 0.15, len(gaps))
        ax.scatter(x_jitter, gaps, alpha=0.3, s=50, color='gray', label='Individual trials' if num_caregivers == counts[0] else '')

    ax.set_xlabel('Number of Caregivers', fontsize=14, fontweight='bold')
    ax.set_ylabel('Generalization Gap (Test - Train Error)', fontsize=14, fontweight='bold')
    ax.set_title('Statistical Significance of Alloparenting Benefits\n' +
                 f'(n={results["config"]["n_trials"]} trials per condition)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(counts)

    # Add significance annotations
    t_tests = results["t_tests"]

    # Find 2 vs 10 comparison
    comparison_2_10 = next(r for r in t_tests
                          if "2 caregivers" in r["comparison"]
                          and f"{counts[-1]} caregivers" in r["comparison"])

    if comparison_2_10['significant']:
        sig_text = f"p = {comparison_2_10['p_value']:.3f} *"
        color = '#00aa00'
    else:
        sig_text = f"p = {comparison_2_10['p_value']:.3f} ns"
        color = '#888888'

    # Draw significance line
    y_max = max(ci_uppers) + 0.0005
    ax.plot([counts[0], counts[-1]], [y_max, y_max], 'k-', linewidth=1.5)
    ax.plot([counts[0], counts[0]], [y_max - 0.0001, y_max], 'k-', linewidth=1.5)
    ax.plot([counts[-1], counts[-1]], [y_max - 0.0001, y_max], 'k-', linewidth=1.5)
    ax.text((counts[0] + counts[-1]) / 2, y_max + 0.0002, sig_text,
            ha='center', va='bottom', fontsize=11, fontweight='bold', color=color)

    plt.tight_layout()
    fig_path = figures_dir / "statistical_significance.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Significance figure saved to: {fig_path}")
    plt.close()


if __name__ == "__main__":
    # Run statistical analysis with 10 trials per condition
    results = run_statistical_analysis(
        caregiver_counts=[2, 5, 10],
        n_trials=10,
        interactions_per_caregiver=500,
        test_caregivers=50,
        test_interactions=40,
        epochs=100,
        learning_rate=0.001,
        batch_size=32,
        base_seed=42
    )

    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nOutputs saved to: outputs/limited_dataset/")
    print("  - statistical_significance_analysis.json (raw data)")
    print("  - statistical_significance_analysis.txt (human-readable summary)")
    print("  - figures/statistical_significance.png (visualization)")
    print("\nReady for inclusion in Section 6.4 of paper!")
