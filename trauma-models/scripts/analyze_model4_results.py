#!/usr/bin/env python3
"""
Analyze Model 4 (Catastrophic Forgetting) results for paper presentation.

Presents results in multiple formats:
- Absolute MSE changes (easier to interpret than percentages)
- Relative changes (shows magnitude of catastrophic forgetting)
- LaTeX table format (for paper appendix)
"""

import json
from pathlib import Path


def load_results(results_path: Path) -> dict:
    """Load results JSON."""
    with open(results_path, 'r') as f:
        return json.load(f)


def print_absolute_changes(results: dict):
    """Print absolute MSE changes (easier to understand)."""
    print("\n" + "=" * 70)
    print("ABSOLUTE MSE CHANGES")
    print("=" * 70)

    baseline_trauma = results['baseline']['trauma_test_mse']
    baseline_therapy = results['baseline']['therapy_test_mse']

    print(f"\nBaseline (Post-Trauma, Pre-Therapy):")
    print(f"  Trauma test MSE: {baseline_trauma:.4f} (low = good trauma pattern learning)")
    print(f"  Therapy test MSE: {baseline_therapy:.4f} (high = poor therapy pattern learning)")

    print(f"\n{'Strategy':<25} {'Trauma MSE':<12} {'Δ Trauma':<12} {'Therapy MSE':<12} {'Δ Therapy':<12}")
    print("-" * 70)

    for strategy in ['naive', 'conservative', 'experience_replay']:
        result = results[strategy]
        trauma_mse = result['trauma_test_mse']
        therapy_mse = result['therapy_test_mse']

        delta_trauma = trauma_mse - baseline_trauma
        delta_therapy = therapy_mse - baseline_therapy

        print(f"{result['description']:<25} {trauma_mse:<12.4f} {delta_trauma:+11.4f} {therapy_mse:<12.4f} {delta_therapy:+11.4f}")


def print_relative_changes(results: dict):
    """Print relative changes (shows scale of forgetting)."""
    print("\n" + "=" * 70)
    print("RELATIVE CHANGES")
    print("=" * 70)

    baseline_trauma = results['baseline']['trauma_test_mse']
    baseline_therapy = results['baseline']['therapy_test_mse']

    print(f"\n{'Strategy':<25} {'Trauma Multiplier':<18} {'Therapy Reduction':<18}")
    print("-" * 70)

    for strategy in ['naive', 'conservative', 'experience_replay']:
        result = results[strategy]
        trauma_mse = result['trauma_test_mse']
        therapy_mse = result['therapy_test_mse']

        trauma_multiplier = trauma_mse / baseline_trauma
        therapy_reduction = (baseline_therapy - therapy_mse) / baseline_therapy * 100

        print(f"{result['description']:<25} {trauma_multiplier:<18.1f}x {therapy_reduction:<17.1f}%")


def print_interpretation(results: dict):
    """Print clinical interpretation."""
    print("\n" + "=" * 70)
    print("CLINICAL INTERPRETATION")
    print("=" * 70)

    print("""
Strategy 1: Naive Retraining (High LR, Therapy Only)
  - Error on trauma patterns increased 124x (0.004 → 0.44)
  - This is CATASTROPHIC FORGETTING in action
  - Clinical analogy: "Forgetting how to recognize danger"
  - Therapy learning: Excellent (99.2%)
  - Overall: UNSAFE - would erase critical survival patterns

Strategy 2: Conservative Retraining (Low LR, Therapy Only)
  - Error on trauma patterns increased 13x (0.004 → 0.047)
  - Still significant forgetting, but less catastrophic
  - Clinical analogy: "Slow, cautious therapy"
  - Therapy learning: Moderate (83.5%)
  - Overall: SLOW - would take years for meaningful change

Strategy 3: Experience Replay (Medium LR, 20% Trauma + 80% Therapy)
  - Error on trauma patterns increased 6x (0.004 → 0.021)
  - Minimal forgetting while achieving therapy goals
  - Clinical analogy: "Processing past while building new patterns"
  - Therapy learning: Excellent (98.9%)
  - Overall: OPTIMAL - balances preservation and change

Key Insight:
Experience replay (revisiting trauma memories during therapy) is not
optional - it's NECESSARY to prevent catastrophic forgetting while
learning new adaptive patterns.

This explains why:
- EMDR uses bilateral stimulation while recalling trauma
- CBT uses exposure therapy (gradual re-exposure to feared stimuli)
- Psychodynamic therapy explores past relationships repeatedly
- DBT combines distress tolerance (old patterns) with emotion regulation (new skills)

Therapy takes months to years because the brain must maintain awareness
of danger (trauma knowledge) while building new responses (therapy learning).
This is the fundamental constraint of neural networks learning contradictory
patterns from imbalanced datasets.
    """)


def generate_latex_table(results: dict):
    """Generate LaTeX table for paper appendix."""
    print("\n" + "=" * 70)
    print("LATEX TABLE (for paper)")
    print("=" * 70)

    baseline_trauma = results['baseline']['trauma_test_mse']
    baseline_therapy = results['baseline']['therapy_test_mse']

    print("""
\\begin{table}[h]
\\centering
\\caption{Catastrophic Forgetting Experiment Results}
\\label{tab:catastrophic_forgetting}
\\begin{tabular}{lcccc}
\\hline
\\textbf{Strategy} & \\textbf{Trauma MSE} & \\textbf{Therapy MSE} & \\textbf{Forgetting} & \\textbf{Learning} \\\\
\\hline
Baseline (pre-therapy) & %.4f & %.4f & -- & -- \\\\
\\hline
""" % (baseline_trauma, baseline_therapy))

    for strategy in ['naive', 'conservative', 'experience_replay']:
        result = results[strategy]
        trauma_mse = result['trauma_test_mse']
        therapy_mse = result['therapy_test_mse']
        trauma_mult = trauma_mse / baseline_trauma
        therapy_reduction = (baseline_therapy - therapy_mse) / baseline_therapy * 100

        strategy_name = result['description'].replace('(', '\\textit{(').replace(')', ')}')
        print(f"{strategy_name} & {trauma_mse:.4f} & {therapy_mse:.4f} & {trauma_mult:.1f}x & {therapy_reduction:.1f}\\% \\\\")

    print("""\\hline
\\end{tabular}
\\begin{tablenotes}
\\small
\\item \\textit{Forgetting}: Multiplier of trauma test error (higher = more forgetting)
\\item \\textit{Learning}: Percentage reduction in therapy test error (higher = better)
\\item \\textit{Optimal strategy}: Experience replay achieves 98.9\\% learning with only 6x forgetting
\\end{tablenotes}
\\end{table}
""")


def print_weight_changes(results: dict):
    """Print weight change statistics."""
    print("\n" + "=" * 70)
    print("WEIGHT CHANGES")
    print("=" * 70)

    print(f"\n{'Strategy':<25} {'Total Weight Δ':<18}")
    print("-" * 50)

    for strategy in ['naive', 'conservative', 'experience_replay']:
        result = results[strategy]
        weight_change = result['weight_changes']['total_weight_change']
        print(f"{result['description']:<25} {weight_change:<18.4f}")

    print("""
Interpretation:
- Naive: Large weight changes (7.45) = aggressive retraining
- Conservative: Small weight changes (1.48) = cautious updates
- Experience Replay: Moderate weight changes (3.93) = balanced approach
""")


def main():
    """Run complete analysis."""
    results_path = Path("outputs/catastrophic_forgetting/data/catastrophic_forgetting_results.json")

    if not results_path.exists():
        print(f"Error: Results file not found at {results_path}")
        print("Run the experiment first: python -m trauma_models.catastrophic_forgetting.experiment")
        return

    results = load_results(results_path)

    print("=" * 70)
    print("MODEL 4: CATASTROPHIC FORGETTING ANALYSIS")
    print("=" * 70)

    print_absolute_changes(results)
    print_relative_changes(results)
    print_weight_changes(results)
    generate_latex_table(results)
    print_interpretation(results)

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
