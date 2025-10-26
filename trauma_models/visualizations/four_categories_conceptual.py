"""
Conceptual diagram showing the four training data categories.

This visualization provides a visual representation of the computational framework's
core taxonomy: extreme penalties, noisy signals, class imbalance, and limited exposure.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set publication-quality style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['font.family'] = 'sans-serif'

def create_four_categories_diagram():
    """Generate 4-panel conceptual diagram of training data problems."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Four Categories of Developmental Training Data Problems',
                 fontsize=16, fontweight='bold', y=0.98)

    # ==================== Panel A: Extreme Penalty ====================
    ax = axes[0, 0]

    # Show normal gradient vs extreme gradient
    epochs = np.arange(0, 20)
    normal_loss = 1.0 * np.exp(-epochs * 0.3) + 0.1

    # Extreme penalty causes huge spike then instability
    extreme_loss = normal_loss.copy()
    extreme_loss[5] = 3.5  # Massive spike at "punishment" event
    extreme_loss[6:] = normal_loss[6:] + np.random.uniform(-0.3, 0.3, len(extreme_loss[6:]))

    ax.plot(epochs, normal_loss, 'g-', linewidth=2.5, label='Normal training', alpha=0.8)
    ax.plot(epochs, extreme_loss, 'r-', linewidth=2.5, label='Extreme penalty', alpha=0.8)

    # Highlight the extreme event
    ax.axvline(x=5, color='darkred', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.text(5.5, 3.2, 'Extreme\nPenalty\nEvent', fontsize=9, color='darkred',
            verticalalignment='top')

    ax.set_xlabel('Training Time (epochs)', fontweight='bold')
    ax.set_ylabel('Loss (Error Signal)', fontweight='bold')
    ax.set_title('Category 1: Extreme Penalties\n(e.g., Harsh Physical Punishment)',
                 fontweight='bold', pad=10)
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax.set_ylim(0, 4)

    # Add text annotation
    ax.text(0.02, 0.97, 'Effect: Weight cascade,\novercorrection, instability',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7, pad=0.5))

    # ==================== Panel B: Noisy Signals ====================
    ax = axes[0, 1]

    # Show consistent vs inconsistent caregiver responses
    time = np.arange(0, 30)

    # Consistent caregiver (smooth pattern)
    consistent_response = np.sin(time * 0.3) + np.random.normal(0, 0.1, len(time))

    # Inconsistent caregiver (same pattern but with big noise)
    inconsistent_response = np.sin(time * 0.3) + np.random.normal(0, 0.6, len(time))

    ax.plot(time, consistent_response, 'b-', linewidth=2.5, alpha=0.7,
            label='Consistent caregiver (5% noise)', marker='o', markersize=4)
    ax.plot(time, inconsistent_response, 'orange', linewidth=2.5, alpha=0.7,
            label='Inconsistent caregiver (60% noise)', marker='s', markersize=4)

    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
    ax.set_xlabel('Similar Contexts Over Time', fontweight='bold')
    ax.set_ylabel('Caregiver Response', fontweight='bold')
    ax.set_title('Category 2: Noisy Signals\n(e.g., Inconsistent Parenting)',
                 fontweight='bold', pad=10)
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax.set_ylim(-2.5, 2.5)

    # Add text annotation
    ax.text(0.02, 0.97, 'Effect: Behavioral instability,\nanxious attachment, uncertainty',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7, pad=0.5))

    # ==================== Panel C: Class Imbalance ====================
    ax = axes[1, 0]

    # Show pie chart of positive vs negative examples
    labels = ['Negative\nExamples\n(Criticism, punishment)',
              'Positive\nExamples\n(Warmth, support)']

    # Typical imbalanced parenting: 85% negative, 15% positive
    sizes = [85, 15]
    colors = ['#ff6b6b', '#51cf66']
    explode = (0, 0.1)  # Explode the positive slice

    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels,
                                       colors=colors, autopct='%1.0f%%',
                                       shadow=True, startangle=90,
                                       textprops={'fontsize': 10, 'weight': 'bold'})

    # Make percentage text bold and larger
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(13)
        autotext.set_weight('bold')

    ax.set_title('Category 3: Class Imbalance\n(e.g., Absent Positive Experiences)',
                 fontweight='bold', pad=10)

    # Add text annotation
    ax.text(0.02, 0.97, 'Effect: Overfitting to\nnegative patterns, lack of\npositive exemplars',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7, pad=0.5))

    # ==================== Panel D: Limited Dataset ====================
    ax = axes[1, 1]

    # Show true function vs learned function from sparse data
    x = np.linspace(0, 10, 100)
    true_function = np.sin(x) + 0.2 * x

    # Limited training data (nuclear family = 2 caregivers)
    x_train_limited = np.array([1, 2, 8, 9])
    y_train_limited = np.sin(x_train_limited) + 0.2 * x_train_limited

    # Diverse training data (community = 10 caregivers)
    x_train_diverse = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5])
    y_train_diverse = np.sin(x_train_diverse) + 0.2 * x_train_diverse

    # Overfitted model (linear fit to limited data)
    from numpy.polynomial import polynomial as P
    coefs_limited = P.polyfit(x_train_limited, y_train_limited, 1)
    overfitted_pred = P.polyval(x, coefs_limited)

    # Better generalization from diverse data
    coefs_diverse = P.polyfit(x_train_diverse, y_train_diverse, 3)
    diverse_pred = P.polyval(x, coefs_diverse)

    # Plot
    ax.plot(x, true_function, 'k--', linewidth=2.5, label='True social patterns', alpha=0.7)
    ax.scatter(x_train_limited, y_train_limited, color='red', s=120, marker='X',
               label='Limited data (2 caregivers)', zorder=5, edgecolor='darkred', linewidth=1.5)
    ax.plot(x, overfitted_pred, 'r-', linewidth=2, alpha=0.6,
            label='Overfitted model', linestyle='-.')

    ax.scatter(x_train_diverse, y_train_diverse, color='green', s=80, marker='o',
               label='Diverse data (10 caregivers)', zorder=5, edgecolor='darkgreen', linewidth=1.5)
    ax.plot(x, diverse_pred, 'g-', linewidth=2, alpha=0.6,
            label='Generalized model')

    ax.set_xlabel('Social Context Diversity', fontweight='bold')
    ax.set_ylabel('Predicted Behavior Quality', fontweight='bold')
    ax.set_title('Category 4: Insufficient Exposure\n(e.g., Nuclear Family Isolation)',
                 fontweight='bold', pad=10)
    ax.legend(loc='lower right', framealpha=0.95, fontsize=8)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)

    # Add text annotation
    ax.text(0.02, 0.97, 'Effect: Overfitting to specific\ncaregivers, poor generalization\nto novel relationships',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7, pad=0.5))

    # ==================== Final Layout ====================
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save figure
    output_dir = Path(__file__).parent.parent.parent / 'essays' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'figure0_four_categories_diagram.png'

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Conceptual diagram saved: {output_path}")

    # Also save to model outputs for reference
    model_output_dir = Path(__file__).parent.parent / 'outputs' / 'conceptual_diagram'
    model_output_dir.mkdir(parents=True, exist_ok=True)
    model_output_path = model_output_dir / 'four_categories_conceptual.png'
    plt.savefig(model_output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Copy saved: {model_output_path}")

    plt.close()

    return str(output_path)

if __name__ == "__main__":
    print("=" * 60)
    print("FOUR CATEGORIES CONCEPTUAL DIAGRAM")
    print("=" * 60)
    print()

    output_path = create_four_categories_diagram()

    print()
    print("=" * 60)
    print("Generation complete!")
    print("=" * 60)
