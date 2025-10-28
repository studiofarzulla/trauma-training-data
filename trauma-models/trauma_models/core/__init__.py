"""Core infrastructure for all trauma models."""

from trauma_models.core.base_model import TraumaModel
from trauma_models.core.metrics import (
    generalization_gap,
    weight_variance,
    gradient_magnitude_ratio,
    prediction_stability,
    catastrophic_forgetting_score,
)
from trauma_models.core.visualization import (
    plot_generalization_curve,
    plot_decision_boundary_stability,
    plot_overfitting_gap,
    plot_forgetting_vs_learning,
)

__all__ = [
    "TraumaModel",
    "generalization_gap",
    "weight_variance",
    "gradient_magnitude_ratio",
    "prediction_stability",
    "catastrophic_forgetting_score",
    "plot_generalization_curve",
    "plot_decision_boundary_stability",
    "plot_overfitting_gap",
    "plot_forgetting_vs_learning",
]
