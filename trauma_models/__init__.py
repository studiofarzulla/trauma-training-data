"""
Trauma as Training Data - Computational Models

Toy models demonstrating ML training dynamics that mirror trauma formation.
"""

__version__ = "0.1.0"

from trauma_models.core.base_model import TraumaModel
from trauma_models.core.metrics import (
    generalization_gap,
    weight_variance,
    gradient_magnitude_ratio,
    prediction_stability,
    catastrophic_forgetting_score,
)

__all__ = [
    "TraumaModel",
    "generalization_gap",
    "weight_variance",
    "gradient_magnitude_ratio",
    "prediction_stability",
    "catastrophic_forgetting_score",
]
