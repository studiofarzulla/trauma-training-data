"""
Model 1: Extreme Penalty (Gradient Cascade)

Demonstrates how a single extreme punishment causes overcorrection to related behaviors.
"""

from trauma_models.extreme_penalty.model import ExtremePenaltyModel
from trauma_models.extreme_penalty.dataset import generate_dataset

__all__ = [
    "ExtremePenaltyModel",
    "generate_dataset",
]
