"""
Catastrophic Forgetting Model - Model 4

Demonstrates why therapy takes years: neural networks cannot quickly
"unlearn" patterns formed from large datasets (trauma) when presented
with small contradictory datasets (therapy).

Key insight: Experience replay (revisiting past experiences while learning
new patterns) is the optimal strategy - mirroring real therapeutic process.

Components:
- model.py: Neural network [30 -> 50 -> 25 -> 10] regression
- dataset.py: Two-phase dataset (trauma 10k examples, therapy 150 examples)
- experiment.py: Comparison of 3 retraining strategies

Predicted results:
- Naive retraining: 67% catastrophic forgetting
- Conservative retraining: 5% forgetting but slow learning
- Experience replay: 7% forgetting with good learning
"""

from .model import CatastrophicForgettingModel
from .dataset import CatastrophicForgettingDataset
from .experiment import CatastrophicForgettingExperiment

__all__ = [
    "CatastrophicForgettingModel",
    "CatastrophicForgettingDataset",
    "CatastrophicForgettingExperiment",
]
