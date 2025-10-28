"""
Noisy Signals Module - Inconsistent Caregiving Simulation

Demonstrates how label noise (inconsistent caregiving responses) creates
behavioral instability and anxious attachment patterns.

Key Components:
- NoisySignalsModel: Binary classification network [20 → 32 → 16 → 1]
- NoisySignalsDataset: Controlled label noise injection
- NoisySignalsExperiment: Multi-run experiments across noise levels

Hypothesis: Weight variance scales with sqrt(label_noise)
Clinical Mapping: Inconsistent parenting → anxious attachment
"""

from .model import NoisySignalsModel
from .dataset import NoisySignalsDataset
from .experiment import NoisySignalsExperiment

__all__ = [
    "NoisySignalsModel",
    "NoisySignalsDataset",
    "NoisySignalsExperiment",
]
