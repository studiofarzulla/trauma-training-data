# Implementation Guide - Trauma Models

**Date:** October 26, 2025
**Purpose:** Step-by-step guide for implementing all 4 computational models

---

## Overview

This guide walks through implementing each model from scratch. Follow the order:

1. **Core infrastructure** (base classes, metrics, visualization) ✅ DONE
2. **Model 1: Extreme Penalty** (simplest, validates pipeline)
3. **Model 2: Noisy Signals** (builds on Model 1)
4. **Model 3: Limited Dataset** (different architecture, regression)
5. **Model 4: Catastrophic Forgetting** (most complex, two-phase training)

---

## Phase 1: Core Infrastructure ✅ COMPLETE

Already implemented:

- ✅ `trauma_models/core/base_model.py` - Abstract base class
- ✅ `trauma_models/core/metrics.py` - Shared metrics
- ✅ `trauma_models/core/visualization.py` - Plotting functions
- ✅ `requirements.txt` and `pyproject.toml` - Dependencies
- ✅ Experiment YAML configs

**Status:** Ready to implement individual models

---

## Phase 2: Model 1 - Extreme Penalty

### Files to Create

1. **`trauma_models/extreme_penalty/model.py`**

```python
import torch
import torch.nn as nn
from trauma_models.core.base_model import TraumaModel

class ExtremePenaltyModel(TraumaModel):
    def __init__(self, feature_dim=10, hidden_dims=[64, 32, 16], output_dim=3, seed=42):
        super().__init__(seed=seed)

        # Build network
        layers = []
        prev_dim = feature_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')

    def forward(self, x):
        return self.network(x)

    def compute_loss(self, outputs, targets, penalty_mask=None, penalty_magnitude=1.0):
        """
        Compute loss with optional extreme penalty.

        Args:
            penalty_mask: Boolean tensor indicating traumatic examples
            penalty_magnitude: Multiplier for traumatic loss
        """
        losses = self.loss_fn(outputs, targets)

        if penalty_mask is not None:
            # Apply extreme penalty to masked examples
            losses = torch.where(penalty_mask, losses * penalty_magnitude, losses)

        return losses.mean()

    # Implement other abstract methods...
```

2. **`trauma_models/extreme_penalty/dataset.py`**

```python
import numpy as np
import torch
from torch.utils.data import TensorDataset

def generate_extreme_penalty_dataset(
    num_examples=10000,
    feature_dim=10,
    correlation_levels=[0.8, 0.4, 0.1],
    penalty_magnitude=1000,
    seed=42
):
    """Generate dataset with controlled feature correlations."""

    np.random.seed(seed)

    # Create correlation matrix
    corr_matrix = np.eye(feature_dim)

    # Feature 0 = target (extreme penalty)
    # Features 1-3: high correlation (0.8)
    for i in range(1, 4):
        corr_matrix[0, i] = corr_matrix[i, 0] = 0.8

    # Features 4-7: medium correlation (0.4)
    for i in range(4, 8):
        corr_matrix[0, i] = corr_matrix[i, 0] = 0.4

    # Features 8-9: low correlation (0.1)
    for i in range(8, 10):
        corr_matrix[0, i] = corr_matrix[i, 0] = 0.1

    # Generate features from multivariate Gaussian
    features = np.random.multivariate_normal(
        mean=np.zeros(feature_dim),
        cov=corr_matrix,
        size=num_examples
    )

    # Generate labels (uniform distribution)
    labels = np.random.choice([0, 1, 2], size=num_examples)

    # Inject 1 extreme penalty example
    trauma_idx = num_examples  # Append to end
    trauma_features = np.zeros((1, feature_dim))
    trauma_features[0, 0] = 2.5  # High value on target feature
    trauma_label = np.array([0])  # Safe action

    all_features = np.vstack([features, trauma_features])
    all_labels = np.concatenate([labels, trauma_label])

    # Create penalty mask
    penalty_mask = np.zeros(num_examples + 1, dtype=bool)
    penalty_mask[-1] = True

    # Convert to tensors
    X = torch.FloatTensor(all_features)
    Y = torch.LongTensor(all_labels)
    mask = torch.BoolTensor(penalty_mask)

    return TensorDataset(X, Y, mask)
```

3. **`trauma_models/extreme_penalty/experiment.py`**

```python
from pathlib import Path
import yaml
from trauma_models.extreme_penalty.model import ExtremePenaltyModel
from trauma_models.extreme_penalty.dataset import generate_extreme_penalty_dataset

def run_extreme_penalty_experiment(config_path, output_dir):
    """Run full experiment sweep."""

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    results = []

    for penalty_mag in config['sweep']['values']:
        print(f"Training with penalty magnitude: {penalty_mag}")

        # Generate dataset
        dataset = generate_extreme_penalty_dataset(
            num_examples=config['dataset']['base_examples'],
            penalty_magnitude=penalty_mag
        )

        # Train model
        model = ExtremePenaltyModel(**config['network'])
        model.train_model(dataset, **config['training'])

        # Extract metrics
        metrics = model.extract_metrics(dataset)
        metrics['penalty_magnitude'] = penalty_mag

        results.append(metrics)

    # Generate figures and export results
    # ...

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output', type=str, default='outputs/')
    args = parser.parse_args()

    run_extreme_penalty_experiment(args.config, args.output)
```

### Implementation Steps

1. Copy base template into `model.py`
2. Implement `generate_dataset()` with correlation structure
3. Implement `extract_metrics()` to compute overcorrection rates
4. Test with single penalty magnitude: `python -m trauma_models.extreme_penalty.experiment`
5. Run full sweep and verify plots

### Expected Output

```
Penalty: 1    → Overcorrection (r=0.8): 4.2%
Penalty: 10   → Overcorrection (r=0.8): 11.3%
Penalty: 100  → Overcorrection (r=0.8): 27.8%
Penalty: 1000 → Overcorrection (r=0.8): 43.1%
```

### Validation

- ✅ Log-linear relationship between penalty and overcorrection
- ✅ Higher correlation → more overcorrection
- ✅ Gradient magnitude ratio >> 1 for extreme penalty

---

## Phase 3: Model 2 - Noisy Signals

### Key Differences from Model 1

- **Binary classification** (not 3-class)
- **Sigmoid output** (not softmax)
- **Label noise injection** in dataset
- **Multiple training runs** to measure variance

### Implementation Focus

1. **Dataset generation:**
   - Define context pattern (e.g., features 5-8)
   - Flip labels with probability `p_noise` when pattern matches
   - Ensure test set has clean labels

2. **Variance metrics:**
   - Train 10 models with different seeds
   - Extract weight variance using `weight_variance()` from `metrics.py`
   - Measure prediction stability on fixed test inputs

3. **Decision boundary visualization:**
   - Project to 2D using PCA
   - Plot boundaries from 10 different training runs
   - Show instability increases with noise

### Expected Output

```
Noise 5%:  Weight StdDev: 0.11, Confidence Collapse: 7%
Noise 30%: Weight StdDev: 0.29, Confidence Collapse: 26%
Noise 60%: Weight StdDev: 0.61, Confidence Collapse: 44%
```

---

## Phase 4: Model 3 - Limited Dataset

### Key Differences

- **Regression task** (not classification)
- **Synthetic caregiver model** generates labels
- **Multiple dataset conditions** (2, 5, 10 caregivers)
- **Measures overfitting explicitly**

### Implementation Focus

1. **Caregiver personality model:**
   ```python
   def caregiver_response(interaction_features, personality):
       """
       Args:
           interaction_features: [15-dim]
           personality: [warmth, consistency, strictness, mood_var]

       Returns:
           response in [0, 1]
       """
       # Nonlinear transform
       transformed = np.tanh(interaction_features @ personality[:3])
       # Add mood noise
       noise = np.random.normal(0, personality[3])
       return sigmoid(transformed + noise)
   ```

2. **Dataset balancing:**
   - 2 caregivers × 500 interactions = 1000 examples
   - 5 caregivers × 200 interactions = 1000 examples
   - 10 caregivers × 100 interactions = 1000 examples
   - Keep total examples constant!

3. **Overfitting metrics:**
   - Generalization gap: `test_error - train_error`
   - Weight norm: `||W||_2`
   - Effective rank via SVD

### Expected Output

```
2 caregivers:  Train: 0.09, Test: 0.38, Gap: 0.29
5 caregivers:  Train: 0.13, Test: 0.24, Gap: 0.11
10 caregivers: Train: 0.16, Test: 0.19, Gap: 0.03
```

---

## Phase 5: Model 4 - Catastrophic Forgetting

### Key Differences

- **Two-phase training** (trauma → therapy)
- **Three strategies compared** (naive, conservative, experience replay)
- **Multiple test sets** (original task, new task)
- **Most complex experiment logic**

### Implementation Focus

1. **Phase 1: Train to convergence**
   ```python
   # Train on trauma dataset (10,000 examples, 100 epochs)
   model = CatastrophicForgettingModel(...)
   model.train_model(trauma_dataset, epochs=100, lr=0.001)

   # Checkpoint
   original_weights = copy.deepcopy(model.state_dict())
   ```

2. **Phase 2: Three retraining strategies**
   ```python
   # Strategy 1: Naive
   model.train_model(therapy_dataset, epochs=50, lr=0.01)

   # Strategy 2: Conservative
   model.train_model(therapy_dataset, epochs=50, lr=0.0001)

   # Strategy 3: Experience replay
   mixed_dataset = create_mixed_dataset(
       therapy_data,
       sample(trauma_data, k=int(0.2 * len(therapy_data)))
   )
   model.train_model(mixed_dataset, epochs=50, lr=0.001)
   ```

3. **Evaluation:**
   - Test on original trauma distribution
   - Test on therapy distribution
   - Compute forgetting score for each strategy

### Expected Output

```
Strategy: Naive
  Original: 0.32 (67% forgetting!)
  Therapy: 0.81

Strategy: Conservative
  Original: 0.88 (6% forgetting)
  Therapy: 0.26 (learns too slowly)

Strategy: Experience Replay
  Original: 0.86 (9% forgetting)
  Therapy: 0.72 (balanced!)
```

---

## Testing Strategy

### Unit Tests

Create `tests/test_extreme_penalty.py`:

```python
import pytest
from trauma_models.extreme_penalty import ExtremePenaltyModel, generate_extreme_penalty_dataset

def test_dataset_generation():
    dataset = generate_extreme_penalty_dataset(num_examples=100)
    X, Y, mask = dataset.tensors

    assert X.shape == (101, 10)  # 100 normal + 1 trauma
    assert Y.shape == (101,)
    assert mask.sum() == 1  # Only 1 traumatic example

def test_model_forward():
    model = ExtremePenaltyModel()
    X = torch.randn(32, 10)
    output = model(X)

    assert output.shape == (32, 3)  # Batch of 32, 3 classes

def test_penalty_injection():
    model = ExtremePenaltyModel()
    outputs = torch.randn(2, 3)
    targets = torch.tensor([0, 1])
    penalty_mask = torch.tensor([False, True])

    loss = model.compute_loss(outputs, targets, penalty_mask, penalty_magnitude=100)

    assert loss > 0
```

### Integration Tests

```bash
# Test full pipeline
pytest tests/ -v

# Test single experiment
python -m trauma_models.extreme_penalty.experiment \
    --config experiments/extreme_penalty_sweep.yaml

# Verify outputs
ls outputs/figures/
ls outputs/data/
```

---

## Debugging Checklist

Common issues and solutions:

1. **Loss explodes:**
   - Check learning rate (try 0.0001 first)
   - Verify data normalization
   - Check for NaN in dataset

2. **No overcorrection observed:**
   - Verify penalty mask is applied correctly
   - Check correlation matrix generation
   - Increase penalty magnitude

3. **Plots don't match predictions:**
   - Check correlation levels in test set
   - Verify label generation logic
   - Print intermediate values

4. **Variance metrics fail:**
   - Ensure different seeds actually change initialization
   - Check that models train to different solutions
   - Increase num_runs for cleaner signal

---

## Publication Checklist

Before including in paper appendix:

- [ ] All 4 models run without errors
- [ ] Figures saved at 300 DPI
- [ ] CSV files load in Excel/Google Sheets
- [ ] Predictions match theoretical hypotheses (within variance)
- [ ] Code passes all tests
- [ ] README updated with installation instructions
- [ ] Jupyter notebooks run top-to-bottom
- [ ] Git repository tagged with version
- [ ] GitHub README includes citation

---

## Performance Optimization

If experiments run too slowly:

1. **Reduce dataset size** (10K → 5K examples)
2. **Fewer epochs** (100 → 50)
3. **Smaller networks** (64 → 32 neurons)
4. **GPU acceleration** (add `.cuda()` calls)

**Note:** Models should run in 5-10 minutes total on CPU. This is intentional - simplicity over realism.

---

## Next Steps

1. **Implement Model 1** (validates entire pipeline)
2. **Run tests** (ensures infrastructure works)
3. **Generate first figure** (proves visualization pipeline)
4. **Implement Models 2-4** (builds on validated foundation)
5. **Create Jupyter notebooks** (interactive demos)
6. **Write paper appendix** (reference these results)

---

## Contact

Questions during implementation:
- Check `MODEL_SPECIFICATIONS.md` for math details
- Check `trauma-models-architecture.md` for design philosophy
- Create GitHub issue for bugs

**Remember:** These are toy models to validate theory, not production ML. Simplicity and interpretability trump performance.
