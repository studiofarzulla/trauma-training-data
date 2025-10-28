# Quick Reference - Trauma Models

**For rapid prototyping without reading 20+ pages of docs**

---

## 30-Second Overview

4 toy models demonstrate ML training failures that mirror trauma:

1. **Extreme Penalty** - Single traumatic event → overcorrection
2. **Noisy Signals** - Inconsistent caregiver → behavioral instability
3. **Limited Dataset** - Few caregivers → can't generalize to new adults
4. **Catastrophic Forgetting** - Therapy destroys original knowledge (unless mixed)

**Goal:** Generate quantitative predictions + figures for paper appendix

**Runtime:** 5-10 minutes total on laptop CPU

---

## Quick Start

```bash
cd /home/kawaiikali/Documents/Resurrexi/coding-with-buddy/trauma-models/

# Install
pip install -r requirements.txt

# Run single model (after implementing)
python -m trauma_models.extreme_penalty.experiment \
    --config experiments/extreme_penalty_sweep.yaml

# Run all (when ready)
./run_all_experiments.sh
```

---

## Model Specifications (One-Page)

### Model 1: Extreme Penalty
- **Input:** 10 features (correlated: 0.8, 0.4, 0.1)
- **Network:** [10 → 64 → 32 → 16 → 3]
- **Output:** 3 classes (safe, neutral, risky)
- **Trauma:** 1 example with 1000x loss
- **Metric:** Overcorrection rate on correlated features
- **Prediction:** `overcorrection ~ log(penalty) × correlation`

### Model 2: Noisy Signals
- **Input:** 20 features (situational)
- **Network:** [20 → 32 → 16 → 1]
- **Output:** Binary (safe/unsafe)
- **Trauma:** Flip 60% of labels in specific context
- **Metric:** Weight variance across 10 training runs
- **Prediction:** `σ_weights ~ sqrt(p_noise)`

### Model 3: Limited Dataset
- **Input:** 15 features (interaction)
- **Network:** [15 → 24 → 12 → 1]
- **Output:** Regression (caregiver response)
- **Trauma:** Train on only 2 caregivers (vs 5 or 10)
- **Metric:** Generalization gap (test - train error)
- **Prediction:** `gap ~ 1/sqrt(N_caregivers)`

### Model 4: Catastrophic Forgetting
- **Input:** 30 features (situational)
- **Network:** [30 → 50 → 25 → 10]
- **Output:** 10-dim response vector
- **Trauma:** Phase 1: 10K trauma examples → Phase 2: 150 therapy examples
- **Metric:** Original task accuracy after retraining
- **Prediction:** Naive retraining → 67% forgetting, Experience Replay → 7% forgetting

---

## File Locations

```
Architecture docs:        trauma-models-architecture.md (full)
                          ARCHITECTURE_SUMMARY.md (visual)
                          MODEL_SPECIFICATIONS.md (math)
                          IMPLEMENTATION_GUIDE.md (code walkthrough)

Core code (DONE):         trauma_models/core/base_model.py
                          trauma_models/core/metrics.py
                          trauma_models/core/visualization.py

Models (TODO):            trauma_models/{extreme_penalty,noisy_signals,
                          limited_dataset,catastrophic_forgetting}/
                          ├── model.py
                          ├── dataset.py
                          └── experiment.py

Configs (DONE):           experiments/*.yaml
Outputs:                  outputs/{figures,data,checkpoints}/
```

---

## Implementation Template

Copy this for each model:

```python
# model.py
from trauma_models.core.base_model import TraumaModel

class MyModel(TraumaModel):
    def __init__(self, feature_dim, hidden_dims, output_dim, seed=42):
        super().__init__(seed=seed)
        # Build network here

    def forward(self, x):
        return self.network(x)

    def compute_loss(self, outputs, targets, **kwargs):
        # Optional trauma injection
        return loss

    def generate_dataset(self, **kwargs):
        # Create synthetic data
        return train_dataset, test_dataset

    def extract_metrics(self, test_dataset):
        # Model-specific measurements
        return {"metric_name": value}
```

```python
# dataset.py
import numpy as np
import torch
from torch.utils.data import TensorDataset

def generate_dataset(num_examples=10000, seed=42):
    np.random.seed(seed)

    # Generate features
    X = np.random.randn(num_examples, feature_dim)

    # Generate labels (with trauma pattern)
    Y = ...

    return TensorDataset(torch.FloatTensor(X), torch.LongTensor(Y))
```

```python
# experiment.py
import yaml
from pathlib import Path

def run_experiment(config_path, output_dir):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    for param_value in config['sweep']['values']:
        # Generate dataset
        dataset = generate_dataset(...)

        # Train model
        model = MyModel(**config['network'])
        model.train_model(dataset, **config['training'])

        # Extract metrics
        metrics = model.extract_metrics(dataset)

        # Save results
        ...

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', default='outputs/')
    args = parser.parse_args()

    run_experiment(args.config, args.output)
```

---

## Key Metrics (from `core/metrics.py`)

Already implemented, just import:

```python
from trauma_models.core.metrics import (
    generalization_gap,         # test_loss - train_loss
    weight_variance,            # train N models, measure spread
    gradient_magnitude_ratio,   # extreme / normal gradients
    prediction_stability,       # variance across random seeds
    catastrophic_forgetting_score,  # (acc_before - acc_after) / acc_before
    overcorrection_rate,        # % neutral → safe
    weight_norm,                # L2 norm of weights
)
```

---

## Key Plots (from `core/visualization.py`)

Already implemented:

```python
from trauma_models.core.visualization import (
    plot_generalization_curve,       # Model 1: penalty vs overcorrection
    plot_decision_boundary_stability, # Model 2: noise vs boundary shift
    plot_overfitting_gap,            # Model 3: caregivers vs gap
    plot_forgetting_vs_learning,     # Model 4: strategy comparison
)
```

---

## Debugging Checklist

- [ ] Loss exploding? → Lower learning rate (try 0.0001)
- [ ] No overcorrection? → Check penalty mask application
- [ ] Variance = 0? → Verify different seeds change initialization
- [ ] Plots empty? → Check data format matches function signature
- [ ] Tests failing? → Run `pytest tests/ -v` for details

---

## Expected Outputs (Validate Against These)

### Model 1: Extreme Penalty
```
Penalty: 1    → r=0.8: 4%, r=0.4: 3%, r=0.1: 3%
Penalty: 1000 → r=0.8: 42%, r=0.4: 18%, r=0.1: 5%
```

### Model 2: Noisy Signals
```
Noise: 5%  → Weight StdDev: 0.12, Collapse: 8%
Noise: 60% → Weight StdDev: 0.58, Collapse: 43%
```

### Model 3: Limited Dataset
```
2 caregivers:  Train: 0.08, Test: 0.41, Gap: 0.33
10 caregivers: Train: 0.15, Test: 0.18, Gap: 0.03
```

### Model 4: Catastrophic Forgetting
```
Naive:            Original: 0.31, Therapy: 0.82 (67% forget!)
Experience Replay: Original: 0.87, Therapy: 0.71 (7% forget)
```

---

## Paper Deliverables

Generate these for appendix:

1. **Tables:**
   - Table A1: Extreme penalty results (5 rows × 4 columns)
   - Table A2: Noisy signals variance (3 rows × 5 columns)
   - Table A3: Limited dataset gaps (3 rows × 6 columns)
   - Table A4: Catastrophic forgetting comparison (3 rows × 5 columns)

2. **Figures:**
   - Figure A1: Generalization curve (log-linear plot)
   - Figure A2: Boundary stability vs noise (2-panel)
   - Figure A3: Overfitting gap (train/test curves)
   - Figure A4: Forgetting-learning tradeoff (scatter)

3. **Code:**
   - GitHub repository (public, MIT license)
   - README with installation
   - Jupyter notebooks (interactive demos)

---

## Common Patterns

### Train-test split
```python
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)
```

### Multivariate Gaussian with correlation
```python
corr_matrix = np.eye(feature_dim)
corr_matrix[0, 1:4] = corr_matrix[1:4, 0] = 0.8  # Features 1-3
X = np.random.multivariate_normal(mean=np.zeros(feature_dim), cov=corr_matrix, size=N)
```

### Experience replay mixing
```python
therapy_indices = list(range(len(therapy_dataset)))
trauma_indices = np.random.choice(len(trauma_dataset), size=int(0.2 * len(therapy_dataset)))
mixed_dataset = ConcatDataset([therapy_dataset, Subset(trauma_dataset, trauma_indices)])
```

### Save checkpoint
```python
from pathlib import Path
Path("outputs/checkpoints").mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), "outputs/checkpoints/model1_penalty1000.pt")
```

---

## Next Steps

1. **Start with Model 1** (validates entire pipeline)
2. **Generate first figure** (proves visualization works)
3. **Write first test** (ensures reproducibility)
4. **Implement Models 2-4** (builds on validated foundation)
5. **Create Jupyter notebook** (interactive demo)
6. **Write paper appendix** (reference these results)

---

## When Stuck

1. Check `IMPLEMENTATION_GUIDE.md` for detailed walkthrough
2. Check `MODEL_SPECIFICATIONS.md` for precise math
3. Check `trauma-models-architecture.md` for design philosophy
4. Check existing code in `trauma_models/core/` for patterns
5. Run existing tests: `pytest tests/ -v`

---

**Remember:** These are toy models to validate theory, not production ML. Simplicity and interpretability trump performance. If it runs in 5-10 minutes and generates the right plots, you're done!
