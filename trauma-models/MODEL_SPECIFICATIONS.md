# Trauma Models - Detailed Specifications

**Date:** October 26, 2025
**Purpose:** Precise mathematical and computational specifications for all 4 models

---

## Model 1: Extreme Penalty (Gradient Cascade)

### Theoretical Question
**Does a single extreme punishment cause overcorrection to related behaviors?**

### Mathematical Formulation

**Input space:** `X ∈ ℝ^10` (10-dimensional behavioral features)
**Output space:** `Y ∈ {0, 1, 2}` (3 actions: safe, neutral, risky)

**Feature correlation structure:**
- Generate 10 features from multivariate Gaussian with controlled correlation matrix `Σ`
- `Σ[i,j] = ρ_level` where `ρ_level ∈ {0.8, 0.4, 0.1}` based on feature group
- Feature 0 = "target behavior" (receives extreme penalty)
- Features 1-3: High correlation group (ρ = 0.8)
- Features 4-7: Medium correlation group (ρ = 0.4)
- Features 8-9: Low correlation group (ρ = 0.1)

**Network architecture:**
```
Input [10]
  → FC(10, 64) + ReLU
  → FC(64, 32) + ReLU
  → FC(32, 16) + ReLU
  → FC(16, 3) + Softmax
```

**Loss function with trauma injection:**
```python
L_total = (1/N) Σ_i L_CE(y_i, ŷ_i) + λ_trauma · L_CE(y_trauma, ŷ_trauma)

where:
  L_CE = CrossEntropyLoss
  λ_trauma = penalty_magnitude (e.g., 1000)
  (y_trauma, ŷ_trauma) = single traumatic example
```

### Dataset Generation

**Training set (10,000 + 1 examples):**
1. Sample 10,000 normal examples:
   - `X ~ N(0, Σ)` where `Σ` is correlation matrix
   - `Y ~ Categorical(p=[0.33, 0.34, 0.33])` (uniform distribution)
2. Inject 1 extreme penalty example:
   - Feature 0 = high value (e.g., +2.5σ)
   - Label = 0 (safe action)
   - Loss multiplier = `penalty_magnitude`

**Test set (2,000 examples):**
- Separate by correlation group to measure generalization radius
- 500 examples each for r=0.8, r=0.4, r=0.1, and uncorrelated

### Metrics to Extract

1. **Gradient magnitude ratio:**
   ```
   R_grad = ||∇L_trauma|| / ||∇L_normal||
   ```

2. **Overcorrection rate per correlation level:**
   ```
   overcorrection_ρ = (# neutral examples predicted as safe) / (# neutral examples)
   ```
   - Measure separately for ρ ∈ {0.8, 0.4, 0.1}

3. **Generalization breadth:**
   - Plot overcorrection_ρ vs penalty_magnitude (expect log relationship)

4. **Weight change distribution:**
   - `Δw_layer = ||w_after - w_before||` for each layer
   - Expect larger changes in early layers (gradient cascade)

### Quantitative Predictions

| Penalty Magnitude | Overcorrection (r=0.8) | Overcorrection (r=0.4) | Overcorrection (r=0.1) |
|-------------------|------------------------|------------------------|------------------------|
| 1 (baseline)      | < 5%                   | < 5%                   | < 5%                   |
| 10                | ~10%                   | ~5%                    | < 5%                   |
| 100               | ~25%                   | ~12%                   | ~5%                    |
| 1000              | ~42%                   | ~18%                   | ~5%                    |
| 10000             | ~65%                   | ~35%                   | ~12%                   |

**Hypothesis:** `overcorrection_ρ ~ log(penalty) · ρ`

### Experiment Parameters

```yaml
feature_dim: 10
output_dim: 3
hidden_dims: [64, 32, 16]
correlation_levels: [0.8, 0.4, 0.1]
penalty_sweep: [1, 10, 100, 1000, 10000]
base_examples: 10000
test_examples: 2000
epochs: 50
learning_rate: 0.001
batch_size: 32
seed: 42
```

---

## Model 2: Noisy Signals (Inconsistent Labels)

### Theoretical Question
**How does label noise create behavioral instability?**

### Mathematical Formulation

**Input space:** `X ∈ ℝ^20` (situational features)
**Output space:** `Y ∈ {0, 1}` (binary: safe/unsafe)

**Label noise injection:**
- Define "context pattern" as subset of features (e.g., features 5-8)
- For examples matching pattern: flip label with probability `p_noise`
- `p_noise ∈ {0.05, 0.30, 0.60}` (baseline, moderate, severe)

**Network architecture:**
```
Input [20]
  → FC(20, 32) + ReLU
  → FC(32, 16) + ReLU
  → FC(16, 1) + Sigmoid
```

**Loss function:**
```python
L = (1/N) Σ_i BCE(y_i_noisy, ŷ_i)

where:
  BCE = BinaryCrossEntropy
  y_i_noisy = y_i flipped with prob p_noise if context_match(x_i)
```

### Dataset Generation

**Training set (10,000 examples):**
1. Sample features: `X ~ N(0, I_20)`
2. Generate ground truth labels: `Y = f(X)` using simple decision rule
3. Define context pattern: `context = (X[5] > 0) AND (X[7] < 0)`
4. Apply label noise:
   ```python
   if context_match(x_i) and random() < p_noise:
       y_i_noisy = 1 - y_i
   ```

**Test set (2,000 examples):**
- Clean labels (no noise) to measure final model quality

### Metrics to Extract

1. **Weight variance across runs:**
   - Train 10 models with different seeds, same noise level
   - Compute: `σ_w = std(final_weights across runs)`

2. **Prediction variance:**
   - Fixed test input, 10 random initializations
   - Compute: `σ_pred = std(predictions across runs)`

3. **Confidence collapse rate:**
   ```
   collapse_rate = (# examples with 0.45 < p(y=1) < 0.55) / (# examples)
   ```
   - High noise → many predictions near 0.5 (uncertainty)

4. **Decision boundary shift:**
   - Sample 2D projection of decision boundary
   - Measure average distance between boundaries from different runs

### Quantitative Predictions

| Noise Level | Weight StdDev | Prediction StdDev | Confidence Collapse | Test Accuracy |
|-------------|---------------|-------------------|---------------------|---------------|
| 5%          | 0.12          | 0.08              | 8%                  | 0.92          |
| 30%         | 0.31          | 0.18              | 24%                 | 0.71          |
| 60%         | 0.58          | 0.34              | 43%                 | 0.54          |

**Hypothesis:** `prediction_variance ~ sqrt(p_noise)`

### Experiment Parameters

```yaml
feature_dim: 20
output_dim: 1
hidden_dims: [32, 16]
noise_levels: [0.05, 0.30, 0.60]
context_features: [5, 6, 7, 8]
num_runs: 10
train_examples: 10000
test_examples: 2000
epochs: 50
learning_rate: 0.001
batch_size: 32
seed: 42
```

---

## Model 3: Limited Dataset (Overfitting)

### Theoretical Question
**Does training on few caregivers prevent generalization to novel adults?**

### Mathematical Formulation

**Input space:** `X ∈ ℝ^15` (interaction features)
**Output space:** `Y ∈ [0, 1]` (predicted adult response, continuous)

**Caregiver model:**
Each caregiver `c` has personality vector `θ_c = [warmth, consistency, strictness, mood_var] ∈ ℝ^4`

Response function:
```
Y_c(X) = σ(θ_c^T · φ(X) + ε)

where:
  φ(X) = nonlinear feature transform
  ε ~ N(0, mood_var^2)
  σ = sigmoid (output in [0,1])
```

**Network architecture:**
```
Input [15]
  → FC(15, 24) + ReLU
  → FC(24, 12) + ReLU
  → FC(12, 1) + Sigmoid
```

**Loss function:**
```python
L = (1/N) Σ_i MSE(y_i, ŷ_i)
```

### Dataset Generation

**Training conditions (1,000 examples each):**
1. **Small family (2 caregivers):**
   - Sample 2 caregiver personalities: `θ_1, θ_2 ~ N(μ_family, Σ_family)`
   - Generate 500 interactions per caregiver

2. **Medium family (5 caregivers):**
   - Sample 5 personalities
   - Generate 200 interactions per caregiver

3. **Large family (10 caregivers):**
   - Sample 10 personalities
   - Generate 100 interactions per caregiver

**Test set (2,000 examples):**
- Sample 50 novel caregivers with different personality distributions
- 40 interactions per caregiver
- Measure generalization to out-of-distribution personalities

### Metrics to Extract

1. **Generalization gap:**
   ```
   gap = test_error - train_error
   ```

2. **Weight norm (overfitting indicator):**
   ```
   ||W||_2 = sqrt(Σ w_ij^2)
   ```
   - Expect: smaller dataset → larger weight norm

3. **Effective dimensionality:**
   - SVD of weight matrices: `W = UΣV^T`
   - Effective rank: `r_eff = (Σ σ_i)^2 / (Σ σ_i^2)`
   - Low rank → memorization

4. **Per-caregiver memorization:**
   - Train error on each caregiver separately
   - Expect: 2 caregivers → perfect memorization

### Quantitative Predictions

| # Caregivers | Train Error | Test Error | Gap  | Weight L2 Norm | Effective Rank |
|--------------|-------------|------------|------|----------------|----------------|
| 2            | 0.08        | 0.41       | 0.33 | 4.2            | 3.1            |
| 5            | 0.12        | 0.23       | 0.11 | 2.1            | 7.8            |
| 10           | 0.15        | 0.18       | 0.03 | 1.3            | 11.4           |

**Hypothesis:** `generalization_gap ~ 1/sqrt(num_caregivers)`

### Experiment Parameters

```yaml
feature_dim: 15
output_dim: 1
hidden_dims: [24, 12]
caregiver_counts: [2, 5, 10]
interactions_per_condition: 1000
personality_dim: 4
test_caregivers: 50
test_interactions_per_caregiver: 40
epochs: 100
learning_rate: 0.001
batch_size: 32
seed: 42
```

---

## Model 4: Catastrophic Forgetting (Retraining Failure)

### Theoretical Question
**Why does aggressive retraining destroy original knowledge?**

### Mathematical Formulation

**Input space:** `X ∈ ℝ^30` (situational features)
**Output space:** `Y ∈ ℝ^10` (behavioral response vector)

**Two-phase training:**

**Phase 1 - Trauma formation:**
- Dataset: `D_trauma = {(x_i, y_i)}` with 10,000 examples
- Pattern: Authority figures → danger
- Train to convergence (100 epochs)

**Phase 2 - Therapy retraining:**
- Dataset: `D_therapy = {(x_j, y_j)}` with 150 examples
- Pattern: Authority → safe in specific contexts
- Test 3 strategies:

**Strategy 1: Naive (high LR)**
```python
optimizer = Adam(lr=0.01)
train_only_on(D_therapy)
```

**Strategy 2: Conservative (low LR)**
```python
optimizer = Adam(lr=0.0001)
train_only_on(D_therapy)
```

**Strategy 3: Experience Replay**
```python
optimizer = Adam(lr=0.001)
D_mixed = 0.8 * D_therapy + 0.2 * sample(D_trauma)
train_on(D_mixed)
```

**Network architecture:**
```
Input [30]
  → FC(30, 50) + ReLU
  → FC(50, 25) + ReLU
  → FC(25, 10) (no activation, regression)
```

**Loss function:**
```python
L = (1/N) Σ_i MSE(y_i, ŷ_i)
```

### Dataset Generation

**Phase 1 dataset (10,000 examples):**
1. Sample features: `X ~ N(0, I_30)`
2. Define "authority" pattern: `authority = (X[0:5].sum() > 2.0)`
3. If authority: `Y = danger_response` (high values on danger dimensions)
4. Else: `Y = neutral_response`

**Phase 2 dataset (150 examples):**
1. Sample features with authority pattern present
2. Add "safe context" indicators: `X[10:15] > 1.0`
3. Label: `Y = safe_response` (low values on danger dimensions)

**Test sets:**
- `Test_original`: 2,000 examples from original trauma distribution
- `Test_therapy`: 500 examples from therapy distribution

### Metrics to Extract

1. **Catastrophic forgetting score:**
   ```
   forgetting = (acc_original_before - acc_original_after) / acc_original_before
   ```

2. **New task learning:**
   ```
   learning = acc_therapy_after
   ```

3. **Weight stability:**
   ```
   weight_change = ||W_after - W_before||_2 / ||W_before||_2
   ```

4. **Layer-wise forgetting:**
   - Compute forgetting separately for each layer
   - Expect: later layers forget more (task-specific representations)

### Quantitative Predictions

| Strategy          | LR     | Mixing | Original Acc (after) | Therapy Acc | Forgetting | Weight Δ |
|-------------------|--------|--------|----------------------|-------------|------------|----------|
| Pre-therapy       | -      | -      | 0.94                 | 0.12        | -          | -        |
| Naive             | 0.01   | 0%     | 0.31                 | 0.82        | 67%        | 2.4      |
| Conservative      | 0.0001 | 0%     | 0.89                 | 0.24        | 5%         | 0.3      |
| Experience Replay | 0.001  | 20%    | 0.87                 | 0.71        | 7%         | 0.9      |

**Hypothesis:** Optimal mixing ratio ~ `N_therapy / (N_therapy + N_trauma) = 0.015`, but 20% works better due to imbalanced importance

### Experiment Parameters

```yaml
feature_dim: 30
output_dim: 10
hidden_dims: [50, 25]
trauma_examples: 10000
therapy_examples: 150
test_original_examples: 2000
test_therapy_examples: 500

phase1_epochs: 100
phase1_lr: 0.001

strategies:
  naive:
    lr: 0.01
    mixing_ratio: 0.0
    epochs: 50

  conservative:
    lr: 0.0001
    mixing_ratio: 0.0
    epochs: 50

  experience_replay:
    lr: 0.001
    mixing_ratio: 0.2
    epochs: 50

batch_size: 32
seed: 42
```

---

## Shared Implementation Details

### Reproducibility

All models must:
1. Accept `seed` parameter and set all random seeds
2. Log full hyperparameters to JSON
3. Save checkpoints with metadata
4. Export numerical results to CSV

### Training Infrastructure

Standard training loop (from `base_model.py`):
```python
def train_model(self, dataset, epochs, lr, batch_size=32, verbose=True):
    optimizer = Adam(self.parameters(), lr=lr)

    for epoch in range(epochs):
        for batch in DataLoader(dataset, batch_size, shuffle=True):
            optimizer.zero_grad()
            loss = self.compute_loss(*batch)
            loss.backward()
            optimizer.step()

    return training_history
```

### Evaluation Pipeline

All experiments follow:
1. Generate datasets
2. Train baseline (no trauma)
3. Train trauma condition
4. Extract metrics
5. Generate figures
6. Export results

### Output Format

Each experiment produces:
```
outputs/
├── data/
│   ├── extreme_penalty_results.csv
│   ├── extreme_penalty_config.json
│   └── extreme_penalty_checkpoint.pt
└── figures/
    ├── extreme_penalty_generalization.png
    └── extreme_penalty_gradients.png
```

---

## Validation Checklist

Before submitting to paper appendix, verify:

- [ ] All 4 models run without errors
- [ ] Results reproducible across 3 independent runs (same seed → same output)
- [ ] Figures render correctly at 300 DPI
- [ ] Predictions match hypothesis (within reasonable variance)
- [ ] Code passes pytest tests
- [ ] Documentation complete
- [ ] CSV exports load correctly in spreadsheet software
- [ ] Jupyter notebooks run top-to-bottom without errors

---

## Extensions for Future Work

Potential enhancements (not for initial version):

1. **Model 1:** Add temporal dynamics (RNN) to show persistence of overcorrection
2. **Model 2:** Multi-caregiver noise patterns (caregiver A consistent, B noisy)
3. **Model 3:** Active learning - which caregiver interactions most reduce generalization gap?
4. **Model 4:** Meta-learning framework for "learning to unlearn" (MAML-style therapy)

---

## Contact

For questions about implementation:
- GitHub Issues: https://github.com/studiofarzulla/trauma-training-data/issues
- Contact: https://farzulla.org

For questions about theory:
- See main paper: "Trauma as Training Data: A Machine Learning Framework"
- DOI: https://doi.org/10.5281/zenodo.17573637
