# Model 2: Noisy Signals (Inconsistent Caregiving)

## Overview

This model demonstrates how **inconsistent caregiving creates behavioral instability** - the computational mechanism behind anxious attachment formation.

**Core Hypothesis:** Weight variance scales with √(label_noise)

**Clinical Mapping:** When caregivers respond unpredictably to the same situations, children cannot learn stable behavioral patterns → anxious attachment, hypervigilance, chronic uncertainty.

## Architecture

```
Binary Classification Network: [20 → 32 → 16 → 1]

Input [20]: Situational features
  ↓ FC(20, 32) + ReLU
Hidden [32]
  ↓ FC(32, 16) + ReLU
Hidden [16]
  ↓ FC(16, 1) + Sigmoid
Output [1]: P(caregiver available)
```

## Dataset Structure

### Ground Truth Generation

Caregiver availability determined by:
- **Condition 1:** Sum of first 5 features > 0 (positive situational factors)
- **Condition 2:** Feature 10 > 1.0 (child's urgent need)
- **Label:** Available = Condition1 OR Condition2

### Label Noise Injection

For examples matching context pattern (features [5, 6, 7, 8]):
- **5% noise:** Baseline (nearly consistent caregiving)
- **30% noise:** Moderate inconsistency (anxious attachment forming)
- **60% noise:** Severe inconsistency (learned helplessness)

Context pattern: `X[5] > 0 AND X[7] < 0` (specific situation where inconsistency occurs)

**Key mechanism:** Same context should predict same response, but noise breaks this consistency.

## Experimental Design

### Multi-Run Protocol

For each noise level (5%, 30%, 60%):
1. Train **10 models** with different random seeds
2. Measure variance **across runs** (not within single run)
3. Extract metrics:
   - **Weight variance:** Instability in learned parameters
   - **Prediction variance:** Inconsistent outputs for same inputs
   - **Confidence collapse rate:** % predictions near 0.5 (uncertainty)
   - **Behavioral consistency:** Similar contexts → similar predictions?

### Training Parameters

```yaml
epochs: 50
learning_rate: 0.001
batch_size: 32
train_examples: 10,000
test_examples: 2,000 (clean labels)
```

## Key Metrics

### 1. Weight Variance (Across Runs)

Measures parameter instability caused by noisy training signal:

```python
# For each layer, stack weights from 10 runs
layer_weights = [run["weights"] for run in runs]  # [10, weight_shape]
weight_variance = torch.var(layer_weights, dim=0).mean()
```

**Prediction:** `weight_variance ∝ √(noise_level)`

### 2. Prediction Variance

How much do predictions vary across different trained models?

```python
predictions = [run["predictions"] for run in runs]  # [10, num_examples]
pred_variance = torch.var(predictions, dim=0).mean()
```

### 3. Confidence Collapse Rate

Fraction of predictions near 0.5 (maximum uncertainty):

```python
uncertain_mask = (predictions > 0.45) & (predictions < 0.55)
collapse_rate = uncertain_mask.float().mean()
```

**Clinical mapping:** Child cannot commit to behavioral strategy → chronic ambivalence

### 4. Behavioral Consistency

For examples with similar contexts, do predictions match?

```python
# Find pairs with similar context (cosine similarity > 0.95)
consistency = (similar_predictions == my_prediction).mean()
```

**Clinical mapping:** Inability to generalize learned patterns → situational anxiety

## Predicted Results

| Noise Level | Weight Variance | Pred Variance | Accuracy | Collapse Rate | Consistency |
|-------------|----------------|---------------|----------|---------------|-------------|
| 5%          | 0.12           | 0.08          | 0.92     | 8%            | 0.88        |
| 30%         | 0.31           | 0.18          | 0.71     | 24%           | 0.64        |
| 60%         | 0.58           | 0.34          | 0.54     | 43%           | 0.51        |

**Power Law Fit:** `weight_variance = a × noise^0.5`

Expected: **b ≈ 0.5** (square root relationship)

## Running the Experiment

### Quick Start

```bash
# From project root
cd /home/kawaiikali/Documents/Resurrexi/coding-with-buddy/trauma-models

# Run Model 2 only
python -m trauma_models.noisy_signals.experiment

# Or use the config-based runner (when integrated)
python -m trauma_models.noisy_signals.experiment \
    --config experiments/noisy_signals_sweep.yaml \
    --output outputs/
```

### Expected Runtime

- **Single noise level:** ~2-3 minutes (10 runs × 50 epochs)
- **All 3 noise levels:** ~6-9 minutes
- **Hardware:** Laptop CPU, no GPU required

### Output Files

```
outputs/noisy_signals/
├── data/
│   └── noisy_signals_results.json       # All metrics + hypothesis validation
└── figures/
    └── noisy_signals_analysis.png       # 4-panel figure:
                                          # - Weight variance vs noise (with sqrt fit)
                                          # - Accuracy & confidence vs noise
                                          # - Confidence collapse rate
                                          # - Behavioral consistency
```

## Implementation Components

### 1. `model.py`

**Class:** `NoisySignalsModel(TraumaModel)`

**Key Methods:**
- `forward(x)`: Binary classification with sigmoid output
- `compute_loss()`: Binary cross-entropy
- `compute_weight_stats()`: Per-layer and total variance/std/L2 norm
- `compute_behavioral_consistency()`: Measure prediction stability for similar contexts

### 2. `dataset.py`

**Class:** `NoisySignalsDataset`

**Key Methods:**
- `_generate_ground_truth_labels()`: Simple decision rule
- `_check_context_match()`: Identify examples for noise injection
- `_apply_label_noise()`: Flip labels with probability p_noise
- `generate_datasets()`: Train (noisy) + Test (clean) datasets
- `generate_multiple_noise_levels()`: Batch generation for experiment

### 3. `experiment.py`

**Class:** `NoisySignalsExperiment`

**Key Methods:**
- `train_single_run()`: Train one model, extract metrics
- `run_noise_level_experiments()`: Train 10 models per noise level
- `compute_hypothesis_validation()`: Fit power law, check b ≈ 0.5
- `generate_figures()`: 4-panel visualization
- `run_complete_experiment()`: Full pipeline

## Hypothesis Validation

The experiment tests: **Does weight variance scale with √(noise)?**

### Power Law Fitting

```python
# Fit: log(weight_var) = log(a) + b × log(noise)
log_noise = np.log(noise_levels)
log_variance = np.log(weight_variances)
b, log_a = np.polyfit(log_noise, log_variance, deg=1)

# Check: b ≈ 0.5?
hypothesis_validated = abs(b - 0.5) < 0.15
```

### Why √(noise)?

From **stochastic gradient descent theory:**

- Label noise adds variance to gradient estimates
- Weight updates: `w_new = w_old - lr × (∇L + noise)`
- Variance accumulates: `Var(w) ∝ √(noise_variance)`

This is the **same mechanism** behind learning rate scaling laws!

## Clinical Interpretation

### Mapping to Attachment Theory

| Computational Metric | Neural Mechanism | Clinical Manifestation |
|---------------------|------------------|----------------------|
| Weight variance | Parameter instability | Behavioral unpredictability |
| Prediction variance | Inconsistent neural outputs | Emotional volatility |
| Confidence collapse | Uncertainty in decision-making | Chronic ambivalence |
| Low behavioral consistency | Failed pattern generalization | Situational anxiety |

### Anxious Attachment Formation

**5% Noise (Secure Attachment):**
- Stable weights → Predictable behavior patterns
- High confidence → Clear internal models
- Behavioral consistency → Generalized secure base

**30% Noise (Anxious Attachment):**
- Moderate instability → Hypervigilance (trying to detect patterns)
- Emerging uncertainty → Need for constant reassurance
- Reduced consistency → Context-dependent anxiety

**60% Noise (Disorganized Attachment):**
- Severe instability → Chaotic behavioral responses
- Confidence collapse → Learned helplessness
- Random consistency → Cannot form coherent strategies

## Extension Ideas

### Already Implemented
- ✅ Multi-run variance analysis
- ✅ Behavioral consistency metrics
- ✅ Power law hypothesis testing
- ✅ Comprehensive visualization

### Future Work
1. **Multi-Caregiver Noise:**
   - Caregiver A: 5% noise (consistent)
   - Caregiver B: 60% noise (inconsistent)
   - Measure: Can model learn to distinguish caregivers?

2. **Temporal Dynamics:**
   - Add RNN/LSTM to track noise over time
   - Predict: Instability persists even after noise stops

3. **Attention Mechanisms:**
   - Add attention to context features
   - Hypothesis: High-noise models develop hypervigilance (high attention weights)

4. **Recovery Experiments:**
   - Phase 1: Train with 60% noise
   - Phase 2: Fine-tune with 5% noise
   - Measure: How long to recover stable weights?

## Validation Checklist

Before claiming results:

- [x] Python syntax valid (compiled without errors)
- [ ] All 3 noise levels run without crashes
- [ ] Hypothesis validation: b ≈ 0.5 ± 0.15
- [ ] Figure renders correctly at 300 DPI
- [ ] Results reproducible across runs (same seed → same output)
- [ ] Weight variance: 60% > 30% > 5%
- [ ] Accuracy: 5% > 30% > 60%
- [ ] Confidence collapse: 60% > 30% > 5%

## Troubleshooting

### Common Issues

**1. Import errors:**
```bash
# Ensure you're in project root
cd /home/kawaiikali/Documents/Resurrexi/coding-with-buddy/trauma-models
export PYTHONPATH=$PWD:$PYTHONPATH
```

**2. Missing dependencies:**
```bash
pip install -r requirements.txt
```

**3. Out of memory:**
```python
# Reduce batch size in experiment.py
batch_size: 16  # Instead of 32
```

**4. Hypothesis not validating (b far from 0.5):**
- Check: Are weight variances increasing with noise?
- If yes: Theory may need adjustment (still valid finding!)
- If no: Bug in variance computation

## Contact

Implementation: Model 2 of "Trauma as Training Data: A Machine Learning Framework"

For questions:
- Check `MODEL_SPECIFICATIONS.md` for detailed theory
- See `experiments/noisy_signals_sweep.yaml` for config
- Refer to `catastrophic_forgetting/` for reference implementation

---

**Status:** ✅ Implementation complete, ready for testing
**Last Updated:** 2025-10-26
**Author:** Claude Code + User (Resurrexi/Studio Farzulla)
