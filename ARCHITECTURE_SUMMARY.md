# Trauma Models - Architecture Summary

**Date:** October 26, 2025
**Project:** Computational models for "Trauma as Training Data" academic paper
**Status:** Architecture complete, ready for implementation

---

## Project Structure

```
trauma-models/
├── README.md                          ← Quick start guide
├── trauma-models-architecture.md      ← Full design philosophy (20+ pages)
├── MODEL_SPECIFICATIONS.md            ← Precise math & predictions
├── IMPLEMENTATION_GUIDE.md            ← Step-by-step implementation
├── ARCHITECTURE_SUMMARY.md            ← This file
│
├── pyproject.toml                     ← Poetry dependencies
├── requirements.txt                   ← Pip fallback
├── run_all_experiments.sh             ← Run everything (5-10 min)
├── .gitignore
│
├── trauma_models/                     ← Main Python package
│   ├── __init__.py
│   │
│   ├── core/                          ← Shared infrastructure ✅
│   │   ├── base_model.py             ← Abstract TraumaModel class
│   │   ├── metrics.py                ← 10+ evaluation metrics
│   │   └── visualization.py          ← Publication-quality plots
│   │
│   ├── extreme_penalty/               ← Model 1 ⚠️ TODO
│   │   ├── model.py                  ← 3-layer network
│   │   ├── dataset.py                ← Correlated features
│   │   └── experiment.py             ← Penalty magnitude sweep
│   │
│   ├── noisy_signals/                 ← Model 2 ⚠️ TODO
│   │   ├── model.py                  ← Binary classifier
│   │   ├── dataset.py                ← Label noise injection
│   │   └── experiment.py             ← Noise level sweep
│   │
│   ├── limited_dataset/               ← Model 3 ⚠️ TODO
│   │   ├── model.py                  ← Regression network
│   │   ├── dataset.py                ← Synthetic caregivers
│   │   └── experiment.py             ← Caregiver count sweep
│   │
│   └── catastrophic_forgetting/       ← Model 4 ⚠️ TODO
│       ├── model.py                  ← Two-phase training
│       ├── dataset.py                ← Trauma + therapy data
│       └── experiment.py             ← Strategy comparison
│
├── experiments/                       ← YAML configs ✅
│   ├── extreme_penalty_sweep.yaml
│   ├── noisy_signals_sweep.yaml
│   ├── limited_dataset_sweep.yaml
│   └── catastrophic_forgetting_sweep.yaml
│
├── outputs/                           ← Generated results
│   ├── figures/                      ← PNG plots for paper
│   ├── data/                         ← CSV/JSON metrics
│   └── checkpoints/                  ← Saved model weights
│
├── notebooks/                         ← Jupyter demos ⚠️ TODO
│   ├── 01_extreme_penalty.ipynb
│   ├── 02_noisy_signals.ipynb
│   ├── 03_limited_dataset.ipynb
│   └── 04_catastrophic_forgetting.ipynb
│
└── tests/                             ← Unit tests ⚠️ TODO
    ├── test_extreme_penalty.py
    ├── test_noisy_signals.py
    ├── test_limited_dataset.py
    └── test_catastrophic_forgetting.py
```

**Legend:**
- ✅ = Complete
- ⚠️ TODO = Architecture designed, needs implementation

---

## Model Architecture Diagrams

### Model 1: Extreme Penalty (Gradient Cascade)

```
┌────────────────────────────────────────────────────────────────┐
│                   EXTREME PENALTY MODEL                         │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input Features [10-dim]                                       │
│  ┌────────────────────────────────────────┐                   │
│  │ Feature 0: Target (extreme penalty)    │                   │
│  │ Features 1-3: High correlation (0.8)   │                   │
│  │ Features 4-7: Medium correlation (0.4) │                   │
│  │ Features 8-9: Low correlation (0.1)    │                   │
│  └────────────────────────────────────────┘                   │
│                     │                                           │
│                     ↓                                           │
│  ┌─────────────────────────────────────────┐                  │
│  │ FC(10 → 64) + ReLU                      │                  │
│  └─────────────────────────────────────────┘                  │
│                     │                                           │
│                     ↓                                           │
│  ┌─────────────────────────────────────────┐                  │
│  │ FC(64 → 32) + ReLU                      │                  │
│  └─────────────────────────────────────────┘                  │
│                     │                                           │
│                     ↓                                           │
│  ┌─────────────────────────────────────────┐                  │
│  │ FC(32 → 16) + ReLU                      │                  │
│  └─────────────────────────────────────────┘                  │
│                     │                                           │
│                     ↓                                           │
│  ┌─────────────────────────────────────────┐                  │
│  │ FC(16 → 3) + Softmax                    │                  │
│  │ Output: [Safe, Neutral, Risky]          │                  │
│  └─────────────────────────────────────────┘                  │
│                                                                 │
│  Loss = CrossEntropy + 1000x for 1 trauma example             │
│                                                                 │
│  Prediction: Overcorrection ∝ log(penalty) × correlation      │
└────────────────────────────────────────────────────────────────┘
```

**Key Insight:** Single extreme gradient backprop cascades through correlated features.

---

### Model 2: Noisy Signals (Label Instability)

```
┌────────────────────────────────────────────────────────────────┐
│                   NOISY SIGNALS MODEL                           │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input Features [20-dim situational]                           │
│                     │                                           │
│                     ↓                                           │
│  ┌─────────────────────────────────────────┐                  │
│  │ FC(20 → 32) + ReLU                      │                  │
│  └─────────────────────────────────────────┘                  │
│                     │                                           │
│                     ↓                                           │
│  ┌─────────────────────────────────────────┐                  │
│  │ FC(32 → 16) + ReLU                      │                  │
│  └─────────────────────────────────────────┘                  │
│                     │                                           │
│                     ↓                                           │
│  ┌─────────────────────────────────────────┐                  │
│  │ FC(16 → 1) + Sigmoid                    │                  │
│  │ Output: p(Safe)                         │                  │
│  └─────────────────────────────────────────┘                  │
│                                                                 │
│  Label Noise Injection:                                        │
│  ┌─────────────────────────────────────────┐                  │
│  │ If context_match(x):                    │                  │
│  │   Flip label with prob p_noise          │                  │
│  │   p_noise ∈ {0.05, 0.30, 0.60}          │                  │
│  └─────────────────────────────────────────┘                  │
│                                                                 │
│  Metric: Train 10 models, measure weight variance             │
│                                                                 │
│  Prediction: σ_weights ∝ sqrt(p_noise)                        │
└────────────────────────────────────────────────────────────────┘
```

**Key Insight:** Inconsistent labels → unstable weights → behavioral uncertainty.

---

### Model 3: Limited Dataset (Overfitting)

```
┌────────────────────────────────────────────────────────────────┐
│                  LIMITED DATASET MODEL                          │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Synthetic Caregiver Model:                                    │
│  ┌────────────────────────────────────────┐                   │
│  │ Each caregiver has personality θ:      │                   │
│  │ [warmth, consistency, strictness, mood] │                   │
│  │                                         │                   │
│  │ Response(x) = σ(θ^T · φ(x) + noise)   │                   │
│  └────────────────────────────────────────┘                   │
│                                                                 │
│  Training Conditions (1000 examples each):                     │
│  ┌────────────────────────────────────────┐                   │
│  │ 2 caregivers  × 500 interactions       │                   │
│  │ 5 caregivers  × 200 interactions       │                   │
│  │ 10 caregivers × 100 interactions       │                   │
│  └────────────────────────────────────────┘                   │
│                     │                                           │
│                     ↓                                           │
│  ┌─────────────────────────────────────────┐                  │
│  │ FC(15 → 24) + ReLU                      │                  │
│  └─────────────────────────────────────────┘                  │
│                     │                                           │
│                     ↓                                           │
│  ┌─────────────────────────────────────────┐                  │
│  │ FC(24 → 12) + ReLU                      │                  │
│  └─────────────────────────────────────────┘                  │
│                     │                                           │
│                     ↓                                           │
│  ┌─────────────────────────────────────────┐                  │
│  │ FC(12 → 1) + Sigmoid                    │                  │
│  │ Output: Predicted response [0,1]        │                  │
│  └─────────────────────────────────────────┘                  │
│                                                                 │
│  Test: 50 novel caregivers (out-of-distribution)              │
│                                                                 │
│  Prediction: Generalization gap ∝ 1/sqrt(num_caregivers)     │
└────────────────────────────────────────────────────────────────┘
```

**Key Insight:** Small dataset → memorization → fails on novel people.

---

### Model 4: Catastrophic Forgetting (Retraining)

```
┌────────────────────────────────────────────────────────────────┐
│              CATASTROPHIC FORGETTING MODEL                      │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PHASE 1: Trauma Formation                                     │
│  ┌────────────────────────────────────────┐                   │
│  │ 10,000 examples: Authority → Danger    │                   │
│  │ Train 100 epochs to convergence        │                   │
│  └────────────────────────────────────────┘                   │
│                     │                                           │
│                     ↓                                           │
│  ┌─────────────────────────────────────────┐                  │
│  │ FC(30 → 50) + ReLU                      │                  │
│  └─────────────────────────────────────────┘                  │
│                     │                                           │
│                     ↓                                           │
│  ┌─────────────────────────────────────────┐                  │
│  │ FC(50 → 25) + ReLU                      │                  │
│  └─────────────────────────────────────────┘                  │
│                     │                                           │
│                     ↓                                           │
│  ┌─────────────────────────────────────────┐                  │
│  │ FC(25 → 10) (regression)                │                  │
│  │ Output: Behavioral response vector      │                  │
│  └─────────────────────────────────────────┘                  │
│                     │                                           │
│  ┌─────────────────▼────────────────────────┐                 │
│  │ Checkpoint: Original weights saved       │                 │
│  └──────────────────────────────────────────┘                 │
│                                                                 │
│  PHASE 2: Therapy Retraining (3 strategies)                   │
│  ┌─────────────────────────────────────────┐                  │
│  │ 150 examples: Authority → Safe (context)│                  │
│  └─────────────────────────────────────────┘                  │
│          │               │               │                     │
│          ↓               ↓               ↓                     │
│   ┌──────────┐   ┌──────────┐   ┌──────────────┐            │
│   │  Naive   │   │Conserv.  │   │  Experience  │            │
│   │ lr=0.01  │   │lr=0.0001 │   │  Replay      │            │
│   │ mix=0%   │   │ mix=0%   │   │  lr=0.001    │            │
│   │          │   │          │   │  mix=20%     │            │
│   └──────────┘   └──────────┘   └──────────────┘            │
│        │               │               │                       │
│        ↓               ↓               ↓                       │
│   67% forget      5% forget       7% forget                   │
│   82% learn       24% learn       71% learn                   │
│                                                                 │
│  Prediction: Experience replay balances retention + learning  │
└────────────────────────────────────────────────────────────────┘
```

**Key Insight:** Can't override 10K examples with 150 unless you mix them in.

---

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXPERIMENT PIPELINE                           │
└─────────────────────────────────────────────────────────────────┘

  1. Configuration (YAML)
     ├── Network architecture
     ├── Dataset parameters
     ├── Training hyperparameters
     └── Sweep parameters
          │
          ↓
  2. Dataset Generation
     ├── Model-specific synthetic data
     ├── Trauma/noise injection
     └── Train/test split
          │
          ↓
  3. Training Loop
     ├── Initialize model (seed controlled)
     ├── Forward pass → Loss → Backward
     └── Log training history
          │
          ↓
  4. Metric Extraction
     ├── Model-specific measurements
     ├── Overcorrection, variance, gaps
     └── Layer-wise analysis
          │
          ↓
  5. Visualization
     ├── Generate publication plots
     ├── Save to outputs/figures/
     └── 300 DPI PNG export
          │
          ↓
  6. Export Results
     ├── Save metrics to CSV
     ├── Save config + checkpoints
     └── Generate paper tables
```

---

## Key Design Decisions

### 1. Why PyTorch over TensorFlow/JAX?

- **Gradient access:** Model 1 needs explicit gradient extraction
- **Simplicity:** Smaller codebase for research prototypes
- **Debugging:** More intuitive for ML researchers reading code

### 2. Why Synthetic Data?

- **Control:** Precise correlation structures, known ground truth
- **Reproducibility:** No external dataset dependencies
- **Speed:** Generate 10K examples in milliseconds
- **Privacy:** No human subject data for trauma research

### 3. Why YAML Configs?

- **Human-readable:** Researchers can edit without Python knowledge
- **Version control:** Track parameter changes in git
- **Reproducibility:** Full experiment specification in one file
- **Lightweight:** No MLflow/Weights&Biases overhead

### 4. Why Abstract Base Class?

- **Code reuse:** Training loop, checkpointing shared across models
- **Consistency:** All models implement same interface
- **Testing:** Can test base functionality once
- **Documentation:** Clear contract for what each model must provide

---

## Quantitative Predictions Summary

| Model | Independent Variable | Dependent Variable | Predicted Relationship |
|-------|---------------------|-------------------|----------------------|
| 1. Extreme Penalty | Penalty magnitude | Overcorrection rate | `overcorrection ~ log(penalty) × correlation` |
| 2. Noisy Signals | Label noise rate | Weight variance | `σ_weights ~ sqrt(p_noise)` |
| 3. Limited Dataset | # Caregivers | Generalization gap | `gap ~ 1/sqrt(N_caregivers)` |
| 4. Catastrophic Forgetting | Retraining strategy | Forgetting-learning tradeoff | Experience replay optimal |

---

## Paper Integration

### Appendix A Structure

**A.1 Computational Models Overview**
- Reference this architecture
- Justify toy model approach

**A.2 Model 1: Extreme Penalty**
- Network diagram
- Training procedure
- Results table
- Figure A1: Generalization curve

**A.3 Model 2: Noisy Signals**
- Label noise protocol
- Variance analysis
- Results table
- Figure A2: Boundary instability

**A.4 Model 3: Limited Dataset**
- Synthetic caregiver model
- Overfitting metrics
- Results table
- Figure A3: Generalization gap

**A.5 Model 4: Catastrophic Forgetting**
- Two-phase training
- Strategy comparison
- Results table
- Figure A4: Forgetting-learning tradeoff

**A.6 Code Availability**
- GitHub repository link
- Installation instructions
- Reproducibility statement

---

## Implementation Roadmap

### Week 1: Core + Model 1
- [x] Design architecture
- [x] Create folder structure
- [x] Implement base classes
- [ ] Implement Model 1
- [ ] Validate against predictions
- [ ] Generate first figure

### Week 2: Models 2-3
- [ ] Implement Model 2 (noisy signals)
- [ ] Implement Model 3 (limited dataset)
- [ ] Create unit tests
- [ ] Generate all figures

### Week 3: Model 4 + Polish
- [ ] Implement Model 4 (catastrophic forgetting)
- [ ] Create Jupyter notebooks
- [ ] Write full documentation
- [ ] Run reproducibility tests

### Week 4: Paper Integration
- [ ] Generate final figures (300 DPI)
- [ ] Export tables for LaTeX
- [ ] Write Appendix A text
- [ ] Submit to GitHub
- [ ] Tag release version

---

## Success Criteria

This architecture succeeds if:

1. **Reproducible:** `./run_all_experiments.sh` completes in <10 minutes
2. **Validated:** All 4 models match theoretical predictions
3. **Documented:** Reviewers understand methodology without asking
4. **Extensible:** Other researchers can build on this
5. **Pedagogical:** ML researchers learn about trauma, psychologists learn about gradients

---

## Files to Reference

| File | Purpose | Status |
|------|---------|--------|
| `README.md` | Quick start | ✅ |
| `trauma-models-architecture.md` | Full design (20 pages) | ✅ |
| `MODEL_SPECIFICATIONS.md` | Precise math | ✅ |
| `IMPLEMENTATION_GUIDE.md` | Step-by-step coding | ✅ |
| `ARCHITECTURE_SUMMARY.md` | This file | ✅ |
| `trauma_models/core/base_model.py` | Abstract base | ✅ |
| `trauma_models/core/metrics.py` | Shared metrics | ✅ |
| `trauma_models/core/visualization.py` | Plotting | ✅ |
| `experiments/*.yaml` | Configs | ✅ |
| `run_all_experiments.sh` | Runner script | ✅ |

---

## Contact

**Architecture designed by:** Claude Code (Anthropic)
**For:** Academic paper "Trauma as Training Data"
**Date:** October 26, 2025
**Next step:** Implement Model 1 following `IMPLEMENTATION_GUIDE.md`
