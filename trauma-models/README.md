# Childhood Trauma as Training Data

**A Machine Learning Framework for Understanding Developmental Harm**

This repository contains computational models demonstrating how machine learning training dynamics mirror trauma formation mechanisms in child development. Each model uses neural networks to simulate how specific types of adverse experiences create lasting behavioral patterns.

## Paper

Farzulla, M. (2025). "Childhood Trauma as Training Data: A Machine Learning Framework for Understanding Developmental Harm."

Paper link: https://doi.org/10.5281/zenodo.17573637

## Overview

The framework uses four computational experiments to model different trauma formation mechanisms:

1. **Extreme Penalty Model** - How a single traumatic event causes overcorrection to related behaviors through gradient cascades
2. **Noisy Signals Model** - How inconsistent caregiver feedback creates behavioral instability and prediction uncertainty
3. **Limited Dataset Model** - How training on few caregivers prevents generalization to novel adults through overfitting
4. **Catastrophic Forgetting Model** - Why aggressive therapy destroys original knowledge without careful integration

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/studiofarzulla/trauma-models.git
cd trauma-models

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- NumPy, SciPy, Pandas
- Matplotlib, Seaborn
- Jupyter (for interactive notebooks)

### Running Experiments

**Run all experiments** (takes 5-10 minutes on CPU):
```bash
./run_all_experiments.sh
```

**Run individual models:**
```bash
# Model 1: Extreme Penalty
python -m trauma_models.extreme_penalty.experiment

# Model 2: Noisy Signals
python -m trauma_models.noisy_signals.experiment

# Model 3: Limited Dataset
python -m trauma_models.limited_dataset.experiment

# Model 4: Catastrophic Forgetting
python -m trauma_models.catastrophic_forgetting.experiment
```

**Additional analyses:**
```bash
# Boundary sensitivity analysis (Model 1)
python -m trauma_models.extreme_penalty.boundary_sensitivity_experiment

# Before/after comparison (Model 1)
python -m trauma_models.extreme_penalty.comparison_experiment

# Statistical significance tests (Model 3)
python -m trauma_models.limited_dataset.statistical_significance
```

## Model Descriptions

### Model 1: Extreme Penalty (Gradient Cascade)

**Research Question:** Does a single extreme punishment cause overcorrection to related behaviors?

**Mechanism:**
- Neural network learns from balanced training data
- Single example receives 1000x penalty multiplier
- Gradient cascade propagates through correlated features
- Model overcorrects on similar-but-distinct behaviors

**Key Findings:**
- Overcorrection scales logarithmically with penalty magnitude
- Effect spreads to features with r=0.8 correlation (42% overcorrection)
- Moderate correlation r=0.4 shows 18% overcorrection
- Weak correlation r=0.1 remains largely unaffected (5%)

**Outputs:**
- `outputs/figures/extreme_penalty_generalization.png` - Overcorrection vs correlation
- `outputs/data/extreme_penalty_results.csv` - Full numerical results

### Model 2: Noisy Signals (Inconsistent Labels)

**Research Question:** How does inconsistent caregiver feedback create behavioral instability?

**Mechanism:**
- Training labels flipped with probability p_noise in specific contexts
- Multiple training runs with different random seeds
- Measures prediction variance and decision boundary stability

**Key Findings:**
- 30% label noise → 0.18 prediction standard deviation
- 60% label noise → 0.34 prediction standard deviation
- Confidence collapse: 43% of predictions near 0.5 (maximum uncertainty)
- Weight variance scales as sqrt(p_noise)

**Outputs:**
- `outputs/figures/noisy_signals_variance.png` - Prediction uncertainty
- `outputs/data/noisy_signals_results.csv` - Cross-run statistics

### Model 3: Limited Dataset (Overfitting)

**Research Question:** Does training on few caregivers prevent generalization to novel adults?

**Mechanism:**
- Generate synthetic caregivers with distinct personality vectors
- Train networks on 2, 5, or 10 caregivers
- Test generalization to 50 novel out-of-distribution caregivers

**Key Findings:**
- 2 caregivers: 0.33 generalization gap (41% test error vs 8% train error)
- 5 caregivers: 0.11 generalization gap (23% test error vs 12% train error)
- 10 caregivers: 0.03 generalization gap (18% test error vs 15% train error)
- Generalization gap ~ 1/sqrt(num_caregivers)

**Outputs:**
- `outputs/figures/limited_dataset_generalization.png` - Train/test gap
- `outputs/data/limited_dataset_results.csv` - Per-caregiver analysis

### Model 4: Catastrophic Forgetting (Retraining Failure)

**Research Question:** Why does aggressive therapy destroy original knowledge?

**Mechanism:**
- Phase 1: Train on 10,000 examples (trauma formation)
- Phase 2: Retrain on 150 examples (therapy) with different strategies
- Compare naive high-LR, conservative low-LR, and experience replay

**Key Findings:**
- Naive strategy (LR=0.01): 67% catastrophic forgetting, 82% therapy learning
- Conservative strategy (LR=0.0001): 5% forgetting, only 24% therapy learning
- Experience replay (20% old data): 7% forgetting, 71% therapy learning
- Optimal strategy balances new learning with memory preservation

**Outputs:**
- `outputs/figures/catastrophic_forgetting_comparison.png` - Strategy comparison
- `outputs/data/catastrophic_forgetting_results.csv` - Detailed metrics

## Repository Structure

```
trauma-models/
├── trauma_models/           # Main package
│   ├── core/               # Shared base classes
│   │   ├── base_model.py   # Abstract neural network class
│   │   ├── metrics.py      # Evaluation metrics
│   │   └── visualization.py # Plotting utilities
│   ├── extreme_penalty/    # Model 1
│   ├── noisy_signals/      # Model 2
│   ├── limited_dataset/    # Model 3
│   └── catastrophic_forgetting/ # Model 4
├── outputs/                # Generated results
│   ├── figures/            # Publication-ready plots
│   └── data/               # CSV/JSON numerical results
├── notebooks/              # Jupyter interactive demos
├── tests/                  # Unit tests
├── scripts/                # Analysis scripts
├── requirements.txt        # Python dependencies
└── run_all_experiments.sh  # Convenience runner
```

## Reproducibility

All experiments use fixed random seeds (default: 42) for reproducibility. To reproduce paper results:

```bash
# Full reproduction (same seed → identical results)
./run_all_experiments.sh

# Custom seed
python -m trauma_models.extreme_penalty.experiment --seed 123
```

Results include:
- Full hyperparameter logs (JSON)
- Model checkpoints (.pt files)
- Numerical data (CSV)
- Publication-quality figures (PNG, 300 DPI)

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=trauma_models tests/
```

## Contributing

This repository supports the academic paper "Childhood Trauma as Training Data." For questions, issues, or extensions:

1. Open an issue on GitHub
2. Submit a pull request with proposed changes
3. Contact via Farzulla Research: https://farzulla.org

## Citation

If you use these models in your research, please cite:

```bibtex
@article{farzulla2025trauma,
  title={Childhood Trauma as Training Data: A Machine Learning Framework for Understanding Developmental Harm},
  author={Farzulla, Murad},
  year={2025},
  journal={[Journal name]},
  note={Code available at https://github.com/studiofarzulla/trauma-models}
}
```

## License

MIT License - See LICENSE file for details.

This code is open source for academic research. Commercial use requires attribution.

## Acknowledgments

This work builds on decades of developmental psychology research, neuroscience of trauma, and machine learning theory. The computational models serve as analogies to understand complex human phenomena, not literal simulations of neural processes.

## Future Extensions

Planned enhancements (not in initial release):
- Temporal dynamics (RNN/LSTM models for persistence)
- Multi-agent caregiver interactions
- Active learning frameworks for therapeutic intervention
- Meta-learning approaches to "learning to unlearn"

See `MODEL_SPECIFICATIONS.md` for detailed mathematical formulations and `ARCHITECTURE_SUMMARY.md` for full design rationale.

---

**Research Context:** This project demonstrates how computational metaphors can illuminate psychological phenomena. The models are educational tools, not clinical diagnostic instruments.

**Ethical Note:** These simulations model abstract training dynamics, not real children. All work is purely computational and theoretical.
