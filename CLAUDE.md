# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a dual-purpose academic research repository containing:
1. **Academic paper** (root level): "Trauma as Bad Training Data: A Computational Framework for Developmental Psychology"
2. **Computational models** (`trauma-models/` subdirectory): PyTorch implementations validating the paper's framework

The paper proposes a novel computational lens for understanding developmental trauma by reframing adverse childhood experiences as "bad training data" problems in learning systems.

## Key Files

### Paper Files (Root Level)
- `trauma-training-data-essay.md` (90KB) - Full academic paper in markdown (8,757 words)
- `trauma-training-data-essay.tex` (67KB) - LaTeX version for publication
- `trauma-training-data.bib` (6KB) - BibTeX bibliography (30+ references)
- `trauma_paper.pdf` (2.8MB) - Compiled PDF output

### Computational Models (trauma-models/ subdirectory)
- `trauma_models/` - Python package with 4 PyTorch models
- `outputs/` - Generated figures and numerical results
- `requirements.txt` - Python dependencies (PyTorch 2.0+, NumPy, Matplotlib, etc.)
- `run_all_experiments.sh` - Execute all 4 models (5-10 minutes)

## Common Commands

### Paper Compilation

**Compile LaTeX to PDF:**
```bash
# From repository root
pdflatex trauma-training-data-essay.tex
bibtex trauma-training-data-essay
pdflatex trauma-training-data-essay.tex
pdflatex trauma-training-data-essay.tex
```

**Convert Markdown to LaTeX:**
Use the `md-to-latex-academic` agent (available in Claude Code) to translate markdown with Unicode math symbols to proper LaTeX commands.

### Running Computational Models

**Setup (one-time):**
```bash
cd trauma-models/
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Run all experiments:**
```bash
cd trauma-models/
./run_all_experiments.sh
```

**Run individual models:**
```bash
# From trauma-models/ directory
python -m trauma_models.extreme_penalty.experiment          # Model 1
python -m trauma_models.noisy_signals.experiment            # Model 2
python -m trauma_models.limited_dataset.experiment          # Model 3
python -m trauma_models.catastrophic_forgetting.experiment  # Model 4
```

**Additional analyses:**
```bash
# Boundary sensitivity analysis (Model 1)
python -m trauma_models.extreme_penalty.boundary_sensitivity_experiment

# Statistical significance tests (Model 3)
python -m trauma_models.limited_dataset.statistical_significance
```

**Testing:**
```bash
cd trauma-models/
pytest tests/                    # Run all tests
pytest --cov=trauma_models tests/  # With coverage
```

## Architecture

### Paper Structure (4 Trauma Categories)

The paper categorizes developmental trauma through machine learning training data failures:

1. **Direct Negative Data** - Abuse as actively harmful training signals (Model 1: Extreme Penalty)
2. **Indirect Negative Data** - Witnessing trauma, environmental stress (Model 2: Noisy Signals)
3. **Absent Positive Data** - Neglect as missing training examples (Model 3: Limited Dataset)
4. **Limited Data Diversity** - Nuclear family isolation preventing generalization (Model 3: Limited Dataset)

Each category maps to specific ML failure modes with empirical predictions validated by computational models.

### Computational Models Architecture

**Shared Core** (`trauma_models/core/`):
- `base_model.py` - Abstract `TraumaModel` class all models inherit from
- `metrics.py` - Evaluation metrics (generalization gap, prediction variance, forgetting rate, etc.)
- `visualization.py` - Publication-quality plotting utilities (300 DPI, consistent styling)

**Model 1: Extreme Penalty** (`trauma_models/extreme_penalty/`):
- **Mechanism:** Single training example receives 1000x penalty multiplier → gradient cascade → overcorrection on correlated features
- **Prediction:** Overcorrection scales logarithmically with penalty magnitude and spreads to r=0.8 correlated features
- **Network:** 3-layer MLP (10 input features, 64 hidden, 1 output)

**Model 2: Noisy Signals** (`trauma_models/noisy_signals/`):
- **Mechanism:** Training labels flipped with probability p_noise in specific contexts → behavioral instability
- **Prediction:** Prediction variance scales as sqrt(p_noise), confidence collapses near 0.5
- **Network:** Binary classifier with cross-entropy loss

**Model 3: Limited Dataset** (`trauma_models/limited_dataset/`):
- **Mechanism:** Train on 2, 5, or 10 synthetic caregivers → test on 50 novel caregivers → generalization gap
- **Prediction:** Generalization gap ~ 1/sqrt(num_caregivers)
- **Network:** Regression model learning caregiver-specific response patterns

**Model 4: Catastrophic Forgetting** (`trauma_models/catastrophic_forgetting/`):
- **Mechanism:** Phase 1 (trauma formation: 10k examples) → Phase 2 (therapy: 150 examples) with different learning strategies
- **Prediction:** Naive high-LR destroys old knowledge, experience replay balances retention and new learning
- **Network:** Two-phase training with strategy comparison (naive, conservative, experience replay)

### Output Structure

All experiments save results to `trauma-models/outputs/`:
- `figures/` - PNG plots (300 DPI, publication-ready)
- `data/` - CSV/JSON numerical results
- `checkpoints/` - Model weights (.pt files)

Naming convention: `{model_name}_{metric}.{ext}`
Example: `extreme_penalty_generalization.png`, `limited_dataset_results.csv`

## Development Notes

### Reproducibility
All experiments use fixed random seeds (default: 42). Custom seed via `--seed` flag:
```bash
python -m trauma_models.extreme_penalty.experiment --seed 123
```

### Code Style
- Black formatter (line length: 100)
- isort for imports (black profile)
- Type hints encouraged but not enforced
- Docstrings for public methods

### Paper Revision Status
As of October 2025, the paper received peer review (8/10 academic rigor). Identified improvements:
- Clearer analogy vs mechanism distinction
- Earlier gene-environment interaction integration
- Stronger empirical operationalization
- More evidence for nuclear family critiques

See `trauma-models/PAPER_REVISION_TEXT.md` for detailed feedback.

### Target Venues
- **Primary:** Zenodo preprint (DOI already assigned)
- **Secondary:** arXiv (cs.AI or q-bio.NC)
- **Potential journals:** Computational Psychiatry, Nature Human Behaviour

## Important Context

**This is research and prototyping, not production code.** The computational models are educational demonstrations of ML training dynamics as analogies for developmental trauma, not clinical diagnostic tools.

**Ethical Note:** All simulations model abstract training dynamics, not real children. Work is purely computational and theoretical.

**Interdisciplinary Nature:** The paper bridges ML, neuroscience, developmental psychology, and cognitive science. Expect specialized notation from multiple fields (gradient descent, synaptic plasticity, attachment theory, etc.).

## File Locations in Refactored Structure

Previously the paper lived in `~/zenodo-packages/02-trauma/`, but the directory structure was refactored in October 2025. Current canonical location: `~/Resurrexi/projects/planned-publish/trauma-training-data/`

The `trauma-models/` subdirectory is a separate git repository (submodule or nested repo) with its own development history and documentation.
