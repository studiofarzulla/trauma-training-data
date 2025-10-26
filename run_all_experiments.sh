#!/bin/bash
# Run all trauma model experiments
# Estimated runtime: 5-10 minutes on laptop CPU

set -e  # Exit on error

echo "========================================"
echo "Trauma Models - Running All Experiments"
echo "========================================"
echo ""

# Create output directories
mkdir -p outputs/figures
mkdir -p outputs/data
mkdir -p outputs/checkpoints

# Model 1: Extreme Penalty
echo "[1/4] Running Extreme Penalty experiment..."
python -m trauma_models.extreme_penalty.experiment \
    --config experiments/extreme_penalty_sweep.yaml \
    --output outputs/

# Model 2: Noisy Signals
echo "[2/4] Running Noisy Signals experiment..."
python -m trauma_models.noisy_signals.experiment \
    --config experiments/noisy_signals_sweep.yaml \
    --output outputs/

# Model 3: Limited Dataset
echo "[3/4] Running Limited Dataset experiment..."
python -m trauma_models.limited_dataset.experiment \
    --config experiments/limited_dataset_sweep.yaml \
    --output outputs/

# Model 4: Catastrophic Forgetting
echo "[4/4] Running Catastrophic Forgetting experiment..."
python -m trauma_models.catastrophic_forgetting.experiment \
    --config experiments/catastrophic_forgetting_sweep.yaml \
    --output outputs/

echo ""
echo "========================================"
echo "All experiments complete!"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - outputs/figures/     (publication-ready plots)"
echo "  - outputs/data/        (CSV/JSON numerical results)"
echo "  - outputs/checkpoints/ (model weights)"
echo ""
echo "Next steps:"
echo "  1. Review figures: ls outputs/figures/"
echo "  2. Check notebooks: jupyter notebook notebooks/"
echo "  3. Run tests: pytest tests/"
echo ""
