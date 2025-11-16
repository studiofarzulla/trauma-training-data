"""
Shared constants for trauma models.

Centralizes magic numbers to improve code maintainability.
"""

# Random seeds
DEFAULT_SEED = 42

# Training hyperparameters
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 50

# Model 1: Extreme Penalty
TRAUMA_FEATURE_STD_DEVIATIONS = 2.5  # How many std devs for trauma feature value
BASELINE_OVERCORRECTION_RATE = 0.05  # Expected random baseline (~5%)

# Model 2: Noisy Signals
NOISE_LEVELS = [0.05, 0.30, 0.60]  # Low, moderate, high noise
CONFIDENCE_THRESHOLD = 0.5  # Prediction confidence boundary

# Model 3: Limited Dataset
CAREGIVER_COUNTS = [2, 5, 10]  # Nuclear, extended, community
PERSONALITY_DIM = 4  # [warmth, consistency, strictness, mood_var]

# Model 4: Catastrophic Forgetting
TRAUMA_EXAMPLES = 10000  # Phase 1 training size
THERAPY_EXAMPLES = 150  # Phase 2 training size
EXPERIENCE_REPLAY_RATIO = 0.2  # 20% old data mixed with new

# Statistical analysis
ALPHA_LEVEL = 0.05  # Standard significance threshold
CONFIDENCE_INTERVAL = 0.95  # 95% CI

# Visualization
FIGURE_DPI = 300  # Publication quality
FIGURE_FORMAT = 'png'
