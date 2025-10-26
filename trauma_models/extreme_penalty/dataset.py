"""
Dataset generation for Extreme Penalty Model.

Creates synthetic data with controlled correlation structure:
- 10 features with multivariate Gaussian distribution
- 3 behaviors: "ask questions" (safe), "explore" (neutral), "express uncertainty" (risky)
- Features 0-3 correlate highly (0.8) with all behaviors
- Configurable number of trauma examples to demonstrate gradient cascade
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset
from typing import Tuple


def generate_correlation_matrix(
    feature_dim: int = 10,
    correlation_levels: list = None,
) -> np.ndarray:
    """
    Generate correlation matrix with controlled structure.

    Feature correlation structure:
    - Feature 0: target behavior (receives extreme penalty)
    - Features 1-3: High correlation (0.8) to target
    - Features 4-7: Medium correlation (0.4) to target
    - Features 8-9: Low correlation (0.1) to target

    Args:
        feature_dim: Number of features (default: 10)
        correlation_levels: [high, medium, low] correlation values

    Returns:
        Correlation matrix [feature_dim, feature_dim]
    """
    if correlation_levels is None:
        correlation_levels = [0.8, 0.4, 0.1]

    # Start with identity (uncorrelated)
    corr_matrix = np.eye(feature_dim, dtype=np.float64)

    # Feature 0 is the target
    # Features 1-3: high correlation (0.8)
    for i in range(1, min(4, feature_dim)):
        corr_matrix[0, i] = corr_matrix[i, 0] = correlation_levels[0]

    # Features 4-7: medium correlation (0.4)
    for i in range(4, min(8, feature_dim)):
        corr_matrix[0, i] = corr_matrix[i, 0] = correlation_levels[1]

    # Features 8-9: low correlation (0.1)
    for i in range(8, min(10, feature_dim)):
        corr_matrix[0, i] = corr_matrix[i, 0] = correlation_levels[2]

    # Ensure positive semi-definite by adding small regularization
    # This is a common technique to handle numerical precision issues
    eigvals = np.linalg.eigvalsh(corr_matrix)
    if np.min(eigvals) < 0:
        corr_matrix = corr_matrix + np.eye(feature_dim) * (abs(np.min(eigvals)) + 1e-6)
        # Re-normalize to correlation matrix (diagonal = 1)
        d = np.sqrt(np.diag(corr_matrix))
        corr_matrix = corr_matrix / np.outer(d, d)

    return corr_matrix


def generate_labels(
    features: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate behavior labels from features.

    Uses simple decision rule based on feature values:
    - Class 0 (safe): feature[0] > 0.5
    - Class 1 (neutral): -0.5 <= feature[0] <= 0.5
    - Class 2 (risky): feature[0] < -0.5

    This creates a natural mapping that will be disrupted by trauma injection.

    Args:
        features: Feature matrix [N, feature_dim]
        seed: Random seed

    Returns:
        Labels [N] with values in {0, 1, 2}
    """
    np.random.seed(seed)
    N = len(features)

    # Base labels on first feature with some noise
    labels = np.zeros(N, dtype=np.int64)

    for i in range(N):
        val = features[i, 0]
        if val > 0.5:
            labels[i] = 0  # safe
        elif val < -0.5:
            labels[i] = 2  # risky
        else:
            labels[i] = 1  # neutral

    # Add some randomness (10% label noise for realism)
    noise_mask = np.random.random(N) < 0.1
    labels[noise_mask] = np.random.randint(0, 3, size=noise_mask.sum())

    return labels


def create_trauma_example(
    correlation_levels: list = None,
) -> Tuple[np.ndarray, int]:
    """
    Create the single traumatic example with extreme penalty.

    This example has:
    - High negative value on target feature (feature 0) = would naturally be risky (2)
    - Label: safe (0) = contradicts natural pattern
    - Will receive 1000x loss multiplier during training
    - Creates strong conflict that should propagate to correlated features

    Args:
        correlation_levels: Correlation structure (unused, for consistency)

    Returns:
        (feature_vector, label)
    """
    # Create features that would naturally suggest "risky" behavior
    trauma_features = np.zeros(10)
    trauma_features[0] = -2.5  # Strong negative value (naturally risky)

    # Add some variation to other features
    trauma_features[1:] = np.random.randn(9) * 0.5

    # Label as SAFE (0) even though features suggest risky
    # This conflict + extreme penalty creates the overcorrection effect
    trauma_label = 0  # Safe - contradicts the negative feature[0]

    return trauma_features, trauma_label


def generate_dataset(
    base_examples: int = 10000,
    test_examples: int = 2000,
    correlation_levels: list = None,
    feature_dim: int = 10,
    num_trauma: int = 5,
    seed: int = 42,
) -> Tuple[TensorDataset, TensorDataset]:
    """
    Generate complete training and test datasets.

    Training set:
    - base_examples normal examples
    - num_trauma traumatic examples (marked with penalty_mask)

    Test set:
    - test_examples stratified by correlation group
    - Used to measure overcorrection at different correlation levels

    Args:
        base_examples: Number of normal training examples
        test_examples: Number of test examples
        correlation_levels: [high, medium, low] correlation values
        feature_dim: Number of features
        num_trauma: Number of traumatic examples (default: 5, pedagogical: 20)
        seed: Random seed for reproducibility

    Returns:
        (train_dataset, test_dataset)
        Each dataset is TensorDataset with:
        - features: [N, feature_dim]
        - labels: [N]
        - penalty_mask: [N] (1 for trauma example, 0 otherwise)
        - correlation_groups: [N] (0=high, 1=medium, 2=low correlation)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    if correlation_levels is None:
        correlation_levels = [0.8, 0.4, 0.1]

    # Generate correlation matrix
    corr_matrix = generate_correlation_matrix(feature_dim, correlation_levels)

    # ========== TRAINING SET ==========
    # Generate normal examples
    X_train = np.random.multivariate_normal(
        mean=np.zeros(feature_dim),
        cov=corr_matrix,
        size=base_examples,
    )
    Y_train = generate_labels(X_train, seed=seed)

    # Create traumatic examples
    trauma_examples = []
    trauma_labels = []

    for i in range(num_trauma):
        trauma_features, trauma_label = create_trauma_example(correlation_levels)
        # Add slight variation to each
        trauma_features = trauma_features + np.random.randn(feature_dim) * 0.1
        trauma_examples.append(trauma_features)
        trauma_labels.append(trauma_label)

    # Add trauma examples to training set
    X_train = np.vstack([X_train] + trauma_examples)
    Y_train = np.concatenate([Y_train, trauma_labels])

    # Create penalty mask (1 for last num_trauma examples)
    penalty_mask_train = np.zeros(len(X_train), dtype=np.float32)
    penalty_mask_train[-num_trauma:] = 1.0

    # Correlation groups for training (mostly for completeness)
    correlation_groups_train = np.zeros(len(X_train), dtype=np.int64)

    # ========== TEST SET ==========
    # Generate test examples stratified by correlation group
    examples_per_group = test_examples // 3

    X_test_list = []
    Y_test_list = []
    corr_groups_list = []

    for group_idx, corr_level in enumerate(correlation_levels):
        # Generate features with specific correlation emphasis
        X_group = np.random.multivariate_normal(
            mean=np.zeros(feature_dim),
            cov=corr_matrix,
            size=examples_per_group,
        )

        # Generate labels
        Y_group = generate_labels(X_group, seed=seed + group_idx)

        # Mark correlation group
        corr_group = np.full(examples_per_group, group_idx, dtype=np.int64)

        X_test_list.append(X_group)
        Y_test_list.append(Y_group)
        corr_groups_list.append(corr_group)

    # Combine test groups
    X_test = np.vstack(X_test_list)
    Y_test = np.concatenate(Y_test_list)
    correlation_groups_test = np.concatenate(corr_groups_list)

    # Penalty mask for test (all zeros, no trauma)
    penalty_mask_test = np.zeros(len(X_test), dtype=np.float32)

    # ========== Convert to PyTorch Tensors ==========
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(Y_train),
        torch.FloatTensor(penalty_mask_train),
        torch.LongTensor(correlation_groups_train),
    )

    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(Y_test),
        torch.FloatTensor(penalty_mask_test),
        torch.LongTensor(correlation_groups_test),
    )

    return train_dataset, test_dataset


def get_correlation_group_indices(
    dataset: TensorDataset,
    group_idx: int,
) -> np.ndarray:
    """
    Get indices of examples in a specific correlation group.

    Args:
        dataset: Dataset with correlation_groups as 4th tensor
        group_idx: 0 (high), 1 (medium), or 2 (low)

    Returns:
        Array of indices
    """
    correlation_groups = dataset.tensors[3].numpy()
    return np.where(correlation_groups == group_idx)[0]


def get_neutral_examples(
    dataset: TensorDataset,
    correlation_group: int = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract neutral examples (label=1) for overcorrection analysis.

    Args:
        dataset: Dataset to filter
        correlation_group: Optional specific correlation group (0, 1, 2)

    Returns:
        (features, labels) of neutral examples
    """
    features, labels, _, corr_groups = dataset.tensors

    # Filter for neutral (label = 1)
    mask = labels == 1

    if correlation_group is not None:
        mask = mask & (corr_groups == correlation_group)

    return features[mask], labels[mask]


def generate_trauma_adjacent_test_set(
    num_examples: int = 300,
    correlation_levels: list = None,
    feature_dim: int = 10,
    seed: int = 42,
) -> TensorDataset:
    """
    Generate test examples in the trauma-adjacent boundary region.

    This set contains examples where feature[0] is in the ambiguous region
    between neutral and the trauma point. These are the examples where
    trauma-induced overcorrection should be most visible.

    Trauma pattern:
    - feature[0] = -2.5 (extreme negative)
    - label = safe (0)

    Trauma-adjacent region:
    - feature[0] ∈ [-1.5, -0.5] (moderately negative)
    - Natural label = risky (2) or neutral (1)
    - Trauma should cause confusion → predict safe (0)

    Args:
        num_examples: Number of test examples per correlation group
        correlation_levels: [high, medium, low] correlation values
        feature_dim: Number of features (should be 10)
        seed: Random seed

    Returns:
        TensorDataset with trauma-adjacent test examples
        Format: (features, labels, penalty_mask, correlation_groups)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    if correlation_levels is None:
        correlation_levels = [0.8, 0.4, 0.1]

    # Generate correlation matrix
    corr_matrix = generate_correlation_matrix(feature_dim, correlation_levels)

    X_test_list = []
    Y_test_list = []
    corr_groups_list = []

    # Create examples for each correlation group
    for group_idx in range(3):
        X_group = []

        # Generate examples with feature[0] in boundary region [-1.5, -0.5]
        # This is between neutral (0) and trauma (-2.5)
        feature_0_values = np.linspace(-1.5, -0.5, num_examples)

        for feature_0_val in feature_0_values:
            # Generate other features from multivariate normal
            features = np.random.multivariate_normal(
                mean=np.zeros(feature_dim),
                cov=corr_matrix,
            )

            # Override feature[0] with controlled value
            features[0] = feature_0_val

            X_group.append(features)

        X_group = np.array(X_group)

        # Generate natural labels (based on our labeling function)
        Y_group = generate_labels(X_group, seed=seed + group_idx)

        # Mark correlation group
        corr_group = np.full(num_examples, group_idx, dtype=np.int64)

        X_test_list.append(X_group)
        Y_test_list.append(Y_group)
        corr_groups_list.append(corr_group)

    # Combine all groups
    X_test = np.vstack(X_test_list)
    Y_test = np.concatenate(Y_test_list)
    correlation_groups = np.concatenate(corr_groups_list)

    # Penalty mask (all zeros, this is test set)
    penalty_mask = np.zeros(len(X_test), dtype=np.float32)

    # Convert to PyTorch tensors
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(Y_test),
        torch.FloatTensor(penalty_mask),
        torch.LongTensor(correlation_groups),
    )

    return test_dataset
