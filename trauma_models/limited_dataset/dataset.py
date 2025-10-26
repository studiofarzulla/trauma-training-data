"""
Dataset generation for Model 3: Limited Dataset (Caregiver Overfitting)

Key idea:
- Each caregiver has a unique "personality" (behavioral pattern)
- Training on 2 caregivers = narrow distribution (nuclear family)
- Training on 10 caregivers = diverse distribution (community child-rearing)
- Test set uses NOVEL caregivers to measure generalization

Caregiver Model:
    Each caregiver c has personality θ_c = [warmth, consistency, strictness, mood_var]
    Response: Y_c(X) = σ(θ_c^T · φ(X) + ε) where ε ~ N(0, mood_var^2)
"""

from typing import Tuple
import numpy as np
import torch
from torch.utils.data import TensorDataset


class CaregiverPersonality:
    """
    Represents a caregiver's behavioral profile.

    Attributes:
        warmth: Base affection level [0, 1]
        consistency: How predictable responses are [0, 1]
        strictness: Tendency toward harsh responses [0, 1]
        mood_variance: Random noise in responses [0, 0.3]
    """

    def __init__(
        self,
        warmth: float,
        consistency: float,
        strictness: float,
        mood_variance: float
    ):
        self.warmth = warmth
        self.consistency = consistency
        self.strictness = strictness
        self.mood_variance = mood_variance

        # Create personality vector for dot product
        self.theta = np.array([warmth, consistency, strictness, mood_variance])

    def respond(self, interaction_features: np.ndarray) -> float:
        """
        Generate caregiver's response to interaction.

        Each caregiver has unique response patterns based on their personality.
        Creates more distinct behaviors to enable overfitting demonstration.

        Args:
            interaction_features: [feature_dim] array describing interaction context

        Returns:
            response: [0, 1] predicted response (relationship success metric)
        """
        # Nonlinear feature transform φ(X)
        phi_x = self._transform_features(interaction_features)

        # Personality-specific response computation
        # Each trait affects response differently

        # Warmth: responds positively to positive child behaviors
        warmth_response = self.warmth * phi_x[0]

        # Consistency: stable responses based on context stability
        consistency_response = self.consistency * phi_x[1]

        # Strictness: reacts strongly to negative behaviors
        strictness_response = self.strictness * phi_x[2]

        # Combine personality components
        base_response = (
            2.0 * warmth_response +
            1.5 * consistency_response -
            2.0 * strictness_response
        )

        # Add mood noise: ε ~ N(0, mood_variance^2)
        # Higher mood variance = more unpredictable
        noise = np.random.normal(0, self.mood_variance)

        # Scale to reasonable range before sigmoid
        scaled_response = base_response + noise

        # Apply sigmoid to bound [0, 1]
        response = 1.0 / (1.0 + np.exp(-scaled_response))

        return response

    def _transform_features(self, features: np.ndarray) -> np.ndarray:
        """
        Nonlinear feature transform φ(X).

        Combines raw features with personality-relevant derived features:
        - Sum of positive features (child's positive behaviors)
        - Sum of negative features (child's challenging behaviors)
        - Interaction complexity (variance in features)
        - Context stability (how consistent situation is)

        Args:
            features: Raw interaction features [feature_dim]

        Returns:
            transformed: [personality_dim] features for personality matching
        """
        # Split features into positive and negative aspects
        mid = len(features) // 2
        positive_behaviors = features[:mid]
        negative_behaviors = features[mid:]

        # Compute derived features with stronger signals
        positive_sum = np.sum(positive_behaviors) / np.sqrt(len(positive_behaviors))
        negative_sum = np.sum(negative_behaviors) / np.sqrt(len(negative_behaviors))
        complexity = np.std(features)
        stability = np.exp(-complexity)  # Exponential decay for stronger contrast

        # Return [warmth-relevant, consistency-relevant, strictness-relevant, mood-relevant]
        return np.array([positive_sum, stability, negative_sum, complexity])


def generate_caregivers(
    num_caregivers: int,
    personality_distribution: str = "train",
    seed: int = 42
) -> list[CaregiverPersonality]:
    """
    Generate caregivers with diverse personalities.

    Args:
        num_caregivers: Number of caregivers to generate
        personality_distribution: "train" or "test" (test has different distribution)
        seed: Random seed

    Returns:
        List of CaregiverPersonality objects
    """
    rng = np.random.RandomState(seed)

    caregivers = []

    for i in range(num_caregivers):
        if personality_distribution == "train":
            # Training caregivers: moderate diversity, centered distribution
            # Create distinct personalities with bimodal-ish distributions
            warmth = rng.beta(2, 2)  # More variance for distinct personalities
            consistency = rng.beta(3, 3)  # Balanced
            strictness = rng.beta(2, 2)  # More variance
            mood_variance = rng.uniform(0.1, 0.3)  # Wider mood range

        else:  # "test"
            # Test caregivers: wider diversity, DIFFERENT distribution
            # This tests whether model learned general patterns or memorized
            warmth = rng.beta(5, 2)  # Skewed high (opposite from typical train)
            consistency = rng.beta(2, 5)  # Skewed low (more inconsistent)
            strictness = rng.beta(5, 2)  # Skewed high (stricter)
            mood_variance = rng.uniform(0.15, 0.35)  # Even higher noise

        caregiver = CaregiverPersonality(
            warmth=warmth,
            consistency=consistency,
            strictness=strictness,
            mood_variance=mood_variance
        )
        caregivers.append(caregiver)

    return caregivers


def generate_interactions(
    caregivers: list[CaregiverPersonality],
    interactions_per_caregiver: int,
    feature_dim: int = 15,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate interaction examples from caregivers.

    Args:
        caregivers: List of caregiver personalities
        interactions_per_caregiver: Number of interactions per caregiver
        feature_dim: Dimensionality of interaction features
        seed: Random seed

    Returns:
        (features, responses): [N, feature_dim] and [N, 1] arrays
    """
    rng = np.random.RandomState(seed)

    all_features = []
    all_responses = []

    for caregiver in caregivers:
        for _ in range(interactions_per_caregiver):
            # Sample interaction context: X ~ N(0, I)
            features = rng.randn(feature_dim)

            # Get caregiver's response
            response = caregiver.respond(features)

            all_features.append(features)
            all_responses.append(response)

    features = np.array(all_features, dtype=np.float32)
    responses = np.array(all_responses, dtype=np.float32).reshape(-1, 1)

    return features, responses


def generate_caregiver_dataset(
    num_train_caregivers: int,
    interactions_per_train_caregiver: int,
    num_test_caregivers: int,
    interactions_per_test_caregiver: int,
    feature_dim: int = 15,
    personality_dim: int = 4,
    seed: int = 42
) -> Tuple[TensorDataset, TensorDataset]:
    """
    Generate complete training and test datasets.

    Training set: N caregivers (e.g., 2, 5, 10)
    Test set: 50 NOVEL caregivers with different personality distribution

    This simulates:
    - Small N → nuclear family (overfitting to parents)
    - Large N → community child-rearing (generalization to adults)

    Args:
        num_train_caregivers: Number of training caregivers (2, 5, 10)
        interactions_per_train_caregiver: Interactions per training caregiver
        num_test_caregivers: Number of test caregivers (recommend 50)
        interactions_per_test_caregiver: Interactions per test caregiver
        feature_dim: Dimensionality of interaction features
        personality_dim: Dimensionality of personality vector
        seed: Random seed

    Returns:
        (train_dataset, test_dataset): TensorDataset objects
    """
    # Generate training caregivers
    train_caregivers = generate_caregivers(
        num_caregivers=num_train_caregivers,
        personality_distribution="train",
        seed=seed
    )

    # Generate training interactions
    train_features, train_responses = generate_interactions(
        caregivers=train_caregivers,
        interactions_per_caregiver=interactions_per_train_caregiver,
        feature_dim=feature_dim,
        seed=seed
    )

    # Generate test caregivers (NOVEL, different distribution)
    test_caregivers = generate_caregivers(
        num_caregivers=num_test_caregivers,
        personality_distribution="test",
        seed=seed + 1000  # Different seed for test
    )

    # Generate test interactions
    test_features, test_responses = generate_interactions(
        caregivers=test_caregivers,
        interactions_per_caregiver=interactions_per_test_caregiver,
        feature_dim=feature_dim,
        seed=seed + 2000
    )

    # Convert to PyTorch tensors
    train_dataset = TensorDataset(
        torch.from_numpy(train_features),
        torch.from_numpy(train_responses)
    )

    test_dataset = TensorDataset(
        torch.from_numpy(test_features),
        torch.from_numpy(test_responses)
    )

    # Print dataset statistics
    print(f"\nDataset Generation Summary:")
    print(f"  Training: {num_train_caregivers} caregivers, "
          f"{len(train_features)} total interactions")
    print(f"  Test: {num_test_caregivers} novel caregivers, "
          f"{len(test_features)} total interactions")
    print(f"  Feature dimensionality: {feature_dim}")
    print(f"  Personality dimensionality: {personality_dim}")

    # Show example caregiver personality
    example = train_caregivers[0]
    print(f"\nExample Training Caregiver Personality:")
    print(f"  Warmth: {example.warmth:.3f}")
    print(f"  Consistency: {example.consistency:.3f}")
    print(f"  Strictness: {example.strictness:.3f}")
    print(f"  Mood Variance: {example.mood_variance:.3f}")

    return train_dataset, test_dataset


def analyze_caregiver_diversity(caregivers: list[CaregiverPersonality]) -> dict:
    """
    Analyze diversity of caregiver personalities.

    More caregivers → higher diversity → better generalization.

    Args:
        caregivers: List of caregiver personalities

    Returns:
        diversity_metrics: Dict with variance and range statistics
    """
    personalities = np.array([c.theta for c in caregivers])

    metrics = {
        "num_caregivers": len(caregivers),
        "warmth_mean": np.mean(personalities[:, 0]),
        "warmth_std": np.std(personalities[:, 0]),
        "consistency_mean": np.mean(personalities[:, 1]),
        "consistency_std": np.std(personalities[:, 1]),
        "strictness_mean": np.mean(personalities[:, 2]),
        "strictness_std": np.std(personalities[:, 2]),
        "mood_variance_mean": np.mean(personalities[:, 3]),
        "mood_variance_std": np.std(personalities[:, 3]),
        "total_variance": np.var(personalities),
    }

    return metrics
