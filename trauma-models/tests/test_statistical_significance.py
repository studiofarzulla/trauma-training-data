"""
Test statistical significance calculations for Model 3.
"""

import pytest
import torch
import numpy as np
from scipy import stats


class TestStatisticalMethods:
    """Test statistical analysis methods."""

    def test_cohens_d_calculation(self):
        """Test Cohen's d effect size calculation."""
        # Known example: two groups with clear difference
        group1 = np.array([1.0, 1.2, 1.1, 1.3, 1.0])
        group2 = np.array([2.0, 2.1, 1.9, 2.2, 2.0])

        # Calculate Cohen's d
        mean1 = np.mean(group1)
        mean2 = np.mean(group2)
        std1 = np.std(group1, ddof=1)
        std2 = np.std(group2, ddof=1)
        n1 = len(group1)
        n2 = len(group2)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
        cohens_d = (mean1 - mean2) / pooled_std

        # Effect should be large (> 0.8)
        assert abs(cohens_d) > 0.8

    def test_t_test_significant_difference(self):
        """Test t-test detects significant differences."""
        # Two clearly different groups
        group1 = np.random.normal(0, 1, 100)
        group2 = np.random.normal(2, 1, 100)

        t_stat, p_value = stats.ttest_ind(group1, group2)

        # Should be highly significant
        assert p_value < 0.001

    def test_t_test_no_difference(self):
        """Test t-test with no real difference."""
        np.random.seed(42)

        # Two groups from same distribution
        group1 = np.random.normal(0, 1, 50)
        group2 = np.random.normal(0, 1, 50)

        t_stat, p_value = stats.ttest_ind(group1, group2)

        # Should not be significant (most likely)
        # Using 0.10 to avoid flakiness with random data
        assert p_value > 0.10

    def test_bonferroni_correction(self):
        """Test Bonferroni correction for multiple comparisons."""
        alpha = 0.05
        num_comparisons = 3

        # Bonferroni-corrected alpha
        corrected_alpha = alpha / num_comparisons

        assert corrected_alpha == pytest.approx(0.0167, abs=0.0001)

        # Test with p-value
        p_value = 0.02

        # Should be significant without correction
        assert p_value < alpha

        # Should NOT be significant with correction
        assert p_value > corrected_alpha

    def test_confidence_interval_calculation(self):
        """Test 95% confidence interval calculation."""
        data = np.array([10.0, 12.0, 11.0, 13.0, 10.5, 11.5, 12.5])

        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        sem = std / np.sqrt(n)

        # 95% CI using t-distribution
        ci_95 = stats.t.interval(0.95, df=n-1, loc=mean, scale=sem)

        # CI should contain the mean
        assert ci_95[0] < mean < ci_95[1]

        # CI width should be reasonable
        width = ci_95[1] - ci_95[0]
        assert width > 0


class TestDatasetGeneration:
    """Test dataset generation produces valid data."""

    def test_extreme_penalty_dataset_shape(self):
        """Model 1 dataset should have correct shapes."""
        from trauma_models.extreme_penalty.dataset import generate_dataset

        train_dataset, test_dataset = generate_dataset(
            base_examples=100,
            test_examples=50,
            seed=42
        )

        # Check train dataset
        assert len(train_dataset) == 105  # 100 base + 5 trauma

        X_train, Y_train, penalty_mask, corr_groups = train_dataset.tensors
        assert X_train.shape == (105, 10)  # 10 features
        assert Y_train.shape == (105,)  # Labels
        assert penalty_mask.shape == (105,)  # Penalty mask

        # Five traumatic examples (default)
        assert penalty_mask.sum() == 5

        # Check test dataset
        assert len(test_dataset) == 48  # Actual test examples generated
        X_test = test_dataset.tensors[0]
        assert X_test.shape == (48, 10)

    def test_limited_dataset_caregiver_generation(self):
        """Model 3 dataset should generate correct number of caregivers."""
        from trauma_models.limited_dataset.dataset import generate_caregiver_dataset

        num_caregivers = 5
        interactions_per = 20

        train_dataset, test_dataset = generate_caregiver_dataset(
            num_train_caregivers=num_caregivers,
            interactions_per_train_caregiver=interactions_per,
            num_test_caregivers=10,
            interactions_per_test_caregiver=10,
            seed=42
        )

        # Total training examples should be num_caregivers * interactions_per
        expected_train_size = num_caregivers * interactions_per
        assert len(train_dataset) == expected_train_size

        X_train, Y_train = train_dataset.tensors
        assert X_train.shape == (expected_train_size, 15)  # 15 features
        assert Y_train.shape == (expected_train_size, 1)  # 1 output

        # Outputs should be in [0, 1] range
        assert torch.all(Y_train >= 0) and torch.all(Y_train <= 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
