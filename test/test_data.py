"""
Unit tests for data processing functionality.

Tests the process_data function which handles feature engineering,
categorical encoding, and label binarization.
"""

import pytest
import numpy as np
from src.ml.data import process_data


class TestDataProcessing:
    """Tests for data processing functionality."""

    def test_process_data_training_mode(self, sample_data, categorical_features):
        """
        Test that process_data correctly encodes categorical features
        and binarizes labels in training mode.
        """
        X, y, encoder, lb = process_data(
            sample_data,
            categorical_features=categorical_features,
            label="salary",
            training=True
        )

        # Check that encoder and lb are fitted
        assert encoder is not None
        assert lb is not None
        assert hasattr(encoder, 'categories_')
        assert hasattr(lb, 'classes_')

        # Check output shapes
        n_samples = len(sample_data)
        assert X.shape[0] == n_samples
        assert y.shape[0] == n_samples

        # Check that y is binarized (0s and 1s)
        assert set(y).issubset({0, 1})

        # Check that continuous features are preserved
        # Should have continuous features + encoded categorical features
        n_continuous = len(sample_data.columns) - len(categorical_features) - 1  # -1 for label
        assert X.shape[1] > n_continuous  # More features after encoding

    def test_process_data_inference_mode(self, sample_data, categorical_features):
        """
        Test that process_data uses pre-fitted encoder and lb in inference mode.
        """
        # First, train mode to get encoder and lb
        _, _, encoder, lb = process_data(
            sample_data,
            categorical_features=categorical_features,
            label="salary",
            training=True
        )

        # Now test inference mode
        X_inf, y_inf, encoder_out, lb_out = process_data(
            sample_data,
            categorical_features=categorical_features,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb
        )

        # Check that same encoder and lb are returned
        assert encoder_out is encoder
        assert lb_out is lb

        # Check shapes
        assert X_inf.shape[0] == len(sample_data)
        assert y_inf.shape[0] == len(sample_data)

    def test_process_data_no_label(self, sample_data, categorical_features):
        """
        Test that process_data handles inference without labels.
        """
        # First get encoder and lb
        _, _, encoder, lb = process_data(
            sample_data,
            categorical_features=categorical_features,
            label="salary",
            training=True
        )

        # Remove label and test
        data_no_label = sample_data.drop('salary', axis=1)
        X, y, _, _ = process_data(
            data_no_label,
            categorical_features=categorical_features,
            label=None,
            training=False,
            encoder=encoder,
            lb=lb
        )

        # y should be empty
        assert len(y) == 0
        # X should still have all samples
        assert X.shape[0] == len(data_no_label)

    def test_process_data_preserves_continuous_features(self, sample_data, categorical_features):
        """
        Test that continuous feature values are preserved after processing.
        """
        X, y, encoder, lb = process_data(
            sample_data,
            categorical_features=categorical_features,
            label="salary",
            training=True
        )

        # The first few columns should be continuous features
        # Check that age values are preserved (first continuous column)
        n_continuous = len(sample_data.columns) - len(categorical_features) - 1

        # Age should be in the first column
        assert np.array_equal(X[:, 0], sample_data['age'].values)
