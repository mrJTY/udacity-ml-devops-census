"""
Unit tests for slice-based metrics computation.

Tests the slice metrics functionality which evaluates model
performance across different subgroups of data.
"""

import pytest
import pandas as pd
from src.ml.slice_metrics import compute_slice_metrics, compute_multiple_slice_metrics


class TestSliceMetrics:
    """Tests for slice-based metrics computation."""

    def test_compute_slice_metrics_education(self, sample_data, categorical_features, trained_components):
        """
        Test that slice metrics are computed correctly for education feature.
        """
        model, encoder, lb, _, _ = trained_components

        results = compute_slice_metrics(
            model=model,
            data=sample_data,
            categorical_features=categorical_features,
            slice_feature="education",
            label="salary",
            encoder=encoder,
            lb=lb
        )

        # Should have results for each unique education value
        unique_education = sample_data['education'].nunique()
        assert len(results) == unique_education

        # Check columns
        expected_columns = ['slice_feature', 'slice_value', 'n_samples', 'precision', 'recall', 'f1']
        assert list(results.columns) == expected_columns

        # Check that all metrics are in valid range
        assert all(0 <= results['precision']) and all(results['precision'] <= 1)
        assert all(0 <= results['recall']) and all(results['recall'] <= 1)
        assert all(0 <= results['f1']) and all(results['f1'] <= 1)

    def test_compute_slice_metrics_sex(self, sample_data, categorical_features, trained_components):
        """
        Test slice metrics for binary protected attribute (sex).
        """
        model, encoder, lb, _, _ = trained_components

        results = compute_slice_metrics(
            model=model,
            data=sample_data,
            categorical_features=categorical_features,
            slice_feature="sex",
            label="salary",
            encoder=encoder,
            lb=lb
        )

        # Should have 2 slices (Male/Female)
        assert len(results) == 2

        # Check that slice values are correct
        assert set(results['slice_value']) == set(sample_data['sex'].unique())

        # Sample counts should sum to total
        assert results['n_samples'].sum() == len(sample_data)

    def test_compute_slice_metrics_invalid_feature(self, sample_data, categorical_features, trained_components):
        """
        Test that invalid slice feature raises error.
        """
        model, encoder, lb, _, _ = trained_components

        with pytest.raises(ValueError, match="not found in data columns"):
            compute_slice_metrics(
                model=model,
                data=sample_data,
                categorical_features=categorical_features,
                slice_feature="invalid_feature",
                encoder=encoder,
                lb=lb
            )

    def test_compute_slice_metrics_requires_encoder(self, sample_data, categorical_features, trained_components):
        """
        Test that encoder and lb are required.
        """
        model, _, _, _, _ = trained_components

        with pytest.raises(ValueError, match="encoder and lb must be provided"):
            compute_slice_metrics(
                model=model,
                data=sample_data,
                categorical_features=categorical_features,
                slice_feature="education",
                encoder=None,
                lb=None
            )

    def test_compute_multiple_slice_metrics(self, sample_data, categorical_features, trained_components):
        """
        Test computing slice metrics for multiple features.
        """
        model, encoder, lb, _, _ = trained_components

        slice_features = ["sex", "race"]
        results_dict = compute_multiple_slice_metrics(
            model=model,
            data=sample_data,
            categorical_features=categorical_features,
            slice_features=slice_features,
            encoder=encoder,
            lb=lb
        )

        # Should have results for each slice feature
        assert set(results_dict.keys()) == set(slice_features)

        # Each should be a DataFrame
        for feature, df in results_dict.items():
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
