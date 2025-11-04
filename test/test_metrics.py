"""
Unit tests for metrics computation.

Tests the compute_model_metrics function which calculates
precision, recall, and F1 score.
"""

import pytest
import numpy as np
from src.ml.model import compute_model_metrics, inference


class TestMetrics:
    """Tests for metrics computation."""

    def test_compute_metrics_perfect_predictions(self):
        """
        Test metrics computation with perfect predictions.
        """
        # Perfect predictions
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 0, 1, 1, 0])

        precision, recall, f1 = compute_model_metrics(y_true, y_pred)

        # All metrics should be 1.0 for perfect predictions
        assert precision == 1.0
        assert recall == 1.0
        assert f1 == 1.0

    def test_compute_metrics_all_wrong(self):
        """
        Test metrics computation with completely wrong predictions.
        """
        # All wrong predictions
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])

        precision, recall, f1 = compute_model_metrics(y_true, y_pred)

        # All metrics should be 0.0
        assert precision == 0.0
        assert recall == 0.0
        assert f1 == 0.0

    def test_compute_metrics_mixed(self):
        """
        Test metrics computation with mixed predictions.
        """
        # 2 TP, 1 FP, 1 FN, 2 TN
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([1, 1, 0, 1, 0, 0])

        precision, recall, f1 = compute_model_metrics(y_true, y_pred)

        # precision = TP / (TP + FP) = 2 / (2 + 1) = 0.667
        # recall = TP / (TP + FN) = 2 / (2 + 1) = 0.667
        # f1 = 2 * (p * r) / (p + r) = 0.667

        assert 0.66 <= precision <= 0.67
        assert 0.66 <= recall <= 0.67
        assert 0.66 <= f1 <= 0.67

    def test_compute_metrics_all_negative_class(self):
        """
        Test metrics computation when all predictions are negative class.
        """
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])

        precision, recall, f1 = compute_model_metrics(y_true, y_pred)

        # With zero_division=1 parameter, should handle gracefully
        assert isinstance(precision, (int, float))
        assert isinstance(recall, (int, float))
        assert isinstance(f1, (int, float))

    def test_metrics_on_trained_model(self, trained_components):
        """
        Test that metrics can be computed on a trained model.
        """
        model, encoder, lb, X_train, y_train = trained_components

        preds = inference(model, X_train)
        precision, recall, f1 = compute_model_metrics(y_train, preds)

        # Metrics should be in valid range [0, 1]
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= f1 <= 1
