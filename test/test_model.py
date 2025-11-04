"""
Unit tests for model training and inference.

Tests the train_model and inference functions.
"""

import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.ml.data import process_data
from src.ml.model import train_model, inference


class TestModel:
    """Tests for model training and inference."""

    def test_train_model_returns_classifier(self, sample_data, categorical_features):
        """
        Test that train_model returns a fitted RandomForestClassifier.
        """
        X, y, _, _ = process_data(
            sample_data,
            categorical_features=categorical_features,
            label="salary",
            training=True
        )

        model = train_model(X, y)

        # Check model type
        assert isinstance(model, RandomForestClassifier)

        # Check that model is fitted
        assert hasattr(model, 'estimators_')

        # Check hyperparameters
        assert model.n_estimators == 50
        assert model.max_depth == 5
        assert model.random_state == 123

    def test_model_inference_shape(self, trained_components):
        """
        Test that inference returns predictions with correct shape.
        """
        model, encoder, lb, X_train, y_train = trained_components

        # Test inference
        preds = inference(model, X_train)

        # Check shape
        assert preds.shape == y_train.shape
        assert len(preds) == len(y_train)

        # Check that predictions are binary
        assert set(preds).issubset({0, 1})

    def test_inference_single_sample(self, trained_components, sample_data, categorical_features):
        """
        Test that inference works with a single sample.
        """
        model, encoder, lb, _, _ = trained_components

        # Process a single sample
        single_sample = sample_data.iloc[[0]]
        X_single, _, _, _ = process_data(
            single_sample,
            categorical_features=categorical_features,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb
        )

        preds = inference(model, X_single)

        # Should return array with one prediction
        assert len(preds) == 1
        assert preds[0] in [0, 1]

    def test_model_determinism(self, sample_data, categorical_features):
        """
        Test that model training is deterministic with fixed random_state.
        """
        X, y, _, _ = process_data(
            sample_data,
            categorical_features=categorical_features,
            label="salary",
            training=True
        )

        # Train two models with same random state
        model1 = train_model(X, y)
        model2 = train_model(X, y)

        # Predictions should be identical
        preds1 = inference(model1, X)
        preds2 = inference(model2, X)

        assert np.array_equal(preds1, preds2)
