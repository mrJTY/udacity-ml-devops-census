"""
Integration tests for end-to-end ML workflows.

Tests complete pipelines from data processing through
model training, inference, and metrics computation.
"""

import pytest
from src.ml.data import process_data
from src.ml.model import train_model, inference, compute_model_metrics
from src.ml.slice_metrics import compute_slice_metrics


class TestIntegration:
    """Integration tests for end-to-end workflows."""

    def test_full_pipeline(self, sample_data, categorical_features):
        """
        Test the complete ML pipeline from data processing to metrics.
        """
        # Step 1: Process data
        X_train, y_train, encoder, lb = process_data(
            sample_data,
            categorical_features=categorical_features,
            label="salary",
            training=True
        )

        # Step 2: Train model
        model = train_model(X_train, y_train)

        # Step 3: Inference
        preds = inference(model, X_train)

        # Step 4: Compute metrics
        precision, recall, f1 = compute_model_metrics(y_train, preds)

        # Step 5: Compute slice metrics
        slice_results = compute_slice_metrics(
            model=model,
            data=sample_data,
            categorical_features=categorical_features,
            slice_feature="education",
            encoder=encoder,
            lb=lb
        )

        # All steps should complete successfully
        assert precision is not None
        assert recall is not None
        assert f1 is not None
        assert len(slice_results) > 0

    def test_inference_on_new_data(self, sample_data, categorical_features):
        """
        Test that model can make predictions on new data.
        """
        # Split data
        train_data = sample_data.iloc[:6]
        test_data = sample_data.iloc[6:]

        # Train
        X_train, y_train, encoder, lb = process_data(
            train_data,
            categorical_features=categorical_features,
            label="salary",
            training=True
        )
        model = train_model(X_train, y_train)

        # Test
        X_test, y_test, _, _ = process_data(
            test_data,
            categorical_features=categorical_features,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb
        )

        preds = inference(model, X_test)
        precision, recall, f1 = compute_model_metrics(y_test, preds)

        # Should produce valid predictions and metrics
        assert len(preds) == len(test_data)
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= f1 <= 1
