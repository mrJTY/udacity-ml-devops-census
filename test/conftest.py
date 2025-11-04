"""
Shared pytest fixtures for ML model tests.

This module contains fixtures that are shared across all test files.
"""

import pytest
import pandas as pd
import os
from src.ml.data import process_data
from src.ml.model import train_model


# Get the directory where this conftest.py file is located
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DATA_PATH = os.path.join(TEST_DIR, "sample_data.csv")


@pytest.fixture
def sample_data():
    """Load sample census-like data from CSV for testing."""
    data = pd.read_csv(SAMPLE_DATA_PATH)
    return data


@pytest.fixture
def categorical_features():
    """Return list of categorical feature names."""
    return [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]


@pytest.fixture
def trained_components(sample_data, categorical_features):
    """Return trained model, encoder, and label binarizer."""
    X_train, y_train, encoder, lb = process_data(
        sample_data,
        categorical_features=categorical_features,
        label="salary",
        training=True
    )
    model = train_model(X_train, y_train)
    return model, encoder, lb, X_train, y_train
