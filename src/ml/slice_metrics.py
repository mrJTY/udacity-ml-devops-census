"""
Compute model performance metrics on slices of data.

This module provides functionality to evaluate model performance across
different subgroups of the data, which is critical for fairness analysis
and understanding model behavior across demographics.
"""

import pandas as pd
from typing import Dict, List, Tuple
from .model import inference, compute_model_metrics
from .data import process_data


def compute_slice_metrics(
    model,
    data: pd.DataFrame,
    categorical_features: List[str],
    slice_feature: str,
    label: str = "salary",
    encoder=None,
    lb=None
) -> pd.DataFrame:
    """
    Compute model performance metrics on slices of the data.

    For each unique value in the slice_feature column, this function:
    1. Filters the data to that slice
    2. Processes the data using the provided encoder/label binariser
    3. Runs model inference
    4. Computes precision, recall, and F1 metrics

    Params:
    model : RandomForestClassifier
        Trained machine learning model.
    data : pd.DataFrame
        Complete dataset with features and labels.
    categorical_features : List[str]
        List of categorical feature names to encode.
    slice_feature : str
        Name of the feature to slice by (e.g., "education", "sex", "race").
    label : str, optional
        Name of the label column (default="salary").
    encoder : OneHotEncoder
        Fitted encoder for categorical features.
    lb : LabelBinarizer
        Fitted label binarizer for target variable.

    Returns
    pd.DataFrame
        DataFrame with columns: slice_feature, slice_value, n_samples,
        precision, recall, f1.
    """
    if encoder is None or lb is None:
        raise ValueError("encoder and lb must be provided for slice analysis")

    if slice_feature not in data.columns:
        raise ValueError(f"slice_feature '{slice_feature}' not found in data columns")

    results = []
    unique_values = data[slice_feature].unique()

    for value in sorted(unique_values):
        # Filter data to this slice
        slice_data = data[data[slice_feature] == value].copy()
        n_samples = len(slice_data)

        # Skip empty slices
        if n_samples == 0:
            continue

        # Process the slice data
        X_slice, y_slice, _, _ = process_data(
            slice_data,
            categorical_features=categorical_features,
            label=label,
            training=False,
            encoder=encoder,
            lb=lb
        )

        # Get predictions
        preds = inference(model, X_slice)

        # Compute metrics
        precision, recall, f1 = compute_model_metrics(y_slice, preds)

        # Store results
        results.append({
            "slice_feature": slice_feature,
            "slice_value": value,
            "n_samples": n_samples,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

    return pd.DataFrame(results)


def print_slice_metrics(slice_metrics_df: pd.DataFrame) -> None:
    """
    Pretty print slice metrics results.
    Params:
    slice_metrics_df : pd.DataFrame
        DataFrame returned by compute_slice_metrics.
    """
    if slice_metrics_df.empty:
        print("No slice metrics to display.")
        return

    slice_feature = slice_metrics_df["slice_feature"].iloc[0]
    print(f"Model Performance by {slice_feature.upper()}")

    for _, row in slice_metrics_df.iterrows():
        print(f"Slice: {slice_feature} = {row['slice_value']}")
        print(f"  n_samples: {row['n_samples']}")
        print(f"  precision: {row['precision']:.4f}")
        print(f"  recall:    {row['recall']:.4f}")
        print(f"  f1:        {row['f1']:.4f}")
        print()


def compute_multiple_slice_metrics(
    model,
    data: pd.DataFrame,
    categorical_features: List[str],
    slice_features: List[str],
    label: str = "salary",
    encoder=None,
    lb=None,
    output_file: str = None
) -> Dict[str, pd.DataFrame]:
    """
    Compute slice metrics for multiple features.

    Params
    model : RandomForestClassifier
        Trained machine learning model.
    data : pd.DataFrame
        Complete dataset with features and labels.
    categorical_features : List[str]
        List of categorical feature names to encode.
    slice_features : List[str]
        List of feature names to slice by.
    label : str, optional
        Name of the label column (default="salary").
    encoder : OneHotEncoder
        Fitted encoder for categorical features.
    lb : LabelBinarizer
        Fitted label binariser for target variable.
    output_file : str, optional
        If provided, saves all results to a text file.

    Returns
    Dict[str, pd.DataFrame]
        Dictionary mapping slice feature names to their results DataFrames.
    """
    all_results = {}

    for slice_feature in slice_features:
        print(f"Computing metrics for: {slice_feature}")
        results_df = compute_slice_metrics(
            model=model,
            data=data,
            categorical_features=categorical_features,
            slice_feature=slice_feature,
            label=label,
            encoder=encoder,
            lb=lb
        )
        all_results[slice_feature] = results_df
        print_slice_metrics(results_df)

    # Save to a file, if requested
    if output_file:
        with open(output_file, 'w') as f:
            for slice_feature, results_df in all_results.items():
                f.write(f"Model Performance by {slice_feature.upper()}")
                f.write(results_df.to_string(index=False))
        print(f"Results saved to: {output_file}")

    return all_results
