"""
Example script to run slice analysis on the trained model.

This demonstrates how to use the slice_metrics module to evaluate
model performance across different demographic groups.
"""

import pandas as pd
import pickle
from src.ml.slice_metrics import compute_slice_metrics, compute_multiple_slice_metrics, print_slice_metrics

CENSUS_FILE = "data/census.csv"
MODEL_FILE = "model/model.pkl"
ENCODER_FILE = "model/encoder.pkl"
LB_FILE = "model/lb.pkl"
OUTPUT_FILE = "doc/slice_analysis_results.txt"

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def main():
    """Run slice analysis on test data."""

    # Load data
    print("Loading data...")
    data = pd.read_csv(CENSUS_FILE, skipinitialspace=True)

    # Load model and encoders
    print("Loading trained model and encoders...")
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)

    with open(ENCODER_FILE, "rb") as f:
        encoder = pickle.load(f)

    with open(LB_FILE, "rb") as f:
        lb = pickle.load(f)

    print(f"Total samples in dataset: {len(data)}")
    print(f"Model type: {type(model).__name__}")

    ######################################################
    print("Slice Analysis for EDUCATION")
    ######################################################

    education_results = compute_slice_metrics(
        model=model,
        data=data,
        categorical_features=CAT_FEATURES,
        slice_feature="education",
        label="salary",
        encoder=encoder,
        lb=lb
    )

    print_slice_metrics(education_results)

    ######################################################
    print("Fairness Analysis for Protected Attributes")
    ######################################################

    protected_features = ["sex", "race"]

    all_results = compute_multiple_slice_metrics(
        model=model,
        data=data,
        categorical_features=CAT_FEATURES,
        slice_features=protected_features,
        label="salary",
        encoder=encoder,
        lb=lb,
        output_file=OUTPUT_FILE
    )

    ############################################
    print("Slice Analysis for WORKCLASS")
    ############################################

    workclass_results = compute_slice_metrics(
        model=model,
        data=data,
        categorical_features=CAT_FEATURES,
        slice_feature="workclass",
        label="salary",
        encoder=encoder,
        lb=lb
    )

    print_slice_metrics(workclass_results)

    #########################################################3
    print("SUMMARY: Performance Variation Across Slices")
    #########################################################3

    for feature_name, results_df in all_results.items():
        f1_min = results_df['f1'].min()
        f1_max = results_df['f1'].max()
        f1_range = f1_max - f1_min

        print(f"\n{feature_name.upper()}:")
        print(f"  F1 Score Range: {f1_min:.4f} to {f1_max:.4f} (range: {f1_range:.4f})")

        # Identify the best and worst performing slices
        best_slice = results_df.loc[results_df['f1'].idxmax()]
        worst_slice = results_df.loc[results_df['f1'].idxmin()]

        print(f"  Best performing: {best_slice['slice_value']} (F1={best_slice['f1']:.4f}, n={best_slice['n_samples']})")
        print(f"  Worst performing: {worst_slice['slice_value']} (F1={worst_slice['f1']:.4f}, n={worst_slice['n_samples']})")

    print("Analysis complete!")
    print(f"Detailed results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
