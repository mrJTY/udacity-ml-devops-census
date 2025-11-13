"""
Script to analyze model bias using Aequitas.

This script loads a trained model and performs fairness analysis
across protected attributes like race, sex, and other demographic features.
"""
import pandas as pd
import pickle
import argparse
import os
from src.ml.data import process_data
from src.ml.model import inference
from src.ml.bias import analyze_model_bias, generate_bias_report, save_bias_metrics_csv


def load_model_artifacts(model_dir='model'):
    """
    Load trained model and preprocessing artifacts.

    Inputs
    ------
    model_dir : str
        Directory containing model artifacts

    Returns
    -------
    model : sklearn model
        Trained model
    encoder : OneHotEncoder
        Fitted encoder for categorical features
    lb : LabelBinarizer
        Fitted label binarizer
    """
    with open(os.path.join(model_dir, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)

    with open(os.path.join(model_dir, 'encoder.pkl'), 'rb') as f:
        encoder = pickle.load(f)

    with open(os.path.join(model_dir, 'lb.pkl'), 'rb') as f:
        lb = pickle.load(f)

    return model, encoder, lb


def main():
    """
    Main function to run bias analysis.
    """
    parser = argparse.ArgumentParser(description='Analyze model bias using Aequitas')
    parser.add_argument(
        '--data',
        type=str,
        default='data/census.csv',
        help='Path to census data file'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='model',
        help='Directory containing model artifacts'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='doc/bias_analysis',
        help='Directory to save bias analysis results'
    )
    parser.add_argument(
        '--attributes',
        type=str,
        nargs='+',
        default=['race', 'sex'],
        help='Protected attributes to analyze for bias'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("MODEL BIAS ANALYSIS")
    print(f"Data file: {args.data}")
    print(f"Model directory: {args.model_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Protected attributes: {', '.join(args.attributes)}")
    print()

    # Load data
    print("Loading census data...")
    data = pd.read_csv(args.data, skipinitialspace=True)
    print(f"Loaded {len(data)} records")
    print()

    # Load model artifacts
    print("Loading model and preprocessing artifacts...")
    model, encoder, lb = load_model_artifacts(args.model_dir)
    print("Model loaded successfully")
    print()

    # Define categorical features (same as training)
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # Process data for inference
    print("Processing data...")
    X, y, _, _ = process_data(
        data,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )
    print(f"Processed {len(X)} samples")
    print()

    # Generate predictions
    print("Generating predictions...")
    predictions = inference(model, X)
    print(f"Generated {len(predictions)} predictions")
    print()

    # Verify protected attributes exist in data
    available_attributes = [attr for attr in args.attributes if attr in data.columns]
    missing_attributes = [attr for attr in args.attributes if attr not in data.columns]

    if missing_attributes:
        print(f"Warning: The following attributes are not in the dataset: {missing_attributes}")

    if not available_attributes:
        print("Error: No valid protected attributes found in dataset")
        return

    print(f"Analyzing bias for attributes: {', '.join(available_attributes)}")
    print()

    # Perform bias analysis
    print("Performing bias analysis with Aequitas...")
    try:
        results = analyze_model_bias(
            predictions=predictions,
            labels=y,
            data=data,
            protected_attributes=available_attributes
        )
        print("Bias analysis completed")
        print()

        # Generate and display report
        print("Generating bias report...")
        report_file = os.path.join(args.output_dir, 'bias_report.txt')
        report = generate_bias_report(results, output_file=report_file)
        print(report)

        # Save metrics to CSV
        print("\nSaving bias metrics to CSV files...")
        save_bias_metrics_csv(results, output_dir=args.output_dir)
        print()

        print("ANALYSIS COMPLETE")
        print(f"Results saved to: {args.output_dir}")

    except Exception as e:
        print(f"Error during bias analysis: {str(e)}")
        raise


if __name__ == "__main__":
    main()
