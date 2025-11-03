"""
Bias analysis utilities using Aequitas for fairness evaluation.
"""
import os
import pandas as pd
import numpy as np
from aequitas.group import Group
from aequitas.bias import Bias
from aequitas.fairness import Fairness


def prepare_aequitas_data(df, label_col, score_col, protected_attributes):
    """
    Prepare data in the format required by Aequitas.

    Inputs
    ------
    df : pd.DataFrame
        DataFrame containing predictions and protected attributes
    label_col : str
        Name of the column containing true labels
    score_col : str
        Name of the column containing predicted scores/labels
    protected_attributes : list[str]
        List of column names representing protected attributes

    Returns
    -------
    aequitas_df : pd.DataFrame
        DataFrame formatted for Aequitas analysis
    """
    aequitas_df = df.copy()

    # Ensure required columns exist
    required_cols = [label_col, score_col] + protected_attributes
    missing_cols = [col for col in required_cols if col not in aequitas_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Rename columns to Aequitas standard names if needed
    if label_col != 'label_value':
        aequitas_df['label_value'] = aequitas_df[label_col]

    if score_col != 'score':
        aequitas_df['score'] = aequitas_df[score_col]

    return aequitas_df


def calculate_bias_metrics(df, protected_attributes, label_col='label_value', score_col='score'):
    """
    Calculate bias and fairness metrics for protected attributes.

    Inputs
    ------
    df : pd.DataFrame
        DataFrame with predictions and protected attributes
    protected_attributes : list[str]
        List of protected attribute column names
    label_col : str
        Name of true label column (default: 'label_value')
    score_col : str
        Name of prediction column (default: 'score')

    Returns
    -------
    results : dict
        Dictionary containing group metrics, bias metrics, and fairness assessments
    """
    # Initialize Aequitas components
    g = Group()
    b = Bias()
    f = Fairness()

    # Calculate group metrics (TPR, FPR, PPV, etc. for each group)
    xtab, _ = g.get_crosstabs(df, attr_cols=protected_attributes)

    # Calculate bias metrics (disparities between groups)
    # Use min_metric for reference group selection (most privileged group)
    bias_df = b.get_disparity_min_metric(xtab, original_df=df)

    # Determine fairness (pass/fail for different fairness criteria)
    fairness_df = f.get_group_value_fairness(bias_df)

    results = {
        'group_metrics': xtab,
        'bias_metrics': bias_df,
        'fairness_assessment': fairness_df
    }

    return results


def analyze_model_bias(predictions, labels, data, protected_attributes):
    """
    Comprehensive bias analysis for a trained model.

    Inputs
    ------
    predictions : np.ndarray
        Model predictions
    labels : np.ndarray
        True labels
    data : pd.DataFrame
        Original data containing protected attributes
    protected_attributes : list[str]
        List of protected attribute column names to analyze

    Returns
    -------
    results : dict
        Dictionary containing bias analysis results
    """
    # Create analysis dataframe
    analysis_df = data[protected_attributes].copy()
    analysis_df['label_value'] = labels
    analysis_df['score'] = predictions

    # Calculate bias metrics
    results = calculate_bias_metrics(
        analysis_df,
        protected_attributes=protected_attributes
    )

    return results


def generate_bias_report(results, output_file=None):
    """
    Generate a human-readable bias analysis report.

    Inputs
    ------
    results : dict
        Dictionary containing bias analysis results from calculate_bias_metrics
    output_file : str, optional
        Path to save the report (default: None, prints to stdout)

    Returns
    -------
    report : str
        Formatted bias analysis report
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("MODEL BIAS ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Group Metrics Summary
    report_lines.append("GROUP METRICS SUMMARY")
    report_lines.append("-" * 80)
    group_metrics = results['group_metrics']

    # Display key metrics for each group
    key_cols = ['attribute_name', 'attribute_value', 'tpr', 'fpr', 'ppv', 'fdr', 'for']
    available_cols = [col for col in key_cols if col in group_metrics.columns]

    if available_cols:
        report_lines.append(group_metrics[available_cols].to_string(index=False))
    report_lines.append("")

    # Bias Metrics Summary
    report_lines.append("BIAS DISPARITY METRICS")
    bias_metrics = results['bias_metrics']

    # Show disparity ratios
    disparity_cols = [col for col in bias_metrics.columns if 'disparity' in col.lower()]
    display_cols = ['attribute_name', 'attribute_value'] + disparity_cols[:5]
    available_disparity_cols = [col for col in display_cols if col in bias_metrics.columns]

    if available_disparity_cols:
        report_lines.append(bias_metrics[available_disparity_cols].to_string(index=False))
    report_lines.append("")

    # Fairness Assessment
    if 'fairness_assessment' in results:
        report_lines.append("FAIRNESS ASSESSMENT")
        fairness = results['fairness_assessment']

        # Count pass/fail by attribute
        for attr in fairness['attribute_name'].unique():
            attr_data = fairness[fairness['attribute_name'] == attr]
            report_lines.append(f"\n{attr}:")
            report_lines.append(attr_data.to_string(index=False))

    report_lines.append("")

    report = "\n".join(report_lines)

    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"Bias report saved to: {output_file}")

    return report


def save_bias_metrics_csv(results, output_dir='.'):
    """
    Save bias metrics to CSV files.

    Inputs
    ------
    results : dict
        Dictionary containing bias analysis results
    output_dir : str
        Directory to save CSV files (default: current directory)
    """

    # Save group metrics
    group_file = os.path.join(output_dir, 'group_metrics.csv')
    results['group_metrics'].to_csv(group_file, index=False)
    print(f"Group metrics saved to: {group_file}")

    # Save bias metrics
    bias_file = os.path.join(output_dir, 'bias_metrics.csv')
    results['bias_metrics'].to_csv(bias_file, index=False)
    print(f"Bias metrics saved to: {bias_file}")

    # Save fairness assessment
    if 'fairness_assessment' in results:
        fairness_file = os.path.join(output_dir, 'fairness_assessment.csv')
        results['fairness_assessment'].to_csv(fairness_file, index=False)
        print(f"Fairness assessment saved to: {fairness_file}")
