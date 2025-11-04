"""
Data transformations for census data normalization.

Converts original census format to snake_case for consistency.
"""

import pandas as pd
import re


def to_snake_case(text: str) -> str:
    """
    Convert text to snake_case.

    Examples:
    - "State-gov" -> "state_gov"
    - "Self-emp-not-inc" -> "self_emp_not_inc"
    - "Married-civ-spouse" -> "married_civ_spouse"
    - "United-States" -> "united_states"
    """
    # Convert to lowercase and replace hyphens with underscores
    return text.lower().replace("-", "_").replace(" ", "_")


def transform_census_data_to_snake_case(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform census DataFrame to use snake_case values.

    Parameters
    df : pd.DataFrame
        Original census data with mixed case and hyphens.

    Returns
    pd.DataFrame
        Transformed data with snake_case categorical values.
    """
    df = df.copy()

    # List of categorical columns to transform
    categorical_columns = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"
    ]

    # Transform each categorical column
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].apply(to_snake_case)

    # Also transform salary column if present
    if "salary" in df.columns:
        df["salary"] = df["salary"].apply(lambda x: x.strip())

    return df
