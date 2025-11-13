# Script to train machine learning model with improvements

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelBinarizer
import pandas as pd
import numpy as np
import pickle

from src.ml.data import process_data
from src.ml.model import compute_model_metrics, inference
from src.ml.transformations import transform_census_data_to_snake_case

CENSUS_FILE = "data/census.csv"


def process_data_with_scaling(
    X, categorical_features=[], label=None, training=True,
    encoder=None, lb=None, scaler=None
):
    """Enhanced data processing with feature scaling."""
    if label is not None:
        y = X[label]
        X_features = X.drop([label], axis=1)
    else:
        y = np.array([])
        X_features = X

    # Separate categorical and continuous
    X_categorical = X_features[categorical_features].values
    continuous_cols = X_features.drop(categorical_features, axis=1).columns.tolist()
    X_continuous = X_features[continuous_cols].values

    if training:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        scaler = StandardScaler()
        lb = LabelBinarizer()

        X_categorical_encoded = encoder.fit_transform(X_categorical)
        X_continuous_scaled = scaler.fit_transform(X_continuous)

        if len(y) > 0:
            y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical_encoded = encoder.transform(X_categorical)
        X_continuous_scaled = scaler.transform(X_continuous)

        if len(y) > 0:
            try:
                y = lb.transform(y.values).ravel()
            except AttributeError:
                pass

    # Concatenate scaled continuous and encoded categorical
    X_processed = np.concatenate([X_continuous_scaled, X_categorical_encoded], axis=1)

    return X_processed, y, encoder, lb, scaler


def main():
    # Load data
    print("Loading data...")
    data = pd.read_csv(CENSUS_FILE, skipinitialspace=True)

    # Transform data to snake_case format
    print("Transforming data to snake_case...")
    data = transform_census_data_to_snake_case(data)

    # Drop census sampling weight
    data = data.drop(columns=['fnlgt'], errors='ignore')

    print(f"Dataset shape: {data.shape}")
    print(f"Class distribution:\n{data['salary'].value_counts()}")

    # Stratified split to maintain class balance
    train, test = train_test_split(data, test_size=0.20, random_state=42, stratify=data['salary'])

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

    # Process data with scaling
    print("Processing data with feature scaling...")
    X_train, y_train, encoder, lb, scaler = process_data_with_scaling(
        train, categorical_features=cat_features, label="salary", training=True
    )

    X_test, y_test, _, _, _ = process_data_with_scaling(
        test, categorical_features=cat_features, label="salary",
        training=False, encoder=encoder, lb=lb, scaler=scaler
    )

    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")

    # Train model with improved parameters
    print("\nTraining Random Forest with improved parameters...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',  # Handle class imbalance
        random_state=42,
        n_jobs=-1
    )

    # Perform 5-fold cross validation
    print("Performing 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    print(f"CV F1 Scores: {cv_scores}")
    print(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Train on full training set
    print("Training on full training set...")
    model.fit(X_train, y_train)

    # Evaluate on test set
    print("Evaluating on test set...")
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    print("MODEL PERFORMANCE")
    print(f"Cross-Validation F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print("Test Set Metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {fbeta:.4f}")

    # Save the model and encoders
    print("Saving model and artifacts...")
    with open("model/model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("model/encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

    with open("model/lb.pkl", "wb") as f:
        pickle.dump(lb, f)

    with open("model/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("Model, encoder, label binarizer, and scaler saved")
    print("Training complete!")


if __name__ == "__main__":
    main()
