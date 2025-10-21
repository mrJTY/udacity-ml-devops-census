# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

CENSUS_FILE = "data/census.csv"

def main():
    data = pd.read_csv(CENSUS_FILE, skipinitialspace=True)

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20)

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
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    # Train and save a model.
    model = train_model(X_train, y_train)

    # Evaluate the model on the test set
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    print(f"Model Performance on Test Set:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F-beta: {fbeta:.4f}")

    # Save the model and encoders
    with open("model/model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("model/encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

    with open("model/lb.pkl", "wb") as f:
        pickle.dump(lb, f)

    print("Model and encoders saved")

if __name__ == "__main__":
    main()