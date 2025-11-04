import pickle
import pandas as pd
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.model_request import ModelRequest
from src.model_response import ModelResponse
from src.ml.data import process_data


# Load model artifacts on startup
MODEL_PATH = Path("model/model.pkl")
ENCODER_PATH = Path("model/encoder.pkl")
LB_PATH = Path("model/lb.pkl")

# Global variables for model artifacts
model = None
encoder = None
lb = None

# Categorical features used in training
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts on startup and cleanup on shutdown."""
    global model, encoder, lb

    # Startup: Load model artifacts
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(ENCODER_PATH, "rb") as f:
            encoder = pickle.load(f)
        with open(LB_PATH, "rb") as f:
            lb = pickle.load(f)
        print("Model artifacts loaded successfully")
    except FileNotFoundError as e:
        print(f"Warning: Could not load model artifacts: {e}")
        print("Run 'python src/train_model.py' to train and save the model")

    yield

    # Shutdown: Cleanup (if needed)
    print("Shutting down application")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Census Income Prediction API",
    description="Predicts whether income exceeds $50K/yr based on census data",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Census Income Prediction API",
        "endpoints": {
            "/": "API information",
            "/predict": "POST - Make income predictions",
            "/health": "GET - Health check",
            "/docs": "API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_loaded = model is not None and encoder is not None and lb is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded
    }


@app.post("/predict", response_model=ModelResponse)
async def predict_income(request: ModelRequest):
    """
    Predict income category based on census features.

    Returns:
        - predicted_salary: "<=50K" or ">50K"
        - prediction_prob: Confidence score (0.0 to 1.0)
    """
    # Check if model is loaded
    if model is None or encoder is None or lb is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )

    try:
        # Convert request to DataFrame (all values already in snake_case)
        input_data = pd.DataFrame([{
            "age": request.age,
            "workclass": request.workclass,
            "fnlgt": request.fnlgt,
            "education": request.education,
            "education-num": request.education_num,
            "marital-status": request.marital_status,
            "occupation": request.occupation,
            "relationship": request.relationship,
            "race": request.race,
            "sex": request.sex,
            "capital-gain": request.capital_gain,
            "capital-loss": request.capital_loss,
            "hours-per-week": request.hours_per_week,
            "native-country": request.native_country
        }])

        # Process data using the same pipeline as training
        X, _, _, _ = process_data(
            input_data,
            categorical_features=CAT_FEATURES,
            label=None,
            training=False,
            encoder=encoder,
            lb=lb
        )

        # Make prediction
        predictions = model.predict(X)
        prediction_proba = model.predict_proba(X)[0]

        # Get the probability of the predicted class
        prediction = predictions[0]
        confidence = prediction_proba[prediction]

        # Convert prediction to salary category
        salary_category = lb.inverse_transform(predictions)[0]

        return ModelResponse(
            predicted_salary=salary_category,
            prediction_prob=float(confidence)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )
