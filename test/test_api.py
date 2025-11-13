"""
Unit tests for FastAPI endpoints.

Tests the API endpoints including prediction and health checks
"""

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture(scope="module")
def client():
    """Create a test client with startup/shutdown events."""
    with TestClient(app) as test_client:
        yield test_client


# Top-level test functions for sanity check script
def test_get_root():
    """Test GET method on root endpoint - should serve frontend HTML."""
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        # Root now serves HTML, not JSON
        assert "text/html" in response.headers["content-type"]


def test_post_prediction_below_50k():
    """Test POST method for prediction of income <=50K."""
    with TestClient(app) as client:
        payload = {
            "age": 25,
            "workclass": "private",
            "education": "hs_grad",
            "education_num": 9,
            "marital_status": "never_married",
            "occupation": "other_service",
            "relationship": "not_in_family",
            "race": "white",
            "sex": "male",
            "capital_gain": 0,
            "capital_loss": 0,
            "hours_per_week": 20,
            "native_country": "united_states"
        }
        response = client.post("/api/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "predicted_salary" in data
        assert data["predicted_salary"] == "<=50K"


def test_post_prediction_above_50k():
    """Test POST method for prediction of income >50K."""
    with TestClient(app) as client:
        payload = {
            "age": 52,
            "workclass": "self_emp_not_inc",
            "education": "masters",
            "education_num": 14,
            "marital_status": "married_civ_spouse",
            "occupation": "exec_managerial",
            "relationship": "husband",
            "race": "white",
            "sex": "male",
            "capital_gain": 15024,
            "capital_loss": 0,
            "hours_per_week": 50,
            "native_country": "united_states"
        }
        response = client.post("/api/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "predicted_salary" in data
        assert data["predicted_salary"] == ">50K"


class TestRootEndpoints:
    """Tests for basic API endpoints."""

    def test_get_api_root(self, client):
        """Test API root endpoint returns API information."""
        response = client.get("/api/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "endpoints" in data
        assert "/api/predict" in data["endpoints"]

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""

    def test_predict_low_income(self, client):
        """Test prediction for a profile likely to earn <=50K."""
        payload = {
            "age": 25,
            "workclass": "private",
            "education": "hs_grad",
            "education_num": 9,
            "marital_status": "never_married",
            "occupation": "other_service",
            "relationship": "not_in_family",
            "race": "white",
            "sex": "male",
            "capital_gain": 0,
            "capital_loss": 0,
            "hours_per_week": 20,
            "native_country": "united_states"
        }

        response = client.post("/api/predict", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "predicted_salary" in data
        assert "prediction_prob" in data
        assert data["predicted_salary"] == "<=50K"
        assert 0.0 <= data["prediction_prob"] <= 1.0

    def test_predict_high_income(self, client):
        """Test prediction for a profile likely to earn >50K."""
        payload = {
            "age": 52,
            "workclass": "self_emp_not_inc",
            "education": "hs_grad",
            "education_num": 9,
            "marital_status": "married_civ_spouse",
            "occupation": "exec_managerial",
            "relationship": "husband",
            "race": "white",
            "sex": "male",
            "capital_gain": 15024,
            "capital_loss": 0,
            "hours_per_week": 45,
            "native_country": "united_states"
        }

        response = client.post("/api/predict", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "predicted_salary" in data
        assert "prediction_prob" in data
        assert data["predicted_salary"]  == ">50K"
        assert 0.0 <= data["prediction_prob"] <= 1.0

    def test_predict_with_doctorate(self, client):
        """Test prediction for highly educated individual."""
        payload = {
            "age": 42,
            "workclass": "private",
            "education": "doctorate",
            "education_num": 16,
            "marital_status": "married_civ_spouse",
            "occupation": "prof_specialty",
            "relationship": "husband",
            "race": "white",
            "sex": "male",
            "capital_gain": 5178,
            "capital_loss": 0,
            "hours_per_week": 50,
            "native_country": "united_states"
        }

        response = client.post("/api/predict", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["predicted_salary"] == ">50K"

    def test_predict_female_profile(self, client):
        """Test prediction for female profile."""
        payload = {
            "age": 37,
            "workclass": "private",
            "education": "masters",
            "education_num": 14,
            "marital_status": "married_civ_spouse",
            "occupation": "exec_managerial",
            "relationship": "wife",
            "race": "white",
            "sex": "female",
            "capital_gain": 0,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "united_states"
        }

        response = client.post("/api/predict", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "predicted_salary" in data
        assert "prediction_prob" in data


class TestPredictValidation:
    """Tests for input validation on /predict endpoint."""

    def test_predict_missing_field(self, client):
        """Test that missing required field returns 422."""
        payload = {
            "age": 39,
            "workclass": "state_gov",
            # Missing other required fields
        }

        response = client.post("/api/predict", json=payload)
        assert response.status_code == 422  # Unprocessable Entity

    def test_predict_invalid_age(self, client):
        """Test that invalid age returns 422."""
        payload = {
            "age": 150,  # Invalid age
            "workclass": "private",
            "education": "bachelors",
            "education_num": 13,
            "marital_status": "never_married",
            "occupation": "adm_clerical",
            "relationship": "not_in_family",
            "race": "white",
            "sex": "male",
            "capital_gain": 0,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "united_states"
        }

        response = client.post("/api/predict", json=payload)
        assert response.status_code == 422

    def test_predict_invalid_workclass(self, client):
        """Test that invalid workclass returns 422."""
        payload = {
            "age": 39,
            "workclass": "InvalidWorkclass",  # Not in allowed values
            "education": "bachelors",
            "education_num": 13,
            "marital_status": "never_married",
            "occupation": "adm_clerical",
            "relationship": "not_in_family",
            "race": "white",
            "sex": "male",
            "capital_gain": 0,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "united_states"
        }

        response = client.post("/api/predict", json=payload)
        assert response.status_code == 422

    def test_predict_negative_capital_gain(self, client):
        """Test that negative capital gain returns 422."""
        payload = {
            "age": 39,
            "workclass": "private",
            "education": "bachelors",
            "education_num": 13,
            "marital_status": "never_married",
            "occupation": "adm_clerical",
            "relationship": "not_in_family",
            "race": "white",
            "sex": "male",
            "capital_gain": -1000,  # Invalid negative value
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "united_states"
        }

        response = client.post("/api/predict", json=payload)
        assert response.status_code == 422


class TestPredictResponseFormat:
    """Tests for prediction response format."""

    def test_response_schema(self, client):
        """Test that response matches expected schema."""
        payload = {
            "age": 39,
            "workclass": "state_gov",
            "education": "bachelors",
            "education_num": 13,
            "marital_status": "never_married",
            "occupation": "adm_clerical",
            "relationship": "not_in_family",
            "race": "white",
            "sex": "male",
            "capital_gain": 2174,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "united_states"
        }

        response = client.post("/api/predict", json=payload)
        assert response.status_code == 200

        data = response.json()

        # Check required fields exist
        assert "predicted_salary" in data
        assert "prediction_prob" in data

        # Check field types and values
        assert isinstance(data["predicted_salary"], str)
        assert isinstance(data["prediction_prob"], float)
        assert data["predicted_salary"] in ["<=50K", ">50K"]
        assert 0.0 <= data["prediction_prob"] <= 1.0

    def test_multiple_predictions_consistency(self, client):
        """Test that same input produces same prediction."""
        payload = {
            "age": 39,
            "workclass": "state_gov",
            "education": "bachelors",
            "education_num": 13,
            "marital_status": "never_married",
            "occupation": "adm_clerical",
            "relationship": "not_in_family",
            "race": "white",
            "sex": "male",
            "capital_gain": 2174,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "united_states"
        }

        # Make two predictions with the same input
        response1 = client.post("/api/predict", json=payload)
        response2 = client.post("/api/predict", json=payload)

        assert response1.status_code == 200
        assert response2.status_code == 200

        data1 = response1.json()
        data2 = response2.json()

        # Results should be identical (or very close for floating point)
        assert data1["predicted_salary"] == data2["predicted_salary"]
        # Use approximate equality for floating point values
        assert abs(data1["prediction_prob"] - data2["prediction_prob"]) < 1e-10
