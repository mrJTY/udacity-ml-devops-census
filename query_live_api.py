#!/usr/bin/env python3
"""
Script to query the live Census Income Prediction API.

This script sends a POST request to the deployed API and prints the response.
"""

import requests
import json
import sys

# API URL (can be overridden with command line argument)
DEFAULT_API_URL = "https://udacity-ml-devops-census.onrender.com/api/predict"

# Sample payload for prediction
SAMPLE_PAYLOAD = {
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


def query_api(api_url: str, payload: dict) -> None:
    """
    Send a POST request to the API and print the response.

    Args:
        api_url: The API endpoint URL
        payload: The JSON payload to send
    """
    print("QUERYING LIVE API")
    print(f"API URL: {api_url}")
    print(f"Payload:")
    print(json.dumps(payload, indent=2))

    try:
        # Send POST request
        response = requests.post(
            api_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30  # 30 second timeout
        )

        # Print status code
        print(f"HTTP Status Code: {response.status_code}")

        # Print response
        print(f"Response:")
        try:
            response_json = response.json()
            print(json.dumps(response_json, indent=2))

            # Print interpretation if successful
            if response.status_code == 200:
                predicted_salary = response_json.get("predicted_salary")
                prediction_prob = response_json.get("prediction_prob")

                print("\n" + "=" * 70)
                print("PREDICTION RESULT")
                print("=" * 70)
                print(f"Predicted Income: {predicted_salary}")
                print(f"Confidence: {prediction_prob:.2%}")
                print("=" * 70)
        except json.JSONDecodeError:
            print(response.text)

    except requests.exceptions.Timeout:
        print("ERROR: Request timed out after 30 seconds")
        print("The API might be spinning up (cold start). Try again in a moment.")
        sys.exit(1)

    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to the API")
        print("Check that the API URL is correct and the service is running.")
        sys.exit(1)

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Request failed: {e}")
        sys.exit(1)

    print()


def main():
    """Main function."""
    # Get API URL from command line or use default
    api_url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_API_URL

    # Query the API
    query_api(api_url, SAMPLE_PAYLOAD)


if __name__ == "__main__":
    main()
