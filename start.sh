#!/usr/bin/env bash

# Startup script for Render.com deployment
# This script starts the FastAPI server which serves both API and frontend

set -e

echo "Starting Census Income Prediction API..."

# Get port from environment variable (Render sets this)
PORT=${PORT:-8000}

echo "Server will run on port $PORT"

# Check if model exists, if not provide helpful message
if [ ! -f "model/model.pkl" ]; then
    echo "WARNING: Model not found at model/model.pkl"
    echo "The API will start but predictions will fail until model is trained"
    echo "To train: python src/train_model.py"
fi

# Start the FastAPI application
exec uvicorn main:app --host 0.0.0.0 --port "$PORT"
