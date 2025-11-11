#!/usr/bin/env bash

# Train model in Docker
# Usage: ./bin/train_model.sh

set -e

cd "$(dirname "$0")/.."

# Build the Docker image
echo "Building Docker image..."
docker build -t census-income-api:latest .

# Run training in Docker
echo "Training model in Docker..."
docker run --rm \
  -v "$(pwd)/model:/app/model" \
  -v "$(pwd)/data:/app/data" \
  census-income-api:latest \
  python src/train_model.py