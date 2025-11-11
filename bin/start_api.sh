#!/usr/bin/env bash

# Start API server in Docker
# Usage: ./bin/start_api.sh

set -e

cd "$(dirname "$0")/.."

# Build the Docker image
echo "Building Docker image..."
docker build -t census-income-api:latest .

# Run the API server in Docker
echo "Starting API server in Docker on port 8000..."
docker run --rm \
  -p 8000:8000 \
  -v "$(pwd)/model:/app/model" \
  -v "$(pwd)/data:/app/data" \
  --name census-api \
  census-income-api:latest