#!/usr/bin/env bash

# Run slice analysis in Docker
# Usage: ./bin/run_slice_analysis.sh

set -e

cd "$(dirname "$0")/.."

# Build the Docker image
echo "Building Docker image..."
docker build -t census-income-api:latest .

# Run slice analysis in Docker
echo "Running slice analysis in Docker..."
docker run --rm \
  -v "$(pwd)/model:/app/model" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/src:/app/src" \
  census-income-api:latest \
  python src/slice_analysis.py