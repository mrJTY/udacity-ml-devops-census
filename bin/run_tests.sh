#!/usr/bin/env bash

# Run tests in Docker
# Usage: ./bin/run_tests.sh [pytest options]

set -e

# Get the project root directory
cd "$(dirname "$0")/.."

# Build the Docker image if it doesn't exist
echo "Building Docker image..."
docker build -t census-income-api:latest .

# Run tests in Docker container
echo "Running tests in Docker..."
docker run --rm \
  -v "$(pwd)/test:/app/test" \
  -v "$(pwd)/src:/app/src" \
  census-income-api:latest \
  pytest test/ "$@"
