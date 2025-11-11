#!/usr/bin/env bash

# Run flake8 linting in Docker
# Usage: ./bin/run_flake8.sh

set -e

cd "$(dirname "$0")/.."

# Build the Docker image
echo "Building Docker image..."
docker build -t census-income-api:latest .

# Run flake8 in Docker
echo "Running flake8 in Docker..."
docker run --rm \
  -v "$(pwd):/app" \
  census-income-api:latest \
  sh -c "pip install flake8 && flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics && flake8 . --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics --exclude=.venv,__pycache__,.pytest_cache,.git"
