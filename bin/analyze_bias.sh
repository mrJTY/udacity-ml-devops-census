#!/usr/bin/env bash
# Wrapper script to run bias analysis in Docker
#
# Usage: ./bin/analyze_bias.sh [options]
#
# Options:
#   --data PATH              Path to census data file (default: data/census.csv)
#   --model-dir PATH         Directory with model artifacts (default: model)
#   --output-dir PATH        Directory for results (default: bias_analysis)
#   --attributes ATTR...     Protected attributes to analyze (default: race sex)
#
# Examples:
#   ./bin/analyze_bias.sh
#   ./bin/analyze_bias.sh --attributes race sex education
#   ./bin/analyze_bias.sh --output-dir custom_output

set -e

cd "$(dirname "$0")/.."

# Build the Docker image
echo "Building Docker image..."
docker build -t census-income-api:latest .

# Run bias analysis in Docker
echo "Running bias analysis in Docker..."
docker run --rm \
  -e PYTHONPATH=/app \
  -v "$(pwd)/model:/app/model" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/src:/app/src" \
  census-income-api:latest \
  python src/analyze_bias.py "$@"
