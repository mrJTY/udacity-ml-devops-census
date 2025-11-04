#!/usr/bin/env bash
# Wrapper script to run bias analysis
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

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Get the project root directory (parent of bin)
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# Run the Python script from the src directory
.venv/bin/python src/analyze_bias.py "$@"
