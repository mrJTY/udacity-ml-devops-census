#!/usr/bin/env bash

# Run tests
# Usage: ./bin/run_tests.sh [pytest options]

cd "$(dirname "$0")/.."
.venv/bin/python -m pytest test/ "$@"
