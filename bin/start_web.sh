#!/usr/bin/env bash

# Start web frontend in Docker
# Usage: ./bin/start_web.sh

set -e

cd "$(dirname "$0")/.."

# Build the Docker image
echo "Building Docker image..."
docker build -t census-income-api:latest .

echo ""
echo "Starting web frontend on http://localhost:3000"
echo "Press Ctrl+C to stop the server"
echo ""
echo "Make sure the API server is running in another terminal with: ./bin/start_api.sh"
echo ""

# Run the web server in Docker
docker run --rm \
  -p 3000:3000 \
  -v "$(pwd)/web:/app/web" \
  --name census-web \
  census-income-api:latest \
  sh -c "cd /app/web && python -m http.server 3000"
