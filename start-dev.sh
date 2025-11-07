#!/bin/bash

# Force development environment
# Usage: ./start-dev.sh [docker-compose-command]
# Example: ./start-dev.sh up -d
# Example: ./start-dev.sh down

set -e

echo "Starting in DEVELOPMENT mode..."
echo ""

# Copy dev environment file
if [ -f ".env.dev" ]; then
    cp .env.dev .env
    echo "Copied .env.dev to .env"
else
    echo "ERROR: .env.dev not found!"
    exit 1
fi

# Default to 'up -d' if no command provided
COMMAND=${1:-up -d}

echo "Running: docker compose -f docker-compose.yml -f docker-compose.dev.yml $COMMAND"
echo ""

docker compose -f docker-compose.yml -f docker-compose.dev.yml $COMMAND

if [ $? -eq 0 ] && [[ "$COMMAND" == *"up"* ]]; then
    echo ""
    echo "================================"
    echo "Development environment started!"
    echo "================================"
    echo ""
    echo "Access URLs:"
    echo "  - Main Application: http://localhost"
    echo "  - API (direct): http://localhost:3000"
    echo "  - Web (direct): http://localhost:3001"
    echo "  - Label Studio: http://localhost/label-studio"
    echo ""
    echo "To view logs: docker compose logs -f"
    echo "To stop: ./start-dev.sh down"
fi
