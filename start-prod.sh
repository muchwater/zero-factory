#!/bin/bash

# Force production environment
# Usage: ./start-prod.sh [docker-compose-command]
# Example: ./start-prod.sh up -d
# Example: ./start-prod.sh down

set -e

echo "Starting in PRODUCTION mode..."
echo ""

# Copy prod environment file
if [ -f ".env.prod" ]; then
    cp .env.prod .env
    echo "Copied .env.prod to .env"
else
    echo "ERROR: .env.prod not found!"
    exit 1
fi

# Default to 'up -d' if no command provided
COMMAND=${1:-up -d}

echo "Running: docker compose -f docker-compose.yml -f docker-compose.prod.yml $COMMAND"
echo ""

docker compose -f docker-compose.yml -f docker-compose.prod.yml $COMMAND

if [ $? -eq 0 ] && [[ "$COMMAND" == *"up"* ]]; then
    echo ""
    echo "================================="
    echo "Production environment started!"
    echo "================================="
    echo ""
    echo "Access URLs:"
    echo "  - Main Application: https://zeromap.store"
    echo "  - Label Studio: https://zeromap.store/label-studio"
    echo ""
    echo "To view logs: docker compose logs -f"
    echo "To stop: ./start-prod.sh down"
fi
