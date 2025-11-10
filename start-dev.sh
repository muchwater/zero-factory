#!/bin/bash

# Force development environment
# Usage: ./start-dev.sh [docker-compose-command]
# Example: ./start-dev.sh up -d
# Example: ./start-dev.sh down

set -e

echo "Starting in DEVELOPMENT mode..."
echo ""

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "ERROR: .env file not found!"
    echo "Please copy .env.dev to .env and configure your API keys:"
    echo "  cp .env.dev .env"
    echo ""
    echo "Then edit .env and set:"
    echo "  - NEXT_PUBLIC_KAKAO_MAP_KEY"
    echo "  - ADMIN_CODE"
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
    echo ""
    echo "To view logs: docker compose logs -f"
    echo "To stop: ./start-dev.sh down"
fi
