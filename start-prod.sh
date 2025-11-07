#!/bin/bash

# Force production environment
# Usage: ./start-prod.sh [docker-compose-command]
# Example: ./start-prod.sh up -d
# Example: ./start-prod.sh down

set -e

echo "Starting in PRODUCTION mode..."
echo ""

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "ERROR: .env file not found!"
    echo "Please copy .env.prod to .env and configure your API keys:"
    echo "  cp .env.prod .env"
    echo ""
    echo "Then edit .env and set:"
    echo "  - NEXT_PUBLIC_KAKAO_MAP_KEY"
    echo "  - ADMIN_CODE"
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
