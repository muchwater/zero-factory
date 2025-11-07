#!/bin/bash

# Auto-detect environment and start services
# This script automatically detects whether to use dev or prod configuration

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect environment
detect_environment() {
    # Check if ENV variable is set
    if [ -n "$ENV" ]; then
        echo "$ENV"
        return
    fi

    # Check if .env.local exists (custom environment marker)
    if [ -f ".env.local" ]; then
        grep -q "NODE_ENV=production" .env.local && echo "prod" || echo "dev"
        return
    fi

    # Check hostname patterns
    if [[ $(hostname) == *"prod"* ]] || [[ $(hostname) == *"production"* ]]; then
        echo "prod"
        return
    fi

    if [[ $(hostname) == *"dev"* ]] || [[ $(hostname) == *"development"* ]]; then
        echo "dev"
        return
    fi

    # Check if SSL certificates exist (indicates production)
    if [ -d "./certbot/conf/live" ] && [ "$(ls -A ./certbot/conf/live)" ]; then
        print_info "SSL certificates detected, assuming production environment"
        echo "prod"
        return
    fi

    # Default to development
    print_warning "Could not detect environment, defaulting to development"
    echo "dev"
}

# Main script
main() {
    print_info "=== Zero Factory Docker Compose Starter ==="
    echo ""

    # Detect environment
    ENV=$(detect_environment)

    if [ "$ENV" == "prod" ]; then
        print_success "Environment: PRODUCTION"
        COMPOSE_FILE="docker-compose.yml"
        COMPOSE_OVERRIDE="docker-compose.prod.yml"
        ENV_FILE=".env.prod"
    else
        print_success "Environment: DEVELOPMENT"
        COMPOSE_FILE="docker-compose.yml"
        COMPOSE_OVERRIDE="docker-compose.dev.yml"
        ENV_FILE=".env.dev"
    fi

    echo ""
    print_info "Configuration:"
    echo "  - Base config: $COMPOSE_FILE"
    echo "  - Override config: $COMPOSE_OVERRIDE"
    echo "  - Environment file: $ENV_FILE"
    echo ""

    # Check if environment file exists
    if [ ! -f "$ENV_FILE" ]; then
        print_error "Environment file $ENV_FILE not found!"
        print_info "Please create $ENV_FILE or copy from .env.example"
        exit 1
    fi

    # Copy environment file to .env (required by docker-compose)
    print_info "Copying $ENV_FILE to .env..."
    cp "$ENV_FILE" .env

    # Parse command line arguments
    COMMAND=${1:-up -d}

    print_info "Running: docker compose -f $COMPOSE_FILE -f $COMPOSE_OVERRIDE $COMMAND"
    echo ""

    # Run docker compose
    docker compose -f "$COMPOSE_FILE" -f "$COMPOSE_OVERRIDE" $COMMAND

    if [ $? -eq 0 ]; then
        echo ""
        print_success "Docker Compose command completed successfully!"

        if [[ "$COMMAND" == *"up"* ]]; then
            echo ""
            print_info "Services are starting..."
            if [ "$ENV" == "prod" ]; then
                print_info "Production URL: https://zeromap.store"
            else
                print_info "Development URL: http://localhost"
                print_info "API Direct Access: http://localhost:3000"
                print_info "Web Direct Access: http://localhost:3001"
            fi
        fi
    else
        print_error "Docker Compose command failed!"
        exit 1
    fi
}

# Run main function
main "$@"
