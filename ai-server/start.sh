#!/bin/bash
# AI Model Server 시작 스크립트

set -e

echo "🚀 Starting AI Model Server..."

# 환경 변수 확인
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Copying from .env.example..."
    cp .env.example .env
fi

# Docker Compose 실행
echo "📦 Starting with Docker Compose..."
docker-compose up -d ai-server

echo "✅ Server started successfully!"
echo ""
echo "🔗 API Server: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "💡 Useful commands:"
echo "  docker-compose logs -f ai-server  # View logs"
echo "  docker-compose down               # Stop server"
echo "  docker-compose restart ai-server  # Restart server"
