#!/bin/bash

# Chess Position Scanner API Deployment Script

set -e

echo "=== Chess Position Scanner API Deployment ==="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Error: Docker Compose is not installed"
    exit 1
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p models
mkdir -p ssl

# Check if models exist
if [ ! -d "models" ] || [ -z "$(ls -A models 2>/dev/null)" ]; then
    echo "Warning: Models directory is empty. You need to download the models first."
    echo "Please run the model download script or copy your models to the models/ directory."
fi

# Build and start the API
echo "Building and starting the API..."
docker-compose -f docker-compose.api.yml build

echo "Starting services..."
docker-compose -f docker-compose.api.yml up -d

# Wait for the API to be ready
echo "Waiting for API to be ready..."
sleep 10

# Test the API
echo "Testing API health..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ API is running successfully!"
    echo "API URL: http://localhost:8000"
    echo "Health check: http://localhost:8000/health"
    echo "API docs: http://localhost:8000/docs"
else
    echo "❌ API health check failed"
    echo "Checking logs..."
    docker-compose -f docker-compose.api.yml logs chess-api
    exit 1
fi

echo ""
echo "=== Deployment Complete ==="
echo "To stop the API: docker-compose -f docker-compose.api.yml down"
echo "To view logs: docker-compose -f docker-compose.api.yml logs -f"
echo "To restart: docker-compose -f docker-compose.api.yml restart" 