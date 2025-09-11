#!/bin/bash

# Server setup script for Chess Position Scanner API
# Run this on the server after copying the deployment files

set -e

echo "🔧 Setting up Chess Position Scanner API on server..."

# Update system
echo "📦 Updating system packages..."
apt-get update

# Install Docker
echo "🐳 Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
fi

# Install Docker Compose
echo "🐳 Installing Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
fi

# Install Nginx
echo "🌐 Installing Nginx..."
apt-get install -y nginx

# Install curl for health checks
echo "🔧 Installing curl..."
apt-get install -y curl

# Create app directory
echo "📁 Creating app directory..."
mkdir -p /opt/chess-api

# Copy files to app directory
echo "📋 Copying application files..."
cp -r * /opt/chess-api/
cd /opt/chess-api

# Set permissions
echo "🔐 Setting permissions..."
chmod +x server_setup.sh

# Build and start the API
echo "🚀 Building and starting the API..."
docker-compose up -d --build

# Wait for API to start
echo "⏳ Waiting for API to start..."
sleep 30

# Check if API is running
echo "🔍 Checking API health..."
if curl -f http://localhost:8000/health; then
    echo "✅ API is running successfully!"
else
    echo "❌ API failed to start. Check logs with: docker-compose logs"
    exit 1
fi

# Configure nginx
echo "🌐 Configuring Nginx..."
cp nginx.conf /etc/nginx/sites-available/chess-api
ln -sf /etc/nginx/sites-available/chess-api /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

# Test nginx configuration
nginx -t

# Reload nginx
systemctl reload nginx

# Install certbot for SSL
echo "🔒 Installing SSL certificate..."
apt-get install -y certbot python3-certbot-nginx

# Get SSL certificate
certbot --nginx -d api.chesspositionscanner.store --non-interactive --agree-tos --email admin@api.chesspositionscanner.store

# Set up systemd service
echo "⚙️ Setting up systemd service..."
cp chess-api.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable chess-api

echo "🎉 Setup completed successfully!"
echo "🌐 API should be available at: https://api.chesspositionscanner.store"
echo "🔍 Health check: https://api.chesspositionscanner.store/health"
echo "📚 Documentation: https://api.chesspositionscanner.store/docs"

# Show status
echo "📊 Service status:"
systemctl status chess-api --no-pager
docker ps
