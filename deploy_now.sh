#!/bin/bash

# Quick deployment script for 159.203.102.249
# This script will copy files and guide you through the deployment

set -e

SERVER_IP="159.203.102.249"
SERVER_USER="root"
DEPLOY_DIR="deployment_20250910_181959"

echo "🚀 Quick Deploy to $SERVER_IP"
echo "=============================="

# Check if deployment package exists
if [ ! -d "$DEPLOY_DIR" ]; then
    echo "❌ Deployment package not found. Run ./deploy_to_server.sh first."
    exit 1
fi

echo "📦 Copying files to server..."
echo "This may take a few minutes due to model files (25MB+)..."

# Copy files to server
scp -r "$DEPLOY_DIR"/* "$SERVER_USER@$SERVER_IP:/tmp/chess-api/"

echo "✅ Files copied successfully!"
echo ""
echo "🔧 Now SSH into your server and run the setup:"
echo ""
echo "ssh $SERVER_USER@$SERVER_IP"
echo "sudo mkdir -p /opt/chess-api"
echo "sudo mv /tmp/chess-api/* /opt/chess-api/"
echo "cd /opt/chess-api"
echo "sudo chmod +x server_setup.sh"
echo "sudo ./server_setup.sh"
echo ""
echo "🎯 After setup, your API will be available at:"
echo "https://api.chesspositionscanner.store/recognize_chess_position_with_corners"
echo ""
echo "🔍 Test with: curl https://api.chesspositionscanner.store/health"
