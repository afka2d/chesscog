#!/bin/bash

echo "🚀 Deploying Fixed Response Format API to Production"
echo "=================================================="

# Configuration
SERVER="root@159.203.102.249"
API_DIR="/opt/chess-api"
DEPLOY_DIR="deployment_fixed_response_20250910_193121"

echo "📦 Uploading files to server..."
scp -r $DEPLOY_DIR/* $SERVER:$API_DIR/

echo "🔧 Restarting API on server..."
ssh $SERVER "cd $API_DIR && docker-compose down && docker-compose up -d"

echo "⏳ Waiting for API to start..."
sleep 10

echo "🧪 Testing deployment..."
python test_fixed_response_deployment.py

echo "✅ Deployment complete!"
echo "🌐 API URL: https://api.chesspositionscanner.store/recognize_chess_position_with_corners"
