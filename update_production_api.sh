#!/bin/bash

echo "🚀 Updating Production API Response Format"
echo "=========================================="

# Configuration
SERVER="root@159.203.102.249"
API_DIR="/opt/chess-api"

echo "📦 Uploading fixed API file..."
scp main_production_fixed_response.py $SERVER:$API_DIR/main_production.py

echo "🔧 Restarting API on server..."
ssh $SERVER "cd $API_DIR && docker-compose restart"

echo "⏳ Waiting for API to restart..."
sleep 15

echo "🧪 Testing updated API..."
python test_fixed_response_deployment.py

echo "✅ Update complete!"
echo "🌐 API URL: https://api.chesspositionscanner.store/recognize_chess_position_with_corners"
