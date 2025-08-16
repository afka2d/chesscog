#!/bin/bash

# Deploy newly trained models to server
echo "🚀 Deploying newly trained models to server..."

# Server details
SERVER="root@159.203.102.249"
SERVER_PATH="/root/chesscog"

# First, create the necessary directories on the server
echo "📁 Creating model directories on server..."
ssh $SERVER "mkdir -p $SERVER_PATH/models/piece_classifier/ResNet && mkdir -p $SERVER_PATH/models/occupancy_classifier/ResNet"

# Deploy piece classifier model
echo "📤 Copying piece classifier model..."
scp /Users/tonyblum/code/chesscog/runs/piece_classifier/ResNet/ResNet.pt $SERVER:$SERVER_PATH/models/piece_classifier/ResNet/ResNet.pt

if [ $? -eq 0 ]; then
    echo "✅ Piece classifier model deployed successfully"
else
    echo "❌ Failed to deploy piece classifier model"
    exit 1
fi

# Deploy occupancy classifier model
echo "📤 Copying occupancy classifier model..."
scp /Users/tonyblum/code/chesscog/runs/occupancy_classifier/ResNet/ResNet.pt $SERVER:$SERVER_PATH/models/occupancy_classifier/ResNet/ResNet.pt

if [ $? -eq 0 ]; then
    echo "✅ Occupancy classifier model deployed successfully"
else
    echo "❌ Failed to deploy occupancy classifier model"
    exit 1
fi

# Restart the chesscog service
echo "🔄 Restarting chesscog service on server..."
ssh $SERVER "systemctl restart chesscog && systemctl status chesscog --no-pager"

if [ $? -eq 0 ]; then
    echo "✅ chesscog service restarted successfully"
else
    echo "❌ Failed to restart chesscog service"
    exit 1
fi

echo "🎉 Deployment complete! The API is now using the new models."