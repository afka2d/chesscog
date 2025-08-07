#!/bin/bash

# Deploy newly trained models to server
echo "🚀 Deploying newly trained models to server..."

# Server details
SERVER="root@159.203.102.249"
SERVER_PATH="/root/chesscog"

echo "📤 Copying piece classifier model..."
scp runs/piece_classifier/ResNet/ResNet.pt $SERVER:$SERVER_PATH/runs/piece_classifier/ResNet/ResNet.pt

if [ $? -eq 0 ]; then
    echo "✅ Piece classifier model deployed successfully"
else
    echo "❌ Failed to deploy piece classifier model"
    exit 1
fi

echo "📤 Copying occupancy classifier model..."
scp runs/occupancy_classifier/ResNet/ResNet.pt $SERVER:$SERVER_PATH/runs/occupancy_classifier/ResNet/ResNet.pt

if [ $? -eq 0 ]; then
    echo "✅ Occupancy classifier model deployed successfully"
else
    echo "❌ Failed to deploy occupancy classifier model"
    exit 1
fi

echo "🔄 Restarting chesscog service on server..."
ssh $SERVER "systemctl restart chesscog && systemctl status chesscog --no-pager"

if [ $? -eq 0 ]; then
    echo "✅ Service restarted successfully"
else
    echo "❌ Failed to restart service"
    exit 1
fi

echo "🎉 Model deployment completed!"
echo ""
echo "📊 Model Performance Summary:"
echo "   • Piece Classifier: 94.1% accuracy (was ~60.5%)"
echo "   • Occupancy Classifier: 99.8% accuracy"
echo ""
echo "🧪 Test the updated API with:"
echo "   curl -X POST \"http://159.203.102.249:8000/recognize\" \\"
echo "     -H \"Content-Type: multipart/form-data\" \\"
echo "     -F \"image=@IMG_4752.JPG\"" 