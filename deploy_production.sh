#!/bin/bash
# Automatic deployment script for production chess APIs
# Deploys to ports 8010 (Marshall API) and 8011 (Corner Detection API)

set -e  # Exit on error

echo "=========================================================================="
echo "🚀 DEPLOYING PRODUCTION CHESS APIS"
echo "=========================================================================="
echo ""
echo "Target Server: 159.203.102.249"
echo "Deployment Directory: /root/chesscog"
echo "APIs: Marshall API (8010), Corner Detection API (8011)"
echo ""

# Server details
SERVER="159.203.102.249"
USER="root"
DEPLOY_DIR="/root/chesscog"

echo "📋 Step 1: Checking server connection..."
ssh -o ConnectTimeout=10 ${USER}@${SERVER} "echo '✅ Connected to server'"

echo ""
echo "📋 Step 2: Creating deployment directory structure..."
ssh ${USER}@${SERVER} "mkdir -p ${DEPLOY_DIR}/production_apis ${DEPLOY_DIR}/models ${DEPLOY_DIR}/models_marshall_improved"

echo ""
echo "📋 Step 3: Uploading API files..."
scp working_marshall_api.py ${USER}@${SERVER}:${DEPLOY_DIR}/production_apis/marshall_api_prod.py
scp robust_corner_api.py ${USER}@${SERVER}:${DEPLOY_DIR}/production_apis/corner_api_prod.py
scp final_optimized_corner_detector.py ${USER}@${SERVER}:${DEPLOY_DIR}/production_apis/
echo "✅ API files uploaded"

echo ""
echo "📋 Step 4: Checking for model files on server..."
ssh ${USER}@${SERVER} "ls -lh ${DEPLOY_DIR}/models/*.pt 2>/dev/null || echo 'Models not found, will upload...'"

echo ""
echo "📋 Step 5: Uploading model files (this may take a few minutes)..."
# Upload original models
if [ -d "models" ]; then
    scp models/color_classifier_simple.pt ${USER}@${SERVER}:${DEPLOY_DIR}/models/ 2>/dev/null || echo "Color model already exists or upload failed"
    scp models/piece_classifier_simple.pt ${USER}@${SERVER}:${DEPLOY_DIR}/models/ 2>/dev/null || echo "Piece model already exists or upload failed"
fi

# Upload Marshall improved models
if [ -d "models_marshall_improved" ]; then
    scp models_marshall_improved/occupancy_marshall.pt ${USER}@${SERVER}:${DEPLOY_DIR}/models_marshall_improved/ 2>/dev/null || echo "Occupancy model already exists or upload failed"
    scp models_marshall_improved/piece_classifier_balanced.pt ${USER}@${SERVER}:${DEPLOY_DIR}/models_marshall_improved/ 2>/dev/null || echo "Balanced piece model already exists or upload failed"
fi

# Upload YOLO model for corner detection
if [ -f "runs/detect/train/weights/best.pt" ]; then
    ssh ${USER}@${SERVER} "mkdir -p ${DEPLOY_DIR}/runs/detect/train/weights"
    scp runs/detect/train/weights/best.pt ${USER}@${SERVER}:${DEPLOY_DIR}/runs/detect/train/weights/ 2>/dev/null || echo "YOLO model already exists or upload failed"
fi

echo "✅ Model files uploaded"

echo ""
echo "📋 Step 6: Setting up Python virtual environment..."
ssh ${USER}@${SERVER} << 'ENDSSH'
cd /root/chesscog
if [ ! -d "venv" ]; then
    echo "Creating new virtual environment..."
    python3 -m venv venv
fi
echo "✅ Virtual environment ready"
ENDSSH

echo ""
echo "📋 Step 7: Installing dependencies..."
ssh ${USER}@${SERVER} << 'ENDSSH'
cd /root/chesscog
source venv/bin/activate
pip install --upgrade pip
pip install fastapi uvicorn torch torchvision opencv-python pillow numpy ultralytics
echo "✅ Dependencies installed"
ENDSSH

echo ""
echo "📋 Step 8: Creating systemd service files..."

# Create Marshall API service
ssh ${USER}@${SERVER} "cat > /etc/systemd/system/marshall-api.service << 'EOF'
[Unit]
Description=Marshall Chess Recognition API (Production Port 8010)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/chesscog
Environment=\"PATH=/root/chesscog/venv/bin\"
ExecStart=/root/chesscog/venv/bin/python3 /root/chesscog/production_apis/marshall_api_prod.py 8010
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
"

# Create Corner Detection API service
ssh ${USER}@${SERVER} "cat > /etc/systemd/system/corner-detection-api.service << 'EOF'
[Unit]
Description=Robust Corner Detection API (Production Port 8011)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/chesscog
Environment=\"PATH=/root/chesscog/venv/bin\"
ExecStart=/root/chesscog/venv/bin/python3 /root/chesscog/production_apis/corner_api_prod.py 8011
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
"

echo "✅ Systemd service files created"

echo ""
echo "📋 Step 9: Enabling and starting services..."
ssh ${USER}@${SERVER} << 'ENDSSH'
systemctl daemon-reload
systemctl enable marshall-api.service
systemctl enable corner-detection-api.service
systemctl restart marshall-api.service
systemctl restart corner-detection-api.service
echo "✅ Services started"
ENDSSH

echo ""
echo "📋 Step 10: Checking service status..."
sleep 3
ssh ${USER}@${SERVER} "systemctl status marshall-api.service --no-pager | head -20"
echo ""
ssh ${USER}@${SERVER} "systemctl status corner-detection-api.service --no-pager | head -20"

echo ""
echo "📋 Step 11: Testing API endpoints..."
sleep 2
echo ""
echo "Testing Marshall API (port 8010)..."
curl -s http://159.203.102.249:8010/health | python3 -m json.tool || echo "⚠️  Marshall API not responding yet"

echo ""
echo "Testing Corner Detection API (port 8011)..."
curl -s http://159.203.102.249:8011/health | python3 -m json.tool || echo "⚠️  Corner API not responding yet"

echo ""
echo "=========================================================================="
echo "✅ DEPLOYMENT COMPLETE!"
echo "=========================================================================="
echo ""
echo "📍 PRODUCTION API ENDPOINTS:"
echo "   Marshall Chess Recognition: http://159.203.102.249:8010"
echo "   Corner Detection: http://159.203.102.249:8011"
echo ""
echo "🔧 MANAGEMENT COMMANDS:"
echo "   Check status: ssh root@159.203.102.249 'systemctl status marshall-api corner-detection-api'"
echo "   View logs: ssh root@159.203.102.249 'journalctl -u marshall-api -f'"
echo "   Restart: ssh root@159.203.102.249 'systemctl restart marshall-api corner-detection-api'"
echo "   Stop: ssh root@159.203.102.249 'systemctl stop marshall-api corner-detection-api'"
echo ""
echo "✅ Services will auto-restart on server reboot"
echo ""
