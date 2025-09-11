#!/bin/bash

# Chess Position Scanner API Deployment Script for 159.203.102.249
# Deploy to: https://api.chesspositionscanner.store

set -e

echo "🚀 Deploying Chess Position Scanner API to 159.203.102.249"
echo "=========================================================="

# Server configuration
SERVER_IP="159.203.102.249"
SERVER_USER="root"  # Change this to your actual username if different
DOMAIN="api.chesspositionscanner.store"
APP_DIR="/opt/chess-api"
DEPLOY_DIR="deployment_$(date +%Y%m%d_%H%M%S)"

echo "📋 Server Details:"
echo "   IP: $SERVER_IP"
echo "   User: $SERVER_USER"
echo "   Domain: $DOMAIN"
echo "   App Directory: $APP_DIR"
echo ""

# Check if required files exist
echo "📋 Checking required files..."
required_files=(
    "main_production.py"
    "Dockerfile"
    "requirements.txt"
    "models/color_classifier_simple.pt"
    "models/piece_classifier_simple.pt"
    "runs/occupancy_classifier/ResNet/ResNet.pt"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Required file missing: $file"
        exit 1
    fi
    echo "✅ Found: $file"
done

# Create deployment package
echo "📦 Creating deployment package..."
mkdir -p "$DEPLOY_DIR"

# Copy necessary files
cp main_production.py "$DEPLOY_DIR/"
cp Dockerfile "$DEPLOY_DIR/"
cp requirements.txt "$DEPLOY_DIR/"
cp -r models "$DEPLOY_DIR/"
cp -r runs "$DEPLOY_DIR/"

# Create production configuration
cat > "$DEPLOY_DIR/.env" << EOF
# Production Environment Variables
PORT=8000
WORKERS=2
LOG_LEVEL=info
PYTHONPATH=/app
PYTHONUNBUFFERED=1
EOF

# Create docker-compose for production
cat > "$DEPLOY_DIR/docker-compose.yml" << EOF
version: '3.8'

services:
  chess-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PORT=8000
      - WORKERS=2
      - LOG_LEVEL=info
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
EOF

# Create nginx configuration for your domain
cat > "$DEPLOY_DIR/nginx.conf" << EOF
server {
    listen 80;
    server_name $DOMAIN;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # Increase timeout for image processing
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # Increase client body size for large images
        client_max_body_size 10M;
    }
}
EOF

# Create systemd service file
cat > "$DEPLOY_DIR/chess-api.service" << EOF
[Unit]
Description=Chess Position Scanner API
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$APP_DIR
ExecStart=/usr/bin/docker-compose up
ExecStop=/usr/bin/docker-compose down
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Create deployment script for the server
cat > "$DEPLOY_DIR/server_setup.sh" << EOF
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
    curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-\$(uname -s)-\$(uname -m)" -o /usr/local/bin/docker-compose
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
mkdir -p $APP_DIR

# Copy files to app directory
echo "📋 Copying application files..."
cp -r * $APP_DIR/
cd $APP_DIR

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
certbot --nginx -d $DOMAIN --non-interactive --agree-tos --email admin@$DOMAIN

# Set up systemd service
echo "⚙️ Setting up systemd service..."
cp chess-api.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable chess-api

echo "🎉 Setup completed successfully!"
echo "🌐 API should be available at: https://$DOMAIN"
echo "🔍 Health check: https://$DOMAIN/health"
echo "📚 Documentation: https://$DOMAIN/docs"

# Show status
echo "📊 Service status:"
systemctl status chess-api --no-pager
docker ps
EOF

chmod +x "$DEPLOY_DIR/server_setup.sh"

# Create deployment instructions
cat > "$DEPLOY_DIR/DEPLOYMENT_STEPS.md" << EOF
# Chess Position Scanner API - Deployment Steps

## Server: 159.203.102.249
## Domain: api.chesspositionscanner.store

### Step 1: Copy files to server
\`\`\`bash
# Copy the deployment package to your server
scp -r $DEPLOY_DIR/* $SERVER_USER@$SERVER_IP:/tmp/chess-api/
\`\`\`

### Step 2: SSH into server and setup
\`\`\`bash
# SSH into your server
ssh $SERVER_USER@$SERVER_IP

# Move files to app directory
sudo mkdir -p $APP_DIR
sudo mv /tmp/chess-api/* $APP_DIR/
cd $APP_DIR

# Run the setup script
sudo chmod +x server_setup.sh
sudo ./server_setup.sh
\`\`\`

### Step 3: Verify deployment
\`\`\`bash
# Check API health
curl https://$DOMAIN/health

# Check API documentation
curl https://$DOMAIN/docs

# Test recognition (replace with your test image)
curl -X POST https://$DOMAIN/recognize_chess_position_with_corners \\
     -F "image=@test_image.jpg" \\
     -F "corners=[[100,100],[1100,100],[1100,1100],[100,1100]]" \\
     -F "turn=white"
\`\`\`

### Step 4: Monitor the API
\`\`\`bash
# View logs
sudo docker-compose logs -f

# Check service status
sudo systemctl status chess-api

# Check containers
sudo docker ps
\`\`\`

## Troubleshooting

### If API doesn't start:
\`\`\`bash
# Check Docker logs
sudo docker-compose logs

# Check if models are loaded
sudo docker-compose exec chess-api ls -la /app/models/
sudo docker-compose exec chess-api ls -la /app/runs/occupancy_classifier/ResNet/
\`\`\`

### If SSL fails:
\`\`\`bash
# Check domain DNS
nslookup $DOMAIN

# Test without SSL first
curl http://$DOMAIN/health
\`\`\`

### If nginx fails:
\`\`\`bash
# Check nginx config
sudo nginx -t

# Check nginx logs
sudo tail -f /var/log/nginx/error.log
\`\`\`

## API Endpoints

- **Health Check:** https://$DOMAIN/health
- **Recognition:** https://$DOMAIN/recognize_chess_position_with_corners
- **Documentation:** https://$DOMAIN/docs

## Files Location

- **App Directory:** $APP_DIR
- **Logs:** $APP_DIR/logs/
- **Nginx Config:** /etc/nginx/sites-available/chess-api
- **SSL Certificates:** /etc/letsencrypt/live/$DOMAIN/
EOF

echo "✅ Deployment package created: $DEPLOY_DIR"
echo ""
echo "📋 Next steps:"
echo "1. Copy files to server:"
echo "   scp -r $DEPLOY_DIR/* $SERVER_USER@$SERVER_IP:/tmp/chess-api/"
echo ""
echo "2. SSH into server and setup:"
echo "   ssh $SERVER_USER@$SERVER_IP"
echo "   sudo mkdir -p $APP_DIR"
echo "   sudo mv /tmp/chess-api/* $APP_DIR/"
echo "   cd $APP_DIR"
echo "   sudo chmod +x server_setup.sh"
echo "   sudo ./server_setup.sh"
echo ""
echo "3. Verify deployment:"
echo "   curl https://$DOMAIN/health"
echo ""
echo "🎉 Your API will be available at: https://$DOMAIN"
