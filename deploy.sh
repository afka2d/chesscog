#!/bin/bash

# Chess Position Scanner API Deployment Script
# Deploy to: https://api.chesspositionscanner.store

set -e

echo "🚀 Starting Chess Position Scanner API Deployment"
echo "=================================================="

# Configuration
APP_NAME="chess-position-scanner-api"
DOMAIN="api.chesspositionscanner.store"
PORT=8000

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
DEPLOY_DIR="deployment_$(date +%Y%m%d_%H%M%S)"
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

# Create nginx configuration
cat > "$DEPLOY_DIR/nginx.conf" << EOF
server {
    listen 80;
    server_name $DOMAIN;

    location / {
        proxy_pass http://localhost:$PORT;
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
User=www-data
WorkingDirectory=/opt/chess-api
ExecStart=/usr/bin/docker-compose up
ExecStop=/usr/bin/docker-compose down
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Create deployment instructions
cat > "$DEPLOY_DIR/DEPLOYMENT_INSTRUCTIONS.md" << EOF
# Chess Position Scanner API - Deployment Instructions

## Server Setup

1. **Install Docker and Docker Compose:**
   \`\`\`bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-\$(uname -s)-\$(uname -m)" -o /usr/local/bin/docker-compose
   sudo chmod +x /usr/local/bin/docker-compose
   \`\`\`

2. **Install Nginx:**
   \`\`\`bash
   sudo apt update
   sudo apt install nginx
   \`\`\`

3. **Deploy the API:**
   \`\`\`bash
   # Copy deployment files to server
   sudo mkdir -p /opt/chess-api
   sudo cp -r * /opt/chess-api/
   cd /opt/chess-api
   
   # Build and start the API
   sudo docker-compose up -d --build
   
   # Configure nginx
   sudo cp nginx.conf /etc/nginx/sites-available/chess-api
   sudo ln -s /etc/nginx/sites-available/chess-api /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl reload nginx
   \`\`\`

4. **Set up SSL (recommended):**
   \`\`\`bash
   sudo apt install certbot python3-certbot-nginx
   sudo certbot --nginx -d $DOMAIN
   \`\`\`

5. **Monitor the API:**
   \`\`\`bash
   # Check logs
   sudo docker-compose logs -f
   
   # Check health
   curl https://$DOMAIN/health
   
   # Test recognition
   curl -X POST https://$DOMAIN/recognize_chess_position_with_corners \\
        -F "image=@test_image.jpg" \\
        -F "corners=[[100,100],[1100,100],[1100,1100],[100,1100]]" \\
        -F "turn=white"
   \`\`\`

## API Endpoints

- **Health Check:** \`GET https://$DOMAIN/health\`
- **Recognition:** \`POST https://$DOMAIN/recognize_chess_position_with_corners\`
- **Documentation:** \`GET https://$DOMAIN/docs\`

## Monitoring

- Logs are stored in \`/opt/chess-api/logs/\`
- Use \`docker-compose logs\` to view real-time logs
- Health check endpoint for monitoring services

## Troubleshooting

- Check Docker containers: \`sudo docker ps\`
- Check logs: \`sudo docker-compose logs\`
- Restart service: \`sudo docker-compose restart\`
- Check nginx: \`sudo nginx -t\`
EOF

echo "✅ Deployment package created: $DEPLOY_DIR"
echo ""
echo "📋 Next steps:"
echo "1. Copy the $DEPLOY_DIR folder to your server"
echo "2. Follow the instructions in $DEPLOY_DIR/DEPLOYMENT_INSTRUCTIONS.md"
echo "3. The API will be available at https://$DOMAIN"
echo ""
echo "🎉 Deployment package ready!"
