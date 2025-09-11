# Chess Position Scanner API - Deployment Steps

## Server: 159.203.102.249
## Domain: api.chesspositionscanner.store

### Step 1: Copy files to server
```bash
# Copy the deployment package to your server
scp -r deployment_20250910_181959/* root@159.203.102.249:/tmp/chess-api/
```

### Step 2: SSH into server and setup
```bash
# SSH into your server
ssh root@159.203.102.249

# Move files to app directory
sudo mkdir -p /opt/chess-api
sudo mv /tmp/chess-api/* /opt/chess-api/
cd /opt/chess-api

# Run the setup script
sudo chmod +x server_setup.sh
sudo ./server_setup.sh
```

### Step 3: Verify deployment
```bash
# Check API health
curl https://api.chesspositionscanner.store/health

# Check API documentation
curl https://api.chesspositionscanner.store/docs

# Test recognition (replace with your test image)
curl -X POST https://api.chesspositionscanner.store/recognize_chess_position_with_corners \
     -F "image=@test_image.jpg" \
     -F "corners=[[100,100],[1100,100],[1100,1100],[100,1100]]" \
     -F "turn=white"
```

### Step 4: Monitor the API
```bash
# View logs
sudo docker-compose logs -f

# Check service status
sudo systemctl status chess-api

# Check containers
sudo docker ps
```

## Troubleshooting

### If API doesn't start:
```bash
# Check Docker logs
sudo docker-compose logs

# Check if models are loaded
sudo docker-compose exec chess-api ls -la /app/models/
sudo docker-compose exec chess-api ls -la /app/runs/occupancy_classifier/ResNet/
```

### If SSL fails:
```bash
# Check domain DNS
nslookup api.chesspositionscanner.store

# Test without SSL first
curl http://api.chesspositionscanner.store/health
```

### If nginx fails:
```bash
# Check nginx config
sudo nginx -t

# Check nginx logs
sudo tail -f /var/log/nginx/error.log
```

## API Endpoints

- **Health Check:** https://api.chesspositionscanner.store/health
- **Recognition:** https://api.chesspositionscanner.store/recognize_chess_position_with_corners
- **Documentation:** https://api.chesspositionscanner.store/docs

## Files Location

- **App Directory:** /opt/chess-api
- **Logs:** /opt/chess-api/logs/
- **Nginx Config:** /etc/nginx/sites-available/chess-api
- **SSL Certificates:** /etc/letsencrypt/live/api.chesspositionscanner.store/
