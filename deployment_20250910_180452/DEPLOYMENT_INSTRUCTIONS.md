# Chess Position Scanner API - Deployment Instructions

## Server Setup

1. **Install Docker and Docker Compose:**
   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   sudo chmod +x /usr/local/bin/docker-compose
   ```

2. **Install Nginx:**
   ```bash
   sudo apt update
   sudo apt install nginx
   ```

3. **Deploy the API:**
   ```bash
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
   ```

4. **Set up SSL (recommended):**
   ```bash
   sudo apt install certbot python3-certbot-nginx
   sudo certbot --nginx -d api.chesspositionscanner.store
   ```

5. **Monitor the API:**
   ```bash
   # Check logs
   sudo docker-compose logs -f
   
   # Check health
   curl https://api.chesspositionscanner.store/health
   
   # Test recognition
   curl -X POST https://api.chesspositionscanner.store/recognize_chess_position_with_corners \
        -F "image=@test_image.jpg" \
        -F "corners=[[100,100],[1100,100],[1100,1100],[100,1100]]" \
        -F "turn=white"
   ```

## API Endpoints

- **Health Check:** `GET https://api.chesspositionscanner.store/health`
- **Recognition:** `POST https://api.chesspositionscanner.store/recognize_chess_position_with_corners`
- **Documentation:** `GET https://api.chesspositionscanner.store/docs`

## Monitoring

- Logs are stored in `/opt/chess-api/logs/`
- Use `docker-compose logs` to view real-time logs
- Health check endpoint for monitoring services

## Troubleshooting

- Check Docker containers: `sudo docker ps`
- Check logs: `sudo docker-compose logs`
- Restart service: `sudo docker-compose restart`
- Check nginx: `sudo nginx -t`
