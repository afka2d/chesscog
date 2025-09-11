# 🚀 Chess Position Scanner API - Deployment Guide

## Server: 159.203.102.249
## Domain: api.chesspositionscanner.store
## Target URL: https://api.chesspositionscanner.store/recognize_chess_position_with_corners

---

## 📋 Quick Deployment (3 Steps)

### Step 1: Copy Files to Server
```bash
# Run this from your local machine
./deploy_now.sh
```

**OR manually:**
```bash
scp -r deployment_20250910_181959/* root@159.203.102.249:/tmp/chess-api/
```

### Step 2: Setup on Server
```bash
# SSH into your server
ssh root@159.203.102.249

# Create app directory and move files
sudo mkdir -p /opt/chess-api
sudo mv /tmp/chess-api/* /opt/chess-api/
cd /opt/chess-api

# Run the automated setup script
sudo chmod +x server_setup.sh
sudo ./server_setup.sh
```

### Step 3: Verify Deployment
```bash
# Check API health
curl https://api.chesspositionscanner.store/health

# Check API documentation
curl https://api.chesspositionscanner.store/docs
```

---

## 🔧 What the Setup Script Does

The `server_setup.sh` script automatically:

1. **Updates system packages**
2. **Installs Docker and Docker Compose**
3. **Installs Nginx**
4. **Installs curl for health checks**
5. **Builds and starts the API container**
6. **Configures Nginx reverse proxy**
7. **Sets up SSL certificate with Let's Encrypt**
8. **Creates systemd service for auto-start**

---

## 📊 Expected Results

After successful deployment:

✅ **API Health Check:** `https://api.chesspositionscanner.store/health`
- Should return: `{"status": "healthy", "models": {...}}`

✅ **API Documentation:** `https://api.chesspositionscanner.store/docs`
- Interactive API documentation

✅ **Recognition Endpoint:** `https://api.chesspositionscanner.store/recognize_chess_position_with_corners`
- Ready for your mobile app integration

---

## 🔍 Testing the API

### Test with curl:
```bash
# Health check
curl https://api.chesspositionscanner.store/health

# Test recognition (replace with your test image)
curl -X POST https://api.chesspositionscanner.store/recognize_chess_position_with_corners \
     -F "image=@test_image.jpg" \
     -F "corners=[[100,100],[1100,100],[1100,1100],[100,1100]]" \
     -F "turn=white"
```

### Expected Response:
```json
{
  "fen": "8/8/8/8/8/8/8/8 w - - 0 1",
  "pieces": [null, null, ...],
  "occupancy": [false, false, ...],
  "success": true,
  "metadata": {
    "request_id": "req_...",
    "processing_time_seconds": 0.44,
    "occupied_squares": 0,
    "classified_pieces": 0
  }
}
```

---

## 📱 Update Your Mobile App

Change your app's API URL from:
```javascript
// OLD
const API_URL = "http://localhost:8000/recognize_chess_position_with_corners";

// NEW
const API_URL = "https://api.chesspositionscanner.store/recognize_chess_position_with_corners";
```

---

## 🛠️ Troubleshooting

### If API doesn't start:
```bash
# Check Docker logs
sudo docker-compose logs

# Check if models are loaded
sudo docker-compose exec chess-api ls -la /app/models/
sudo docker-compose exec chess-api ls -la /app/runs/occupancy_classifier/ResNet/

# Restart the service
sudo docker-compose restart
```

### If SSL certificate fails:
```bash
# Check domain DNS
nslookup api.chesspositionscanner.store

# Test without SSL first
curl http://api.chesspositionscanner.store/health

# Manually get SSL certificate
sudo certbot --nginx -d api.chesspositionscanner.store
```

### If nginx fails:
```bash
# Check nginx config
sudo nginx -t

# Check nginx logs
sudo tail -f /var/log/nginx/error.log

# Restart nginx
sudo systemctl restart nginx
```

---

## 📊 Monitoring Commands

```bash
# View API logs
sudo docker-compose logs -f

# Check service status
sudo systemctl status chess-api

# Check containers
sudo docker ps

# Check disk space
df -h

# Check memory usage
free -h
```

---

## 🔒 Security Notes

- The API is configured with SSL/TLS encryption
- File uploads are limited to 10MB
- CORS is enabled for web app integration
- All requests are logged for monitoring

---

## 📞 Support

### Log Locations:
- **API Logs:** `/opt/chess-api/logs/chess_api.log`
- **Docker Logs:** `sudo docker-compose logs`
- **Nginx Logs:** `/var/log/nginx/`

### Health Monitoring:
- **Health Endpoint:** `https://api.chesspositionscanner.store/health`
- **Response Time:** Monitor with external services
- **Uptime:** Use services like UptimeRobot

---

## 🎯 Success Criteria

Your deployment is successful when:

✅ **Health check returns "healthy"**  
✅ **All models are loaded successfully**  
✅ **SSL certificate is active**  
✅ **API responds to recognition requests**  
✅ **Your mobile app can connect successfully**

---

## 🚀 Ready for App Store!

Once deployed, your API will be available at:
**`https://api.chesspositionscanner.store/recognize_chess_position_with_corners`**

You can now update your mobile app and submit to the App Store! 🎉

---

**Deployment Package:** `deployment_20250910_181959/`  
**Server:** `159.203.102.249`  
**Domain:** `api.chesspositionscanner.store`  
**Quick Deploy:** `./deploy_now.sh`
