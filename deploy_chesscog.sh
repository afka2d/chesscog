#!/bin/bash
set -e

# Update and install system packages
apt update && apt upgrade -y
apt install -y python3 python3-venv python3-pip git nginx

# Clone your chesscog repo (replace with your repo URL)
cd /opt
git clone https://github.com/afka2d/chesscog.git
cd chesscog

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements (adjust if needed)
pip install --upgrade pip
pip install -r requirements.txt

# Deactivate for now
deactivate

# Setup systemd service for uvicorn
cat > /etc/systemd/system/chesscog.service << EOF
[Unit]
Description=ChessCog FastAPI service
After=network.target

[Service]
User=root
WorkingDirectory=/opt/chesscog
ExecStart=/opt/chesscog/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Start and enable service
systemctl daemon-reload
systemctl start chesscog
systemctl enable chesscog

# Configure Nginx as a reverse proxy
cat > /etc/nginx/sites-available/chesscog << EOF
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
    }
}
EOF

ln -s /etc/nginx/sites-available/chesscog /etc/nginx/sites-enabled/
nginx -t && systemctl restart nginx

echo "Deployment completed. Your API should be available at http://159.203.102.249/"
