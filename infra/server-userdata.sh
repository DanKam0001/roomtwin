#!/bin/bash
# First-boot setup for the RoomTwin server (Ubuntu 24.04 arm64 on Graviton).
# The app code is copied in afterwards by infra/deploy.sh.
set -eux
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y python3-venv python3-pip caddy libgl1 libglib2.0-0 unzip
useradd -m -s /bin/bash roomtwin || true
mkdir -p /opt/roomtwin /var/lib/roomtwin
chown -R roomtwin:roomtwin /opt/roomtwin /var/lib/roomtwin

# 2 GB swap so a big point cloud can't OOM the box
if [ ! -f /swapfile ]; then
  fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

cat > /etc/systemd/system/roomtwin.service <<'EOF'
[Unit]
Description=RoomTwin API
After=network.target

[Service]
User=roomtwin
WorkingDirectory=/opt/roomtwin/server
EnvironmentFile=-/etc/roomtwin.env
Environment=ROOMTWIN_DATA=/var/lib/roomtwin
ExecStart=/opt/roomtwin/venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000 --proxy-headers
Restart=always

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
echo "===USERDATA_DONE==="
