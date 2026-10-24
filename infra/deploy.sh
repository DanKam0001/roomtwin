#!/usr/bin/env bash
# Copy the app to the RoomTwin server and (re)start it.
# usage: infra/deploy.sh <public-ip>
# Needs ~/.ssh/roomtwin.pem and ANTHROPIC_API_KEY in the environment (written to
# /etc/roomtwin.env on the server, never into the repo).
set -euo pipefail
IP="$1"
HOST="$(echo "$IP" | tr . -).sslip.io"     # free DNS name that resolves to the IP, for HTTPS
SSH="ssh -i $HOME/.ssh/roomtwin.pem -o StrictHostKeyChecking=accept-new ubuntu@$IP"

tar --exclude='__pycache__' --exclude='.pytest_cache' --exclude='*.pyc' \
    -czf /tmp/roomtwin.tgz server pipeline web
scp -i "$HOME/.ssh/roomtwin.pem" -o StrictHostKeyChecking=accept-new /tmp/roomtwin.tgz "ubuntu@$IP:/tmp/"

$SSH "sudo bash -s" <<EOF
set -e
until grep -q USERDATA_DONE /var/log/cloud-init-output.log 2>/dev/null; do sleep 5; done
cd /opt/roomtwin && tar -xzf /tmp/roomtwin.tgz && chown -R roomtwin:roomtwin /opt/roomtwin
[ -d venv ] || sudo -u roomtwin python3 -m venv venv
sudo -u roomtwin venv/bin/pip install -q -r server/requirements.txt
[ -f pipeline/requirements.txt ] && sudo -u roomtwin venv/bin/pip install -q -r pipeline/requirements.txt
umask 077; echo "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}" > /etc/roomtwin.env
cat > /etc/caddy/Caddyfile <<CADDY
$HOST {
    request_body {
        max_size 400MB
    }
    reverse_proxy 127.0.0.1:8000
}
CADDY
systemctl enable --now roomtwin && systemctl restart roomtwin
systemctl reload caddy || systemctl restart caddy
EOF
echo "deployed: https://$HOST/"
