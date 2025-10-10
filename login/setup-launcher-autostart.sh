#!/bin/bash

# Setup Launcher Auto-Start Service
# This makes the launcher start automatically on boot

set -e

echo "🧠 BrAIn-nw1 Launcher Auto-Start Setup"
echo "======================================="
echo ""

# Check if cloudflared tunnel is set up
if ! sudo systemctl list-unit-files | grep -q "cloudflared-brain.service"; then
    echo "⚠️  WARNING: Cloudflared tunnel service not found!"
    echo "   Please run ./setup-persistent-tunnel.sh first"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if password is set
if [ -z "$LAUNCHER_PASSWORD" ]; then
    echo "⚠️  LAUNCHER_PASSWORD not set in environment"
    echo ""
    read -sp "Enter launcher password to use: " LAUNCHER_PASSWORD
    echo ""
    echo ""
    echo "💡 To make this permanent, add to your ~/.bashrc:"
    echo "   echo 'export LAUNCHER_PASSWORD=\"$LAUNCHER_PASSWORD\"' >> ~/.bashrc"
    echo ""
fi

# Update service file with actual password
echo "📝 Creating systemd service file..."
cat > /tmp/brain-launcher.service << EOF
[Unit]
Description=BrAIn-nw1 Launcher App
After=network.target cloudflared-brain.service
Wants=cloudflared-brain.service

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER/ai/dev/langgraph/KB_BS_local-rag-he/login
Environment="LAUNCHER_PASSWORD=$LAUNCHER_PASSWORD"
Environment="LAUNCHER_URL=https://brain-nw1-launcher.trycloudflare.com"
Environment="MAIN_APP_URL=https://brain-nw1-app.trycloudflare.com"
Environment="PATH=/home/$USER/.local/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/home/$USER/.local/bin/uv run --with streamlit --with psutil streamlit run launcher_app.py --server.port 8502 --server.headless true
Restart=always
RestartSec=10s
StandardOutput=append:/tmp/launcher.log
StandardError=append:/tmp/launcher.log

[Install]
WantedBy=multi-user.target
EOF

# Install service
echo "📦 Installing systemd service..."
sudo cp /tmp/brain-launcher.service /etc/systemd/system/brain-launcher.service
sudo chmod 644 /etc/systemd/system/brain-launcher.service
rm /tmp/brain-launcher.service

# Reload systemd
echo "🔄 Reloading systemd..."
sudo systemctl daemon-reload

# Stop any existing launcher processes
echo "🧹 Stopping any existing launcher processes..."
lsof -ti:8502 | xargs -r kill -9 2>/dev/null || true
sleep 2

# Enable and start service
echo "🚀 Enabling and starting launcher service..."
sudo systemctl enable brain-launcher
sudo systemctl start brain-launcher

# Wait for service to start
echo "⏳ Waiting for launcher to start..."
sleep 5

# Check status
if sudo systemctl is-active brain-launcher >/dev/null 2>&1; then
    echo "✅ Launcher service is running!"
else
    echo "⚠️  Launcher status unclear. Checking..."
    sudo systemctl status brain-launcher --no-pager
fi

echo ""
echo "======================================"
echo "✅ Auto-Start Setup Complete!"
echo "======================================"
echo ""
echo "The launcher will now start automatically on boot!"
echo ""
echo "Your URLs (always accessible):"
echo "🔐 Launcher:  https://brain-nw1-launcher.trycloudflare.com"
echo "🚀 Main App:  https://brain-nw1-app.trycloudflare.com"
echo ""
echo "Useful commands:"
echo "  sudo systemctl status brain-launcher   # Check launcher status"
echo "  sudo systemctl restart brain-launcher  # Restart launcher"
echo "  sudo systemctl stop brain-launcher     # Stop launcher"
echo "  sudo journalctl -u brain-launcher -f   # View launcher logs"
echo "  tail -f /tmp/launcher.log               # View launcher app logs"
echo ""
echo "To disable auto-start:"
echo "  sudo systemctl disable brain-launcher"
echo "  sudo systemctl stop brain-launcher"
echo ""
