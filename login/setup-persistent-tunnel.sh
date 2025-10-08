#!/bin/bash

# Setup Persistent Named Tunnel for RAG Researcher
# Creates a named tunnel "brain-nw1" with consistent URLs

set -e

echo "🧠 RAG Researcher - Persistent Tunnel Setup"
echo "============================================"
echo ""
echo "This will create a NAMED tunnel called 'brain-nw1'"
echo "with persistent URLs for both launcher and main app."
echo ""

# Check if cloudflared is installed
if ! command -v cloudflared &> /dev/null; then
    echo "❌ cloudflared not found. Installing..."
    wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
    sudo dpkg -i cloudflared-linux-amd64.deb
    rm cloudflared-linux-amd64.deb
    echo "✅ cloudflared installed"
fi

# Create .cloudflared directory
mkdir -p ~/.cloudflared

# Check if already authenticated
if [ ! -f ~/.cloudflared/cert.pem ]; then
    echo ""
    echo "🔐 Step 1: Authenticate with Cloudflare"
    echo "This will open a browser for authentication..."
    echo ""
    cloudflared tunnel login
    
    if [ ! -f ~/.cloudflared/cert.pem ]; then
        echo "❌ Authentication failed. Please try again."
        exit 1
    fi
    echo "✅ Authenticated successfully"
else
    echo "✅ Already authenticated with Cloudflare"
fi

echo ""
echo "📦 Step 2: Creating named tunnel 'brain-nw1'..."

# Check if tunnel already exists
if cloudflared tunnel list 2>/dev/null | grep -q "brain-nw1"; then
    echo "⚠️  Tunnel 'brain-nw1' already exists"
    TUNNEL_ID=$(cloudflared tunnel list | grep "brain-nw1" | awk '{print $1}')
    echo "   Using existing tunnel ID: $TUNNEL_ID"
else
    # Create the tunnel
    cloudflared tunnel create brain-nw1
    TUNNEL_ID=$(cloudflared tunnel list | grep "brain-nw1" | awk '{print $1}')
    echo "✅ Tunnel created with ID: $TUNNEL_ID"
fi

echo ""
echo "📝 Step 3: Creating tunnel configuration..."

# Create config file with ingress rules for both services
cat > ~/.cloudflared/config.yml << EOF
tunnel: $TUNNEL_ID
credentials-file: /home/$USER/.cloudflared/$TUNNEL_ID.json

ingress:
  # Launcher app on port 8502
  - hostname: brain-nw1-launcher.trycloudflare.com
    service: http://localhost:8502
  
  # Main RAG app on port 8501
  - hostname: brain-nw1-app.trycloudflare.com
    service: http://localhost:8501
  
  # Catch-all rule (required)
  - service: http_status:404
EOF

echo "✅ Configuration created at ~/.cloudflared/config.yml"

echo ""
echo "🌐 Step 4: Setting up DNS routes..."

# Route DNS for both hostnames
cloudflared tunnel route dns brain-nw1 brain-nw1-launcher.trycloudflare.com 2>/dev/null || true
cloudflared tunnel route dns brain-nw1 brain-nw1-app.trycloudflare.com 2>/dev/null || true

echo "✅ DNS routes configured"

echo ""
echo "📋 Step 5: Creating systemd service..."

# Create systemd service
sudo tee /etc/systemd/system/cloudflared-brain.service > /dev/null << EOF
[Unit]
Description=Cloudflare Tunnel - BrAIn-nw1
After=network.target

[Service]
Type=simple
User=$USER
ExecStart=/usr/bin/cloudflared tunnel --config /home/$USER/.cloudflared/config.yml run brain-nw1
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
EOF

echo "✅ Systemd service created"

# Reload systemd
sudo systemctl daemon-reload

echo ""
echo "🚀 Step 6: Starting the tunnel..."
sudo systemctl enable cloudflared-brain
sudo systemctl start cloudflared-brain

sleep 3

# Check status
if sudo systemctl is-active cloudflared-brain >/dev/null 2>&1; then
    echo "✅ Tunnel is running!"
else
    echo "⚠️  Tunnel status unclear. Check with: sudo systemctl status cloudflared-brain"
fi

echo ""
echo "======================================"
echo "✅ Setup Complete!"
echo "======================================"
echo ""
echo "Your persistent URLs:"
echo ""
echo "🔐 Launcher:  https://brain-nw1-launcher.trycloudflare.com"
echo "🚀 Main App:  https://brain-nw1-app.trycloudflare.com"
echo ""
echo "Note: If using trycloudflare.com, URLs may change on restart."
echo "      For truly persistent URLs, add a custom domain to Cloudflare."
echo ""
echo "Next steps:"
echo "1. Set your password: export LAUNCHER_PASSWORD='your_password'"
echo "2. Start the launcher: ./start-persistent-launcher.sh"
echo "3. Access via: https://brain-nw1-launcher.trycloudflare.com"
echo ""
echo "Useful commands:"
echo "  sudo systemctl status cloudflared-brain  # Check tunnel status"
echo "  sudo systemctl restart cloudflared-brain # Restart tunnel"
echo "  sudo journalctl -u cloudflared-brain -f  # View tunnel logs"
echo ""
