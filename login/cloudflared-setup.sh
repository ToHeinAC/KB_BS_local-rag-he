#!/bin/bash

# Cloudflare Tunnel Setup Script
# This script helps set up a Cloudflare Tunnel to access your Streamlit apps remotely

set -e

echo "🚀 Cloudflare Tunnel Setup"
echo "=========================="
echo ""

# Check if cloudflared is installed
if ! command -v cloudflared &> /dev/null; then
    echo "❌ cloudflared is not installed!"
    echo "📥 Installing cloudflared..."
    
    # Install for Debian/Ubuntu
    if command -v apt-get &> /dev/null; then
        wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
        sudo dpkg -i cloudflared-linux-amd64.deb
        rm cloudflared-linux-amd64.deb
    else
        echo "Please install cloudflared manually from: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation"
        exit 1
    fi
fi

echo "✅ cloudflared is installed"
echo ""

# Authenticate
echo "🔐 Step 1: Authenticate with Cloudflare"
echo "This will open a browser window for authentication..."
read -p "Press Enter to continue..."
cloudflared tunnel login

echo ""
echo "✅ Authentication complete"
echo ""

# Create tunnel
echo "🔧 Step 2: Create tunnel"
read -p "Enter a name for your tunnel (e.g., rag-researcher): " TUNNEL_NAME

if [ -z "$TUNNEL_NAME" ]; then
    echo "❌ Tunnel name cannot be empty"
    exit 1
fi

echo "Creating tunnel: $TUNNEL_NAME"
cloudflared tunnel create $TUNNEL_NAME

# Get tunnel ID
TUNNEL_ID=$(cloudflared tunnel list | grep $TUNNEL_NAME | awk '{print $1}')

if [ -z "$TUNNEL_ID" ]; then
    echo "❌ Failed to get tunnel ID"
    exit 1
fi

echo "✅ Tunnel created with ID: $TUNNEL_ID"
echo ""

# Create config directory
echo "📁 Step 3: Setting up configuration"
mkdir -p ~/.cloudflared

# Create config file
cat > ~/.cloudflared/config.yml <<EOF
tunnel: $TUNNEL_ID
credentials-file: /home/$USER/.cloudflared/$TUNNEL_ID.json

ingress:
  # Main RAG app on port 8501
  - hostname: rag-app-$TUNNEL_NAME.your-domain.com
    service: http://localhost:8501
  
  # Launcher app on port 8502 (optional)
  - hostname: launcher-$TUNNEL_NAME.your-domain.com
    service: http://localhost:8502
  
  # Catch-all rule
  - service: http_status:404
EOF

echo "✅ Config file created at ~/.cloudflared/config.yml"
echo ""

# Setup DNS
echo "🌐 Step 4: Setting up DNS routes"
echo "You need to configure DNS for your hostnames..."
read -p "Enter your domain (e.g., example.com): " DOMAIN

if [ ! -z "$DOMAIN" ]; then
    echo "Setting up DNS for rag-app-$TUNNEL_NAME.$DOMAIN"
    cloudflared tunnel route dns $TUNNEL_NAME rag-app-$TUNNEL_NAME.$DOMAIN || true
    
    read -p "Also setup launcher route? (y/n): " SETUP_LAUNCHER
    if [ "$SETUP_LAUNCHER" = "y" ]; then
        cloudflared tunnel route dns $TUNNEL_NAME launcher-$TUNNEL_NAME.$DOMAIN || true
    fi
    
    echo "✅ DNS routes configured"
else
    echo "⚠️  Skipping DNS setup - you'll need to configure this manually"
fi

echo ""

# Create systemd service
echo "🔧 Step 5: Setting up systemd service"

sudo tee /etc/systemd/system/cloudflared.service > /dev/null <<EOF
[Unit]
Description=Cloudflare Tunnel
After=network.target

[Service]
Type=simple
User=$USER
ExecStart=/usr/bin/cloudflared tunnel --config /home/$USER/.cloudflared/config.yml run
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
EOF

echo "✅ Systemd service created"
echo ""

# Enable and start service
echo "🚀 Step 6: Starting tunnel service"
sudo systemctl daemon-reload
sudo systemctl enable cloudflared
sudo systemctl start cloudflared

echo ""
echo "✅ Cloudflare Tunnel setup complete!"
echo ""
echo "📊 Service Status:"
sudo systemctl status cloudflared --no-pager
echo ""
echo "🌐 Your tunnel is now running!"
echo ""
echo "📝 Next steps:"
echo "1. Update the hostnames in ~/.cloudflared/config.yml to match your domain"
echo "2. Start your Streamlit apps on the configured ports"
echo "3. Access your apps via the configured hostnames"
echo ""
echo "💡 Useful commands:"
echo "  - Check status: sudo systemctl status cloudflared"
echo "  - View logs: sudo journalctl -u cloudflared -f"
echo "  - Restart: sudo systemctl restart cloudflared"
echo "  - Stop: sudo systemctl stop cloudflared"
echo ""
