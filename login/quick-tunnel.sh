#!/bin/bash

# Quick Cloudflare Tunnel Setup (No Domain Required)
# This creates temporary tunnels with auto-generated URLs

echo "🚀 Quick Cloudflare Tunnel Setup"
echo "================================="
echo ""
echo "This will create temporary Cloudflare tunnels with auto-generated URLs."
echo "Perfect for testing! No domain or authentication needed."
echo ""

# Set password if not already set
if [ -z "$LAUNCHER_PASSWORD" ]; then
    echo "⚠️  LAUNCHER_PASSWORD not set"
    read -sp "Enter a password for the launcher: " LAUNCHER_PASSWORD
    echo ""
    export LAUNCHER_PASSWORD
fi

# Start the launcher in the background
echo ""
echo "📦 Step 1: Starting the launcher app..."
cd /home/he/ai/dev/langgraph/KB_BS_local-rag-he/login

# Kill any existing launcher
lsof -ti:8502 | xargs -r kill -9 2>/dev/null

# Start launcher in background
LAUNCHER_PASSWORD="$LAUNCHER_PASSWORD" streamlit run launcher_app.py --server.port 8502 --server.headless true > /tmp/launcher.log 2>&1 &
LAUNCHER_PID=$!

echo "✅ Launcher started (PID: $LAUNCHER_PID)"
echo "   Waiting for it to be ready..."
sleep 5

# Check if launcher is running
if ! lsof -Pi :8502 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "❌ Launcher failed to start. Check logs:"
    cat /tmp/launcher.log
    exit 1
fi

echo "✅ Launcher is running on port 8502"
echo ""

# Create tunnel for launcher
echo "📦 Step 2: Creating Cloudflare Quick Tunnel for launcher..."
echo ""
echo "Starting tunnel (this will show the URL)..."
echo "============================================"
echo ""

# Start the tunnel (this will output the URL)
cloudflared tunnel --url http://localhost:8502

# Note: This command runs in foreground and will show the URL
# Press Ctrl+C to stop the tunnel when done
