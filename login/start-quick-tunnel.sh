#!/bin/bash

# Quick Start - Cloudflare Tunnel without Domain
# This uses Cloudflare Quick Tunnels (temporary URLs, no setup needed)

set -e

echo "🚀 RAG Researcher - Quick Tunnel Launcher"
echo "=========================================="
echo ""
echo "This will:"
echo "1. Start the launcher app on port 8502"
echo "2. Create a Cloudflare tunnel with auto-generated URL"
echo "3. Give you instant access - no domain needed!"
echo ""

# Check if cloudflared is installed
if ! command -v cloudflared &> /dev/null; then
    echo "❌ cloudflared not found. Installing..."
    wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
    sudo dpkg -i cloudflared-linux-amd64.deb
    rm cloudflared-linux-amd64.deb
    echo "✅ cloudflared installed"
fi

# Set password
if [ -z "$LAUNCHER_PASSWORD" ]; then
    echo ""
    read -sp "🔐 Set a password for the launcher: " LAUNCHER_PASSWORD
    echo ""
    export LAUNCHER_PASSWORD
fi

# Kill any existing processes
echo ""
echo "🧹 Cleaning up any existing processes..."
lsof -ti:8502 | xargs -r kill -9 2>/dev/null || true
lsof -ti:8501 | xargs -r kill -9 2>/dev/null || true

# Start launcher
echo ""
echo "📦 Starting launcher app..."
cd /home/he/ai/dev/langgraph/KB_BS_local-rag-he/login

# Use uv to run streamlit (avoid venv issues)
LAUNCHER_PASSWORD="$LAUNCHER_PASSWORD" \
  uv run --with streamlit --with psutil \
  streamlit run launcher_app.py --server.port 8502 --server.headless true \
  > /tmp/launcher.log 2>&1 &

LAUNCHER_PID=$!
echo "✅ Launcher starting (PID: $LAUNCHER_PID)..."

# Wait for launcher to be ready
echo "⏳ Waiting for launcher to be ready..."
for i in {1..10}; do
    if lsof -Pi :8502 -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "✅ Launcher is ready!"
        break
    fi
    sleep 1
done

if ! lsof -Pi :8502 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "❌ Launcher failed to start. Logs:"
    cat /tmp/launcher.log
    exit 1
fi

echo ""
echo "🌐 Creating Cloudflare Quick Tunnel..."
echo "========================================"
echo ""
echo "Your launcher will be accessible via the URL shown below:"
echo ""

# Start the tunnel - this runs in foreground and shows the URL
cloudflared tunnel --url http://localhost:8502

# When user presses Ctrl+C, cleanup
trap "echo ''; echo '🛑 Stopping...'; kill $LAUNCHER_PID 2>/dev/null; exit 0" SIGINT SIGTERM
