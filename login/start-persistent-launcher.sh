#!/bin/bash

# Start the launcher with persistent tunnel URLs

set -e

echo "🧠 Starting RAG Researcher Launcher (Persistent URLs)"
echo "======================================================"
echo ""

# Set the persistent URLs
export LAUNCHER_URL="https://brain-nw1-launcher.trycloudflare.com"
export MAIN_APP_URL="https://brain-nw1-app.trycloudflare.com"

# Check if password is set
if [ -z "$LAUNCHER_PASSWORD" ]; then
    echo "🔐 Password not set in environment."
    read -sp "Enter launcher password: " LAUNCHER_PASSWORD
    echo ""
    export LAUNCHER_PASSWORD
fi

# Check if tunnel is running
if ! sudo systemctl is-active cloudflared-brain >/dev/null 2>&1; then
    echo "⚠️  Tunnel is not running. Starting it..."
    sudo systemctl start cloudflared-brain
    sleep 3
fi

if sudo systemctl is-active cloudflared-brain >/dev/null 2>&1; then
    echo "✅ Tunnel is running"
else
    echo "❌ Tunnel failed to start. Check: sudo systemctl status cloudflared-brain"
    exit 1
fi

# Kill any existing launcher
echo ""
echo "🧹 Cleaning up existing processes..."
lsof -ti:8502 | xargs -r kill -9 2>/dev/null || true
lsof -ti:8501 | xargs -r kill -9 2>/dev/null || true

# Start the launcher
echo ""
echo "📦 Starting launcher app..."
cd /home/he/ai/dev/langgraph/KB_BS_local-rag-he/login

LAUNCHER_PASSWORD="$LAUNCHER_PASSWORD" \
LAUNCHER_URL="$LAUNCHER_URL" \
MAIN_APP_URL="$MAIN_APP_URL" \
  uv run --with streamlit --with psutil \
  streamlit run launcher_app.py --server.port 8502 --server.headless true \
  > /tmp/launcher.log 2>&1 &

LAUNCHER_PID=$!
echo "✅ Launcher starting (PID: $LAUNCHER_PID)..."

# Wait for launcher to be ready
echo "⏳ Waiting for launcher to be ready..."
for i in {1..15}; do
    if lsof -Pi :8502 -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "✅ Launcher is ready!"
        break
    fi
    if [ $i -eq 15 ]; then
        echo "❌ Launcher failed to start. Check logs:"
        tail -20 /tmp/launcher.log
        exit 1
    fi
    sleep 1
done

echo ""
echo "======================================"
echo "✅ Launcher is Running!"
echo "======================================"
echo ""
echo "🔐 Launcher URL:  $LAUNCHER_URL"
echo "🚀 Main App URL:  $MAIN_APP_URL"
echo ""
echo "Both URLs are accessible through the same tunnel!"
echo ""
echo "Next steps:"
echo "1. Open the launcher: $LAUNCHER_URL"
echo "2. Login with your password"
echo "3. Click 'Start App' to launch the main RAG app"
echo "4. Click 'Open RAG Researcher App' to access it"
echo "   (It will open: $MAIN_APP_URL)"
echo ""
echo "The launcher is running in the background."
echo "To stop it: lsof -ti:8502 | xargs -r kill -9"
echo "To view logs: tail -f /tmp/launcher.log"
echo ""
