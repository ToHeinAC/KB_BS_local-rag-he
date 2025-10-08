#!/bin/bash

# Dual Quick Tunnel Setup
# Creates two quick tunnels with BrAIn naming, no domain needed

set -e

echo "🧠 BrAIn-nw1 Dual Quick Tunnel Launcher"
echo "========================================"
echo ""
echo "This creates TWO quick tunnels:"
echo "1. Launcher tunnel (port 8502)"
echo "2. Main app tunnel (port 8501)"
echo ""
echo "URLs will include 'brain' identifier but change on restart."
echo "For permanent URLs, you need to add a domain to Cloudflare."
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
    read -sp "🔐 Set launcher password: " LAUNCHER_PASSWORD
    echo ""
    export LAUNCHER_PASSWORD
fi

# Kill existing processes
echo ""
echo "🧹 Cleaning up existing processes..."
lsof -ti:8502 | xargs -r kill -9 2>/dev/null || true
lsof -ti:8501 | xargs -r kill -9 2>/dev/null || true
pkill -f "cloudflared tunnel --url" 2>/dev/null || true
sleep 2

# Start launcher app
echo ""
echo "📦 Step 1: Starting launcher app..."
cd /home/he/ai/dev/langgraph/KB_BS_local-rag-he/login

LAUNCHER_PASSWORD="$LAUNCHER_PASSWORD" \
  uv run --with streamlit --with psutil \
  streamlit run launcher_app.py --server.port 8502 --server.headless true \
  > /tmp/launcher.log 2>&1 &

LAUNCHER_PID=$!
echo "✅ Launcher starting (PID: $LAUNCHER_PID)..."

# Wait for launcher
echo "⏳ Waiting for launcher to be ready..."
for i in {1..15}; do
    if lsof -Pi :8502 -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "✅ Launcher is ready!"
        break
    fi
    sleep 1
done

if ! lsof -Pi :8502 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "❌ Launcher failed to start. Logs:"
    tail -20 /tmp/launcher.log
    exit 1
fi

# Create tunnel for launcher
echo ""
echo "🌐 Step 2: Creating Quick Tunnel for LAUNCHER..."
echo ""

cloudflared tunnel --url http://localhost:8502 > /tmp/tunnel-launcher.log 2>&1 &
TUNNEL1_PID=$!

# Wait for tunnel URL
echo "⏳ Waiting for launcher tunnel URL (this may take 30 seconds)..."
for i in {1..40}; do
    if grep -q "trycloudflare.com" /tmp/tunnel-launcher.log 2>/dev/null; then
        # Extract URL - handle potential line breaks
        LAUNCHER_URL=$(cat /tmp/tunnel-launcher.log | tr -d '\n' | grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' | head -1)
        if [ -n "$LAUNCHER_URL" ]; then
            break
        fi
    fi
    sleep 1
    if [ $((i % 5)) -eq 0 ]; then
        echo "   Still waiting... ($i seconds)"
    fi
done

if [ -z "$LAUNCHER_URL" ]; then
    echo "❌ Failed to get launcher tunnel URL"
    cat /tmp/tunnel-launcher.log
    kill $LAUNCHER_PID $TUNNEL1_PID 2>/dev/null
    exit 1
fi

echo "✅ Launcher tunnel created!"
echo "   URL: $LAUNCHER_URL"

# Create tunnel for main app (pre-emptively, it will be used when app starts)
echo ""
echo "🌐 Step 3: Creating Quick Tunnel for MAIN APP..."
echo ""

cloudflared tunnel --url http://localhost:8501 > /tmp/tunnel-app.log 2>&1 &
TUNNEL2_PID=$!

# Wait for tunnel URL
echo "⏳ Waiting for main app tunnel URL (this may take 30 seconds)..."
for i in {1..40}; do
    if grep -q "trycloudflare.com" /tmp/tunnel-app.log 2>/dev/null; then
        # Extract URL - handle potential line breaks
        APP_URL=$(cat /tmp/tunnel-app.log | tr -d '\n' | grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' | head -1)
        if [ -n "$APP_URL" ]; then
            break
        fi
    fi
    sleep 1
    if [ $((i % 5)) -eq 0 ]; then
        echo "   Still waiting... ($i seconds)"
    fi
done

if [ -z "$APP_URL" ]; then
    echo "❌ Failed to get app tunnel URL"
    cat /tmp/tunnel-app.log
    kill $LAUNCHER_PID $TUNNEL1_PID $TUNNEL2_PID 2>/dev/null
    exit 1
fi

echo "✅ Main app tunnel created!"
echo "   URL: $APP_URL"

# Update environment variables for the launcher
export LAUNCHER_URL="$LAUNCHER_URL"
export MAIN_APP_URL="$APP_URL"

# Save URLs to file for reference
cat > /tmp/brain-tunnel-urls.txt << EOF
LAUNCHER_URL=$LAUNCHER_URL
MAIN_APP_URL=$APP_URL
LAUNCHER_PID=$LAUNCHER_PID
TUNNEL1_PID=$TUNNEL1_PID
TUNNEL2_PID=$TUNNEL2_PID
EOF

echo ""
echo "=============================================="
echo "🎉 BrAIn-nw1 System is Running!"
echo "=============================================="
echo ""
echo "🔐 Launcher:   $LAUNCHER_URL"
echo "🚀 Main App:   $APP_URL"
echo ""
echo "⚠️  NOTE: These URLs will change on restart!"
echo "    To get persistent URLs, add a domain to Cloudflare."
echo ""
echo "Next steps:"
echo "1. Open launcher: $LAUNCHER_URL"
echo "2. Login with your password"
echo "3. Click 'Start App' to launch the RAG app"
echo "4. Click 'Open RAG Researcher App'"
echo "   (It will open: $APP_URL)"
echo ""
echo "To stop everything:"
echo "  Press Ctrl+C (or run: ./stop-brain-tunnels.sh)"
echo ""
echo "Processes:"
echo "  Launcher PID: $LAUNCHER_PID"
echo "  Tunnel 1 PID: $TUNNEL1_PID"
echo "  Tunnel 2 PID: $TUNNEL2_PID"
echo ""
echo "URLs saved to: /tmp/brain-tunnel-urls.txt"
echo ""

# Setup cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping all processes..."
    kill $LAUNCHER_PID 2>/dev/null || true
    kill $TUNNEL1_PID 2>/dev/null || true
    kill $TUNNEL2_PID 2>/dev/null || true
    lsof -ti:8502 | xargs -r kill -9 2>/dev/null || true
    lsof -ti:8501 | xargs -r kill -9 2>/dev/null || true
    echo "✅ Stopped"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Keep script running
echo "Press Ctrl+C to stop all services..."
echo ""
wait
