#!/bin/bash

# Quick Cloudflare Tunnels with Auto-Generated URLs
# URLs are random and change each time, but this works immediately!

set -e

echo "🚀 Starting Quick Cloudflare Tunnels"
echo "====================================="
echo ""

# Kill any existing cloudflared processes
echo "🧹 Cleaning up existing tunnels..."
pkill -f "cloudflared tunnel" 2>/dev/null || true
sleep 2

# Backup config files to force true quick tunnels
CONFIG_BACKUP=""
CRED_BACKUP=""

if [ -f ~/.cloudflared/config.yml ]; then
    CONFIG_BACKUP=~/.cloudflared/config.yml.quickbackup
    mv ~/.cloudflared/config.yml "$CONFIG_BACKUP"
    echo "📝 Temporarily moved config.yml"
fi

if [ -f ~/.cloudflared/5bc36850-dc39-48be-81ab-fd15f8071bd0.json ]; then
    CRED_BACKUP=~/.cloudflared/5bc36850-dc39-48be-81ab-fd15f8071bd0.json.quickbackup
    mv ~/.cloudflared/5bc36850-dc39-48be-81ab-fd15f8071bd0.json "$CRED_BACKUP"
    echo "📝 Temporarily moved tunnel credentials"
fi

# Start launcher tunnel (random URL)
echo "📦 Starting launcher tunnel on port 8502..."
nohup cloudflared tunnel --url http://localhost:8502 > /tmp/launcher-tunnel.log 2>&1 &
LAUNCHER_PID=$!
sleep 8

# Start main app tunnel (random URL)
echo "📦 Starting main app tunnel on port 8501..."
nohup cloudflared tunnel --url http://localhost:8501 > /tmp/app-tunnel.log 2>&1 &
APP_PID=$!
sleep 8

# Restore config files
if [ -n "$CONFIG_BACKUP" ] && [ -f "$CONFIG_BACKUP" ]; then
    mv "$CONFIG_BACKUP" ~/.cloudflared/config.yml
    echo "📝 Restored config.yml"
fi

if [ -n "$CRED_BACKUP" ] && [ -f "$CRED_BACKUP" ]; then
    mv "$CRED_BACKUP" ~/.cloudflared/5bc36850-dc39-48be-81ab-fd15f8071bd0.json
    echo "📝 Restored tunnel credentials"
fi

# Extract URLs from logs
echo ""
echo "⏳ Extracting tunnel URLs..."
sleep 2

LAUNCHER_URL=$(grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' /tmp/launcher-tunnel.log | head -1 || echo "")
APP_URL=$(grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' /tmp/app-tunnel.log | head -1 || echo "")

if [ -z "$LAUNCHER_URL" ] || [ -z "$APP_URL" ]; then
    echo "❌ Failed to extract URLs. Checking logs..."
    echo ""
    echo "=== Launcher Log ==="
    tail -20 /tmp/launcher-tunnel.log
    echo ""
    echo "=== App Log ==="
    tail -20 /tmp/app-tunnel.log
    exit 1
fi

# Save URLs to file
echo "$LAUNCHER_URL" > /tmp/launcher-url.txt
echo "$APP_URL" > /tmp/app-url.txt

echo ""
echo "======================================"
echo "✅ Quick Tunnels Running!"
echo "======================================"
echo ""
echo "🔐 Launcher URL:  $LAUNCHER_URL"
echo "🚀 Main App URL:  $APP_URL"
echo ""
echo "⚠️  IMPORTANT: These URLs are temporary!"
echo "   They will change each time you restart the tunnels."
echo ""
echo "📋 URLs saved to:"
echo "   /tmp/launcher-url.txt"
echo "   /tmp/app-url.txt"
echo ""
echo "To stop tunnels:"
echo "  pkill -f 'cloudflared tunnel'"
echo ""
echo "To view logs:"
echo "  tail -f /tmp/launcher-tunnel.log"
echo "  tail -f /tmp/app-tunnel.log"
echo ""
