#!/bin/bash

# Quick start script for the RAG Researcher Launcher

set -e

echo "🚀 Starting RAG Researcher Launcher"
echo "===================================="
echo ""

# Check for password
if [ -z "$LAUNCHER_PASSWORD" ]; then
    echo "⚠️  LAUNCHER_PASSWORD environment variable not set"
    read -sp "Enter launcher password: " LAUNCHER_PASSWORD
    echo ""
    export LAUNCHER_PASSWORD
fi

# Check if port 8502 is available
if lsof -Pi :8502 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  Port 8502 is already in use"
    read -p "Kill existing process? (y/n): " KILL_PROC
    if [ "$KILL_PROC" = "y" ]; then
        lsof -ti:8502 | xargs -r kill -9
        echo "✅ Killed process on port 8502"
        sleep 1
    else
        echo "❌ Cannot start - port 8502 is in use"
        exit 1
    fi
fi

# Change to login directory
cd "$(dirname "$0")"

# Check dependencies
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Installing..."
    pip install -r requirements.txt
fi

echo ""
echo "✅ Starting launcher on port 8502..."
echo "🌐 Access URL: http://localhost:8502"
echo ""
echo "💡 To access remotely, use your Cloudflare Tunnel URL"
echo "   (See README.md for setup instructions)"
echo ""

# Start the launcher
streamlit run launcher_app.py --server.port 8502 --server.headless true
