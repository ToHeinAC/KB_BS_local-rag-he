#!/bin/bash

# Test script to verify the remote access setup

echo "🔍 RAG Researcher Remote Access - Setup Verification"
echo "====================================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check cloudflared
echo "1. Checking cloudflared installation..."
if command -v cloudflared &> /dev/null; then
    echo -e "${GREEN}✅ cloudflared is installed${NC}"
    cloudflared --version
else
    echo -e "${RED}❌ cloudflared is NOT installed${NC}"
    echo "   Install with: wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb && sudo dpkg -i cloudflared-linux-amd64.deb"
fi
echo ""

# Check config file
echo "2. Checking Cloudflare Tunnel configuration..."
if [ -f "$HOME/.cloudflared/config.yml" ]; then
    echo -e "${GREEN}✅ Config file exists${NC}"
    echo "   Location: $HOME/.cloudflared/config.yml"
    
    # Check if it's still the template
    if grep -q "YOUR_TUNNEL_ID_HERE" "$HOME/.cloudflared/config.yml"; then
        echo -e "${YELLOW}⚠️  Config file still has placeholder values${NC}"
        echo "   Run ./cloudflared-setup.sh to configure"
    else
        echo -e "${GREEN}✅ Config file is configured${NC}"
    fi
else
    echo -e "${RED}❌ Config file does NOT exist${NC}"
    echo "   Run ./cloudflared-setup.sh to create"
fi
echo ""

# Check systemd service
echo "3. Checking cloudflared service..."
if systemctl list-unit-files | grep -q cloudflared.service; then
    echo -e "${GREEN}✅ Service file exists${NC}"
    
    # Check if enabled
    if systemctl is-enabled cloudflared &> /dev/null; then
        echo -e "${GREEN}✅ Service is enabled (auto-start on boot)${NC}"
    else
        echo -e "${YELLOW}⚠️  Service is NOT enabled${NC}"
        echo "   Enable with: sudo systemctl enable cloudflared"
    fi
    
    # Check if running
    if systemctl is-active cloudflared &> /dev/null; then
        echo -e "${GREEN}✅ Service is RUNNING${NC}"
    else
        echo -e "${RED}❌ Service is NOT running${NC}"
        echo "   Start with: sudo systemctl start cloudflared"
    fi
else
    echo -e "${RED}❌ Service file does NOT exist${NC}"
    echo "   Run ./cloudflared-setup.sh to create"
fi
echo ""

# Check Python dependencies
echo "4. Checking Python dependencies..."
if python3 -c "import streamlit" &> /dev/null; then
    echo -e "${GREEN}✅ Streamlit is installed${NC}"
else
    echo -e "${RED}❌ Streamlit is NOT installed${NC}"
    echo "   Install with: pip install -r requirements.txt"
fi

if python3 -c "import psutil" &> /dev/null; then
    echo -e "${GREEN}✅ psutil is installed${NC}"
else
    echo -e "${RED}❌ psutil is NOT installed${NC}"
    echo "   Install with: pip install -r requirements.txt"
fi
echo ""

# Check ports
echo "5. Checking port availability..."
if lsof -Pi :8502 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Port 8502 is IN USE${NC}"
    echo "   Process: $(lsof -Pi :8502 -sTCP:LISTEN | tail -n +2 | awk '{print $1, $2}')"
else
    echo -e "${GREEN}✅ Port 8502 is available (for launcher)${NC}"
fi

if lsof -Pi :8501 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Port 8501 is IN USE${NC}"
    echo "   Process: $(lsof -Pi :8501 -sTCP:LISTEN | tail -n +2 | awk '{print $1, $2}')"
else
    echo -e "${GREEN}✅ Port 8501 is available (for main app)${NC}"
fi
echo ""

# Check environment variables
echo "6. Checking environment variables..."
if [ -z "$LAUNCHER_PASSWORD" ]; then
    echo -e "${YELLOW}⚠️  LAUNCHER_PASSWORD is NOT set${NC}"
    echo "   Set with: export LAUNCHER_PASSWORD='your_password'"
else
    echo -e "${GREEN}✅ LAUNCHER_PASSWORD is set${NC}"
fi

if [ -z "$LAUNCHER_URL" ]; then
    echo -e "${YELLOW}⚠️  LAUNCHER_URL is NOT set (will use localhost)${NC}"
    echo "   Set with: export LAUNCHER_URL='https://launcher.your-domain.com'"
else
    echo -e "${GREEN}✅ LAUNCHER_URL is set: $LAUNCHER_URL${NC}"
fi

if [ -z "$MAIN_APP_URL" ]; then
    echo -e "${YELLOW}⚠️  MAIN_APP_URL is NOT set (will use localhost)${NC}"
    echo "   Set with: export MAIN_APP_URL='https://rag-app.your-domain.com'"
else
    echo -e "${GREEN}✅ MAIN_APP_URL is set: $MAIN_APP_URL${NC}"
fi
echo ""

# Check uv
echo "7. Checking uv (for running the main app)..."
if command -v uv &> /dev/null; then
    echo -e "${GREEN}✅ uv is installed${NC}"
    uv --version
else
    echo -e "${RED}❌ uv is NOT installed${NC}"
    echo "   Install from: https://github.com/astral-sh/uv"
fi
echo ""

# Check main app
echo "8. Checking main RAG app..."
if [ -f "/home/he/ai/dev/langgraph/KB_BS_local-rag-he/apps/app_v2_0g.py" ]; then
    echo -e "${GREEN}✅ app_v2_0g.py exists${NC}"
else
    echo -e "${RED}❌ app_v2_0g.py NOT found${NC}"
fi
echo ""

# Summary
echo "======================================================"
echo "📊 Setup Summary"
echo "======================================================"
echo ""
echo "Next steps:"
echo ""

if ! command -v cloudflared &> /dev/null; then
    echo "1️⃣  Install cloudflared"
fi

if [ ! -f "$HOME/.cloudflared/config.yml" ] || grep -q "YOUR_TUNNEL_ID_HERE" "$HOME/.cloudflared/config.yml"; then
    echo "2️⃣  Run ./cloudflared-setup.sh"
fi

if ! systemctl is-active cloudflared &> /dev/null; then
    echo "3️⃣  Start tunnel: sudo systemctl start cloudflared"
fi

if [ -z "$LAUNCHER_PASSWORD" ]; then
    echo "4️⃣  Set password: export LAUNCHER_PASSWORD='your_password'"
fi

echo "5️⃣  Start launcher: ./start-launcher.sh"
echo "6️⃣  Access via: https://launcher.your-domain.com"
echo ""
