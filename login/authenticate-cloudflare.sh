#!/bin/bash

# Cloudflare Authentication Helper
# Run this to authenticate with Cloudflare

set -e

echo "🔐 Cloudflare Tunnel Authentication"
echo "===================================="
echo ""

# Check if already authenticated
if [ -f ~/.cloudflared/cert.pem ]; then
    echo "✅ Already authenticated!"
    echo "   Certificate found at: ~/.cloudflared/cert.pem"
    echo ""
    echo "To re-authenticate (optional):"
    echo "  rm ~/.cloudflared/cert.pem"
    echo "  cloudflared tunnel login"
    echo ""
    exit 0
fi

# Create directory if it doesn't exist
mkdir -p ~/.cloudflared

echo "🌐 Starting authentication process..."
echo ""
echo "📋 IMPORTANT INSTRUCTIONS:"
echo "   1. A browser window will open (or a URL will be shown)"
echo "   2. Login with your Cloudflare account"
echo "   3. Click 'Authorize' on the permission page"
echo "   4. Wait for 'success' message in browser"
echo "   5. Certificate will be saved automatically"
echo ""
echo "⚠️  DO NOT press Ctrl+C - let the process complete!"
echo ""
read -p "Press Enter to start authentication..." 

# Run authentication
cloudflared tunnel login

# Check if successful
if [ -f ~/.cloudflared/cert.pem ]; then
    echo ""
    echo "======================================"
    echo "✅ Authentication Successful!"
    echo "======================================"
    echo ""
    echo "Certificate saved to: ~/.cloudflared/cert.pem"
    echo ""
    echo "Next step:"
    echo "  ./setup-persistent-tunnel.sh"
    echo ""
else
    echo ""
    echo "❌ Authentication failed - cert.pem not found"
    echo ""
    echo "Possible issues:"
    echo "  1. You cancelled the process (don't press Ctrl+C)"
    echo "  2. Browser didn't open (try copying the URL manually)"
    echo "  3. You didn't click 'Authorize' in browser"
    echo ""
    echo "Try again:"
    echo "  ./authenticate-cloudflare.sh"
    echo ""
    exit 1
fi
