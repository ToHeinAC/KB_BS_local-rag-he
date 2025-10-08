#!/bin/bash

# Stop all BrAIn tunnel processes

echo "🛑 Stopping BrAIn-nw1 Tunnels..."
echo ""

# Read PIDs from file if available
if [ -f /tmp/brain-tunnel-urls.txt ]; then
    source /tmp/brain-tunnel-urls.txt
    
    echo "Stopping processes..."
    kill $LAUNCHER_PID 2>/dev/null && echo "  ✅ Stopped launcher (PID: $LAUNCHER_PID)" || true
    kill $TUNNEL1_PID 2>/dev/null && echo "  ✅ Stopped launcher tunnel (PID: $TUNNEL1_PID)" || true
    kill $TUNNEL2_PID 2>/dev/null && echo "  ✅ Stopped app tunnel (PID: $TUNNEL2_PID)" || true
fi

# Kill by port (backup method)
echo ""
echo "Cleaning up ports..."
lsof -ti:8502 | xargs -r kill -9 2>/dev/null && echo "  ✅ Cleaned port 8502" || true
lsof -ti:8501 | xargs -r kill -9 2>/dev/null && echo "  ✅ Cleaned port 8501" || true

# Kill any cloudflared quick tunnels
echo ""
echo "Stopping cloudflared tunnels..."
pkill -f "cloudflared tunnel --url" && echo "  ✅ Stopped cloudflared tunnels" || true

echo ""
echo "✅ All stopped!"
